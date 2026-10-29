"""
    Exposed API for easily aligning/stacking multiple images.
    Decoupled from Qt — can be used standalone or from CLI.

    Supports three stacking methods:
      - laplacian: Laplacian Pyramid fusion (best for fine detail)
      - weighted_average: Contrast-weighted blending (smooth results)
      - depth_map: Per-pixel selection from sharpest source (best color fidelity)
"""
import os
import tempfile
import time
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

import src.utilities as utilities
import src.algorithms as algorithms
import src.algorithms.stacking_algorithms.cpu as CPU
from src.config import AlgorithmConfig

try:
    import src.algorithms.stacking_algorithms.gpu as _GPU_module
    _HAS_CUPY = _GPU_module.HAS_CUPY
except Exception:
    _HAS_CUPY = False

logger = logging.getLogger(__name__)


class LaplacianPyramid:
    """Main stacking API. Name kept for backward compatibility."""

    def __init__(self, config=None):
        if config is None:
            config = AlgorithmConfig()
        self.config = config
        self.output_image = None
        self.depth_map = None  # Populated by depth_map method
        self.image_paths = []
        self.sequential_log = None   # per-frame chain log (sequential alignment)
        self._breathing_transforms = None
        self._breathing_index = {}   # path → frame index
        self._progress_offset = 0
        self._progress_total = None
        self.Algorithm = algorithms.Algorithm()

    @property
    def fusion_kernel_size(self):
        return self.config.fusion_kernel_size

    @property
    def pyramid_num_levels(self):
        return self.config.pyramid_num_levels

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def apply_gpu_settings(self):
        self.Algorithm.toggle_cpu_gpu(
            self.config.use_gpu, self.config.selected_gpu_id,
        )

    def cancel(self):
        self.Algorithm.cancel()

    def pause(self):
        self.Algorithm.pause()

    def resume(self):
        self.Algorithm.resume()

    def update_image_paths(self, new_image_paths):
        self.image_paths = sorted(new_image_paths, key=utilities.int_string_sorting)

    def get_crop_bounds(self):
        """
        Compute crop rectangle to remove black edges from alignment.
        Works for both translation-only shifts (signed x,y) and
        RST shifts (max corner displacement, always positive).
        Returns (top, bottom, left, right) pixel counts.
        """
        shifts = self.Algorithm.alignment_shifts
        if not shifts:
            return None
        import math
        # For RST, shifts store (max_abs_dx, max_abs_dy) — always positive.
        # For translation, shifts can be negative.
        # Handle both: use max absolute value in each direction.
        max_x = max(abs(s[0]) for s in shifts)
        max_y = max(abs(s[1]) for s in shifts)
        # Apply symmetrically + 1px margin
        crop_x = math.ceil(max_x) + 1
        crop_y = math.ceil(max_y) + 1
        return (crop_y, crop_y, crop_x, crop_x)

    def auto_crop_output(self):
        if self.output_image is None:
            return None
        bounds = self.get_crop_bounds()
        if bounds is None:
            return None
        top, bottom, left, right = bounds
        h, w = self.output_image.shape[:2]
        if top + bottom >= h or left + right >= w:
            return None
        self.output_image = self.output_image[top:h - bottom, left:w - right].copy()
        return bounds

    def _get_reference_index(self):
        strategy = self.config.alignment_reference
        n = len(self.image_paths)
        if strategy == "middle" and n > 0:
            return n // 2
        return 0

    # ─── Sequential (chained) alignment: focus breathing compensation ───

    def _is_sequential(self):
        """True when chaining frame-to-frame estimates instead of one global ref.

        Activated by alignment_mode="breathing", or by the alignment_reference
        "previous" strategy with any non-auto mode.
        """
        mode = self._resolve_alignment_mode()
        if mode == "breathing":
            return True
        if self.config.alignment_reference == "previous":
            return mode in ("translation", "euclidean", "similarity", "affine")
        return False

    def _prepare_alignment(self, signals=None, progress_callback=None, total_mult=2,
                           later_steps=None):
        """Precompute per-frame transforms for sequential (breathing) alignment.

        Pair deltas are estimated in parallel by the Algorithm, then composed
        into a chain anchored at the configured reference frame.

        later_steps: progress steps reported after estimation; defaults to
        (total_mult - 1) * frames. Set up front so the bar stays monotonic.

        Returns False if the stack was cancelled.
        """
        self._progress_offset = 0
        self._progress_total = None
        self._breathing_transforms = None
        self._breathing_index = {}
        self.sequential_log = None

        if not self._is_sequential() or len(self.image_paths) < 2:
            return True

        n = len(self.image_paths)
        if later_steps is None:
            later_steps = (total_mult - 1) * n  # apply (n-1 or n)
        self._progress_total = n + later_steps  # estimation (n) + later phases
        transforms, log = self.Algorithm.estimate_sequential_transforms(
            self.image_paths,
            ref_index=self._get_reference_index(),
            alignment_mode=self._resolve_alignment_mode(),
            scale_factor=self.config.alignment_scale_factor,
            max_scale_step=self.config.breathing_max_scale_step,
            precision="max" if self.config.breathing_high_precision else "default",
            emit=lambda cur, tot, t: self._emit_progress(
                signals, progress_callback, cur, tot, t),
        )
        if transforms is None:
            logger.info("Sequential alignment cancelled")
            return False

        self._breathing_transforms = transforms
        self._breathing_index = {p: i for i, p in enumerate(self.image_paths)}
        self.sequential_log = log
        self._progress_offset = n  # later phases report on top of estimation
        return True

    def _sequential_warp_kwargs(self):
        """Warp options for the sequential path.

        Lanczos + replicate in both precision modes: the resampling, not the
        estimator, accounts for nearly all of the sharpness gain over
        bilinear, at little extra cost (measured on a 50-frame 24MP stack).
        """
        return {"interpolation": cv2.INTER_LANCZOS4,
                "border_mode": cv2.BORDER_REPLICATE}

    def _load_ref_frame(self):
        """Load frame 0, warped into reference space when chaining."""
        img = self.Algorithm.load_image(self.image_paths[0])
        if self._is_sequential() and self._breathing_transforms is not None:
            img = self.Algorithm.apply_transform_matrix(
                img, self._breathing_transforms[0],
                **self._sequential_warp_kwargs(),
            )
        return img

    def _resolve_alignment_mode(self):
        """Resolve alignment mode from config, handling backward compat."""
        mode = self.config.alignment_mode
        if mode != "auto":
            return mode
        # Legacy: align_rotation_scale boolean → similarity (upgraded from old euclidean)
        if self.config.align_rotation_scale:
            return "similarity"
        return "translation"

    def _load_and_align(self, ref_image, path):
        """Load an image and align it to the reference. Returns float32."""
        if self._is_sequential() and self._breathing_transforms is not None:
            # Sequential mode: each frame has its composed transform ready.
            # Transforms were estimated on frames matched to frame 0's
            # dimensions (#29), so match before warping too.
            img = self.Algorithm.load_image(path)
            if img is not None and ref_image is not None:
                img = self.Algorithm._match_dimensions(ref_image, img)
            idx = self._breathing_index.get(path)
            M = self._breathing_transforms[idx] if idx is not None else None
            return self.Algorithm.apply_transform_matrix(
                img, M, **self._sequential_warp_kwargs(),
            )
        return self.Algorithm.align_image_pair(
            ref_image, path,
            scale_factor=self.config.alignment_scale_factor,
            alignment_mode=self._resolve_alignment_mode(),
        )

    # ─── Align + Stack (dispatches to chosen method) ───

    def align_and_stack_images(self, signals=None, progress_callback=None):
        """Align and stack using the configured stacking method."""
        if self._slabbing_active():
            return self._align_and_stack_slabbed(signals, progress_callback)
        method = self.config.stacking_method
        if method == "weighted_average":
            self._align_and_stack_weighted_average(signals, progress_callback)
        elif method == "depth_map":
            self._align_and_stack_depthmap(signals, progress_callback)
        elif method == "exposure_fusion":
            self._align_and_stack_exposure(signals, progress_callback)
        elif method == "hybrid":
            self._align_and_stack_hybrid(signals, progress_callback)
        else:
            self._align_and_stack_laplacian(signals, progress_callback)

    def stack_images(self, signals=None, progress_callback=None):
        """Stack without alignment using the configured method."""
        # No alignment phase: clear progress offsets left by a previous run
        self._progress_offset = 0
        self._progress_total = None
        if self._slabbing_active():
            return self._stack_slabbed(signals, progress_callback)
        method = self.config.stacking_method
        if method == "weighted_average":
            self._stack_weighted_average(signals, progress_callback)
        elif method == "depth_map":
            self._stack_depthmap(signals, progress_callback)
        elif method == "exposure_fusion":
            self._stack_exposure(signals, progress_callback)
        elif method == "hybrid":
            self._stack_hybrid(signals, progress_callback)
        else:
            self._stack_laplacian(signals, progress_callback)

    # ─── Laplacian Pyramid Method ───

    def _can_use_cupy_path(self):
        """Check if the fully GPU-resident CuPy path can be used."""
        return _HAS_CUPY and self.config.use_gpu

    def _align_and_stack_laplacian(self, signals=None, progress_callback=None):
        self.apply_gpu_settings()
        self.Algorithm.reset_cancel()
        self.Algorithm.alignment_shifts = []

        # Use fully GPU-resident path when CuPy is available
        use_cupy = self._can_use_cupy_path()
        # Sequential (breathing) alignment adds an estimation phase up front
        if not self._prepare_alignment(signals, progress_callback,
                                       total_mult=3 if use_cupy else 2):
            return

        if use_cupy:
            logger.info("Using CuPy GPU-resident pipeline (use_gpu=True, CuPy available)")
            return self._align_and_stack_laplacian_cupy(signals, progress_callback)
        else:
            logger.info(f"Using CPU pipeline (use_gpu={self.config.use_gpu}, CuPy={_HAS_CUPY})")

        ref_image = self._load_ref_frame()
        fused = self._laplacian_core(
            ref_image, lambda path: self._load_and_align(ref_image, path),
            self.image_paths, signals, progress_callback,
        )
        if fused is not None:
            self.output_image = CPU.local_tone_map(fused, strength=0.3)

    def _laplacian_core(self, first_image, load_fn, keys, signals, progress_callback):
        """CPU pairwise pyramid fusion over keys[0..] (first_image is keys[0]).

        Returns the fused image before tone mapping, or None if cancelled.
        """
        fused_pyr = self.Algorithm.generate_laplacian_pyramid(
            first_image, self.pyramid_num_levels
        )
        del first_image
        new_pyr = None

        def _load_and_pyramid(key):
            """Load (and align) and build the pyramid in one worker call."""
            img = load_fn(key)
            pyr = self.Algorithm.generate_laplacian_pyramid(img, self.pyramid_num_levels)
            del img
            return pyr

        try:
            # Use 2 workers: pre-compute load + pyramid for the next images
            # while the main thread fuses the current pair. cv2 ops release GIL.
            with ThreadPoolExecutor(max_workers=2) as pool:
                pending = {}
                lookahead = 3
                for j in range(1, min(1 + lookahead, len(keys))):
                    pending[j] = pool.submit(_load_and_pyramid, keys[j])

                for i in range(1, len(keys)):
                    self.Algorithm.wait_if_paused()
                    if self.Algorithm.is_cancelled:
                        logger.info("Stacking cancelled")
                        for f in pending.values():
                            f.cancel()
                        return None

                    start_time = time.time()

                    if i in pending:
                        new_pyr = pending.pop(i).result()
                    else:
                        new_pyr = _load_and_pyramid(keys[i])

                    # Refill lookahead
                    for j in range(i + 1, min(i + 1 + lookahead, len(keys))):
                        if j not in pending:
                            pending[j] = pool.submit(_load_and_pyramid, keys[j])

                    fused_pyr = self.Algorithm.focus_fuse_pyramid_pair(
                        fused_pyr, new_pyr, self.fusion_kernel_size,
                        self.config.contrast_threshold, self.config.feather_radius,
                    )
                    del new_pyr
                    new_pyr = None
                    elapsed = time.time() - start_time
                    self._emit_progress(signals, progress_callback, i + 1, len(keys), elapsed)
        finally:
            del new_pyr

        if self.Algorithm.is_cancelled:
            return None
        return self.Algorithm.reconstruct_pyramid(fused_pyr)

    def _align_and_stack_laplacian_cupy(self, signals=None, progress_callback=None):
        """Two-phase GPU pipeline: align ALL on CPU, then stream-fuse on GPU.

        Phase 1: Align all images using ALL CPU cores in parallel.
                 Saturates CPU — all cores busy with alignment.
        Phase 2: Stream aligned images through GPU one by one for fusion.
                 GPU runs at ~100ms/image, continuously fed from RAM.

        Separated phases avoid CPU↔GPU contention and maximize both.
        """
        import cupy as cp
        GPU = _GPU_module
        import os

        paths = self.image_paths
        n = len(paths)
        n_workers = min(max(4, os.cpu_count() // 2), 12)
        logger.info(f"[CuPy GPU] Two-phase pipeline: {n_workers} CPU workers, then GPU stream")
        t_total = time.time()

        # ── Phase 1: Align ALL images on CPU (parallel) ──
        t_align = time.time()
        ref_image = self._load_ref_frame()
        aligned_images = [ref_image]

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for j in range(1, n):
                futures[j] = pool.submit(self._load_and_align, ref_image, paths[j])

            t_last = time.time()
            for i in range(1, n):
                self.Algorithm.wait_if_paused()
                if self.Algorithm.is_cancelled:
                    logger.info("Stacking cancelled during alignment")
                    for f in futures.values():
                        f.cancel()
                    return

                aligned = futures[i].result()
                aligned_images.append(aligned)
                t_now = time.time()
                per_image_time = t_now - t_last
                t_last = t_now
                logger.info(f"[CuPy GPU] Aligned {i+1}/{n} ({t_now - t_align:.1f}s elapsed)")
                # Report progress over total pipeline (n items align + n items fuse)
                # Pass per-image time so TimeRemainingHandler estimates correctly
                self._emit_progress(signals, progress_callback, i + 1, n * 2, per_image_time)

        align_time = time.time() - t_align
        logger.info(f"[CuPy GPU] Phase 1 (align): {align_time:.2f}s "
                     f"({align_time/max(n-1,1):.3f}s/image avg)")
        del ref_image

        if self.Algorithm.is_cancelled:
            return

        # ── Phase 2: Stream-fuse on GPU (pairwise, all data in RAM already) ──
        t_gpu = time.time()
        GPU._cupy_warmup()

        img0_gpu = cp.asarray(aligned_images[0])
        fused_pyr = GPU._cupy_laplacian_pyramid(img0_gpu, self.pyramid_num_levels)
        del img0_gpu

        t_last_gpu = time.time()
        for i in range(1, n):
            if self.Algorithm.is_cancelled:
                return
            img_gpu = cp.asarray(aligned_images[i])
            aligned_images[i] = None  # free RAM as we go
            new_pyr = GPU._cupy_laplacian_pyramid(img_gpu, self.pyramid_num_levels)
            del img_gpu
            fused_pyr = GPU._cupy_fuse_pyramid_pair(
                fused_pyr, new_pyr, self.fusion_kernel_size,
                self.config.contrast_threshold, self.config.feather_radius
            )
            del new_pyr
            t_now = time.time()
            self._emit_progress(signals, progress_callback, n + i, n * 2,
                                t_now - t_last_gpu)
            t_last_gpu = t_now

        cp.cuda.Stream.null.synchronize()
        del aligned_images
        gpu_time = time.time() - t_gpu
        logger.info(f"[CuPy GPU] Phase 2 (GPU fuse): {gpu_time:.2f}s "
                     f"({gpu_time/max(n-1,1):.3f}s/image)")

        # ── Reconstruct ──
        t_recon = time.time()
        result_gpu = GPU._cupy_reconstruct(fused_pyr)
        self.output_image = cp.asnumpy(result_gpu)
        del result_gpu, fused_pyr
        self.output_image = CPU.local_tone_map(self.output_image, strength=0.3)
        logger.info(f"[CuPy GPU] Reconstruct+tonemap: {time.time()-t_recon:.3f}s")
        logger.info(f"[CuPy GPU] Total: {time.time()-t_total:.2f}s "
                     f"(align={align_time:.1f}s + gpu={gpu_time:.1f}s)")

    def _stack_laplacian(self, signals=None, progress_callback=None):
        self.apply_gpu_settings()
        self.Algorithm.reset_cancel()

        # Use fully GPU-resident path when CuPy is available
        if self._can_use_cupy_path():
            logger.info("Using CuPy GPU-resident pipeline (stack-only)")
            return self._stack_laplacian_cupy(signals, progress_callback)
        else:
            logger.info(f"Using CPU pipeline (use_gpu={self.config.use_gpu}, CuPy={_HAS_CUPY})")

        fused = self._laplacian_core(
            self.Algorithm.load_image(self.image_paths[0]), self.Algorithm.load_image,
            self.image_paths, signals, progress_callback,
        )
        if fused is not None:
            self.output_image = CPU.local_tone_map(fused, strength=0.3)

    def _stack_laplacian_cupy(self, signals=None, progress_callback=None):
        """GPU-resident stacking with parallel image loading."""
        import cupy as cp
        GPU = _GPU_module
        import os

        paths = self.image_paths
        n = len(paths)
        n_workers = min(max(3, os.cpu_count() // 3), 8)
        logger.info(f"[CuPy GPU] Pipelined stack: {n_workers} load workers")
        t_start = time.time()
        GPU._cupy_warmup()

        im0 = self.Algorithm.load_image(paths[0])
        img0_gpu = cp.asarray(im0)
        fused_pyr = GPU._cupy_laplacian_pyramid(img0_gpu, self.pyramid_num_levels)
        del im0, img0_gpu

        try:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                pending = {}
                lookahead = n_workers + 1
                for j in range(1, min(1 + lookahead, n)):
                    pending[j] = pool.submit(self.Algorithm.load_image, paths[j])

                for i in range(1, n):
                    self.Algorithm.wait_if_paused()
                    if self.Algorithm.is_cancelled:
                        for f in pending.values():
                            f.cancel()
                        return

                    start_time = time.time()

                    if i in pending:
                        im1 = pending.pop(i).result()
                    else:
                        im1 = self.Algorithm.load_image(paths[i])

                    for j in range(i + 1, min(i + 1 + lookahead, n)):
                        if j not in pending:
                            pending[j] = pool.submit(self.Algorithm.load_image, paths[j])

                    t_gpu = time.time()
                    img_gpu = cp.asarray(im1)
                    del im1
                    new_pyr = GPU._cupy_laplacian_pyramid(img_gpu, self.pyramid_num_levels)
                    del img_gpu
                    fused_pyr = GPU._cupy_fuse_pyramid_pair(
                        fused_pyr, new_pyr, self.fusion_kernel_size,
                        self.config.contrast_threshold, self.config.feather_radius
                    )
                    del new_pyr
                    cp.cuda.Stream.null.synchronize()

                    elapsed = time.time() - start_time
                    gpu_elapsed = time.time() - t_gpu
                    logger.info(f"[CuPy GPU] Image {i+1}/{n}: "
                                f"total={elapsed:.3f}s (load={elapsed-gpu_elapsed:.3f}s, "
                                f"gpu={gpu_elapsed:.3f}s)")
                    self._emit_progress(signals, progress_callback, i + 1, n, elapsed)
        finally:
            pass

        if self.Algorithm.is_cancelled:
            return

        t_recon = time.time()
        result_gpu = GPU._cupy_reconstruct(fused_pyr)
        self.output_image = cp.asnumpy(result_gpu)
        del result_gpu, fused_pyr
        self.output_image = CPU.local_tone_map(self.output_image, strength=0.3)
        logger.info(f"[CuPy GPU] Reconstruct+tonemap: {time.time()-t_recon:.3f}s")
        logger.info(f"[CuPy GPU] Total: {time.time()-t_start:.2f}s")

    # ─── Slabbing ───
    #
    # Deep stacks are fused in overlapping chunks ("slabs"), then the slab
    # results are fused with the same method. Every frame is aligned once to
    # the global reference, so slabs need no second alignment/resample. Slab
    # results go to temp files to keep memory flat for very deep stacks.

    SLAB_METHODS = ("laplacian", "hybrid", "weighted_average", "depth_map")

    @staticmethod
    def slab_groups(n, size, overlap):
        """Index ranges of overlapping slabs covering frames 0..n-1."""
        size = max(2, int(size))
        overlap = min(max(0, int(overlap)), size - 1)
        groups, start = [], 0
        while True:
            end = min(start + size, n)
            groups.append(list(range(start, end)))
            if end >= n:
                return groups
            start = end - overlap

    def _slabbing_active(self):
        return (self.config.slab_size >= 2
                and self.config.stacking_method in self.SLAB_METHODS
                and len(self.image_paths) > self.config.slab_size)

    def _fuse_keys(self, first_image, load_fn, keys, signals, progress_callback):
        """Fuse keys with the configured method's CPU core; returns the image."""
        method = self.config.stacking_method
        if method == "hybrid":
            return self._hybrid_core(first_image, load_fn, keys, signals, progress_callback)
        if method == "laplacian":
            return self._laplacian_core(first_image, load_fn, keys, signals, progress_callback)

        def image_iter():
            yield 0, first_image
            for i in range(1, len(keys)):
                yield i, load_fn(keys[i])

        self.output_image = None
        core = self._weighted_avg_core if method == "weighted_average" else self._depthmap_core
        core(image_iter(), len(keys), signals, progress_callback)
        return self.output_image

    def _slab_plan(self):
        """(slab groups, progress steps: every slab frame + the final pass)."""
        groups = self.slab_groups(len(self.image_paths), self.config.slab_size,
                                  self.config.slab_overlap)
        return groups, sum(len(g) for g in groups) + len(groups)

    def _run_slabs(self, first_image, load_fn, signals, progress_callback):
        groups, steps = self._slab_plan()
        logger.info(f"Slabbing: {len(self.image_paths)} frames in {len(groups)} slabs")
        base = self._progress_offset   # > 0 after a sequential alignment phase
        self._progress_total = base + steps
        self.output_image = None
        fused = None
        with tempfile.TemporaryDirectory(prefix="chimpstackr_slabs_") as tmp:
            slab_files = []
            done = 0
            for s, group in enumerate(groups):
                keys = [self.image_paths[k] for k in group]
                first = first_image if group[0] == 0 else load_fn(keys[0])
                self._progress_offset = base + done
                slab = self._fuse_keys(first, load_fn, keys, signals, progress_callback)
                del first
                if slab is None:
                    return
                path = os.path.join(tmp, f"slab_{s:03d}.npy")
                np.save(path, slab.astype(np.float32, copy=False))
                slab_files.append(path)
                del slab
                done += len(group)
            first_image = None

            self._progress_offset = base + done
            fused = self._fuse_keys(np.load(slab_files[0]), np.load, slab_files,
                                    signals, progress_callback)
        if fused is None:
            return
        if self.config.stacking_method == "laplacian":
            fused = CPU.local_tone_map(fused, strength=0.3)
        self.output_image = fused

    def _align_and_stack_slabbed(self, signals=None, progress_callback=None):
        self.apply_gpu_settings()
        self.Algorithm.reset_cancel()
        self.Algorithm.alignment_shifts = []
        if not self._prepare_alignment(signals, progress_callback,
                                       later_steps=self._slab_plan()[1]):
            return
        ref_image = self._load_ref_frame()
        self._run_slabs(ref_image, lambda path: self._load_and_align(ref_image, path),
                        signals, progress_callback)

    def _stack_slabbed(self, signals=None, progress_callback=None):
        self.Algorithm.reset_cancel()
        self._run_slabs(self.Algorithm.load_image(self.image_paths[0]),
                        self.Algorithm.load_image, signals, progress_callback)

    # ─── Hybrid Method ───
    #
    # Pyramid detail on the fine levels, focus-weighted blend on the coarse
    # levels (see CPU.HybridFuser). Single pass; frames are loaded, aligned
    # and decomposed in worker threads while the main thread accumulates.

    def _hybrid_core(self, first_image, load_fn, keys, signals, progress_callback):
        """Hybrid fusion over keys[0..] (first_image is keys[0]).

        Returns the fused image, or None if cancelled.
        """
        if self.config.use_gpu:
            logger.info("Hybrid stacking runs on the CPU")
        fuser = CPU.HybridFuser(self.pyramid_num_levels, self.fusion_kernel_size)
        fuser.add(*fuser.prepare(first_image))
        del first_image
        self._emit_progress(signals, progress_callback, 1, len(keys), 0.0)

        def _prepare(key):
            img = load_fn(key)
            prepared = fuser.prepare(img)
            del img
            return prepared

        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = {}
            lookahead = 3
            for j in range(1, min(1 + lookahead, len(keys))):
                pending[j] = pool.submit(_prepare, keys[j])

            for i in range(1, len(keys)):
                self.Algorithm.wait_if_paused()
                if self.Algorithm.is_cancelled:
                    logger.info("Stacking cancelled")
                    for f in pending.values():
                        f.cancel()
                    return None
                start_time = time.time()

                prepared = pending.pop(i).result() if i in pending else _prepare(keys[i])
                for j in range(i + 1, min(i + 1 + lookahead, len(keys))):
                    if j not in pending:
                        pending[j] = pool.submit(_prepare, keys[j])

                fuser.add(*prepared)
                del prepared
                self._emit_progress(signals, progress_callback, i + 1, len(keys),
                                    time.time() - start_time)

        if self.Algorithm.is_cancelled:
            return None
        return fuser.result()

    def _align_and_stack_hybrid(self, signals=None, progress_callback=None):
        self.apply_gpu_settings()
        self.Algorithm.reset_cancel()
        self.Algorithm.alignment_shifts = []
        if not self._prepare_alignment(signals, progress_callback, total_mult=2):
            return
        ref_image = self._load_ref_frame()
        fused = self._hybrid_core(
            ref_image, lambda path: self._load_and_align(ref_image, path),
            self.image_paths, signals, progress_callback,
        )
        if fused is not None:
            self.output_image = fused

    def _stack_hybrid(self, signals=None, progress_callback=None):
        self.Algorithm.reset_cancel()
        fused = self._hybrid_core(
            self.Algorithm.load_image(self.image_paths[0]), self.Algorithm.load_image,
            self.image_paths, signals, progress_callback,
        )
        if fused is not None:
            self.output_image = fused

    # ─── Weighted Average Method ───
    #
    # Correct approach: accumulate weighted_sum and weight_total across all
    # images, then divide once at the end. This avoids the compounding blur
    # from pairwise blending of already-blended results.

    def _weighted_avg_core(self, image_iter, total, signals, progress_callback):
        """Core weighted average: accumulate weights and weighted pixels."""
        weighted_sum = None
        weight_total = None

        for i, img in image_iter:
            self.Algorithm.wait_if_paused()
            if self.Algorithm.is_cancelled:
                return
            start_time = time.time()

            w = CPU.compute_focus_weights(img, self.fusion_kernel_size)
            w3 = w[:, :, np.newaxis] if img.ndim == 3 else w

            if weighted_sum is None:
                weighted_sum = img.astype(np.float64) * w3
                weight_total = w.astype(np.float64)
            else:
                weighted_sum += img.astype(np.float64) * w3
                weight_total += w.astype(np.float64)
            del img

            elapsed = time.time() - start_time
            self._emit_progress(signals, progress_callback, i + 1, total, elapsed)

        if self.Algorithm.is_cancelled or weighted_sum is None:
            return

        # Divide once: proper weighted average
        denom = weight_total + 1e-12
        if weighted_sum.ndim == 3:
            denom = denom[:, :, np.newaxis]
        self.output_image = (weighted_sum / denom).astype(np.float32)

    def _align_and_stack_weighted_average(self, signals=None, progress_callback=None):
        self.apply_gpu_settings()
        self.Algorithm.reset_cancel()
        self.Algorithm.alignment_shifts = []
        if not self._prepare_alignment(signals, progress_callback, total_mult=2):
            return

        ref_image = self._load_ref_frame()

        def image_iter():
            yield 0, ref_image
            for i, path in enumerate(self.image_paths):
                if i == 0:
                    continue
                aligned = self._load_and_align(ref_image, path)
                yield i, aligned

        self._weighted_avg_core(image_iter(), len(self.image_paths), signals, progress_callback)

    def _stack_weighted_average(self, signals=None, progress_callback=None):
        self.Algorithm.reset_cancel()

        def image_iter():
            for i, path in enumerate(self.image_paths):
                img = self.Algorithm.load_image(path)
                if img is not None:
                    yield i, img

        self._weighted_avg_core(image_iter(), len(self.image_paths), signals, progress_callback)

    # ─── Depth Map Method ───
    #
    # Correct approach: for each pixel, keep the pixel from whichever
    # *original* source image has the highest sharpness. Since we compare
    # each fresh source against the running best (not a blended result),
    # this is correct incrementally.

    def _depthmap_core(self, image_iter, total, signals, progress_callback):
        """Core depth map: per-pixel keep sharpest source."""
        result = None
        best_sharpness = None

        for i, img in image_iter:
            self.Algorithm.wait_if_paused()
            if self.Algorithm.is_cancelled:
                return
            start_time = time.time()

            sharpness = CPU.compute_multires_sharpness(
                img,
                scales=(5, self.fusion_kernel_size | 1, self.fusion_kernel_size * 3 + 1),
                smoothing=self.config.depthmap_smoothing,
            )

            if result is None:
                result = img.copy()
                best_sharpness = sharpness
            else:
                mask = sharpness > best_sharpness
                if img.ndim == 3:
                    mask_3ch = mask[:, :, np.newaxis]
                else:
                    mask_3ch = mask
                result = np.where(mask_3ch, img, result)
                best_sharpness = np.maximum(best_sharpness, sharpness)
            del img

            elapsed = time.time() - start_time
            self._emit_progress(signals, progress_callback, i + 1, total, elapsed)

        if self.Algorithm.is_cancelled or result is None:
            return
        self.output_image = result.astype(np.float32)
        self.depth_map = None

    def _align_and_stack_depthmap(self, signals=None, progress_callback=None):
        self.apply_gpu_settings()
        self.Algorithm.reset_cancel()
        self.Algorithm.alignment_shifts = []
        if not self._prepare_alignment(signals, progress_callback, total_mult=2):
            return

        ref_image = self._load_ref_frame()

        def image_iter():
            yield 0, ref_image
            for i, path in enumerate(self.image_paths):
                if i == 0:
                    continue
                aligned = self._load_and_align(ref_image, path)
                yield i, aligned

        self._depthmap_core(image_iter(), len(self.image_paths), signals, progress_callback)

    def _stack_depthmap(self, signals=None, progress_callback=None):
        self.Algorithm.reset_cancel()

        def image_iter():
            for i, path in enumerate(self.image_paths):
                img = self.Algorithm.load_image(path)
                if img is not None:
                    yield i, img

        self._depthmap_core(image_iter(), len(self.image_paths), signals, progress_callback)

    # ─── Exposure Fusion (HDR/Mertens) ───
    #
    # For HDR/exposure blending, not focus stacking.
    # Uses batched Mertens (groups of 4) for memory efficiency.

    EXPOSURE_BATCH_SIZE = 4

    def _exposure_core(self, image_iter, total, signals, progress_callback):
        batch = []
        intermediates = []
        for i, img in image_iter:
            self.Algorithm.wait_if_paused()
            if self.Algorithm.is_cancelled:
                return
            start_time = time.time()
            batch.append(img)
            if len(batch) >= self.EXPOSURE_BATCH_SIZE:
                intermediates.append(CPU.mertens_fuse_batch(batch))
                batch = []
            elapsed = time.time() - start_time
            self._emit_progress(signals, progress_callback, i + 1, total, elapsed)

        if self.Algorithm.is_cancelled:
            return
        if batch:
            if len(batch) == 1:
                intermediates.append(batch[0].astype(np.float32))
            else:
                intermediates.append(CPU.mertens_fuse_batch(batch))
        if not intermediates:
            return
        if len(intermediates) == 1:
            self.output_image = intermediates[0]
        else:
            self.output_image = CPU.mertens_fuse_batch(intermediates)

    def _align_and_stack_exposure(self, signals=None, progress_callback=None):
        self.apply_gpu_settings()
        self.Algorithm.reset_cancel()
        self.Algorithm.alignment_shifts = []
        if not self._prepare_alignment(signals, progress_callback, total_mult=2):
            return

        ref_image = self._load_ref_frame()

        def image_iter():
            yield 0, ref_image
            for i, path in enumerate(self.image_paths):
                if i == 0:
                    continue
                yield i, self._load_and_align(ref_image, path)

        self._exposure_core(image_iter(), len(self.image_paths), signals, progress_callback)

    def _stack_exposure(self, signals=None, progress_callback=None):
        self.Algorithm.reset_cancel()

        def image_iter():
            for i, path in enumerate(self.image_paths):
                img = self.Algorithm.load_image(path)
                if img is not None:
                    yield i, img

        self._exposure_core(image_iter(), len(self.image_paths), signals, progress_callback)

    # ─── Progress ───

    def _emit_progress(self, signals, progress_callback, current, total, time_taken):
        # Sequential alignment has extra phases (estimation); progress totals
        # and offsets are adjusted so the bar stays monotonic.
        current += self._progress_offset
        total = self._progress_total if self._progress_total is not None else total
        if signals is not None:
            signals.finished_inter_task.emit(
                ["finished_image", current, total, time_taken]
            )
        if progress_callback is not None:
            progress_callback(current, total, time_taken)
