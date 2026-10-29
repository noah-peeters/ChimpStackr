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
import src.algorithms.stacking_algorithms.ocl as OCL
from src.config import AlgorithmConfig

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

    def _align_and_stack_laplacian(self, signals=None, progress_callback=None):
        self.Algorithm.reset_cancel()
        self.Algorithm.alignment_shifts = []
        # Sequential (breathing) alignment adds an estimation phase up front
        if not self._prepare_alignment(signals, progress_callback, total_mult=2):
            return
        if self._gpu_run(True, signals, progress_callback):
            return

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
        workers = max(2, self._frame_workers(first_image.shape, self.PYRAMID_BYTES_PER_PIXEL))
        del first_image
        new_pyr = None

        def _load_and_pyramid(key):
            """Load (and align) and build the pyramid in one worker call."""
            img = load_fn(key)
            pyr = self.Algorithm.generate_laplacian_pyramid(img, self.pyramid_num_levels)
            del img
            return pyr

        try:
            # Workers pre-compute load + pyramid for the next images while the
            # main thread fuses the current pair. cv2 ops release the GIL.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending = {}
                lookahead = workers + 1
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

    def _stack_laplacian(self, signals=None, progress_callback=None):
        self.Algorithm.reset_cancel()
        if self._gpu_run(False, signals, progress_callback):
            return
        fused = self._laplacian_core(
            self.Algorithm.load_image(self.image_paths[0]), self.Algorithm.load_image,
            self.image_paths, signals, progress_callback,
        )
        if fused is not None:
            self.output_image = CPU.local_tone_map(fused, strength=0.3)

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

    # Peak memory of one in-flight frame: loaded/aligned image plus its
    # prepared pyramids, weights and variances (~1.33x for all levels)
    HYBRID_BYTES_PER_PIXEL = 80
    PYRAMID_BYTES_PER_PIXEL = 40   # aligned image + its pyramid

    @staticmethod
    def _frame_workers(shape, bytes_per_pixel, cap=6):
        """Worker threads for per-frame preparation: about half the CPU
        threads, capped so in-flight frames fit in half the available RAM."""
        workers = max(1, min(cap, (os.cpu_count() or 4) // 2))
        avail = algorithms._available_memory_bytes()
        if avail is not None:
            per_frame = shape[0] * shape[1] * bytes_per_pixel
            # in flight: up to workers + 1 frames (lookahead) + the one being added
            workers = min(workers, max(1, int(avail * 0.5 // per_frame) - 2))
        return max(1, workers)

    def _hybrid_core(self, first_image, load_fn, keys, signals, progress_callback):
        """Hybrid fusion over keys[0..] (first_image is keys[0]).

        Returns the fused image, or None if cancelled.
        """
        fuser = CPU.HybridFuser(self.pyramid_num_levels, self.fusion_kernel_size)
        frame_shape = first_image.shape
        fuser.add(fuser.prepare(first_image))
        del first_image
        self._emit_progress(signals, progress_callback, 1, len(keys), 0.0)

        # With RAM for 2+ in-flight frames, workers do all per-frame work in
        # parallel (prepare() is independent per frame). Otherwise workers
        # only load/align (cheap on memory) while the main thread prepares,
        # so loading still overlaps with fusion.
        workers = self._frame_workers(frame_shape, self.HYBRID_BYTES_PER_PIXEL)
        prepare_in_workers = workers >= 2
        if not prepare_in_workers:
            workers = 2

        def _load(key):
            img = load_fn(key)
            if not prepare_in_workers:
                return img
            prepared = fuser.prepare(img)
            del img
            return prepared

        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {}
            lookahead = workers + 1
            for j in range(1, min(1 + lookahead, len(keys))):
                pending[j] = pool.submit(_load, keys[j])

            for i in range(1, len(keys)):
                self.Algorithm.wait_if_paused()
                if self.Algorithm.is_cancelled:
                    logger.info("Stacking cancelled")
                    for f in pending.values():
                        f.cancel()
                    return None
                start_time = time.time()

                prepared = pending.pop(i).result() if i in pending else _load(keys[i])
                for j in range(i + 1, min(i + 1 + lookahead, len(keys))):
                    if j not in pending:
                        pending[j] = pool.submit(_load, keys[j])

                if not prepare_in_workers:
                    prepared = fuser.prepare(prepared)
                fuser.add(prepared)
                del prepared
                self._emit_progress(signals, progress_callback, i + 1, len(keys),
                                    time.time() - start_time)

        if self.Algorithm.is_cancelled:
            return None
        return fuser.result()

    def _align_and_stack_hybrid(self, signals=None, progress_callback=None):
        self.Algorithm.reset_cancel()
        self.Algorithm.alignment_shifts = []
        if not self._prepare_alignment(signals, progress_callback, total_mult=2):
            return
        if self._gpu_run(True, signals, progress_callback):
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
        if self._gpu_run(False, signals, progress_callback):
            return
        fused = self._hybrid_core(
            self.Algorithm.load_image(self.image_paths[0]), self.Algorithm.load_image,
            self.image_paths, signals, progress_callback,
        )
        if fused is not None:
            self.output_image = fused

    # ─── GPU (OpenCL) ───
    #
    # Pyramid and Hybrid can run on any OpenCL GPU: worker threads decode
    # frames (and estimate alignment), the GPU warps, decomposes and fuses
    # them, and only the result comes back. Falls back to the CPU path when
    # no GPU is available or anything goes wrong on it.

    def _gpu_wanted(self):
        c = self.config
        if not c.use_gpu or not OCL.available():
            return False
        if c.stacking_method == "laplacian":
            return c.contrast_threshold <= 0   # thresholded focus map is CPU-only
        return c.stacking_method == "hybrid"

    def _gpu_run(self, align, signals, progress_callback):
        """Stack on the GPU. Returns False if the CPU path should run instead."""
        if not self._gpu_wanted():
            return False
        try:
            gpu = OCL.get(self.config.selected_gpu_id)
            logger.info(f"Stacking on GPU: {gpu.name}")
            fused = self._gpu_core(gpu, align, signals, progress_callback)
        except Exception as exc:   # OpenCL/driver errors, out of GPU memory, ...
            logger.warning(f"GPU stacking failed ({exc}); falling back to the CPU")
            self.Algorithm.alignment_shifts = []
            return False
        if fused is not None:
            if self.config.stacking_method == "laplacian":
                fused = CPU.local_tone_map(fused, strength=0.3)
            self.output_image = fused
        return True

    def _gpu_core(self, gpu, align, signals, progress_callback):
        paths = self.image_paths
        load = self.Algorithm.ImageLoadingHandler.read_image_native
        sequential = align and self._breathing_transforms is not None
        mode = self._resolve_alignment_mode()
        ref = load(paths[0])
        if ref is None:
            raise RuntimeError(f"failed to load {paths[0]}")
        # single-reference alignment estimates from float frames, exactly as
        # the CPU path does (the estimators are sensitive to their input)
        ref_f32 = ref.astype(np.float32) if align and not sequential else None
        if self.config.stacking_method == "hybrid":
            fuser = OCL.HybridGPU(gpu, self.pyramid_num_levels, self.fusion_kernel_size)
        else:
            fuser = OCL.PyramidGPU(gpu, self.pyramid_num_levels, self.fusion_kernel_size,
                                   self.config.feather_radius)

        def frame(i):
            """Decode (and, for single-reference alignment, estimate) frame i."""
            img = ref if i == 0 else load(paths[i])
            if img is None:
                raise RuntimeError(f"failed to load {paths[i]}")
            if img.dtype != ref.dtype:
                img = img.astype(np.float32) if ref.dtype == np.float32 else                     np.clip(img, 0, 255).astype(np.uint8)
            img = self.Algorithm._match_dimensions(ref, img)
            M = None
            if sequential:
                M = self._breathing_transforms[i]
            elif align and i > 0:
                M, _ = self.Algorithm.estimate_pair_transform(
                    ref_f32, img.astype(np.float32), alignment_mode=mode,
                    scale_factor=self.config.alignment_scale_factor,
                )
            return img, M

        workers = 4
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {j: pool.submit(frame, j) for j in range(min(workers + 1, len(paths)))}
            for i in range(len(paths)):
                self.Algorithm.wait_if_paused()
                if self.Algorithm.is_cancelled:
                    logger.info("Stacking cancelled")
                    for f in pending.values():
                        f.cancel()
                    return None
                start_time = time.time()
                img, M = pending.pop(i).result()
                j = i + workers + 1
                if j < len(paths):
                    pending[j] = pool.submit(frame, j)
                # breathing: Lanczos + replicated borders, as on the CPU path
                fuser.add(gpu.warp(img, M, lanczos=sequential, replicate=sequential))
                if M is not None:
                    self.Algorithm._track_warp_shifts(M, img.shape)
                del img
                self._emit_progress(signals, progress_callback, i + 1, len(paths),
                                    time.time() - start_time)
        return fuser.result()

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
