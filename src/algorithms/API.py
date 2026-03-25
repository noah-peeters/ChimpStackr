"""
    Exposed API for easily aligning/stacking multiple images.
    Decoupled from Qt — can be used standalone or from CLI.

    Supports three stacking methods:
      - laplacian: Laplacian Pyramid fusion (best for fine detail)
      - weighted_average: Contrast-weighted blending (smooth results)
      - depth_map: Per-pixel selection from sharpest source (best color fidelity)
"""
import time
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
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
        if strategy == "middle":
            return len(self.image_paths) // 2
        return 0

    def _load_and_align(self, ref_image, path):
        """Load an image and align it to the reference. Returns float32."""
        return self.Algorithm.align_image_pair(
            ref_image, path,
            scale_factor=self.config.alignment_scale_factor,
            use_rst=self.config.align_rotation_scale,
        )

    # ─── Align + Stack (dispatches to chosen method) ───

    def align_and_stack_images(self, signals=None, progress_callback=None):
        """Align and stack using the configured stacking method."""
        method = self.config.stacking_method
        if method == "weighted_average":
            self._align_and_stack_weighted_average(signals, progress_callback)
        elif method == "depth_map":
            self._align_and_stack_depthmap(signals, progress_callback)
        elif method == "exposure_fusion":
            self._align_and_stack_exposure(signals, progress_callback)
        else:
            self._align_and_stack_laplacian(signals, progress_callback)

    def stack_images(self, signals=None, progress_callback=None):
        """Stack without alignment using the configured method."""
        method = self.config.stacking_method
        if method == "weighted_average":
            self._stack_weighted_average(signals, progress_callback)
        elif method == "depth_map":
            self._stack_depthmap(signals, progress_callback)
        elif method == "exposure_fusion":
            self._stack_exposure(signals, progress_callback)
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
        if self._can_use_cupy_path():
            logger.info("Using CuPy GPU-resident pipeline (use_gpu=True, CuPy available)")
            return self._align_and_stack_laplacian_cupy(signals, progress_callback)
        else:
            logger.info(f"Using CPU pipeline (use_gpu={self.config.use_gpu}, CuPy={_HAS_CUPY})")

        ref_image = self.Algorithm.align_image_pair(self.image_paths[0], self.image_paths[0])
        fused_pyr = self.Algorithm.generate_laplacian_pyramid(
            ref_image, self.pyramid_num_levels
        )
        new_pyr = None
        paths = self.image_paths

        def _align_and_pyramid(path):
            """Load, align, and build pyramid in one worker call."""
            aligned = self._load_and_align(ref_image, path)
            pyr = self.Algorithm.generate_laplacian_pyramid(
                aligned, self.pyramid_num_levels
            )
            del aligned
            return pyr

        try:
            # Use 2 workers: pre-compute alignment + pyramid for next images
            # while main thread fuses current pair. cv2 ops release GIL.
            with ThreadPoolExecutor(max_workers=2) as pool:
                pending = {}
                lookahead = 3
                for j in range(1, min(1 + lookahead, len(paths))):
                    pending[j] = pool.submit(_align_and_pyramid, paths[j])

                for i in range(1, len(paths)):
                    self.Algorithm.wait_if_paused()
                    if self.Algorithm.is_cancelled:
                        logger.info("Stacking cancelled")
                        for f in pending.values():
                            f.cancel()
                        return

                    start_time = time.time()

                    if i in pending:
                        new_pyr = pending.pop(i).result()
                    else:
                        new_pyr = _align_and_pyramid(paths[i])

                    # Refill lookahead
                    for j in range(i + 1, min(i + 1 + lookahead, len(paths))):
                        if j not in pending:
                            pending[j] = pool.submit(_align_and_pyramid, paths[j])

                    fused_pyr = self.Algorithm.focus_fuse_pyramid_pair(
                        fused_pyr, new_pyr, self.fusion_kernel_size,
                        self.config.contrast_threshold, self.config.feather_radius,
                    )
                    del new_pyr
                    new_pyr = None
                    elapsed = time.time() - start_time
                    self._emit_progress(signals, progress_callback, i + 1, len(paths), elapsed)
        finally:
            del new_pyr, ref_image

        if self.Algorithm.is_cancelled:
            return

        self.output_image = self.Algorithm.reconstruct_pyramid(fused_pyr)
        self.output_image = CPU.local_tone_map(self.output_image, strength=0.3)

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
        ref_image = self.Algorithm.align_image_pair(paths[0], paths[0])
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

        im0 = self.Algorithm.load_image(self.image_paths[0])
        fused_pyr = self.Algorithm.generate_laplacian_pyramid(im0, self.pyramid_num_levels)
        del im0
        new_pyr = None
        paths = self.image_paths

        def _load_and_pyramid(path):
            """Load image and build pyramid in worker thread."""
            img = self.Algorithm.load_image(path)
            pyr = self.Algorithm.generate_laplacian_pyramid(img, self.pyramid_num_levels)
            del img
            return pyr

        try:
            # Pre-compute load+pyramid in background while main thread fuses
            with ThreadPoolExecutor(max_workers=2) as pool:
                pending = {}
                lookahead = 3
                for j in range(1, min(1 + lookahead, len(paths))):
                    pending[j] = pool.submit(_load_and_pyramid, paths[j])

                for i in range(1, len(paths)):
                    self.Algorithm.wait_if_paused()
                    if self.Algorithm.is_cancelled:
                        for f in pending.values():
                            f.cancel()
                        return

                    start_time = time.time()

                    if i in pending:
                        new_pyr = pending.pop(i).result()
                    else:
                        new_pyr = _load_and_pyramid(paths[i])

                    for j in range(i + 1, min(i + 1 + lookahead, len(paths))):
                        if j not in pending:
                            pending[j] = pool.submit(_load_and_pyramid, paths[j])

                    fused_pyr = self.Algorithm.focus_fuse_pyramid_pair(
                        fused_pyr, new_pyr, self.fusion_kernel_size,
                        self.config.contrast_threshold, self.config.feather_radius,
                    )
                    del new_pyr
                    new_pyr = None
                    elapsed = time.time() - start_time
                    self._emit_progress(signals, progress_callback, i + 1, len(paths), elapsed)
        finally:
            del new_pyr

        if self.Algorithm.is_cancelled:
            return
        self.output_image = self.Algorithm.reconstruct_pyramid(fused_pyr)
        self.output_image = CPU.local_tone_map(self.output_image, strength=0.3)

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

        ref_image = self.Algorithm.load_image(self.image_paths[0])

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

        ref_image = self.Algorithm.load_image(self.image_paths[0])

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
        ref_image = self.Algorithm.load_image(self.image_paths[0])

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
        if signals is not None:
            signals.finished_inter_task.emit(
                ["finished_image", current, total, time_taken]
            )
        if progress_callback is not None:
            progress_callback(current, total, time_taken)
