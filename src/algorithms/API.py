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

    def _align_and_stack_laplacian(self, signals=None, progress_callback=None):
        self.apply_gpu_settings()
        self.Algorithm.reset_cancel()
        self.Algorithm.alignment_shifts = []

        ref_image = self.Algorithm.align_image_pair(self.image_paths[0], self.image_paths[0])
        fused_pyr = self.Algorithm.generate_laplacian_pyramid(
            ref_image, self.pyramid_num_levels
        )
        new_pyr = None

        try:
            # Pre-fetch: load+align next image in background while GPU fuses current.
            # cv2 and Numba CUDA both release the GIL, so this overlaps I/O with compute.
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = None
                paths = self.image_paths

                for i in range(1, len(paths)):
                    self.Algorithm.wait_if_paused()
                    if self.Algorithm.is_cancelled:
                        logger.info("Stacking cancelled")
                        return

                    start_time = time.time()

                    # Get aligned image (from pre-fetch or synchronous)
                    if future is not None:
                        aligned = future.result()
                    else:
                        aligned = self._load_and_align(ref_image, paths[i])

                    # Submit pre-fetch for next image (overlaps with pyramid+fuse below)
                    if i + 1 < len(paths):
                        future = pool.submit(self._load_and_align, ref_image, paths[i + 1])
                    else:
                        future = None

                    new_pyr = self.Algorithm.generate_laplacian_pyramid(
                        aligned, self.pyramid_num_levels
                    )
                    del aligned
                    fused_pyr = self.Algorithm.focus_fuse_pyramid_pair(
                        fused_pyr, new_pyr, self.fusion_kernel_size,
                        self.config.contrast_threshold, self.config.feather_radius,
                    )
                    del new_pyr
                    new_pyr = None
                    elapsed = time.time() - start_time
                    self._emit_progress(signals, progress_callback, i + 1, len(paths), elapsed)
        finally:
            # Ensure intermediate arrays are freed even on cancel/error
            del new_pyr, ref_image

        if self.Algorithm.is_cancelled:
            return

        self.output_image = self.Algorithm.reconstruct_pyramid(fused_pyr)
        # Local tone-mapping to compensate for PMax contrast boost
        self.output_image = CPU.local_tone_map(self.output_image, strength=0.3)

    def _stack_laplacian(self, signals=None, progress_callback=None):
        self.apply_gpu_settings()
        self.Algorithm.reset_cancel()

        im0 = self.Algorithm.load_image(self.image_paths[0])
        fused_pyr = self.Algorithm.generate_laplacian_pyramid(im0, self.pyramid_num_levels)
        del im0
        new_pyr = None

        try:
            # Pre-fetch: load next image in background while GPU fuses current
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = None
                paths = self.image_paths

                for i in range(1, len(paths)):
                    self.Algorithm.wait_if_paused()
                    if self.Algorithm.is_cancelled:
                        return

                    start_time = time.time()

                    if future is not None:
                        im1 = future.result()
                    else:
                        im1 = self.Algorithm.load_image(paths[i])

                    # Submit pre-fetch for next image
                    if i + 1 < len(paths):
                        future = pool.submit(self.Algorithm.load_image, paths[i + 1])
                    else:
                        future = None

                    new_pyr = self.Algorithm.generate_laplacian_pyramid(im1, self.pyramid_num_levels)
                    del im1
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
