"""
    Exposed API for easily aligning/stacking multiple images.
    Decoupled from Qt — can be used standalone or from CLI.
"""
import time
import logging

import numpy as np
import src.utilities as utilities
import src.algorithms as algorithms
from src.config import AlgorithmConfig

logger = logging.getLogger(__name__)


class LaplacianPyramid:
    def __init__(self, config=None):
        if config is None:
            config = AlgorithmConfig()
        self.config = config
        self.output_image = None
        self.image_paths = []
        self.Algorithm = algorithms.Algorithm()

    @property
    def fusion_kernel_size(self):
        return self.config.fusion_kernel_size

    @property
    def pyramid_num_levels(self):
        return self.config.pyramid_num_levels

    def configure(self, **kwargs):
        """Update algorithm configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def apply_gpu_settings(self):
        """Apply GPU settings from config (no Qt dependency)."""
        self.Algorithm.toggle_cpu_gpu(
            self.config.use_gpu,
            self.config.selected_gpu_id,
        )

    def cancel(self):
        """Cancel any running operation."""
        self.Algorithm.cancel()

    def pause(self):
        """Pause any running operation."""
        self.Algorithm.pause()

    def resume(self):
        """Resume a paused operation."""
        self.Algorithm.resume()

    def update_image_paths(self, new_image_paths):
        """Set new image paths (sorted by name)."""
        self.image_paths = sorted(new_image_paths, key=utilities.int_string_sorting)

    def get_crop_bounds(self):
        """
        Compute crop rectangle that removes black edges from alignment shifts.
        Returns (top, bottom, left, right) pixel counts to crop,
        or None if no alignment was performed.
        """
        shifts = self.Algorithm.alignment_shifts
        if not shifts:
            return None

        # Find the max absolute shift in each direction
        max_x_pos = max(s[0] for s in shifts if s[0] > 0) if any(s[0] > 0 for s in shifts) else 0
        max_x_neg = abs(min(s[0] for s in shifts if s[0] < 0)) if any(s[0] < 0 for s in shifts) else 0
        max_y_pos = max(s[1] for s in shifts if s[1] > 0) if any(s[1] > 0 for s in shifts) else 0
        max_y_neg = abs(min(s[1] for s in shifts if s[1] < 0)) if any(s[1] < 0 for s in shifts) else 0

        # Ceiling to be safe — add 1px margin
        import math
        left = math.ceil(max_x_pos) + 1
        right = math.ceil(max_x_neg) + 1
        top = math.ceil(max_y_pos) + 1
        bottom = math.ceil(max_y_neg) + 1

        return (top, bottom, left, right)

    def auto_crop_output(self):
        """
        Crop the output image to remove black edges caused by alignment shifts.
        Modifies self.output_image in place. Returns the crop bounds used, or None.
        """
        if self.output_image is None:
            return None

        bounds = self.get_crop_bounds()
        if bounds is None:
            return None

        top, bottom, left, right = bounds
        h, w = self.output_image.shape[:2]

        # Only crop if bounds are meaningful
        if top + bottom >= h or left + right >= w:
            return None

        self.output_image = self.output_image[top:h - bottom, left:w - right].copy()
        return bounds

    def _get_reference_index(self):
        """Get the reference image index based on alignment strategy."""
        strategy = self.config.alignment_reference
        if strategy == "middle":
            return len(self.image_paths) // 2
        elif strategy == "first":
            return 0
        else:
            return 0

    def align_and_stack_images(self, signals=None, progress_callback=None):
        """
        Align and stack images.
        Uses chain-alignment: each image aligns to the previous aligned result,
        matching the original algorithm behavior.
        signals: Qt WorkerSignals (optional, for GUI mode)
        progress_callback: callable(current, total, time_taken) (optional, for CLI mode)
        """
        self.apply_gpu_settings()
        self.Algorithm.reset_cancel()
        self.Algorithm.alignment_shifts = []  # Reset shift tracking

        # Load first image (no alignment needed — it's the reference)
        aligned_images = [
            self.Algorithm.align_image_pair(self.image_paths[0], self.image_paths[0])
        ]
        fused_pyr = self.Algorithm.generate_laplacian_pyramid(
            aligned_images[0], self.pyramid_num_levels
        )

        for i, path in enumerate(self.image_paths):
            if i == 0:
                continue  # First image already in list

            self.Algorithm.wait_if_paused()
            if self.Algorithm.is_cancelled:
                logger.info("Stacking cancelled by user")
                return

            start_time = time.time()

            # Align to previous aligned image (chain alignment)
            aligned_images.append(
                self.Algorithm.align_image_pair(
                    aligned_images[0], path,
                    scale_factor=self.config.alignment_scale_factor
                )
            )

            # Generate pyramid for the newly aligned image
            new_pyr = self.Algorithm.generate_laplacian_pyramid(
                aligned_images[1], self.pyramid_num_levels
            )

            # Remove first aligned image — shifts [1] to [0] for next iteration
            # This means next image aligns to THIS aligned result (chain alignment)
            del aligned_images[0]

            # Fuse this new pyramid with the running result
            fused_pyr = self.Algorithm.focus_fuse_pyramid_pair(
                fused_pyr, new_pyr, self.fusion_kernel_size
            )

            elapsed = time.time() - start_time
            self._emit_progress(signals, progress_callback, i + 1, len(self.image_paths), elapsed)

        if self.Algorithm.is_cancelled:
            return

        # Copy back from GPU if needed
        if self.Algorithm.useGpu:
            fused_pyr = [level.copy_to_host() for level in fused_pyr]

        fused_image = self.Algorithm.reconstruct_pyramid(fused_pyr)
        self.output_image = fused_image

    def stack_images(self, signals=None, progress_callback=None):
        """
        Stack images without alignment.
        signals: Qt WorkerSignals (optional, for GUI mode)
        progress_callback: callable(current, total, time_taken) (optional, for CLI mode)
        """
        self.apply_gpu_settings()
        self.Algorithm.reset_cancel()

        im0 = self.Algorithm.align_image_pair(self.image_paths[0], self.image_paths[0])
        fused_pyr = self.Algorithm.generate_laplacian_pyramid(
            im0, self.pyramid_num_levels
        )

        for i, path in enumerate(self.image_paths):
            if i == 0:
                continue

            self.Algorithm.wait_if_paused()
            if self.Algorithm.is_cancelled:
                logger.info("Stacking cancelled by user")
                return

            start_time = time.time()

            im1 = self.Algorithm.align_image_pair(path, path)
            new_pyr = self.Algorithm.generate_laplacian_pyramid(
                im1, self.pyramid_num_levels
            )
            del im1

            fused_pyr = self.Algorithm.focus_fuse_pyramid_pair(
                fused_pyr, new_pyr, self.fusion_kernel_size
            )

            elapsed = time.time() - start_time
            self._emit_progress(signals, progress_callback, i + 1, len(self.image_paths), elapsed)

        if self.Algorithm.is_cancelled:
            return

        fused_image = self.Algorithm.reconstruct_pyramid(fused_pyr)
        self.output_image = fused_image

    def _emit_progress(self, signals, progress_callback, current, total, time_taken):
        """Emit progress via Qt signals or plain callback."""
        if signals is not None:
            signals.finished_inter_task.emit(
                ["finished_image", current, total, time_taken]
            )
        if progress_callback is not None:
            progress_callback(current, total, time_taken)
