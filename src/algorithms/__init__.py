"""
    Main pyramid stacking + image alignment algorithm(s).
"""
import math
import threading
import os
import cv2
import numpy as np

# Enable OpenCV multi-threading
_cv_threads = os.cpu_count() or 4
cv2.setNumThreads(_cv_threads)

try:
    import numba.cuda as cuda
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

import src.algorithms.dft_imreg as dft_imreg
import src.ImageLoadingHandler as ImageLoadingHandler
import src.algorithms.stacking_algorithms.cpu as CPU

try:
    import src.algorithms.stacking_algorithms.gpu as GPU
    HAS_GPU = True
except Exception:
    HAS_GPU = False


class Algorithm:
    def __init__(self):
        self.ImageLoadingHandler = ImageLoadingHandler.ImageLoadingHandler()
        self.DFT_Imreg = dft_imreg.im_reg()
        self.DFT_Imreg.last_shift = (0.0, 0.0)
        self.useGpu = False
        self._cancel_event = threading.Event()
        self._pause_event = threading.Event()
        self.alignment_shifts = []  # Track all (x, y) shifts for auto-crop

    def cancel(self):
        """Signal the algorithm to stop processing."""
        self._cancel_event.set()
        self._pause_event.set()  # Unblock if paused

    def pause(self):
        """Pause the algorithm. Call resume() to continue."""
        self._pause_event.clear()

    def resume(self):
        """Resume a paused algorithm."""
        self._pause_event.set()

    def reset_cancel(self):
        """Reset cancel/pause flags before starting a new operation."""
        self._cancel_event.clear()
        self._pause_event.set()  # Not paused by default

    @property
    def is_cancelled(self):
        return self._cancel_event.is_set()

    @property
    def is_paused(self):
        return not self._pause_event.is_set()

    def wait_if_paused(self):
        """Block until resumed or cancelled."""
        self._pause_event.wait()

    def toggle_cpu_gpu(self, use_gpu, selected_gpu_id):
        if use_gpu and HAS_CUDA and HAS_GPU:
            cuda.select_device(selected_gpu_id)
            self.useGpu = True
        else:
            self.useGpu = False

    def align_image_pair(self, ref_im, im_to_align, scale_factor=10, coarse_fine=False):
        """
        Fast Fourier Transform (FFT) image translational registration ((x, y)-shift only!)
        'ref_im' and 'im_to_align' can be an image array (np.ndarray), or an image path (str).
        If coarse_fine=True, does a fast alignment at 1/4 resolution first, then refines.
        """
        if isinstance(ref_im, str) and isinstance(im_to_align, str):
            return self.ImageLoadingHandler.read_image_from_path(im_to_align)

        if isinstance(ref_im, str):
            ref_im = self.ImageLoadingHandler.read_image_from_path(ref_im)
        if isinstance(im_to_align, str):
            im_to_align = self.ImageLoadingHandler.read_image_from_path(im_to_align)

        if coarse_fine and min(ref_im.shape[:2]) > 1000:
            small_ref = cv2.resize(ref_im, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            small_align = cv2.resize(im_to_align, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            self.DFT_Imreg.register_image_translation(
                small_ref, small_align, scale_factor=max(1, scale_factor // 2)
            )
            del small_ref, small_align

        result = self.DFT_Imreg.register_image_translation(
            ref_im, im_to_align, scale_factor=scale_factor
        )
        # Track shift for auto-crop
        self.alignment_shifts.append(self.DFT_Imreg.last_shift)
        return result

    def generate_laplacian_pyramid(self, im1, num_levels):
        """Generate a laplacian pyramid for an image."""
        if isinstance(im1, str):
            im1 = self.ImageLoadingHandler.read_image_from_path(im1)

        if self.useGpu and HAS_GPU:
            return GPU.generate_laplacian_pyramid(im1, num_levels)
        else:
            return CPU.generate_laplacian_pyramid(im1, num_levels)

    def reconstruct_pyramid(self, laplacian_pyr):
        """Reconstruct the original image from a laplacian pyramid."""
        if self.useGpu and HAS_GPU:
            return GPU.reconstruct_pyramid(laplacian_pyr)
        else:
            return CPU.reconstruct_pyramid(laplacian_pyr)

    def focus_fuse_pyramid_pair(self, pyr1, pyr2, kernel_size):
        """
        Fuse 2 image pyramids into one.
        Each pyramid level will be compared between the two pyramids,
        and the sharpest pixels/parts of each image will be placed in the output pyramid.
        """
        if self.useGpu and HAS_GPU:
            return self._fuse_gpu(pyr1, pyr2, kernel_size)
        else:
            return self._fuse_cpu(pyr1, pyr2, kernel_size)

    def _fuse_cpu(self, pyr1, pyr2, kernel_size):
        threshold_index = len(pyr1) - 1
        new_pyr = []
        current_focusmap = None
        for pyramid_level in range(len(pyr1)):
            if pyramid_level < threshold_index:
                current_focusmap = CPU.compute_focusmap(
                    cv2.cvtColor(pyr1[pyramid_level], cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(pyr2[pyramid_level], cv2.COLOR_BGR2GRAY),
                    kernel_size,
                )
            else:
                s = pyr2[pyramid_level].shape
                current_focusmap = cv2.resize(
                    current_focusmap, (s[1], s[0]), interpolation=cv2.INTER_AREA
                )
            new_pyr_level = CPU.fuse_pyramid_levels_using_focusmap(
                pyr1[pyramid_level],
                pyr2[pyramid_level],
                current_focusmap,
            )
            new_pyr.append(new_pyr_level)
        return new_pyr

    def _fuse_gpu(self, pyr1, pyr2, kernel_size):
        threshold_index = len(pyr1) - 1
        new_pyr = []
        previous_focusmap = None

        if not cuda.is_cuda_array(pyr1[0]):
            inter_pyr = []
            for i in pyr1:
                inter_pyr.append(cuda.to_device(i))
            pyr1 = inter_pyr.copy()

        inter_pyr = []
        for i in pyr2:
            inter_pyr.append(cuda.to_device(i))
        pyr2 = inter_pyr.copy()
        del inter_pyr

        for pyramid_level in range(len(pyr1)):
            if pyramid_level < threshold_index:
                previous_focusmap = GPU.compute_focusmap(
                    pyr1[pyramid_level],
                    pyr2[pyramid_level],
                    kernel_size,
                )
                new_pyr_level = GPU.fuse_pyramid_levels_using_focusmap(
                    pyr1[pyramid_level],
                    pyr2[pyramid_level],
                    previous_focusmap,
                )
            else:
                s = pyr2[pyramid_level].shape
                array_out = cuda.device_array(
                    (s[0], s[1]),
                    previous_focusmap.dtype,
                )

                threadsperblock = (16, 16)
                blockspergrid_x = math.ceil(s[0] / threadsperblock[0])
                blockspergrid_y = math.ceil(s[1] / threadsperblock[1])
                blockspergrid = (blockspergrid_x, blockspergrid_y)

                GPU.resize_image[blockspergrid, threadsperblock](
                    previous_focusmap,
                    array_out,
                    s[1],
                    s[0],
                )

                new_pyr_level = GPU.fuse_pyramid_levels_using_focusmap(
                    pyr1[pyramid_level],
                    pyr2[pyramid_level],
                    array_out,
                )

            new_pyr.append(new_pyr_level)
        return new_pyr
