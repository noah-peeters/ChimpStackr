"""
    Main pyramid stacking + image alignment algorithm(s).
    Supports translation-only and full rotation+scale+translation alignment.
    Pipeline operates in float32 to preserve full bit depth.
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
        self._cancel_event.set()
        self._pause_event.set()

    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def reset_cancel(self):
        self._cancel_event.clear()
        self._pause_event.set()

    @property
    def is_cancelled(self):
        return self._cancel_event.is_set()

    @property
    def is_paused(self):
        return not self._pause_event.is_set()

    def wait_if_paused(self):
        self._pause_event.wait()

    def toggle_cpu_gpu(self, use_gpu, selected_gpu_id):
        if use_gpu and HAS_CUDA and HAS_GPU:
            cuda.select_device(selected_gpu_id)
            self.useGpu = True
        else:
            self.useGpu = False

    def load_image(self, path):
        """Load image from path as float32 (preserving full bit depth)."""
        return self.ImageLoadingHandler.read_image_as_float32(path)

    def align_image_pair(self, ref_im, im_to_align, scale_factor=10,
                         coarse_fine=False, use_rst=False):
        """
        Align im_to_align to ref_im.
        Returns float32 aligned image.

        If use_rst=True, uses rotation+scale+translation alignment.
        Otherwise, translation only (DFT phase correlation).
        """
        # Handle path loading
        if isinstance(ref_im, str) and isinstance(im_to_align, str) and ref_im == im_to_align:
            return self.load_image(im_to_align)
        if isinstance(ref_im, str):
            ref_im = self.load_image(ref_im)
        if isinstance(im_to_align, str):
            im_to_align = self.load_image(im_to_align)

        if use_rst:
            result = self._align_rst(ref_im, im_to_align, scale_factor)
        else:
            result = self._align_translation(ref_im, im_to_align, scale_factor, coarse_fine)

        return result

    def _align_translation(self, ref_im, im_to_align, scale_factor, coarse_fine):
        """Translation-only alignment using DFT phase correlation."""
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
        self.alignment_shifts.append(self.DFT_Imreg.last_shift)
        return result

    def _align_rst(self, ref_im, im_to_align, scale_factor):
        """
        Rotation + Scale + Translation alignment using multi-scale ECC.
        Uses MOTION_EUCLIDEAN (translation + rotation, 3 DOF) which is
        the right model for focus stacking — no shear, no anisotropic scale.
        Focus breathing (uniform scale) is handled by ORB feature matching fallback.
        """
        # Convert to grayscale
        if ref_im.ndim == 3:
            ref_gray = cv2.cvtColor(ref_im, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = ref_im.copy()
        if im_to_align.ndim == 3:
            align_gray = cv2.cvtColor(im_to_align, cv2.COLOR_BGR2GRAY)
        else:
            align_gray = im_to_align.copy()

        # Ensure uint8
        if ref_gray.dtype != np.uint8:
            ref_u8 = np.clip(ref_gray, 0, 255).astype(np.uint8)
            align_u8 = np.clip(align_gray, 0, 255).astype(np.uint8)
        else:
            ref_u8 = ref_gray
            align_u8 = align_gray

        h_full, w_full = ref_u8.shape[:2]

        # Multi-scale ECC: coarse → fine
        # Level 0: 1/4 res (coarse alignment)
        # Level 1: 1/2 res (refine)
        # Level 2: full or capped at 2048px (final refinement)
        scales = []
        for max_dim in [512, 1024, min(2048, max(h_full, w_full))]:
            s = min(max_dim / max(h_full, w_full), 1.0)
            if not scales or s > scales[-1]:
                scales.append(s)

        warp_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            for i, s in enumerate(scales):
                if s < 1.0:
                    ref_s = cv2.resize(ref_u8, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                    align_s = cv2.resize(align_u8, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                else:
                    ref_s = ref_u8
                    align_s = align_u8

                # Scale warp matrix translation from previous level
                if i > 0:
                    prev_s = scales[i - 1]
                    warp_matrix[0, 2] *= (s / prev_s)
                    warp_matrix[1, 2] *= (s / prev_s)

                # More iterations at coarse, fewer at fine
                max_iter = 300 if i == 0 else 150
                gauss = 11 if i == 0 else 5
                criteria = (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    max_iter, 1e-7,
                )

                # EUCLIDEAN: translation + rotation (3 DOF) — correct for focus stacking
                _, warp_matrix = cv2.findTransformECC(
                    ref_s, align_s, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria,
                    inputMask=None, gaussFiltSize=gauss,
                )

            # Scale translation back to full resolution
            final_s = scales[-1]
            if final_s < 1.0:
                warp_matrix[0, 2] /= final_s
                warp_matrix[1, 2] /= final_s

        except cv2.error:
            # ECC failed — fall back to translation-only DFT
            return self._align_translation(ref_im, im_to_align, scale_factor)

        # Apply the warp to the full-resolution color image
        h, w = im_to_align.shape[:2]
        result = cv2.warpAffine(
            im_to_align, warp_matrix, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        # Track shifts for auto-crop.
        # For RST we compute the max displacement at any corner of the image,
        # since rotation/scale moves corners more than the center.
        h, w = im_to_align.shape[:2]
        corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], dtype=np.float32)
        # warp_matrix maps ref→src (WARP_INVERSE_MAP), so the inverse maps src→ref.
        # Displacement = transformed_corner - original_corner
        M = warp_matrix  # 2x3
        max_dx = 0.0
        max_dy = 0.0
        for cx, cy, _ in corners:
            # Since we used WARP_INVERSE_MAP, the actual forward transform is the inverse
            # The border black region is determined by which src pixels map outside the image
            # Approximate: just use the translation + scale/rotation effect at corners
            tx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2] - cx
            ty = M[1, 0] * cx + M[1, 1] * cy + M[1, 2] - cy
            max_dx = max(max_dx, abs(tx))
            max_dy = max(max_dy, abs(ty))

        self.alignment_shifts.append((max_dx, max_dy))

        return result.astype(np.float32)

    def generate_laplacian_pyramid(self, im1, num_levels):
        if isinstance(im1, str):
            im1 = self.load_image(im1)
        if self.useGpu and HAS_GPU:
            return GPU.generate_laplacian_pyramid(im1, num_levels)
        return CPU.generate_laplacian_pyramid(im1, num_levels)

    def reconstruct_pyramid(self, laplacian_pyr):
        if self.useGpu and HAS_GPU:
            return GPU.reconstruct_pyramid(laplacian_pyr)
        return CPU.reconstruct_pyramid(laplacian_pyr)

    def focus_fuse_pyramid_pair(self, pyr1, pyr2, kernel_size,
                               contrast_threshold=0.0, feather_radius=0):
        if self.useGpu and HAS_GPU:
            return self._fuse_gpu(pyr1, pyr2, kernel_size)
        return self._fuse_cpu(pyr1, pyr2, kernel_size, contrast_threshold, feather_radius)

    def _fuse_cpu(self, pyr1, pyr2, kernel_size, contrast_threshold=0.0, feather_radius=0):
        threshold_index = len(pyr1) - 1
        new_pyr = []
        current_focusmap = None
        use_soft = feather_radius > 0

        for pyramid_level in range(len(pyr1)):
            if pyramid_level < threshold_index:
                gray1 = cv2.cvtColor(pyr1[pyramid_level], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(pyr2[pyramid_level], cv2.COLOR_BGR2GRAY)
                if contrast_threshold > 0:
                    current_focusmap = CPU.compute_focusmap_thresholded(
                        gray1, gray2, kernel_size, np.float32(contrast_threshold),
                    )
                else:
                    current_focusmap = CPU.compute_focusmap(
                        gray1, gray2, kernel_size,
                    )
            else:
                s = pyr2[pyramid_level].shape
                current_focusmap = cv2.resize(
                    current_focusmap, (s[1], s[0]), interpolation=cv2.INTER_AREA
                )

            if use_soft:
                soft_map = CPU.feather_focusmap(current_focusmap, feather_radius)
                new_pyr_level = CPU.fuse_pyramid_levels_soft(
                    pyr1[pyramid_level], pyr2[pyramid_level], soft_map,
                )
            else:
                new_pyr_level = CPU.fuse_pyramid_levels_using_focusmap(
                    pyr1[pyramid_level], pyr2[pyramid_level], current_focusmap,
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
                    pyr1[pyramid_level], pyr2[pyramid_level], kernel_size,
                )
                new_pyr_level = GPU.fuse_pyramid_levels_using_focusmap(
                    pyr1[pyramid_level], pyr2[pyramid_level], previous_focusmap,
                )
            else:
                s = pyr2[pyramid_level].shape
                array_out = cuda.device_array((s[0], s[1]), previous_focusmap.dtype)
                threadsperblock = (16, 16)
                blockspergrid_x = math.ceil(s[0] / threadsperblock[0])
                blockspergrid_y = math.ceil(s[1] / threadsperblock[1])
                blockspergrid = (blockspergrid_x, blockspergrid_y)
                GPU.resize_image[blockspergrid, threadsperblock](
                    previous_focusmap, array_out, s[1], s[0],
                )
                new_pyr_level = GPU.fuse_pyramid_levels_using_focusmap(
                    pyr1[pyramid_level], pyr2[pyramid_level], array_out,
                )
            new_pyr.append(new_pyr_level)
        return new_pyr
