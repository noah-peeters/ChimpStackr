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
        self._pause_event.set()  # Start unpaused (set = running)
        self.alignment_shifts = []  # Track all (x, y) shifts for auto-crop
        self._ref_gray_cache = (None, None)  # (id(ref_im), gray) cache

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
        """Block until unpaused. Times out after 60s to prevent deadlocks."""
        while not self._pause_event.wait(timeout=60.0):
            if self._cancel_event.is_set():
                return

    def toggle_cpu_gpu(self, use_gpu, selected_gpu_id):
        if use_gpu and HAS_CUDA and HAS_GPU:
            cuda.select_device(selected_gpu_id)
            self.useGpu = True
        else:
            self.useGpu = False

    def load_image(self, path):
        """Load image from path as float32 (preserving full bit depth)."""
        return self.ImageLoadingHandler.read_image_as_float32(path)

    @staticmethod
    def _match_dimensions(ref_im, im_to_align):
        """Resize im_to_align to match ref_im dimensions if they differ.

        Handles mixed image dimensions (#29) by center-cropping or padding
        the image to align to match the reference dimensions. Uses
        center-crop for larger images and zero-padding for smaller ones.
        """
        if ref_im.shape[:2] == im_to_align.shape[:2]:
            return im_to_align

        rh, rw = ref_im.shape[:2]
        ah, aw = im_to_align.shape[:2]

        # If dimensions are very different (>2x), resize to match
        if ah > rh * 2 or aw > rw * 2 or ah < rh // 2 or aw < rw // 2:
            return cv2.resize(im_to_align, (rw, rh), interpolation=cv2.INTER_AREA)

        # Center-crop if larger, zero-pad if smaller
        result = np.zeros_like(ref_im)
        # Source region (from im_to_align)
        sy = max(0, (ah - rh) // 2)
        sx = max(0, (aw - rw) // 2)
        # Destination region (in result)
        dy = max(0, (rh - ah) // 2)
        dx = max(0, (rw - aw) // 2)
        # Copy region size
        ch = min(rh - dy, ah - sy)
        cw = min(rw - dx, aw - sx)
        result[dy:dy+ch, dx:dx+cw] = im_to_align[sy:sy+ch, sx:sx+cw]
        return result

    def align_image_pair(self, ref_im, im_to_align, scale_factor=10,
                         coarse_fine=False, use_rst=False, alignment_mode="translation"):
        """
        Align im_to_align to ref_im.
        Returns float32 aligned image.

        alignment_mode: "translation", "euclidean", "similarity", "affine"
        use_rst: legacy flag — if True and alignment_mode not explicitly set, uses euclidean
        """
        # Handle path loading
        if isinstance(ref_im, str) and isinstance(im_to_align, str) and ref_im == im_to_align:
            return self.load_image(im_to_align)
        if isinstance(ref_im, str):
            ref_im = self.load_image(ref_im)
        if isinstance(im_to_align, str):
            im_to_align = self.load_image(im_to_align)

        # Handle different image dimensions (#29)
        if ref_im is not None and im_to_align is not None:
            im_to_align = self._match_dimensions(ref_im, im_to_align)

        # Legacy compat: use_rst maps to euclidean if no explicit mode
        if use_rst and alignment_mode == "translation":
            alignment_mode = "euclidean"

        if alignment_mode == "similarity":
            result = self._align_similarity(ref_im, im_to_align, scale_factor)
        elif alignment_mode == "affine":
            result = self._align_affine(ref_im, im_to_align, scale_factor)
        elif alignment_mode == "euclidean":
            result = self._align_rst(ref_im, im_to_align, scale_factor)
        else:
            result = self._align_translation(ref_im, im_to_align, scale_factor, coarse_fine)

        return result

    def _get_ref_gray(self, ref_im):
        """Get cached grayscale of reference image (avoids redundant cvtColor)."""
        ref_id = id(ref_im)
        if self._ref_gray_cache[0] != ref_id:
            if ref_im.ndim == 3:
                gray = cv2.cvtColor(ref_im, cv2.COLOR_BGR2GRAY)
            else:
                gray = ref_im
            self._ref_gray_cache = (ref_id, gray)
        return self._ref_gray_cache[1]

    def _align_translation(self, ref_im, im_to_align, scale_factor, coarse_fine):
        """Translation-only alignment using DFT phase correlation."""
        ref_gray = self._get_ref_gray(ref_im)

        if coarse_fine and min(ref_im.shape[:2]) > 1000:
            small_ref = cv2.resize(ref_im, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            small_align = cv2.resize(im_to_align, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            self.DFT_Imreg.register_image_translation(
                small_ref, small_align, scale_factor=max(1, scale_factor // 2)
            )
            del small_ref, small_align

        result = self.DFT_Imreg.register_image_translation(
            ref_im, im_to_align, scale_factor=scale_factor,
            ref_gray=ref_gray,
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
        # Convert to grayscale (reuse cached ref grayscale when available)
        ref_gray = self._get_ref_gray(ref_im)
        if im_to_align.ndim == 3:
            align_gray = cv2.cvtColor(im_to_align, cv2.COLOR_BGR2GRAY)
        else:
            align_gray = im_to_align

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
                max_iter = 200 if i == 0 else 100
                gauss = 11 if i == 0 else 5
                criteria = (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    max_iter, 1e-6,
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

        self._track_warp_shifts(warp_matrix, im_to_align.shape)
        return result

    def _track_warp_shifts(self, warp_matrix, shape):
        """Track max corner displacement for auto-crop (shared by RST/similarity/affine)."""
        h, w = shape[:2]
        M = warp_matrix
        max_dx = max_dy = 0.0
        for cx, cy in [(0, 0), (w, 0), (0, h), (w, h)]:
            tx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2] - cx
            ty = M[1, 0] * cx + M[1, 1] * cy + M[1, 2] - cy
            max_dx = max(max_dx, abs(tx))
            max_dy = max(max_dy, abs(ty))
        self.alignment_shifts.append((max_dx, max_dy))

    def _to_gray_u8(self, im):
        """Convert image to grayscale uint8 for feature matching / ECC."""
        if im.ndim == 3:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            gray = im
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        return gray

    def _feature_match(self, ref_u8, align_u8, mode="similarity"):
        """
        ORB feature matching → estimateAffinePartial2D (4 DOF) or estimateAffine2D (6 DOF).
        Returns 2x3 warp matrix or None if insufficient matches.
        """
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(ref_u8, None)
        kp2, des2 = orb.detectAndCompute(align_u8, None)

        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        min_matches = 6 if mode == "similarity" else 10
        if len(good) < min_matches:
            return None

        pts_ref = np.float32([kp1[m.queryIdx].pt for m in good])
        pts_align = np.float32([kp2[m.trainIdx].pt for m in good])

        # estimateAffine*2D(src, dst) computes transform FROM src TO dst.
        # warpAffine with WARP_INVERSE_MAP expects the map FROM dst TO src.
        # We want to warp align→ref space, so the inverse map is ref→align.
        # Therefore: estimate(ref→align) = estimate(pts_ref, pts_align)
        if mode == "affine":
            M, inliers = cv2.estimateAffine2D(
                pts_ref, pts_align,
                method=cv2.RANSAC, ransacReprojThreshold=3.0,
                maxIters=2000, confidence=0.99,
            )
        else:
            M, inliers = cv2.estimateAffinePartial2D(
                pts_ref, pts_align,
                method=cv2.RANSAC, ransacReprojThreshold=3.0,
                maxIters=2000, confidence=0.99,
            )

        if M is None or (inliers is not None and inliers.sum() < min_matches):
            return None
        return M.astype(np.float32)

    def _ecc_refine(self, ref_u8, prealigned_u8, motion_type=cv2.MOTION_EUCLIDEAN):
        """
        Multi-scale ECC refinement on a pre-aligned image.
        Returns residual 2x3 warp matrix (identity if ECC fails).
        """
        h_full, w_full = ref_u8.shape[:2]
        scales = []
        for max_dim in [512, 1024, min(2048, max(h_full, w_full))]:
            s = min(max_dim / max(h_full, w_full), 1.0)
            if not scales or s > scales[-1]:
                scales.append(s)

        warp = np.eye(2, 3, dtype=np.float32)

        try:
            for i, s in enumerate(scales):
                if s < 1.0:
                    ref_s = cv2.resize(ref_u8, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                    align_s = cv2.resize(prealigned_u8, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                else:
                    ref_s = ref_u8
                    align_s = prealigned_u8

                if i > 0:
                    prev_s = scales[i - 1]
                    warp[0, 2] *= (s / prev_s)
                    warp[1, 2] *= (s / prev_s)

                max_iter = 200 if i == 0 else 100
                gauss = 11 if i == 0 else 5
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, 1e-6)

                _, warp = cv2.findTransformECC(
                    ref_s, align_s, warp, motion_type, criteria,
                    inputMask=None, gaussFiltSize=gauss,
                )

            final_s = scales[-1]
            if final_s < 1.0:
                warp[0, 2] /= final_s
                warp[1, 2] /= final_s

        except cv2.error:
            pass  # Return identity (no refinement)

        return warp

    def _align_similarity(self, ref_im, im_to_align, scale_factor):
        """
        4 DOF similarity alignment: tx, ty, rotation, uniform scale.
        Hybrid: ORB features for coarse alignment, ECC for sub-pixel refinement.
        """
        ref_u8 = self._to_gray_u8(self._get_ref_gray(ref_im))
        align_u8 = self._to_gray_u8(im_to_align)

        # Stage 1: Feature-based coarse alignment
        coarse_warp = self._feature_match(ref_u8, align_u8, mode="similarity")
        if coarse_warp is None:
            # Fallback to euclidean ECC
            return self._align_rst(ref_im, im_to_align, scale_factor)

        # Stage 2: Pre-warp and refine with ECC
        h, w = align_u8.shape[:2]
        prealigned = cv2.warpAffine(
            align_u8, coarse_warp, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        residual = self._ecc_refine(ref_u8, prealigned, cv2.MOTION_EUCLIDEAN)

        # Stage 3: Compose coarse + residual
        C = np.vstack([coarse_warp, [0, 0, 1]])
        R = np.vstack([residual, [0, 0, 1]])
        combined = (R @ C)[:2, :].astype(np.float32)

        # Apply to full-res color image
        h, w = im_to_align.shape[:2]
        result = cv2.warpAffine(
            im_to_align, combined, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        self._track_warp_shifts(combined, im_to_align.shape)
        return result

    def _align_affine(self, ref_im, im_to_align, scale_factor):
        """
        6 DOF affine alignment: tx, ty, rotation, scale_x, scale_y, shear.
        Hybrid: ORB features for coarse alignment, ECC MOTION_AFFINE for refinement.
        """
        ref_u8 = self._to_gray_u8(self._get_ref_gray(ref_im))
        align_u8 = self._to_gray_u8(im_to_align)

        # Stage 1: Feature-based coarse alignment (full 6 DOF)
        coarse_warp = self._feature_match(ref_u8, align_u8, mode="affine")
        if coarse_warp is None:
            # Fallback to similarity
            return self._align_similarity(ref_im, im_to_align, scale_factor)

        # Stage 2: Pre-warp and refine with MOTION_AFFINE ECC
        h, w = align_u8.shape[:2]
        prealigned = cv2.warpAffine(
            align_u8, coarse_warp, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        residual = self._ecc_refine(ref_u8, prealigned, cv2.MOTION_AFFINE)

        # Stage 3: Compose
        C = np.vstack([coarse_warp, [0, 0, 1]])
        R = np.vstack([residual, [0, 0, 1]])
        combined = (R @ C)[:2, :].astype(np.float32)

        h, w = im_to_align.shape[:2]
        result = cv2.warpAffine(
            im_to_align, combined, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        self._track_warp_shifts(combined, im_to_align.shape)
        return result

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
        # GPU path doesn't support contrast_threshold/feather_radius —
        # fall back to CPU when those are enabled for consistent results
        if self.useGpu and HAS_GPU and contrast_threshold <= 0 and feather_radius <= 0:
            return self._fuse_gpu(pyr1, pyr2, kernel_size)
        return self._fuse_cpu(pyr1, pyr2, kernel_size, contrast_threshold, feather_radius)

    def _fuse_cpu(self, pyr1, pyr2, kernel_size, contrast_threshold=0.0, feather_radius=0):
        threshold_index = len(pyr1) - 1
        new_pyr = []
        current_focusmap = None
        use_soft = feather_radius > 0

        for pyramid_level in range(len(pyr1)):
            if pyramid_level < threshold_index:
                if contrast_threshold > 0:
                    # Thresholded focusmap needs the Numba path
                    gray1 = cv2.cvtColor(pyr1[pyramid_level], cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(pyr2[pyramid_level], cv2.COLOR_BGR2GRAY)
                    current_focusmap = CPU.compute_focusmap_thresholded(
                        gray1, gray2, kernel_size, np.float32(contrast_threshold),
                    )
                elif HAS_GPU:
                    # Use fast vectorized focusmap (cv2.blur variance, O(1)/pixel)
                    current_focusmap = GPU.compute_focusmap_fast(
                        pyr1[pyramid_level], pyr2[pyramid_level], kernel_size,
                    )
                else:
                    gray1 = cv2.cvtColor(pyr1[pyramid_level], cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(pyr2[pyramid_level], cv2.COLOR_BGR2GRAY)
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
        """GPU fusion — batched pipeline that keeps data on-GPU.
        Batch-uploads all pyramid levels, runs all kernels without
        intermediate host↔device transfers, downloads only at the end."""
        return GPU.fuse_pyramid_pair_gpu(pyr1, pyr2, kernel_size)
