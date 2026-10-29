"""
    Main pyramid stacking + image alignment algorithm(s).
    Supports translation-only and full rotation+scale+translation alignment.
    Pipeline operates in float32 to preserve full bit depth.
"""
import math
import threading
import logging
import os
import sys
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

# Enable OpenCV multi-threading
_cv_threads = os.cpu_count() or 4
cv2.setNumThreads(_cv_threads)

logger = logging.getLogger(__name__)

import src.algorithms.dft_imreg as dft_imreg
import src.ImageLoadingHandler as ImageLoadingHandler
import src.algorithms.stacking_algorithms.cpu as CPU


def _available_memory_bytes():
    """Available physical memory in bytes, or None if it can't be determined."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
            return None
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        # macOS / other POSIX: free pages only (underestimates, which is safe)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (OSError, ValueError, AttributeError):
        return None


class Algorithm:
    def __init__(self):
        self.ImageLoadingHandler = ImageLoadingHandler.ImageLoadingHandler()
        self.DFT_Imreg = dft_imreg.im_reg()
        self.DFT_Imreg.last_shift = (0.0, 0.0)
        self._cancel_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start unpaused (set = running)
        self.alignment_shifts = []  # Track all (x, y) shifts for auto-crop
        self._tls = threading.local()  # per-thread caches (parallel pair estimation)
        # ORB features per source frame (weak refs: frames aren't kept alive)
        self._features = []
        self._features_lock = threading.Lock()
        # Optional ocl.EstimationGPU: pre-warp + ECC cascade levels on the GPU
        self.gpu_estimation = None
        # Default resampling for alignment warps (match original behavior);
        # the breathing pipeline overrides with Lanczos + replicate borders.
        self.warp_interpolation = cv2.INTER_LINEAR
        self.warp_border_mode = cv2.BORDER_CONSTANT

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

    def load_image(self, path):
        """Load image from path as float32 (preserving full bit depth)."""
        return self.ImageLoadingHandler.read_image_as_float32(path)

    def load_gray_u8(self, path):
        """Load image as 8-bit grayscale (fast path for alignment estimation)."""
        return self.ImageLoadingHandler.read_image_as_gray_u8(path)

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

    # ─── Affine matrix helpers ───
    #
    # Convention: a transform matrix is a 2x3 *forward map* from reference
    # (template) coordinates to source coordinates:
    #     dst(x, y) = src(M · [x, y, 1])
    # so `cv2.warpAffine(src, M, (w, h), flags=... | cv2.WARP_INVERSE_MAP)`
    # warps `src` into the reference frame. Estimators *estimate* these maps,
    # apply/composition helpers *use* them.

    @staticmethod
    def _hom(M):
        """3x3 homogeneous version of a 2x3 affine map."""
        M = np.asarray(M, dtype=np.float64)
        return np.vstack([M, [0.0, 0.0, 1.0]])

    @classmethod
    def compose_affine(cls, first, then):
        """Compose two forward maps: apply `first`, then `then`.

        Either may be None (treated as identity in that position).
        Returns a 2x3 float64 matrix (keeps full precision through the
        sequential chain — the final cast happens only in warpAffine, which
        accepts float64 matrices natively).
        """
        if first is None:
            return None if then is None else np.asarray(then, np.float64)
        if then is None:
            return np.asarray(first, np.float64)
        M = cls._hom(then) @ cls._hom(first)
        return M[:2, :].astype(np.float64)

    @classmethod
    def invert_affine(cls, M):
        """Invert a 2x3 affine forward map."""
        H = np.linalg.inv(cls._hom(M))
        return H[:2, :].astype(np.float32)

    @staticmethod
    def transform_scale(M):
        """Uniform scale factor of a (partial-)affine forward map."""
        L = np.asarray(M, dtype=np.float64)[:, :2]
        return float(np.sqrt(abs(np.linalg.det(L))))

    @staticmethod
    def transform_rotation(M):
        """Rotation in degrees of a (partial-)affine forward map.

        Uses the ``cv2.getRotationMatrix2D`` sign convention (positive =
        counter-clockwise visually, i.e. ``atan2(M[0, 1], M[0, 0])`` in
        y-flipped image coordinates).
        """
        L = np.asarray(M, dtype=np.float64)[:, :2]
        return float(np.degrees(np.arctan2(L[0, 1], L[0, 0])))

    @staticmethod
    def project_to_similarity(M, anchor=None):
        """Project a 6-DOF affine onto the similarity manifold (4 DOF).

        Replaces the linear part with its closest uniform-scale rotation
        (scale = sqrt(det(L)), rotation via the getRotationMatrix2D sign
        convention, shear/scale anisotropy dropped) and re-anchors the
        translation so the mapping of `anchor` is preserved.

        Used by the scale refinement, where ECC propagates in 6 DOF
        (MOTION_AFFINE) but the breathing chain must stay 4 DOF.
        """
        M = np.asarray(M, dtype=np.float64)
        L = M[:, :2]
        s = float(np.sqrt(abs(np.linalg.det(L))))
        th = float(np.arctan2(L[0, 1], L[0, 0]))
        c, sn = np.cos(th), np.sin(th)
        R = s * np.array([[c, sn], [-sn, c]])
        out = np.hstack([R, M[:, 2:3].copy()])
        if anchor is not None:
            # preserve M's mapping of the anchor point: R·a + t' = L·a + t
            a = np.asarray(anchor, dtype=np.float64)
            out[:, 2] = (L - R) @ a + M[:, 2]
        return out

    def apply_transform_matrix(self, im_to_align, M, track=True,
                               interpolation=None, border_mode=None):
        """Warp `im_to_align` into the reference frame using forward map M.

        Defaults (linear interpolation, constant border) match the original
        alignment behavior. The breathing path overrides these with
        Lanczos resampling + replicated borders for maximum sharpness.
        """
        if M is None:
            return im_to_align
        if interpolation is None:
            interpolation = self.warp_interpolation
        if border_mode is None:
            border_mode = self.warp_border_mode
        h, w = im_to_align.shape[:2]
        result = cv2.warpAffine(
            im_to_align, M, (w, h),
            flags=interpolation + cv2.WARP_INVERSE_MAP,
            borderMode=border_mode, borderValue=0,
        )
        if track:
            self._track_warp_shifts(M, im_to_align.shape)
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

        if ref_im is None or im_to_align is None:
            return im_to_align

        # Handle different image dimensions (#29) before estimating, so the
        # transform is applied to the same image it was estimated on
        im_to_align = self._match_dimensions(ref_im, im_to_align)

        # Legacy compat: use_rst maps to euclidean if no explicit mode
        if use_rst and alignment_mode == "translation":
            alignment_mode = "euclidean"

        M, _meta = self._estimate_pair(
            ref_im, im_to_align, alignment_mode, scale_factor, coarse_fine
        )
        if M is None:
            return im_to_align
        return self.apply_transform_matrix(im_to_align, M)

    def estimate_pair_transform(self, ref_im, im_to_align,
                                alignment_mode="similarity", scale_factor=10,
                                coarse_fine=False, precision="default"):
        """
        Estimate the 2x3 forward map aligning `im_to_align` onto `ref_im`.
        Matrix only — no warping, no shift tracking. Accepts paths or arrays.

        precision: "default" (fast u8 ECC, ≤2048px cascade) or "max"
            (float32 ECC, Lanczos pre-warp, ≤4096px cascade — used by the
            sequential breathing engine).

        Returns:
            (matrix, meta): matrix may be None if all estimators failed;
            meta is a dict with "method", "scale", "angle", "success".
        """
        if isinstance(ref_im, str):
            ref_im = self.load_image(ref_im)
        if isinstance(im_to_align, str):
            im_to_align = self.load_image(im_to_align)
        if ref_im is None or im_to_align is None:
            return None, {
                "method": "load_error", "scale": 1.0, "angle": 0.0, "success": 0.0,
            }
        return self._estimate_pair(
            ref_im, im_to_align, alignment_mode, scale_factor, coarse_fine,
            precision=precision,
        )

    def _estimate_pair(self, ref_im, im_to_align, alignment_mode,
                       scale_factor, coarse_fine=False, precision="default"):
        """Dispatch to a pairwise estimator (arrays only)."""
        # Handle different image dimensions (#29)
        im_to_align = self._match_dimensions(ref_im, im_to_align)

        if alignment_mode in ("similarity", "breathing"):
            return self._estimate_similarity(ref_im, im_to_align, scale_factor,
                                             precision=precision)
        if alignment_mode == "affine":
            return self._estimate_affine(ref_im, im_to_align, scale_factor,
                                         precision=precision)
        if alignment_mode == "euclidean":
            return self._estimate_rst(ref_im, im_to_align, scale_factor,
                                      precision=precision)
        return self._estimate_translation(ref_im, im_to_align, scale_factor, coarse_fine)

    def _get_ref_gray(self, ref_im):
        """Get grayscale of the reference image, cached per thread.

        Sequential alignment estimates pairs in parallel worker threads, so
        the cache must be thread-local. The cache holds a STRONG reference
        to the source image and compares by identity: keying on id() alone
        is unsafe because a freed image's address can be reused by the next
        pair's image in the same worker, silently returning the previous
        pair's grayscale (i.e. estimating the wrong transform).
        """
        cache = getattr(self._tls, "gray_cache", None)
        if cache is None:
            # [source image (strong ref), its grayscale, 8-bit grayscale]
            cache = [None, None, None]
            self._tls.gray_cache = cache
        if cache[0] is ref_im:
            return cache[1]
        if ref_im.ndim == 3:
            gray = cv2.cvtColor(ref_im, cv2.COLOR_BGR2GRAY)
        else:
            gray = ref_im
        cache[:] = [ref_im, gray, None]
        return gray

    def _ref_gray_u8(self, ref_im):
        """8-bit grayscale of the reference, cached like _get_ref_gray."""
        gray = self._get_ref_gray(ref_im)
        cache = self._tls.gray_cache
        if cache[2] is None:
            cache[2] = self._to_gray_u8(gray)
        return cache[2]

    def _estimate_translation(self, ref_im, im_to_align, scale_factor, coarse_fine=False):
        """Translation-only DFT phase correlation estimate."""
        ref_gray = self._get_ref_gray(ref_im)

        if coarse_fine and min(ref_im.shape[:2]) > 1000:
            small_ref = cv2.resize(ref_im, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            small_align = cv2.resize(im_to_align, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            self.DFT_Imreg.estimate_translation_matrix(
                small_ref, small_align, scale_factor=max(1, scale_factor // 2)
            )
            del small_ref, small_align

        M, success = self.DFT_Imreg.estimate_translation_matrix(
            ref_im, im_to_align, scale_factor=scale_factor,
            ref_gray=ref_gray,
        )
        meta = {
            "method": "dft_translation",
            "scale": 1.0,
            "angle": 0.0,
            "success": success,
        }
        return M, meta

    def _estimate_rst(self, ref_im, im_to_align, scale_factor, precision="default",
                      align_u8=None):
        """
        Rotation + translation (3 DOF) estimate via multi-scale ECC.
        Uses MOTION_EUCLIDEAN (translation + rotation) which is
        the right model for focus stacking — no shear, no anisotropic scale.
        Uniform scale (focus breathing) is estimated by _estimate_similarity.
        align_u8: im_to_align's 8-bit gray, if the caller already has it.
        """
        hq = precision == "max"
        if hq:
            # High precision: native-bit-depth float32 grayscale
            warp_matrix = self._ecc_refine(
                self._to_gray_f32(self._get_ref_gray(ref_im)), self._to_gray_f32(im_to_align),
                cv2.MOTION_EUCLIDEAN, cascade_dims=(512, 1024, 2048, 4096), eps=1e-7,
                iters=(200, 100), none_on_error=True,
            )
        else:
            if align_u8 is None:
                align_u8 = self._to_gray_u8(im_to_align)
            warp_matrix = self._prewarp_and_refine(
                self._ref_gray_u8(ref_im), align_u8, None,
                cv2.MOTION_EUCLIDEAN, iters=(200, 100), none_on_error=True,
            )
        if warp_matrix is None:
            # ECC failed — fall back to translation-only DFT
            return self._estimate_translation(ref_im, im_to_align, scale_factor)

        meta = {
            "method": "ecc_euclidean_hp" if hq else "ecc_euclidean",
            "scale": self.transform_scale(warp_matrix),
            "angle": self.transform_rotation(warp_matrix),
            "success": 1.0,
        }
        return warp_matrix.astype(np.float32), meta

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
        """Convert image to grayscale uint8 for feature matching."""
        if im.ndim == 3:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            gray = im
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        return gray

    def _to_gray_f32(self, im):
        """Convert image to grayscale float32 WITHOUT 8-bit quantization.

        ECC precision is limited by the bit depth it optimizes on: u8
        quantization noise (~0.5 LSB) directly caps the sub-pixel accuracy
        the optimizer can steer toward.  Keeping the native float/bit depth
        preserves 16-bit RAW and unstretched low-contrast gradients.
        """
        if im.ndim == 3:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            gray = im
        if gray.dtype != np.float32:
            # uint8 stays in 0-255 range; float16/64 and int types are
            # converted preserving value scale
            gray = np.asarray(gray, dtype=np.float32)
        return gray

    @staticmethod
    def _is_plausible_warp(M, shape):
        """Reject degenerate feature-match transforms (e.g. near-zero scale
        from junk matches on a blurry frame). Limits are generous: focus
        stacks only ever need modest scale, rotation and shift."""
        h, w = shape[:2]
        L = M[:, :2].astype(np.float64)
        scale = math.sqrt(abs(np.linalg.det(L)))
        if not 0.7 <= scale <= 1.4:
            return False
        sv = np.linalg.svd(L, compute_uv=False)
        if sv[1] <= 0 or sv[0] / sv[1] > 1.25:  # strong shear / anisotropy
            return False
        if abs(math.degrees(math.atan2(L[1, 0], L[0, 0]))) > 15:
            return False
        return abs(M[0, 2]) <= 0.25 * w and abs(M[1, 2]) <= 0.25 * h

    def _orb_features(self, u8, source=None):
        """ORB keypoints and descriptors of u8, the 8-bit gray of `source`.

        Cached per source frame: a frame takes part in two pairs of a
        sequential chain, and the reference in every pair of a single-reference
        stack, so detection runs once per frame instead of once per pair.
        """
        if source is not None:
            with self._features_lock:
                for ref, feats in self._features:
                    if ref() is source:
                        return feats
        feats = cv2.ORB_create(nfeatures=2000).detectAndCompute(u8, None)
        if source is not None:
            with self._features_lock:
                self._features = [e for e in self._features if e[0]() is not None][-31:]
                self._features.append((weakref.ref(source), feats))
        return feats

    def _feature_match(self, ref_u8, align_u8, mode="similarity", ref_source=None,
                       align_source=None):
        """
        ORB feature matching → estimateAffinePartial2D (4 DOF) or estimateAffine2D (6 DOF).
        Returns 2x3 warp matrix, or None if there are insufficient matches or
        the result is implausible (callers then fall back to ECC).
        ref_source/align_source: the frames the grays came from (feature cache keys).
        """
        kp1, des1 = self._orb_features(ref_u8, ref_source)
        kp2, des2 = self._orb_features(align_u8, align_source)
        # OpenCV's RANSAC draws samples from the per-thread global RNG. In
        # parallel pair estimation the RNG history depends on which pairs
        # landed on which worker, making the coarse estimate (and hence the
        # whole breathing chain) nondeterministic run-to-run. Pinning the
        # seed here makes every pair's estimate depend only on its images.
        cv2.setRNGSeed(42)

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
        if not self._is_plausible_warp(M, ref_u8.shape):
            return None
        return M.astype(np.float32)

    def _warp_score(self, ref_g, align_g, M):
        """Alignment quality of transform M between float grays.

        Returns (ncc, highpass_mad): cross-correlation plus the mean
        absolute difference of highpass (detail) content. NCC is robust to
        focus-dependent brightness change but nearly blind to sub-0.1%
        scale errors; the highpass MAD resolves exactly those ties.
        Higher NCC / lower MAD = better.
        """
        if M is None:
            return (-2.0, float("inf"))
        h, w = ref_g.shape[:2]
        warped = cv2.warpAffine(
            align_g, M, (w, h),
            flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        ncc = self._ncc(ref_g, warped)

        k = 31
        ref_hp = ref_g - cv2.GaussianBlur(ref_g, (k, k), 8.0)
        warped_hp = warped - cv2.GaussianBlur(warped, (k, k), 8.0)
        ty0, ty1 = h // 16, h - max(1, h // 16)
        tx0, tx1 = w // 16, w - max(1, w // 16)
        mad = float(np.mean(np.abs(
            ref_hp[ty0:ty1, tx0:tx1] - warped_hp[ty0:ty1, tx0:tx1])))
        return (ncc, mad)

    @staticmethod
    def _warp_score_better(a, b, ncc_tie=5e-4):
        """True when score tuple `a` is convincingly better than `b`.

        NCC decides when it clearly separates the candidates (> ncc_tie);
        otherwise the misalignment-sensitive highpass MAD (lower better).
        """
        (ncc_a, mad_a), (ncc_b, mad_b) = a, b
        if ncc_a > ncc_b + ncc_tie:
            return True
        if ncc_b > ncc_a + ncc_tie:
            return False
        return mad_a < mad_b

    @staticmethod
    def _ncc(a, b):
        """Normalized cross-correlation between two same-shape float images."""
        a = np.asarray(a, np.float64).ravel()
        b = np.asarray(b, np.float64).ravel()
        a -= a.mean()
        b -= b.mean()
        d = np.sqrt(float(a @ a) * float(b @ b))
        if d <= 0.0:
            return -2.0
        return float((a @ b) / d)

    @staticmethod
    def _cascade_scales(shape, cascade_dims=(512, 1024, 2048)):
        """Downscale factor per ECC cascade level (coarse → fine)."""
        h_full, w_full = shape[:2]
        scales = []
        for max_dim in cascade_dims:
            s = min(max_dim / max(h_full, w_full), 1.0)
            if not scales or s > scales[-1]:
                scales.append(s)
        return scales

    def _prewarp_and_refine(self, ref_u8, align_u8, coarse_warp, motion_type,
                            iters=(240, 160), none_on_error=False):
        """ECC refinement (_ecc_refine) of align_u8, pre-warped by coarse_warp
        (8-bit, bilinear) unless that is None. On the GPU when available:
        pre-warp, cascade levels and ECC iterations."""
        gpu = self.gpu_estimation
        if gpu is not None:
            try:
                scales = self._cascade_scales(ref_u8.shape)
                levels = gpu.ecc_levels(ref_u8, align_u8, coarse_warp, scales)
                return self._ecc_refine(None, None, motion_type,
                                        levels=[(s, r, a) for s, (r, a) in zip(scales, levels)],
                                        ecc=gpu.find_transform_ecc, iters=iters,
                                        none_on_error=none_on_error)
            except Exception as exc:   # OpenCL errors: carry on without the GPU
                logger.warning(f"GPU alignment help failed ({exc}); using the CPU")
                self.gpu_estimation = None
        prealigned = align_u8
        if coarse_warp is not None:
            h, w = align_u8.shape[:2]
            prealigned = cv2.warpAffine(
                align_u8, coarse_warp, (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
        return self._ecc_refine(ref_u8, prealigned, motion_type, iters=iters,
                                none_on_error=none_on_error)

    def _ecc_refine(self, ref, prealigned, motion_type=cv2.MOTION_EUCLIDEAN,
                    cascade_dims=None, eps=1e-6, none_on_error=False, levels=None, ecc=None,
                    iters=(240, 160)):
        """
        Multi-scale ECC refinement on a pre-aligned image.

        Args:
            ref, prealigned: same-shape grayscale images (uint8 or float32).
            levels: instead of ref/prealigned, (scale, ref, prealigned) per
                cascade level, already downscaled (see _prewarp_and_refine).
            ecc: findTransformECC replacement for the levels (same arguments
                as `ecc(template, image, warp, motion, criteria, gauss)`).
            iters: max ECC iterations on the coarsest / the other levels.
            cascade_dims: max image size per cascade level. Default
                (512, 1024, 2048); the high-precision path passes a level
                extending to 4096 so ECC optimizes on (nearly) full res.
            eps: ECC convergence epsilon — smaller = more precise but
                more iterations.

        Returns:
            Residual 2x3 warp matrix (float32, identity if ECC fails), or
            None when `none_on_error=True` and ECC raised (lets callers
            fall back to a different estimator).
        """
        if levels is not None:
            scales = [s for s, _, _ in levels]
        else:
            scales = self._cascade_scales(ref.shape, cascade_dims or (512, 1024, 2048))

        warp = np.eye(2, 3, dtype=np.float32)

        try:
            for i, s in enumerate(scales):
                if levels is not None:
                    _, ref_s, align_s = levels[i]
                elif s < 1.0:
                    ref_s = cv2.resize(ref, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                    align_s = cv2.resize(prealigned, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                else:
                    ref_s = ref
                    align_s = prealigned

                if i > 0:
                    prev_s = scales[i - 1]
                    warp[0, 2] *= (s / prev_s)
                    warp[1, 2] *= (s / prev_s)

                max_iter = iters[0] if i == 0 else iters[1]
                gauss = 11 if i == 0 else 5
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)

                if ecc is not None:
                    _, warp = ecc(ref_s, align_s, warp, motion_type, criteria, gauss)
                else:
                    _, warp = cv2.findTransformECC(
                        ref_s, align_s, warp, motion_type, criteria,
                        inputMask=None, gaussFiltSize=gauss,
                    )

            final_s = scales[-1]
            if final_s < 1.0:
                warp[0, 2] /= final_s
                warp[1, 2] /= final_s

        except cv2.error:
            if none_on_error:
                return None
            # Return identity (no refinement)

        return warp

    def _estimate_similarity(self, ref_im, im_to_align, scale_factor,
                             precision="default"):
        """
        4 DOF similarity estimate: tx, ty, rotation, uniform scale.
        Hybrid: ORB features for coarse alignment, ECC for sub-pixel refinement
        of scale, rotation and translation. Falls back to euclidean ECC when
        feature matching fails.

        precision="max" runs the ECC refinement on float32 grayscale (native
        bit depth, no 8-bit quantization) with Lanczos pre-warping and a
        cascade extending to 4096px — the maximum-accuracy breathing path.
        ORB matching still uses u8 (ORB requires 8-bit descriptors).
        """
        ref_u8 = self._ref_gray_u8(ref_im)
        align_u8 = self._to_gray_u8(im_to_align)
        hq = precision == "max"

        # Stage 1: Feature-based coarse alignment
        coarse_warp = self._feature_match(ref_u8, align_u8, mode="similarity",
                                          ref_source=ref_im, align_source=im_to_align)
        if coarse_warp is None:
            # Fallback to euclidean ECC
            return self._estimate_rst(ref_im, im_to_align, scale_factor,
                                      precision=precision, align_u8=align_u8)

        # Stage 2: Pre-warp and refine with ECC
        h, w = align_u8.shape[:2]
        if hq:
            # Refine on native-bit-depth grayscale, NOT the u8 copy
            ref_g = self._to_gray_f32(self._get_ref_gray(ref_im))
            align_g = self._to_gray_f32(im_to_align)
            prealigned = cv2.warpAffine(
                align_g, coarse_warp, (w, h),
                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )

            # Scale refinement via 6-DOF ECC: ECC has no 4-DOF mode, so run
            # MOTION_AFFINE on float32 grayscale and project onto the
            # similarity manifold (this is where precise breathing
            # compensation comes from — the scale was otherwise frozen at
            # the noisy ORB estimate).
            residual_hq = self._ecc_refine(
                ref_g, prealigned, cv2.MOTION_AFFINE,
                cascade_dims=(512, 1024, 2048, 4096), eps=1e-7,
            )
            scaled = self.project_to_similarity(
                self.compose_affine(coarse_warp, residual_hq),
                anchor=(w * 0.5, h * 0.5),
            )

            # Re-refine rotation + translation AT the refined scale: EUCLIDEAN
            # ECC restores full translation/rotation precision that the 6-DOF
            # projection may have traded away, without touching the scale.
            pre_scaled = cv2.warpAffine(
                align_g, scaled, (w, h),
                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
            residual_rot = self._ecc_refine(
                ref_g, pre_scaled, cv2.MOTION_EUCLIDEAN,
                cascade_dims=(512, 1024, 2048, 4096), eps=1e-7,
            )
            combined_hq = self.compose_affine(scaled, residual_rot)

            # Safe candidate: 3-DOF refinement only (scale frozen at the ORB
            # estimate — can never drift). Serves as the objective fallback
            # for texture-poor content where the 6-DOF scale estimate is noise.
            residual_euc = self._ecc_refine(
                ref_g, prealigned, cv2.MOTION_EUCLIDEAN,
                cascade_dims=(512, 1024, 2048), eps=1e-6,
            )
            combined_euc = self.compose_affine(coarse_warp, residual_euc)

            # Objective arbitration on actual overlap
            score_hq = self._warp_score(ref_g, align_g, combined_hq)
            score_euc = self._warp_score(ref_g, align_g, combined_euc)
            use_scale_refined = self._warp_score_better(score_hq, score_euc)
            combined = combined_hq if use_scale_refined else combined_euc
            logger.debug(
                "HQ similarity: ncc %.5f/%.5f mad %.5f/%.5f → %s",
                score_hq[0], score_euc[0], score_hq[1], score_euc[1],
                "scale-refined" if use_scale_refined else "euclidean",
            )
        else:
            # Refine the scale too: ORB's scale is noisy (~3e-4 per pair),
            # and in a breathing chain that noise accumulates into pixels of
            # drift. 6-DOF ECC projected onto a similarity, then EUCLIDEAN
            # ECC at that scale for full rotation/translation precision.
            residual = self._prewarp_and_refine(ref_u8, align_u8, coarse_warp,
                                                cv2.MOTION_AFFINE)
            scaled = self.project_to_similarity(
                self.compose_affine(coarse_warp, residual), anchor=(w * 0.5, h * 0.5))
            if abs(self.transform_scale(scaled) / self.transform_scale(coarse_warp) - 1) > 0.01:
                # implausible jump (texture-poor frames): keep ORB's scale
                scaled = coarse_warp
            residual = self._prewarp_and_refine(ref_u8, align_u8, scaled, cv2.MOTION_EUCLIDEAN)
            combined = self.compose_affine(scaled, residual)

        meta = {
            "method": "orb_ecc_similarity_hp" if hq else "orb_ecc_similarity",
            "scale": self.transform_scale(combined),
            "angle": self.transform_rotation(combined),
            "success": 1.0,
        }
        return combined, meta

    def _estimate_affine(self, ref_im, im_to_align, scale_factor,
                         precision="default"):
        """
        6 DOF affine estimate: tx, ty, rotation, scale_x, scale_y, shear.
        Hybrid: ORB features for coarse alignment, ECC MOTION_AFFINE for refinement.
        Falls back to similarity when feature matching fails.
        """
        ref_u8 = self._ref_gray_u8(ref_im)
        align_u8 = self._to_gray_u8(im_to_align)
        hq = precision == "max"

        # Stage 1: Feature-based coarse alignment (full 6 DOF)
        coarse_warp = self._feature_match(ref_u8, align_u8, mode="affine",
                                          ref_source=ref_im, align_source=im_to_align)
        if coarse_warp is None:
            # Fallback to similarity
            return self._estimate_similarity(ref_im, im_to_align, scale_factor,
                                             precision=precision)

        # Stage 2: Pre-warp and refine with MOTION_AFFINE ECC
        h, w = align_u8.shape[:2]
        if hq:
            ref_g = self._to_gray_f32(self._get_ref_gray(ref_im))
            align_g = self._to_gray_f32(im_to_align)
            prealigned = cv2.warpAffine(
                align_g, coarse_warp, (w, h),
                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
            residual = self._ecc_refine(
                ref_g, prealigned, cv2.MOTION_AFFINE,
                cascade_dims=(512, 1024, 2048, 4096), eps=1e-7,
            )
        else:
            residual = self._prewarp_and_refine(ref_u8, align_u8, coarse_warp,
                                                cv2.MOTION_AFFINE)

        # Stage 3: Compose
        combined = self.compose_affine(coarse_warp, residual)

        meta = {
            "method": "orb_ecc_affine_hp" if hq else "orb_ecc_affine",
            "scale": self.transform_scale(combined),
            "angle": self.transform_rotation(combined),
            "success": 1.0,
        }
        return combined, meta

    # ─── Sequential (chained) alignment: focus breathing compensation ───
    #
    # Focus breathing is a *cumulative* scale drift: frame N vs. frame 0 can
    # differ by 10-20% while frame N vs. frame N-1 differs by well under 1%.
    # Aligning each frame to a single reference therefore fails on strongly
    # breathing lenses; instead we estimate a small transform per consecutive
    # pair and compose the chain onto the reference frame.

    # Peak working memory per pair-estimation worker, in bytes per image
    # pixel: default works on 8-bit gray frames (less when the GPU does the
    # ECC part), max on float color frames (24MP: ~1.68 GB/worker max)
    _PAIR_BYTES_PER_PIXEL = {"max": 70, "default": 12, "gpu": 6}

    def _sequential_worker_count(self, ref_template, precision, n_pairs):
        """Parallel pair-estimation workers: one per physical core (OpenCV
        runs single-threaded inside each), capped so the workers' combined
        working set stays within 70% of the currently available RAM."""
        workers = min(8, n_pairs, max(1, (os.cpu_count() or 4) // 2))
        avail = _available_memory_bytes()
        if avail is not None:
            h, w = ref_template.shape[:2]
            key = "gpu" if precision == "default" and self.gpu_estimation else precision
            per_worker = h * w * self._PAIR_BYTES_PER_PIXEL.get(key, 70)
            workers = min(workers, max(1, int(avail * 0.7 // per_worker)))
        return max(1, workers)

    def estimate_sequential_transforms(self, paths, ref_index=0,
                                       alignment_mode="similarity",
                                       scale_factor=10,
                                       max_scale_step=0.05,
                                       precision="max",
                                       emit=None, on_ready=None):
        """
        Estimate per-frame transforms for sequential (chained) alignment.

        Args:
            paths: ordered image paths (focus order matters).
            ref_index: frame all transforms are composed into ("first"/"middle").
            alignment_mode: per-pair estimator mode
                ("similarity" [default, handles breathing scale], "euclidean",
                "affine", "translation"). "breathing" is treated as "similarity".
            scale_factor: DFT precision parameter for translation fallback.
            max_scale_step: max plausible scale change per pair (0.05 = 5%).
                Larger suspected steps are re-estimated with translation-only
                correlation; if that also fails, the previous delta is held
                (constant breathing rate assumption).
            precision: "max" (default) refines scale with float32 ECC at up to
                4096px and warps with Lanczos — the slow, maximum-accuracy
                path. "default" uses the fast u8/2048px estimators for quick
                stacks on easy subjects.
            emit: optional callable (current, total, seconds) for progress.
            on_ready: optional callable (i, matrix), called as soon as frame
                i's final transform is known, so stacking can start before the
                whole chain is estimated. Frames up to the reference arrive
                together once the pairs below it are done, later frames one
                by one in order.

        Returns:
            (transforms, log):
            transforms[i] is the 2x3 forward map (reference coords → frame-i
            coords) — cv2.warpAffine(frame_i, transforms[i], WARP_INVERSE_MAP)
            maps frame i into the reference frame.
            log is a list of per-frame dicts (step_scale, cumulative scale,
            method, fallback count). (None, None) if cancelled.
        """
        n = len(paths)

        def _identity_chain():
            transforms = [np.eye(2, 3, dtype=np.float32) for _ in range(n)]
            if on_ready is not None:
                for i, M in enumerate(transforms):
                    on_ready(i, M)
            return transforms, []

        if n < 2:
            return _identity_chain()

        # The default-precision estimators work on 8-bit grayscale, so frames
        # are decoded straight to gray: much faster, and ~12x less memory per
        # worker. Max precision optimises on native-bit-depth float data.
        load = self.load_image if precision == "max" else self.load_gray_u8

        # Every frame is matched to frame 0's dimensions (#29) so all pair
        # transforms live in one coordinate system and the chain composes
        ref_template = load(paths[0])
        if ref_template is None:
            return _identity_chain()

        r = max(0, min(ref_index, n - 1))
        pair_mode = (alignment_mode
                     if alignment_mode in ("similarity", "affine", "euclidean", "translation")
                     else "similarity")

        # Scale plausibility bounds per pair
        lo, hi = 1.0 - max_scale_step, 1.0 + max_scale_step

        # Each frame takes part in two pairs: it is loaded once and shared
        # (with its ORB features, see _orb_features) until both are done.
        # Entries: [lock, image, pairs still to use it]
        frames = {0: [threading.Lock(), ref_template, 1]}
        frames_lock = threading.Lock()

        def _frame(i):
            with frames_lock:
                entry = frames.setdefault(i, [threading.Lock(), None, 2 if i < n - 1 else 1])
            with entry[0]:
                if entry[1] is None:
                    img = load(paths[i])
                    if img is None:
                        raise ValueError("Image failed to load")
                    entry[1] = self._match_dimensions(ref_template, img)
                return entry[1]

        def _release(i):
            with frames_lock:
                entry = frames.get(i)
                if entry is not None:
                    entry[2] -= 1
                    if entry[2] <= 0:
                        del frames[i]

        def _estimate_pair_step(i):
            """Estimate the frame i → i+1 forward map (template = frame i)."""
            t0 = time.time()
            meta = {"method": "load_error", "scale": 1.0, "angle": 0.0, "success": 0.0}
            M = None
            if self.is_cancelled:
                return i, M, meta, 0.0
            try:
                ref = _frame(i)
                src = _frame(i + 1)
                # Maximum precision for the breathing chain: float32 ECC
                # + Lanczos pre-warp + 4096px cascade per pair (or the fast
                # default estimators when the user opts for a quick stack)
                M, meta = self.estimate_pair_transform(
                    ref, src, alignment_mode=pair_mode, scale_factor=scale_factor,
                    precision=precision,
                )
                meta["scale"] = self.transform_scale(M) if M is not None else 1.0

                # Plausibility check: focus breathing never changes
                # magnification by large steps between consecutive frames.
                if M is not None and not lo <= meta["scale"] <= hi:
                    meta["rejected_scale"] = meta["scale"]
                    Mt, tmeta = self._estimate_translation(ref, src, scale_factor)
                    if tmeta["success"] >= 0.3:
                        M = Mt
                        meta = {**tmeta, "scale": 1.0,
                                "fallback": "translation",
                                "rejected_scale": meta["rejected_scale"],
                                "pair": i}
                    else:
                        M = None
                        meta["fallback"] = "hold"
                del ref, src
            except Exception as exc:  # defensive: keep the stack running
                M, meta = None, {**meta, "method": "error", "error": str(exc), "pair": i}
            finally:
                _release(i)
                _release(i + 1)

            return i, M, meta, time.time() - t0

        # Pair deltas are independent, so they are estimated in parallel; the
        # chain is composed as they arrive in order: pair i-1 ↔ i is the
        # forward map frame i-1 coords → frame i coords.
        deltas = [None] * (n - 1)
        metas = [None] * (n - 1)
        transforms = [np.eye(2, 3, dtype=np.float32) for _ in range(n)]
        log = [None] * n
        log[r] = {"frame": r, "step_scale": 1.0, "scale": 1.0, "angle": 0.0,
                  "method": "reference", "fallbacks": 0}
        identity = np.eye(2, 3, dtype=np.float32)
        up_delta = [None]   # last successful delta above the reference

        def _chain_up(i):
            """transforms[i] = Δ(i-1→i) · transforms[i-1], for i > r."""
            meta = metas[i - 1] or {}
            M_delta, fallbacks = deltas[i - 1], (1 if meta.get("fallback") else 0)
            if M_delta is None:
                # Hold: assume a constant breathing rate (reuse the last
                # successful delta); identity if none so far.
                M_delta = up_delta[0] if up_delta[0] is not None else identity
                fallbacks += 1
                meta = {**meta, "held": True}
            else:
                up_delta[0] = M_delta
            transforms[i] = self.compose_affine(transforms[i - 1], M_delta)
            log[i] = {
                "frame": i,
                "step_scale": self.transform_scale(M_delta),
                "scale": self.transform_scale(transforms[i]),
                "angle": self.transform_rotation(transforms[i]),
                "method": meta.get("method", "?"),
                "fallbacks": fallbacks,
            }

        def _chain_down():
            """Reference → start of stack: M_ref→(i-1) = Δ(i-1→i)⁻¹ · M_ref→i."""
            prev_delta = None
            for i in range(r, 0, -1):
                meta = metas[i - 1] or {}
                fallbacks = 1 if meta.get("fallback") else 0
                M_delta = deltas[i - 1]
                if M_delta is None:
                    # Hold: reuse the last successful delta (constant breathing
                    # rate); identity if none so far.
                    M_delta = prev_delta if prev_delta is not None else identity
                    fallbacks += 1
                    meta = {**meta, "held": True}
                else:
                    prev_delta = M_delta
                D = self.invert_affine(M_delta)  # maps frame i coords → frame i-1 coords
                transforms[i - 1] = self.compose_affine(transforms[i], D)
                log[i - 1] = {
                    "frame": i - 1,
                    "step_scale": self.transform_scale(M_delta),
                    "scale": self.transform_scale(transforms[i - 1]),
                    "angle": self.transform_rotation(transforms[i - 1]),
                    "method": meta.get("method", "?"),
                    "fallbacks": fallbacks,
                }

        def _ready(i):
            if on_ready is not None:
                on_ready(i, transforms[i])

        n_workers = self._sequential_worker_count(ref_template, precision, n - 1)
        total = n - 1
        # Run OpenCV single-threaded inside the workers: nested thread pools
        # both oversubscribe the CPU and introduce run-to-run numeric jitter
        # in ECC (parallel reduction order depends on scheduling). With
        # single-threaded OpenCV, each pair estimate is a pure function of
        # its two images — the breathing chain becomes reproducible.
        prev_cv_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        executor = ThreadPoolExecutor(max_workers=n_workers)
        try:
            futs = [executor.submit(_estimate_pair_step, i) for i in range(n - 1)]
            if r == 0:
                _ready(0)
            t_last = time.time()
            for i in range(n - 1):
                self.wait_if_paused()
                if self.is_cancelled:
                    logger.info(f"Sequential alignment cancelled at pair {i}")
                    return None, None
                idx, M, meta, _dt = futs[i].result()
                deltas[idx] = M
                metas[idx] = meta
                if emit is not None:
                    emit(i + 1, total, time.time() - t_last)
                    t_last = time.time()
                if i == r - 1:
                    _chain_down()
                    for k in range(r + 1):
                        _ready(k)
                elif i >= r:
                    _chain_up(i + 1)
                    _ready(i + 1)
        finally:
            # Wait for pairs already running (pending ones are dropped) so a
            # cancelled stack doesn't leave workers holding full-size images
            executor.shutdown(wait=True, cancel_futures=True)
            cv2.setNumThreads(prev_cv_threads)

        return transforms, log

    def generate_laplacian_pyramid(self, im1, num_levels):
        if isinstance(im1, str):
            im1 = self.load_image(im1)
        return CPU.generate_laplacian_pyramid(im1, num_levels)

    def reconstruct_pyramid(self, laplacian_pyr):
        return CPU.reconstruct_pyramid(laplacian_pyr)

    def focus_fuse_pyramid_pair(self, pyr1, pyr2, kernel_size,
                               contrast_threshold=0.0, feather_radius=0):
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
                else:
                    # Vectorized focusmap (cv2.blur variance, O(1)/pixel)
                    current_focusmap = CPU.compute_focusmap_fast(
                        pyr1[pyramid_level], pyr2[pyramid_level], kernel_size,
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
