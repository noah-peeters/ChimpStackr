"""
Focus stacking algorithms on CPU accelerated with Numba's njit.

Supports these stacking methods:
  - Laplacian Pyramid: Best for fine detail (hairs, bristles)
  - Hybrid: Pyramid detail with focus-weighted low frequencies (no halos)
  - Weighted Average: Smooth blending based on local contrast weights
  - Depth Map: Selects entire regions from the sharpest source frame
"""
import math

import numpy as np
import numba as nb
import cv2


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

@nb.njit(
    nb.float32[:, :](nb.float32[:, :], nb.int64),
    fastmath=True, cache=True,
)
def pad_array(array, kernel_size):
    y_shape = array.shape[0]
    x_shape = array.shape[1]
    y_pad = kernel_size - y_shape
    x_pad = kernel_size - x_shape
    if y_pad > 0 or x_pad > 0:
        padded_array = np.zeros((y_shape + y_pad, x_shape + x_pad), dtype=array.dtype)
        padded_array[0:y_shape, 0:x_shape] = array
        return padded_array
    return array


@nb.njit(
    nb.float32(nb.float32[:, :]),
    fastmath=True, cache=True,
)
def get_deviation(matrix):
    summed_deviation = float(0)
    average_value = np.mean(matrix)
    kernel_area = matrix.shape[0] * matrix.shape[1]
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            summed_deviation += (matrix[y, x] - average_value) ** 2 / kernel_area
    return summed_deviation


def gaussian_pyramid(img, num_levels):
    """Calculate Gaussian pyramid."""
    lower = img if img.dtype == np.float32 else img.astype(np.float32)
    gaussian_pyr = [lower]
    for _ in range(num_levels):
        lower = cv2.pyrDown(lower)  # returns float32 when input is float32
        gaussian_pyr.append(lower)
    return gaussian_pyr


# ──────────────────────────────────────────────
# Laplacian Pyramid stacking (original algorithm)
# ──────────────────────────────────────────────

@nb.njit(
    nb.uint8[:, :](nb.float32[:, :], nb.float32[:, :], nb.int64),
    fastmath=True, parallel=True, cache=True,
)
def compute_focusmap(pyr_level1, pyr_level2, kernel_size):
    y_range = pyr_level1.shape[0]
    x_range = pyr_level1.shape[1]
    focusmap = np.empty((y_range, x_range), dtype=np.uint8)
    k = int(kernel_size / 2)
    for y in nb.prange(y_range):
        for x in nb.prange(x_range):
            patch = pyr_level1[y - k : y + k, x - k : x + k]
            padded_patch = pad_array(patch, kernel_size)
            dev1 = get_deviation(padded_patch)
            patch = pyr_level2[y - k : y + k, x - k : x + k]
            padded_patch = pad_array(patch, kernel_size)
            dev2 = get_deviation(padded_patch)
            value_to_insert = 0
            if dev2 > dev1:
                value_to_insert = 1
            focusmap[y, x] = value_to_insert
    return focusmap


@nb.njit(
    nb.uint8[:, :](nb.float32[:, :], nb.float32[:, :], nb.int64, nb.float32),
    fastmath=True, parallel=True, cache=True,
)
def compute_focusmap_thresholded(pyr_level1, pyr_level2, kernel_size, contrast_threshold):
    """
    Like compute_focusmap but with a contrast threshold.
    If neither patch has deviation above the threshold, defaults to pyr1 (0).
    This reduces noise in smooth/out-of-focus areas.
    """
    y_range = pyr_level1.shape[0]
    x_range = pyr_level1.shape[1]
    focusmap = np.empty((y_range, x_range), dtype=np.uint8)
    k = int(kernel_size / 2)
    for y in nb.prange(y_range):
        for x in nb.prange(x_range):
            patch = pyr_level1[y - k : y + k, x - k : x + k]
            padded_patch = pad_array(patch, kernel_size)
            dev1 = get_deviation(padded_patch)
            patch = pyr_level2[y - k : y + k, x - k : x + k]
            padded_patch = pad_array(patch, kernel_size)
            dev2 = get_deviation(padded_patch)
            # If both are below threshold, keep pyr1 (no switching in flat areas)
            if dev1 < contrast_threshold and dev2 < contrast_threshold:
                focusmap[y, x] = 0
            elif dev2 > dev1:
                focusmap[y, x] = 1
            else:
                focusmap[y, x] = 0
    return focusmap


def feather_focusmap(focusmap, radius=3):
    """
    Blur a binary focusmap to create soft transitions between sources.
    Returns a float32 map in [0, 1] range.
    """
    if radius <= 0:
        return focusmap.astype(np.float32) / 255.0
    k = radius * 2 + 1
    soft = cv2.GaussianBlur(focusmap.astype(np.float32), (k, k), 0)
    # Normalize to 0-1
    max_val = soft.max()
    if max_val > 0:
        soft /= max_val
    return soft


def fuse_pyramid_levels_soft(pyr_level1, pyr_level2, soft_focusmap):
    """
    Fuse two pyramid levels using a soft (float) focusmap.
    soft_focusmap: float32 in [0, 1], where 1 = use pyr_level2.
    """
    mask = soft_focusmap
    if pyr_level1.ndim == 3:
        mask = mask[:, :, np.newaxis]
    return (pyr_level1 * (1.0 - mask) + pyr_level2 * mask).astype(np.float32)


@nb.njit(
    nb.float32[:, :, :](nb.float32[:, :, :], nb.float32[:, :, :], nb.uint8[:, :]),
    fastmath=True, parallel=True, cache=True,
)
def fuse_pyramid_levels_using_focusmap(pyr_level1, pyr_level2, focusmap):
    output = pyr_level1.copy()
    for y in nb.prange(focusmap.shape[0]):
        for x in nb.prange(focusmap.shape[1]):
            if focusmap[y, x] != 0:
                output[y, x, :] = pyr_level2[y, x, :]
    return output


def generate_laplacian_pyramid(img, num_levels):
    """Generate Laplacian pyramid (from Gaussian pyramid)."""
    gaussian_pyr = gaussian_pyramid(img, num_levels)
    laplacian_top = gaussian_pyr[-1]
    laplacian_pyr = [laplacian_top]
    for i in range(num_levels, 0, -1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
        cv2.subtract(gaussian_pyr[i - 1], gaussian_expanded, dst=gaussian_expanded)
        laplacian_pyr.append(gaussian_expanded)
    return laplacian_pyr


def reconstruct_pyramid(laplacian_pyr):
    """Reconstruct original image from Laplacian pyramid."""
    top = laplacian_pyr[0]
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        expanded = cv2.pyrUp(top, dstsize=size)
        top = cv2.add(laplacian_pyr[i + 1], expanded)
    return top


def local_tone_map(image, strength=0.5):
    """
    Local tone-mapping to compensate for the contrast boost inherent
    in Laplacian pyramid max-contrast selection.

    Max-contrast selection pushes darks darker and brights brighter.
    This uses CLAHE (Contrast Limited Adaptive Histogram Equalization)
    blended with the original to bring the dynamic range back.

    Args:
        image: float32 BGR in 0-255 range
        strength: 0.0 = no tone-mapping, 1.0 = full CLAHE. Default 0.5.
    """
    if strength <= 0:
        return image

    # Work in float LAB and evaluate the CLAHE curves at full precision, so
    # the result isn't quantized to 256 levels (which an 8-bit round trip
    # would do to every stacked output). Same curves as cv2's CLAHE.
    img = np.clip(image, 0, 255).astype(np.float32) / 255.0
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)   # L in 0..100
    l_255 = np.clip(lab[:, :, 0] * 2.55, 0, 255)

    # CLAHE with moderate clip limit
    luts = _clahe_luts(np.round(l_255).astype(np.uint8), clip_limit=2.0, grid=8)
    l_corrected = _clahe_apply(luts, l_255)

    # Blend original L with CLAHE L
    lab[:, :, 0] = (l_255 * (1.0 - strength) + l_corrected * strength) / 2.55

    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return np.clip(result * 255.0, 0, 255).astype(np.float32)


def _clahe_luts(gray_u8, clip_limit=2.0, grid=8):
    """Per-tile CLAHE lookup tables, computed like cv2.createCLAHE().

    Returns (luts[grid, grid, 256] float32, tile_w, tile_h).
    """
    h, w = gray_u8.shape
    pad_y, pad_x = (-h) % grid, (-w) % grid
    src = cv2.copyMakeBorder(gray_u8, 0, pad_y, 0, pad_x, cv2.BORDER_REFLECT_101) \
        if pad_x or pad_y else gray_u8
    th, tw = src.shape[0] // grid, src.shape[1] // grid
    total = tw * th
    limit = max(int(clip_limit * total / 256), 1)
    luts = np.empty((grid, grid, 256), np.float32)
    for ty in range(grid):
        for tx in range(grid):
            tile = src[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw]
            hist = np.bincount(tile.ravel(), minlength=256).astype(np.int64)
            clipped = int(np.maximum(hist - limit, 0).sum())
            hist = np.minimum(hist, limit)
            hist += clipped // 256
            residual = clipped - (clipped // 256) * 256
            if residual:
                step = max(256 // residual, 1)
                idx = np.arange(0, 256, step)[:residual]
                hist[idx] += 1
            luts[ty, tx] = np.clip(np.round(np.cumsum(hist) * (255.0 / total)), 0, 255)
    return luts, tw, th


def _clahe_apply(lut_info, values):
    """Evaluate CLAHE at float values (0..255): bilinear between tiles, linear
    between LUT entries."""
    luts, tw, th = lut_info
    grid = luts.shape[0]
    h, w = values.shape

    def axis(n, size):
        f = np.arange(n, dtype=np.float32) / size - 0.5
        t1 = np.floor(f)
        a = f - t1
        t1 = t1.astype(np.int64)
        t2 = np.minimum(t1 + 1, grid - 1)
        return np.maximum(t1, 0), t2, a.astype(np.float32)

    x1, x2, xa = axis(w, tw)
    y1, y2, ya = axis(h, th)
    lo = np.floor(values).astype(np.int64)
    np.clip(lo, 0, 255, out=lo)
    hi = np.minimum(lo + 1, 255)
    frac = (values - lo).astype(np.float32)

    table = luts.reshape(-1)
    out = np.zeros((h, w), np.float32)
    for ty, wy in ((y1, 1.0 - ya), (y2, ya)):
        for tx, wx in ((x1, 1.0 - xa), (x2, xa)):
            offset = (ty[:, None] * grid + tx[None, :]) * 256    # start of each pixel's tile LUT
            v = table[offset + lo] * (1.0 - frac) + table[offset + hi] * frac
            out += v * (wy[:, None] * wx[None, :])
    return out


# ──────────────────────────────────────────────
# Hybrid pyramid stacking
# ──────────────────────────────────────────────
#
# Max-contrast selection on every pyramid level makes halos: at coarse
# levels an out-of-focus frame often has *more* local contrast next to an
# edge, because the blurred subject spills into the background. Halos only
# form where there is no in-focus detail, so Hybrid gates on that:
#   - on the subject (some frame has real fine detail there), every level
#     keeps max-contrast selection, for full crispness;
#   - elsewhere (background, the area around edges) coarse levels and the
#     base are a blend of all frames weighted by each frame's fine-detail
#     sharpness, so the frame where a nearby edge is in focus supplies the
#     low frequencies and out-of-focus spill can't leak in;
#   - on the fine levels, areas with no detail above the noise level are
#     blended instead of picking whichever frame is noisiest.
# A mild contrast boost is then applied to the fine detail bands on the
# subject only (at full precision), so smooth backgrounds stay clean.

HYBRID_WEIGHT_POWER = 4.0      # sharpness^p: higher = closer to picking one frame
HYBRID_NOISE_FACTOR = 4.0      # detail must exceed this x noise variance to be max-selected
HYBRID_SHARPNESS_KSIZE = 5     # window for the fine-detail sharpness measure
HYBRID_SPREAD_SIGMA = 2.0      # extra blur of coarse-level weights (in that level's pixels)
HYBRID_DETAIL_GATE = 2.0       # "subject" = sharpness above this x the background floor
HYBRID_DETAIL_GAIN = 0.08      # contrast boost of the fine detail bands on the subject


def hybrid_fine_level_count(shape, num_levels):
    """Number of finest pyramid levels that keep max-contrast selection.

    Halo width grows with resolution, so this grows with it too (3 levels
    at ~3000 px on the long side, 4 at ~6000 px).
    """
    longest = max(shape[:2])
    n = 3 + int(round(math.log2(max(longest, 1) / 3000.0)))
    return int(min(max(n, 2), num_levels))


def fine_sharpness(image, ksize=HYBRID_SHARPNESS_KSIZE):
    """Per-pixel fine-detail sharpness (sum-modified Laplacian, box-filtered)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float32, copy=False)
    lap_x = cv2.filter2D(gray, cv2.CV_32F, np.array([[1, -2, 1]], dtype=np.float32))
    lap_y = cv2.filter2D(gray, cv2.CV_32F, np.array([[1], [-2], [1]], dtype=np.float32))
    return cv2.boxFilter(np.abs(lap_x) + np.abs(lap_y), -1, (ksize, ksize))


def _level_variance(level, kernel_size):
    """Local variance of a pyramid level (grayscale)."""
    gray = cv2.cvtColor(level, cv2.COLOR_BGR2GRAY) if level.ndim == 3 else level
    k = (kernel_size, kernel_size)
    mean = cv2.blur(gray, k)
    return cv2.blur(gray * gray, k) - mean * mean


def _per_channel(weights, like):
    return weights[:, :, np.newaxis] if like.ndim == 3 else weights


class HybridFuser:
    """Streaming hybrid fusion: add frames one at a time, then call result().

    prepare() (pyramid + sharpness) is independent per frame and can run in a
    worker thread; add() must be called from one thread.
    """

    def __init__(self, num_levels, kernel_size=6, detail_gain=HYBRID_DETAIL_GAIN):
        self.num_levels = num_levels
        self.kernel_size = max(2, int(kernel_size))
        self.detail_gain = detail_gain
        self._n_fine = None
        self._num = []         # per level: sum of weight * coefficient
        self._den = []         # per level: sum of weights
        self._best = []        # per level: max-contrast coefficients
        self._best_var = []    # per level: their local variance
        self._max_sharp = []   # per level: max fine-detail sharpness over frames
        self._noise_var = {}   # fine levels: noise variance estimate
        self._sharp_floor = None  # background sharpness level (first frame)
        self.frames = 0

    def prepare(self, image):
        """Image pyramid and matching sharpness pyramid for one frame (thread-safe).

        The sharpness pyramid is a Gaussian pyramid of the fine-detail
        sharpness map, so a coarse level's weight aggregates in-focus evidence
        over the area that level covers: a sharp edge then also claims the
        surrounding area where out-of-focus frames would spill their blur.
        """
        sharp = [fine_sharpness(image)]
        for _ in range(self.num_levels):
            sharp.append(cv2.pyrDown(sharp[-1]))
        return generate_laplacian_pyramid(image, self.num_levels), sharp[::-1]

    def add(self, pyramid, sharp_pyramid):
        n = len(pyramid)
        if self._n_fine is None:
            self._n_fine = hybrid_fine_level_count(pyramid[-1].shape, n - 1)
            # Background sharpness: a low percentile of the first frame, which in
            # a focus stack is mostly out of focus
            self._sharp_floor = float(np.percentile(sharp_pyramid[-1], 10)) + 1e-3
        first_fine = n - self._n_fine

        for i, level in enumerate(pyramid):
            sharp = sharp_pyramid[i]
            weight = sharp
            if i < first_fine:
                weight = cv2.GaussianBlur(weight, (0, 0), HYBRID_SPREAD_SIGMA)
            weight = np.power(weight + 1e-3, HYBRID_WEIGHT_POWER, dtype=np.float32)
            weighted = level * _per_channel(weight, level)
            var = _level_variance(level, self.kernel_size)
            if self.frames == 0:
                self._num.append(weighted)
                self._den.append(weight)
                self._best.append(level.copy())
                self._best_var.append(var)
                self._max_sharp.append(sharp.copy())
                if i >= first_fine:
                    # Robust (MAD) noise estimate from the first frame's coefficients
                    gray = cv2.cvtColor(level, cv2.COLOR_BGR2GRAY) if level.ndim == 3 else level
                    sigma = float(np.median(np.abs(gray))) / 0.6745
                    self._noise_var[i] = max(sigma * sigma, 1e-6)
            else:
                self._num[i] += weighted
                self._den[i] += weight
                mask = var > self._best_var[i]
                self._best[i][mask] = level[mask]
                np.maximum(self._best_var[i], var, out=self._best_var[i])
                np.maximum(self._max_sharp[i], sharp, out=self._max_sharp[i])
        self.frames += 1

    def _subject_mask(self, i, blur):
        """Soft 0..1 map of where some frame has real fine detail (level i)."""
        gate = self._max_sharp[i] / (HYBRID_DETAIL_GATE * self._sharp_floor) - 1.0
        np.clip(gate, 0.0, 1.0, out=gate)
        return cv2.GaussianBlur(gate, (0, 0), blur)

    def result(self):
        """Fused image (float32), or None if no frames were added."""
        if self.frames == 0:
            return None
        n = len(self._num)
        first_fine = n - self._n_fine
        levels = []
        for i in range(n):
            blend = self._num[i] / _per_channel(self._den[i], self._num[i])
            if i >= first_fine:
                # fine levels: max contrast wherever there is detail above noise
                alpha = self._best_var[i] / (HYBRID_NOISE_FACTOR * self._noise_var[i]) - 1.0
                np.clip(alpha, 0.0, 1.0, out=alpha)
                alpha = cv2.GaussianBlur(alpha, (0, 0), 2.0)
            else:
                # coarse levels: max contrast only on the subject
                alpha = self._subject_mask(i, 1.0)
            alpha = _per_channel(alpha, blend)
            fused = alpha * self._best[i] + (1.0 - alpha) * blend
            if i >= first_fine and self.detail_gain > 0:
                # fine bands only: boosting coarse bands near strong edges
                # would itself create halos (like heavy unsharp masking)
                fused *= 1.0 + self.detail_gain * _per_channel(self._subject_mask(i, 2.0), fused)
            levels.append(fused.astype(np.float32, copy=False))
        return reconstruct_pyramid(levels)


# ──────────────────────────────────────────────
# Weighted Average stacking
# ──────────────────────────────────────────────

def compute_focus_weights(image, kernel_size=5):
    """
    Compute per-pixel focus weight map using local Laplacian energy.
    Higher weight = more in-focus.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = gray.astype(np.float32)

    # Laplacian gives high response at edges/detail
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    # Square to get energy, then blur to get local focus measure
    energy = lap * lap
    weights = cv2.GaussianBlur(energy, (kernel_size | 1, kernel_size | 1), 0)
    return weights


def weighted_average_fuse_pair(img1, img2, kernel_size=5):
    """
    Fuse two images using weighted average based on local contrast.
    Each pixel is a weighted blend from the two sources.
    Returns float32 image.
    """
    w1 = compute_focus_weights(img1, kernel_size)
    w2 = compute_focus_weights(img2, kernel_size)

    # Normalize weights (avoid division by zero)
    total = w1 + w2 + 1e-12
    w1_norm = w1 / total
    w2_norm = w2 / total

    # Expand weights for color channels
    if img1.ndim == 3:
        w1_norm = w1_norm[:, :, np.newaxis]
        w2_norm = w2_norm[:, :, np.newaxis]

    result = img1.astype(np.float32) * w1_norm + img2.astype(np.float32) * w2_norm
    return result.astype(np.float32)


def weighted_average_fuse_multi(images, kernel_size=5):
    """
    Fuse N images at once using weighted average (more accurate than pairwise).
    images: list of float32 ndarrays (same shape).
    """
    if len(images) == 0:
        return None
    if len(images) == 1:
        return images[0].astype(np.float32)

    weights = []
    for img in images:
        weights.append(compute_focus_weights(img, kernel_size))

    # Stack and normalize
    weight_stack = np.stack(weights, axis=0)  # (N, H, W)
    total = weight_stack.sum(axis=0, keepdims=True) + 1e-12
    weight_stack = weight_stack / total  # (N, H, W) normalized

    result = np.zeros_like(images[0], dtype=np.float32)
    for i, img in enumerate(images):
        w = weight_stack[i]
        if img.ndim == 3:
            w = w[:, :, np.newaxis]
        result += img.astype(np.float32) * w

    return result


# ──────────────────────────────────────────────
# Depth Map stacking
# ──────────────────────────────────────────────

def compute_depthmap_index(images, kernel_size=11):
    """
    For each pixel, determine which source image is sharpest.
    Returns an index map (H, W) of dtype int32 where each value
    is the index of the sharpest image at that pixel.
    """
    if len(images) == 0:
        return None

    # Compute sharpness maps for all images
    sharpness_maps = []
    for img in images:
        if img.ndim == 3:
            gray = cv2.cvtColor(img.astype(np.float32) if img.dtype != np.float32 else img,
                                cv2.COLOR_BGR2GRAY)
        else:
            gray = img.astype(np.float32)

        # Modified Laplacian (better than standard Laplacian for depth estimation)
        lap_x = cv2.filter2D(gray, cv2.CV_32F, np.array([[1, -2, 1]], dtype=np.float32))
        lap_y = cv2.filter2D(gray, cv2.CV_32F, np.array([[1], [-2], [1]], dtype=np.float32))
        ml = np.abs(lap_x) + np.abs(lap_y)

        # Sum in local window for robustness
        ksize = kernel_size | 1  # ensure odd
        sharpness = cv2.boxFilter(ml, -1, (ksize, ksize))
        sharpness_maps.append(sharpness)

    # Stack and argmax
    stack = np.stack(sharpness_maps, axis=0)  # (N, H, W)
    depth_index = np.argmax(stack, axis=0).astype(np.int32)  # (H, W)

    # Optional: smooth the depth map to reduce noise
    depth_smoothed = cv2.medianBlur(depth_index.astype(np.uint8), kernel_size | 1)
    return depth_smoothed.astype(np.int32)


def depthmap_fuse_multi(images, kernel_size=11):
    """
    Fuse N images using depth map selection.
    For each pixel, picks the pixel from the sharpest source image.
    """
    if len(images) == 0:
        return None
    if len(images) == 1:
        return images[0].astype(np.float32)

    depth_index = compute_depthmap_index(images, kernel_size)
    h, w = depth_index.shape

    # Build output by selecting from each image according to depth map
    result = np.empty_like(images[0], dtype=np.float32)
    image_stack = np.stack([img.astype(np.float32) for img in images], axis=0)

    # Vectorized fancy indexing
    rows = np.arange(h)[:, np.newaxis]
    cols = np.arange(w)[np.newaxis, :]
    if images[0].ndim == 3:
        result = image_stack[depth_index, rows, cols, :]
    else:
        result = image_stack[depth_index, rows, cols]

    return result.astype(np.float32), depth_index


# ──────────────────────────────────────────────
# Mertens Exposure Fusion
# ──────────────────────────────────────────────

def _to_u8(img):
    if img.dtype != np.uint8:
        return np.clip(img, 0, 255).astype(np.uint8)
    return img


def mertens_fuse_batch(images, contrast_weight=1.0, saturation_weight=1.0, exposure_weight=0.0):
    """
    Fuse a small batch of images (2-6) using Mertens exposure fusion.
    All images must be in memory. Returns float32 BGR in 0-255 range.
    """
    merger = cv2.createMergeMertens(
        contrast_weight=contrast_weight,
        saturation_weight=saturation_weight,
        exposure_weight=exposure_weight,
    )
    imgs_u8 = [_to_u8(img) for img in images]
    fused = merger.process(imgs_u8)
    return np.clip(fused * 255.0, 0, 255).astype(np.float32)


# ──────────────────────────────────────────────
# Multi-resolution Depth Map
# ──────────────────────────────────────────────

def compute_multires_sharpness(image, scales=(5, 11, 21), smoothing=0):
    """
    Compute sharpness at multiple window sizes and combine.
    Uses Sum Modified Laplacian (Nayar & Nakagawa) — the standard
    focus measure operator used by commercial stackers.

    Args:
        smoothing: edge-aware smoothing radius for the focus map.
            Uses bilateral filter to smooth flat areas while preserving
            depth discontinuities at edges.
            Higher = smoother transitions, fewer artifacts.
            Lower = sharper but more halos at boundaries.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(
            image.astype(np.float32) if image.dtype != np.float32 else image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image.astype(np.float32)

    # Sum Modified Laplacian (SML) — better than standard Laplacian
    # because abs() prevents cancellation of opposing-sign derivatives
    lap_x = cv2.filter2D(gray, cv2.CV_32F, np.array([[1, -2, 1]], dtype=np.float32))
    lap_y = cv2.filter2D(gray, cv2.CV_32F, np.array([[1], [-2], [1]], dtype=np.float32))
    ml = np.abs(lap_x) + np.abs(lap_y)

    # Multi-scale aggregation — evaluate focus at multiple window sizes
    combined = np.zeros_like(ml)
    for s in scales:
        ksize = s | 1
        combined += cv2.boxFilter(ml, -1, (ksize, ksize))
    combined /= len(scales)

    # Edge-aware smoothing using bilateral filter
    # Preserves depth discontinuities at object edges while smoothing
    # noisy depth estimates in flat areas (the key difference from Gaussian)
    if smoothing > 0:
        # Bilateral: spatial sigma from smoothing, range sigma auto-scaled
        d = smoothing * 2 + 1
        sigma_space = float(smoothing * 2)
        sigma_color = float(combined.std() * 2 + 1e-6)
        combined = cv2.bilateralFilter(combined, d, sigma_color, sigma_space)

    return combined
