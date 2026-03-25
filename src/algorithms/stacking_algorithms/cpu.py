"""
Focus stacking algorithms on CPU accelerated with Numba's njit.

Supports three stacking methods:
  - Laplacian Pyramid (PMax-style): Best for fine detail (hairs, bristles)
  - Weighted Average: Smooth blending based on local contrast weights
  - Depth Map: Selects entire regions from the sharpest source frame
"""
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
    laplacian_top = laplacian_pyr[0]
    laplacian_lst = [laplacian_top]
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
        laplacian_top = cv2.add(laplacian_pyr[i + 1], laplacian_expanded)
        laplacian_lst.append(laplacian_top)
    return laplacian_lst[num_levels]


def local_tone_map(image, strength=0.5):
    """
    Local tone-mapping to compensate for the contrast boost inherent
    in Laplacian pyramid max-contrast selection (PMax-style).

    Max-contrast selection pushes darks darker and brights brighter.
    This uses CLAHE (Contrast Limited Adaptive Histogram Equalization)
    blended with the original to bring the dynamic range back.

    Args:
        image: float32 BGR in 0-255 range
        strength: 0.0 = no tone-mapping, 1.0 = full CLAHE. Default 0.5.
    """
    if strength <= 0:
        return image

    img_u8 = np.clip(image, 0, 255).astype(np.uint8)

    # Convert to LAB — only adjust L channel (luminance), preserve color
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # CLAHE with moderate clip limit
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_corrected = clahe.apply(l_channel)

    # Blend original L with CLAHE L
    l_blended = cv2.addWeighted(l_channel, 1.0 - strength, l_corrected, strength, 0)
    lab[:, :, 0] = l_blended

    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result.astype(np.float32)


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
            depth discontinuities at edges (like Zerene's DMap).
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
