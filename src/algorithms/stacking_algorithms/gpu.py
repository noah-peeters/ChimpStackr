"""
GPU-accelerated focus stacking algorithms.

Three acceleration tiers (auto-selected based on available hardware):

  Tier 1 - CuPy GPU-resident (best, ~5x faster):
    All data stays on GPU. Upload image once, download result once.
    Requires: pip install cupy-cuda12x (or cupy-cuda11x)
    Measured: ~1.1s total for 10x 24MP images vs ~5.4s CPU

  Tier 2 - cv2.cuda pyramid + CPU vectorized compute:
    Pyramid ops on GPU, focusmap/fuse on CPU with cv2.blur.
    Requires: OpenCV built with CUDA (custom compile)

  Tier 3 - CPU vectorized (fallback):
    Everything on CPU using cv2.blur variance (O(1)/pixel) and
    Numba parallel fuse. Still 1.35x faster than old per-pixel kernel.
"""
import math
import numpy as np
import cv2

# ── Detect available backends ──
try:
    import cupy as cp
    from cupyx.scipy.ndimage import uniform_filter as _cp_uniform_filter
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    import numba.cuda as cuda
    HAS_NUMBA_CUDA = cuda.is_available()
except ImportError:
    HAS_NUMBA_CUDA = False

HAS_CV_CUDA = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, 'cuda') else False
HAS_CV_CUDA_PYRUP = HAS_CV_CUDA and hasattr(cv2.cuda, 'pyrUp')

# CuPy kernel warmup flag
_cupy_warmed_up = False


# ══════════════════════════════════════════════
# Tier 1: CuPy GPU-resident pipeline
# ══════════════════════════════════════════════

def _cupy_warmup():
    """Compile CuPy kernels on first use (adds ~2s one-time cost)."""
    global _cupy_warmed_up
    if _cupy_warmed_up:
        return
    # Trigger kernel compilation with small arrays
    small = cp.zeros((16, 16), dtype=cp.float32)
    _ = _cp_uniform_filter(small, size=3)
    _ = small * small
    _ = (small > small).astype(cp.uint8)
    _ = cp.where(small > 0, small, small)
    from cupyx.scipy.ndimage import convolve1d as _c1d
    _k = cp.array([1, 4, 6, 4, 1], dtype=cp.float32) / 16.0
    _ = _c1d(small, _k, axis=0)
    cp.cuda.Stream.null.synchronize()
    _cupy_warmed_up = True


# Gaussian 5-tap kernel for pyrDown/pyrUp (same as OpenCV)
_GAUSS_KERNEL_1D = None

def _get_gauss_kernel():
    global _GAUSS_KERNEL_1D
    if _GAUSS_KERNEL_1D is None:
        _GAUSS_KERNEL_1D = cp.array([1, 4, 6, 4, 1], dtype=cp.float32) / 16.0
    return _GAUSS_KERNEL_1D


def _cupy_pyrdown(img_gpu):
    """GPU pyrDown: separable Gaussian blur + 2x subsample."""
    from cupyx.scipy.ndimage import convolve1d
    k = _get_gauss_kernel()
    if img_gpu.ndim == 3:
        channels = []
        for ch in range(img_gpu.shape[2]):
            tmp = convolve1d(img_gpu[:, :, ch], k, axis=0)
            tmp = convolve1d(tmp, k, axis=1)
            channels.append(tmp[::2, ::2])
        return cp.stack(channels, axis=2)
    else:
        tmp = convolve1d(img_gpu, k, axis=0)
        tmp = convolve1d(tmp, k, axis=1)
        return tmp[::2, ::2]


def _cupy_pyrup(img_gpu, target_shape):
    """GPU pyrUp: 2x upsample (zero-insert) + Gaussian blur."""
    from cupyx.scipy.ndimage import convolve1d
    k = _get_gauss_kernel()
    th, tw = target_shape[:2]

    if img_gpu.ndim == 3:
        up = cp.zeros((th, tw, img_gpu.shape[2]), dtype=cp.float32)
        up[::2, ::2, :] = img_gpu * 4  # scale by 4 to match OpenCV convention
        channels = []
        for ch in range(img_gpu.shape[2]):
            tmp = convolve1d(up[:, :, ch], k, axis=0)
            channels.append(convolve1d(tmp, k, axis=1))
        return cp.stack(channels, axis=2)
    else:
        up = cp.zeros((th, tw), dtype=cp.float32)
        up[::2, ::2] = img_gpu * 4
        tmp = convolve1d(up, k, axis=0)
        return convolve1d(tmp, k, axis=1)


def _cupy_gaussian_pyramid(img_gpu, num_levels):
    """Gaussian pyramid entirely on GPU. Returns list of CuPy arrays."""
    pyr = [img_gpu]
    current = img_gpu
    for _ in range(num_levels):
        current = _cupy_pyrdown(current)
        pyr.append(current)
    return pyr


def _cupy_laplacian_pyramid(img_gpu, num_levels):
    """Laplacian pyramid entirely on GPU. Returns list of CuPy arrays."""
    gauss = _cupy_gaussian_pyramid(img_gpu, num_levels)
    lap_pyr = [gauss[-1]]  # lowpass residual
    for i in range(num_levels, 0, -1):
        target_shape = gauss[i - 1].shape
        expanded = _cupy_pyrup(gauss[i], target_shape)
        lap_pyr.append(gauss[i - 1] - expanded)
    return lap_pyr


def _cupy_reconstruct(lap_pyr):
    """Reconstruct from Laplacian pyramid on GPU. Returns CuPy array."""
    top = lap_pyr[0]
    for i in range(len(lap_pyr) - 1):
        target_shape = lap_pyr[i + 1].shape
        expanded = _cupy_pyrup(top, target_shape)
        top = lap_pyr[i + 1] + expanded
    return top


def _cupy_focusmap(level1_gpu, level2_gpu, kernel_size):
    """Compute focusmap on GPU using blur-based variance."""
    k = kernel_size | 1
    if level1_gpu.ndim == 3:
        gray1 = level1_gpu[:, :, 0] * 0.114 + level1_gpu[:, :, 1] * 0.587 + level1_gpu[:, :, 2] * 0.299
        gray2 = level2_gpu[:, :, 0] * 0.114 + level2_gpu[:, :, 1] * 0.587 + level2_gpu[:, :, 2] * 0.299
    else:
        gray1 = level1_gpu
        gray2 = level2_gpu

    mean1 = _cp_uniform_filter(gray1, size=k)
    var1 = _cp_uniform_filter(gray1 * gray1, size=k) - mean1 * mean1

    mean2 = _cp_uniform_filter(gray2, size=k)
    var2 = _cp_uniform_filter(gray2 * gray2, size=k) - mean2 * mean2

    return (var2 > var1).astype(cp.uint8)


def _cupy_fuse(pyr1_level, pyr2_level, focusmap):
    """Fuse two pyramid levels on GPU."""
    if pyr1_level.ndim == 3:
        mask = focusmap[:, :, None].astype(bool)
    else:
        mask = focusmap.astype(bool)
    return cp.where(mask, pyr2_level, pyr1_level)


def _cupy_fuse_pyramid_pair(pyr1, pyr2, kernel_size):
    """Fuse two Laplacian pyramids entirely on GPU.

    All computation stays on GPU — no host transfers between steps.
    """
    threshold_index = len(pyr1) - 1
    new_pyr = []
    current_focusmap = None

    for level in range(len(pyr1)):
        if level < threshold_index:
            current_focusmap = _cupy_focusmap(
                pyr1[level], pyr2[level], kernel_size
            )
        else:
            # Resize focusmap for largest level(s)
            # CuPy doesn't have cv2.resize — use nearest-neighbor via repeat
            s = pyr2[level].shape[:2]
            fm_h, fm_w = current_focusmap.shape
            # Simple nearest-neighbor resize on GPU
            y_idx = (cp.arange(s[0]) * fm_h // s[0]).astype(cp.int32)
            x_idx = (cp.arange(s[1]) * fm_w // s[1]).astype(cp.int32)
            current_focusmap = current_focusmap[cp.ix_(y_idx, x_idx)]

        new_pyr.append(_cupy_fuse(pyr1[level], pyr2[level], current_focusmap))

    return new_pyr


# ──────────────────────────────────────────────
# N-way batch fusion (process many images at once)
# ──────────────────────────────────────────────

def _cupy_variance_gpu(level_gpu, kernel_size):
    """Compute local variance of a single pyramid level on GPU."""
    k = kernel_size | 1
    if level_gpu.ndim == 3:
        gray = level_gpu[:, :, 0] * 0.114 + level_gpu[:, :, 1] * 0.587 + level_gpu[:, :, 2] * 0.299
    else:
        gray = level_gpu
    mean = _cp_uniform_filter(gray, size=k)
    return _cp_uniform_filter(gray * gray, size=k) - mean * mean


def _estimate_gpu_memory_per_image(shape, num_levels):
    """Estimate GPU memory needed per image for N-way fusion (bytes)."""
    h, w = shape[:2]
    channels = shape[2] if len(shape) > 2 else 1
    bytes_per_pixel = 4  # float32

    # Laplacian pyramid: sum of all levels (geometric series ≈ 4/3 of full res)
    pyr_bytes = int(h * w * channels * bytes_per_pixel * 1.34)
    # Variance map (grayscale, one per level)
    var_bytes = int(h * w * bytes_per_pixel * 1.34)
    # Working memory overhead
    overhead = int(h * w * channels * bytes_per_pixel * 0.5)

    return pyr_bytes + var_bytes + overhead


def get_max_batch_size(image_shape, num_levels):
    """Calculate how many images can fit in GPU memory for N-way fusion."""
    if not HAS_CUPY:
        return 2  # CPU path: pairwise only

    free_mem, total_mem = cp.cuda.Device(0).mem_info
    # Reserve 500MB for CuPy overhead, kernel code, etc.
    usable = free_mem - 500 * 1024**2
    per_image = _estimate_gpu_memory_per_image(image_shape, num_levels)
    # Need at least 2 images for fusion
    return max(2, int(usable / per_image))


def cupy_fuse_n_way(images_np, num_levels, kernel_size, progress_callback=None):
    """N-way Laplacian pyramid fusion: process all images in one pass on GPU.

    Instead of N-1 sequential pairwise fusions, computes variance for ALL
    images at each pyramid level and selects pixels from the sharpest source.
    This is both faster (one pass) and more accurate (compares clean originals
    instead of fused intermediates).

    If images don't fit in GPU memory, processes in batches and merges.

    Args:
        images_np: list of numpy float32 images (all same shape)
        num_levels: number of pyramid levels
        kernel_size: focus comparison kernel size
        progress_callback: optional fn(current, total) for progress updates

    Returns:
        list of numpy arrays (fused Laplacian pyramid)
    """
    _cupy_warmup()
    N = len(images_np)
    if N == 0:
        return None
    if N == 1:
        img_gpu = cp.asarray(images_np[0])
        pyr = _cupy_laplacian_pyramid(img_gpu, num_levels)
        return [cp.asnumpy(l) for l in pyr]

    # Check batch size
    max_batch = get_max_batch_size(images_np[0].shape, num_levels)

    if N <= max_batch:
        # All fit in GPU — single pass
        return _cupy_fuse_n_way_batch(images_np, num_levels, kernel_size, progress_callback)
    else:
        # Process in batches, then merge batch results
        results = []
        for batch_start in range(0, N, max_batch):
            batch = images_np[batch_start:batch_start + max_batch]
            batch_pyr = _cupy_fuse_n_way_batch(batch, num_levels, kernel_size)
            results.append(batch_pyr)
            if progress_callback:
                progress_callback(min(batch_start + max_batch, N), N)

        # Merge batch results pairwise on GPU
        while len(results) > 1:
            merged = []
            for i in range(0, len(results), 2):
                if i + 1 < len(results):
                    # Upload both pyramids, fuse on GPU
                    g1 = [cp.asarray(l) for l in results[i]]
                    g2 = [cp.asarray(l) for l in results[i + 1]]
                    fused = _cupy_fuse_pyramid_pair(g1, g2, kernel_size)
                    cp.cuda.Stream.null.synchronize()
                    merged.append([cp.asnumpy(l) for l in fused])
                else:
                    merged.append(results[i])
            results = merged

        return results[0]


def _cupy_fuse_n_way_batch(images_np, num_levels, kernel_size, progress_callback=None):
    """Fuse N images in one pass on GPU using argmax over variances.

    All images must fit in GPU memory simultaneously.
    """
    N = len(images_np)
    k = kernel_size | 1

    # Upload all images and build pyramids on GPU
    pyramids = []
    for idx, img in enumerate(images_np):
        img_gpu = cp.asarray(img if img.dtype == np.float32 else img.astype(np.float32))
        pyr = _cupy_laplacian_pyramid(img_gpu, num_levels)
        pyramids.append(pyr)
        del img_gpu  # free the raw image (pyramid holds the data now)
        if progress_callback:
            progress_callback(idx + 1, N * 2)  # first half = building pyramids

    num_total_levels = len(pyramids[0])
    threshold_index = num_total_levels - 1
    fused_pyr = []
    best_idx = None

    for level in range(num_total_levels):
        if level < threshold_index:
            # Compute variance for ALL images at this level
            variances = []
            for p in range(N):
                variances.append(_cupy_variance_gpu(pyramids[p][level], kernel_size))

            # Stack and argmax: which image is sharpest at each pixel
            var_stack = cp.stack(variances, axis=0)  # (N, H, W)
            best_idx = cp.argmax(var_stack, axis=0)  # (H, W) values in [0, N-1]
            del var_stack, variances
        else:
            # Resize best_idx for largest level(s) (nearest-neighbor on GPU)
            s = pyramids[0][level].shape[:2]
            fm_h, fm_w = best_idx.shape
            y_idx = (cp.arange(s[0]) * fm_h // s[0]).astype(cp.int32)
            x_idx = (cp.arange(s[1]) * fm_w // s[1]).astype(cp.int32)
            best_idx = best_idx[cp.ix_(y_idx, x_idx)]

        # Select pixels from the winning image at this level
        # Stack all images' level: (N, H, W, C) or (N, H, W)
        level_stack = cp.stack([pyramids[p][level] for p in range(N)], axis=0)

        h, w = best_idx.shape
        rows = cp.arange(h)[:, None]
        cols = cp.arange(w)[None, :]
        if level_stack.ndim == 4:  # (N, H, W, C)
            fused_level = level_stack[best_idx, rows, cols, :]
        else:  # (N, H, W)
            fused_level = level_stack[best_idx, rows, cols]

        fused_pyr.append(fused_level)
        del level_stack

    # Free pyramid memory
    del pyramids
    cp.cuda.Stream.null.synchronize()

    # Download result
    return [cp.asnumpy(l) for l in fused_pyr]


# ══════════════════════════════════════════════
# Tier 3: CPU vectorized (fallback)
# ══════════════════════════════════════════════

def _local_variance(gray, kernel_size):
    """Per-pixel local variance: Var(X) = E[X^2] - E[X]^2, O(1)/pixel."""
    k = kernel_size | 1
    ksize = (k, k)
    mean = cv2.blur(gray, ksize)
    sq_mean = cv2.blur(gray * gray, ksize)
    var = sq_mean - mean * mean
    np.maximum(var, 0, out=var)
    return var


def compute_focusmap_fast(level1, level2, kernel_size):
    """Compute focus map using fast vectorized local variance (CPU)."""
    if level1.ndim == 3:
        gray1 = cv2.cvtColor(level1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(level2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = level1
        gray2 = level2
    if gray1.dtype != np.float32:
        gray1 = gray1.astype(np.float32)
    if gray2.dtype != np.float32:
        gray2 = gray2.astype(np.float32)
    var1 = _local_variance(gray1, kernel_size)
    var2 = _local_variance(gray2, kernel_size)
    return (var2 > var1).astype(np.uint8)


def fuse_levels_fast(pyr_level1, pyr_level2, focusmap):
    """Fuse using Numba parallel in-place copy (CPU, faster than np.where)."""
    from src.algorithms.stacking_algorithms import cpu as _CPU
    return _CPU.fuse_pyramid_levels_using_focusmap(pyr_level1, pyr_level2, focusmap)


# ══════════════════════════════════════════════
# Pyramid operations
# ══════════════════════════════════════════════

def gaussian_pyramid(img, num_levels):
    """Calculate Gaussian pyramid."""
    if HAS_CV_CUDA:
        return _gaussian_pyramid_cuda(img, num_levels)
    lower = img if img.dtype == np.float32 else img.astype(np.float32)
    pyr = [lower]
    for _ in range(num_levels):
        lower = cv2.pyrDown(lower)
        pyr.append(lower)
    return pyr


def _gaussian_pyramid_cuda(img, num_levels):
    """Gaussian pyramid using OpenCV CUDA."""
    gpu_mat = cv2.cuda_GpuMat()
    gpu_mat.upload(img.astype(np.float32))
    gpu_mats = [gpu_mat]
    for _ in range(num_levels):
        gpu_mat = cv2.cuda.pyrDown(gpu_mat)
        gpu_mats.append(gpu_mat)
    return [g.download().astype(np.float32) for g in gpu_mats]


def generate_laplacian_pyramid(img, num_levels):
    """Generate Laplacian pyramid."""
    if HAS_CV_CUDA_PYRUP:
        return _generate_laplacian_pyramid_cuda(img, num_levels)
    return _generate_laplacian_pyramid_cpu(img, num_levels)


def _generate_laplacian_pyramid_cuda(img, num_levels):
    gpu_mat = cv2.cuda_GpuMat()
    gpu_mat.upload(img.astype(np.float32))
    gpu_gauss = [gpu_mat]
    for _ in range(num_levels):
        gpu_mat = cv2.cuda.pyrDown(gpu_mat)
        gpu_gauss.append(gpu_mat)
    lap_pyr = [gpu_gauss[-1].download().astype(np.float32)]
    for i in range(num_levels, 0, -1):
        size = (gpu_gauss[i - 1].size()[0], gpu_gauss[i - 1].size()[1])
        expanded = cv2.cuda.pyrUp(gpu_gauss[i], dstsize=size)
        lap_gpu = cv2.cuda.subtract(gpu_gauss[i - 1], expanded)
        lap_pyr.append(lap_gpu.download().astype(np.float32))
    return lap_pyr


def _generate_laplacian_pyramid_cpu(img, num_levels):
    gauss = gaussian_pyramid(img, num_levels)
    lap_pyr = [gauss[-1]]
    for i in range(num_levels, 0, -1):
        size = (gauss[i - 1].shape[1], gauss[i - 1].shape[0])
        expanded = cv2.pyrUp(gauss[i], dstsize=size)
        cv2.subtract(gauss[i - 1], expanded, dst=expanded)
        lap_pyr.append(expanded)
    return lap_pyr


def reconstruct_pyramid(laplacian_pyr):
    """Reconstruct image from Laplacian pyramid."""
    if HAS_CV_CUDA_PYRUP:
        return _reconstruct_pyramid_cuda(laplacian_pyr)
    return _reconstruct_pyramid_cpu(laplacian_pyr)


def _reconstruct_pyramid_cuda(laplacian_pyr):
    gpu_top = cv2.cuda_GpuMat()
    gpu_top.upload(laplacian_pyr[0].astype(np.float32))
    for i in range(len(laplacian_pyr) - 1):
        next_level = laplacian_pyr[i + 1]
        size = (next_level.shape[1], next_level.shape[0])
        expanded = cv2.cuda.pyrUp(gpu_top, dstsize=size)
        gpu_next = cv2.cuda_GpuMat()
        gpu_next.upload(next_level.astype(np.float32))
        gpu_top = cv2.cuda.add(gpu_next, expanded)
    return gpu_top.download()


def _reconstruct_pyramid_cpu(laplacian_pyr):
    top = laplacian_pyr[0]
    for i in range(len(laplacian_pyr) - 1):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        expanded = cv2.pyrUp(top, dstsize=size)
        top = cv2.add(laplacian_pyr[i + 1], expanded)
    return top


# ══════════════════════════════════════════════
# Main entry points (auto-select best backend)
# ══════════════════════════════════════════════

def fuse_pyramid_pair_gpu(pyr1, pyr2, kernel_size):
    """Fuse two Laplacian pyramids using best available backend.

    CuPy path: upload both pyramids, fuse entirely on GPU, download result.
    CPU path: cv2.blur variance + Numba fuse (still fast, 1.35x over old code).
    """
    if HAS_CUPY:
        return _fuse_cupy(pyr1, pyr2, kernel_size)
    return _fuse_cpu_vectorized(pyr1, pyr2, kernel_size)


def generate_and_fuse_gpu(img1, img2, num_levels, kernel_size):
    """Build pyramids AND fuse in one call — optimal for CuPy path.

    Keeps everything on GPU from raw image to fused pyramid.
    Only useful with CuPy; falls back to separate steps otherwise.
    """
    if HAS_CUPY:
        _cupy_warmup()
        g1 = cp.asarray(img1 if img1.dtype == np.float32 else img1.astype(np.float32))
        g2 = cp.asarray(img2 if img2.dtype == np.float32 else img2.astype(np.float32))
        pyr1 = _cupy_laplacian_pyramid(g1, num_levels)
        pyr2 = _cupy_laplacian_pyramid(g2, num_levels)
        fused = _cupy_fuse_pyramid_pair(pyr1, pyr2, kernel_size)
        cp.cuda.Stream.null.synchronize()
        return fused
    # Fallback
    pyr1 = generate_laplacian_pyramid(img1, num_levels)
    pyr2 = generate_laplacian_pyramid(img2, num_levels)
    return fuse_pyramid_pair_gpu(pyr1, pyr2, kernel_size)


def _fuse_cupy(pyr1, pyr2, kernel_size):
    """Fuse using CuPy: upload, compute on GPU, download."""
    _cupy_warmup()
    # Upload all levels
    g_pyr1 = [cp.asarray(l) for l in pyr1]
    g_pyr2 = [cp.asarray(l) for l in pyr2]
    # Fuse on GPU
    fused_gpu = _cupy_fuse_pyramid_pair(g_pyr1, g_pyr2, kernel_size)
    # Download
    cp.cuda.Stream.null.synchronize()
    return [cp.asnumpy(l) for l in fused_gpu]


def _fuse_cpu_vectorized(pyr1, pyr2, kernel_size):
    """Fuse using CPU vectorized ops (Tier 3 fallback)."""
    threshold_index = len(pyr1) - 1
    new_pyr = []
    current_focusmap = None

    for level in range(len(pyr1)):
        if level < threshold_index:
            current_focusmap = compute_focusmap_fast(
                pyr1[level], pyr2[level], kernel_size
            )
        else:
            s = pyr2[level].shape
            current_focusmap = cv2.resize(
                current_focusmap, (s[1], s[0]), interpolation=cv2.INTER_AREA
            )
        new_pyr.append(fuse_levels_fast(pyr1[level], pyr2[level], current_focusmap))

    return new_pyr


# ══════════════════════════════════════════════
# Legacy API (backward compatibility)
# ══════════════════════════════════════════════

def compute_focusmap(array1, array2, kernel_size):
    """Compute focus map comparing two pyramid levels."""
    return compute_focusmap_fast(array1, array2, kernel_size)


def fuse_pyramid_levels_using_focusmap(pyr_level1, pyr_level2, focusmap):
    """Fuse two pyramid levels using focusmap."""
    return fuse_levels_fast(pyr_level1, pyr_level2, focusmap)
