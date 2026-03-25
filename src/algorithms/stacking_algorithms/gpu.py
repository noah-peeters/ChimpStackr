"""
GPU-accelerated focus stacking algorithms.

Optimized pipeline that keeps data on-GPU to minimize host↔device transfers.
The main entry point is `fuse_pyramid_pair_gpu()` which batch-uploads pyramid
levels, runs all kernels (bgr2gray → focusmap → fuse) without intermediate
downloads, and batch-downloads only at the end.

Uses OpenCV CUDA for pyramid ops (pyrDown/pyrUp/subtract/add) when available,
and Numba CUDA kernels for the custom focusmap/fuse logic.

Falls back gracefully:
  - No cv2.cuda → CPU OpenCV for pyramid ops
  - No Numba CUDA → full CPU fallback via cpu module
"""
import math
import numpy as np
import cv2

try:
    import numba.cuda as cuda
    HAS_NUMBA_CUDA = cuda.is_available()
except ImportError:
    HAS_NUMBA_CUDA = False

# Check for OpenCV CUDA support (opencv-contrib with CUDA build)
HAS_CV_CUDA = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, 'cuda') else False

# Check for cv2.cuda.pyrUp (not all CUDA builds include it)
HAS_CV_CUDA_PYRUP = HAS_CV_CUDA and hasattr(cv2.cuda, 'pyrUp')

# Optimal block size: 32-wide for warp coalescing, 8 high for occupancy
_BLOCK = (32, 8)


# ──────────────────────────────────────────────
# Numba CUDA kernels (focusmap + fuse)
# ──────────────────────────────────────────────

if HAS_NUMBA_CUDA:
    @cuda.jit(device=True, fastmath=True)
    def _variance_welford(matrix, kernel_size):
        """
        Single-pass variance using Welford's algorithm.
        Accounts for virtual zero-padding at borders.
        Halves global memory reads vs. the two-pass mean-then-variance approach.
        """
        y_shape = matrix.shape[0]
        x_shape = matrix.shape[1]
        num_elements = y_shape * x_shape

        y_pad = max(0, kernel_size - y_shape)
        x_pad = max(0, kernel_size - x_shape)
        total_elements = num_elements + y_pad * x_shape + x_pad * y_shape

        if total_elements == 0:
            return 0.0

        # Welford single-pass: accumulate mean and M2 in one loop
        mean = 0.0
        m2 = 0.0
        n = 0
        for y in range(y_shape):
            for x in range(x_shape):
                n += 1
                val = matrix[y, x]
                delta = val - mean
                mean += delta / n
                delta2 = val - mean
                m2 += delta * delta2

        # Virtual zero-padded elements (value = 0)
        pad_count = total_elements - num_elements
        for _ in range(pad_count):
            n += 1
            delta = 0.0 - mean
            mean += delta / n
            delta2 = 0.0 - mean
            m2 += delta * delta2

        return m2 / total_elements

    @cuda.jit(fastmath=True)
    def _focusmap_kernel(gray1, gray2, kernel_size, focusmap):
        """Per-pixel focus comparison between two grayscale images."""
        y, x = cuda.grid(2)
        if y < gray1.shape[0] and x < gray1.shape[1]:
            k = kernel_size // 2
            # Clamp patch bounds
            y0 = max(0, y - k)
            y1 = min(gray1.shape[0], y + k)
            x0 = max(0, x - k)
            x1 = min(gray1.shape[1], x + k)

            patch1 = gray1[y0:y1, x0:x1]
            patch2 = gray2[y0:y1, x0:x1]

            v1 = _variance_welford(patch1, kernel_size)
            v2 = _variance_welford(patch2, kernel_size)

            focusmap[y, x] = 1 if v2 > v1 else 0

    @cuda.jit(fastmath=True)
    def _fuse_kernel(pyr1, pyr2, focusmap):
        """Fuse two pyramid levels: copy pyr2 pixels where focusmap == 1."""
        y, x = cuda.grid(2)
        if y < pyr1.shape[0] and x < pyr1.shape[1]:
            if focusmap[y, x] != 0:
                for c in range(pyr1.shape[2]):
                    pyr1[y, x, c] = pyr2[y, x, c]

    @cuda.jit(fastmath=True)
    def _bgr2gray_kernel(bgr_in, gray_out):
        """BGR to grayscale on GPU."""
        y, x = cuda.grid(2)
        if y < bgr_in.shape[0] and x < bgr_in.shape[1]:
            gray_out[y, x] = (
                0.114 * bgr_in[y, x, 0]
                + 0.587 * bgr_in[y, x, 1]
                + 0.299 * bgr_in[y, x, 2]
            )


def _cuda_grid(shape_2d, block=_BLOCK):
    """Compute CUDA grid dimensions for a 2D array."""
    return (
        math.ceil(shape_2d[0] / block[0]),
        math.ceil(shape_2d[1] / block[1]),
    )


def _to_device(arr):
    """Upload numpy array to GPU, avoiding unnecessary copy if already contiguous."""
    if arr.flags['C_CONTIGUOUS']:
        return cuda.to_device(arr)
    return cuda.to_device(np.ascontiguousarray(arr))


# ──────────────────────────────────────────────
# Pyramid operations (GPU-accelerated where possible)
# ──────────────────────────────────────────────

def gaussian_pyramid(img, num_levels):
    """Calculate Gaussian pyramid, using cv2.cuda if available."""
    if HAS_CV_CUDA:
        return _gaussian_pyramid_cuda(img, num_levels)
    # Fallback: CPU OpenCV (still fast, uses SSE/AVX)
    lower = img.astype(np.float32)
    pyr = [lower]
    for _ in range(num_levels):
        lower = cv2.pyrDown(lower)
        pyr.append(lower.astype(np.float32))
    return pyr


def _gaussian_pyramid_cuda(img, num_levels):
    """Gaussian pyramid using OpenCV CUDA — download only at the end."""
    gpu_mat = cv2.cuda_GpuMat()
    gpu_mat.upload(img.astype(np.float32))

    # Keep all levels on GPU as GpuMat
    gpu_mats = [gpu_mat]
    for _ in range(num_levels):
        gpu_mat = cv2.cuda.pyrDown(gpu_mat)
        gpu_mats.append(gpu_mat)

    # Batch download at the end
    return [g.download().astype(np.float32) for g in gpu_mats]


def generate_laplacian_pyramid(img, num_levels):
    """Generate Laplacian pyramid, using GPU pyrUp/subtract when available."""
    if HAS_CV_CUDA_PYRUP:
        return _generate_laplacian_pyramid_cuda(img, num_levels)
    return _generate_laplacian_pyramid_cpu(img, num_levels)


def _generate_laplacian_pyramid_cuda(img, num_levels):
    """Laplacian pyramid on GPU using cv2.cuda.pyrUp + cv2.cuda.subtract."""
    # Build Gaussian pyramid on GPU (keep as GpuMat)
    gpu_mat = cv2.cuda_GpuMat()
    gpu_mat.upload(img.astype(np.float32))
    gpu_gauss = [gpu_mat]
    for _ in range(num_levels):
        gpu_mat = cv2.cuda.pyrDown(gpu_mat)
        gpu_gauss.append(gpu_mat)

    # Build Laplacian: smallest level first (lowpass residual)
    lap_pyr = [gpu_gauss[-1].download().astype(np.float32)]
    for i in range(num_levels, 0, -1):
        size = (gpu_gauss[i - 1].size()[0], gpu_gauss[i - 1].size()[1])
        expanded = cv2.cuda.pyrUp(gpu_gauss[i], dstsize=size)
        lap_gpu = cv2.cuda.subtract(gpu_gauss[i - 1], expanded)
        lap_pyr.append(lap_gpu.download().astype(np.float32))

    return lap_pyr


def _generate_laplacian_pyramid_cpu(img, num_levels):
    """Laplacian pyramid using CPU OpenCV (fallback)."""
    gauss = gaussian_pyramid(img, num_levels)
    lap_pyr = [gauss[-1]]
    for i in range(num_levels, 0, -1):
        size = (gauss[i - 1].shape[1], gauss[i - 1].shape[0])
        expanded = cv2.pyrUp(gauss[i], dstsize=size)
        lap = np.subtract(gauss[i - 1], expanded)
        lap_pyr.append(lap)
    return lap_pyr


def reconstruct_pyramid(laplacian_pyr):
    """Reconstruct image from Laplacian pyramid, using GPU when available."""
    if HAS_CV_CUDA_PYRUP:
        return _reconstruct_pyramid_cuda(laplacian_pyr)
    return _reconstruct_pyramid_cpu(laplacian_pyr)


def _reconstruct_pyramid_cuda(laplacian_pyr):
    """Reconstruct on GPU using cv2.cuda.pyrUp + cv2.cuda.add."""
    gpu_top = cv2.cuda_GpuMat()
    gpu_top.upload(laplacian_pyr[0].astype(np.float32))
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        next_level = laplacian_pyr[i + 1]
        size = (next_level.shape[1], next_level.shape[0])
        expanded = cv2.cuda.pyrUp(gpu_top, dstsize=size)
        gpu_next = cv2.cuda_GpuMat()
        gpu_next.upload(next_level.astype(np.float32))
        gpu_top = cv2.cuda.add(gpu_next, expanded)
    # Single download of the final full-resolution image
    return gpu_top.download()


def _reconstruct_pyramid_cpu(laplacian_pyr):
    """Reconstruct using CPU OpenCV (fallback)."""
    top = laplacian_pyr[0]
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        expanded = cv2.pyrUp(top, dstsize=size)
        top = cv2.add(laplacian_pyr[i + 1], expanded)
    return top


# ──────────────────────────────────────────────
# Batched GPU fusion pipeline (Phase 1 optimization)
# ──────────────────────────────────────────────

def fuse_pyramid_pair_gpu(pyr1, pyr2, kernel_size):
    """
    Fuse two Laplacian pyramids entirely on GPU with minimal transfers.

    Instead of per-level upload→compute→download round-trips, this:
      1. Batch-uploads all pyramid levels at the start
      2. Runs bgr2gray → focusmap → fuse kernels per level on-device
      3. Single cuda.synchronize() + batch download at the end

    Returns list of numpy arrays (fused pyramid).
    Falls back to per-level CPU path if Numba CUDA is unavailable.
    """
    if not HAS_NUMBA_CUDA:
        from src.algorithms.stacking_algorithms import cpu as CPU
        # CPU fallback: per-level compute
        threshold_index = len(pyr1) - 1
        new_pyr = []
        current_focusmap = None
        for level in range(len(pyr1)):
            if level < threshold_index:
                gray1 = cv2.cvtColor(pyr1[level], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(pyr2[level], cv2.COLOR_BGR2GRAY)
                current_focusmap = CPU.compute_focusmap(gray1, gray2, kernel_size)
            else:
                s = pyr2[level].shape
                current_focusmap = cv2.resize(
                    current_focusmap, (s[1], s[0]), interpolation=cv2.INTER_AREA
                )
            new_pyr.append(CPU.fuse_pyramid_levels_using_focusmap(
                pyr1[level], pyr2[level], current_focusmap
            ))
        return new_pyr

    # --- GPU pipeline: batch upload, process, batch download ---
    threshold_index = len(pyr1) - 1

    # Batch upload all pyramid levels
    d_pyr1 = [_to_device(level) for level in pyr1]
    d_pyr2 = [_to_device(level) for level in pyr2]

    d_focusmap = None

    for level in range(len(pyr1)):
        shape_2d = pyr1[level].shape[:2]
        grid = _cuda_grid(shape_2d)

        if level < threshold_index:
            # BGR → grayscale on GPU
            if pyr1[level].ndim == 3:
                d_gray1 = cuda.device_array(shape_2d, dtype=np.float32)
                d_gray2 = cuda.device_array(shape_2d, dtype=np.float32)
                _bgr2gray_kernel[grid, _BLOCK](d_pyr1[level], d_gray1)
                _bgr2gray_kernel[grid, _BLOCK](d_pyr2[level], d_gray2)
            else:
                d_gray1 = d_pyr1[level]
                d_gray2 = d_pyr2[level]

            # Focusmap on GPU (no download)
            d_focusmap = cuda.device_array(shape_2d, dtype=np.uint8)
            _focusmap_kernel[grid, _BLOCK](d_gray1, d_gray2, kernel_size, d_focusmap)
        else:
            # For the largest level(s), resize the previous focusmap.
            # Download the small focusmap, resize on CPU, re-upload.
            # This is fine: smallest levels are tiny (e.g. 4×6 pixels).
            fm_host = d_focusmap.copy_to_host()
            s = pyr2[level].shape
            fm_resized = cv2.resize(fm_host, (s[1], s[0]), interpolation=cv2.INTER_AREA)
            d_focusmap = _to_device(fm_resized)

        # Fuse in-place on d_pyr1[level] (no download)
        _fuse_kernel[grid, _BLOCK](d_pyr1[level], d_pyr2[level], d_focusmap)

    # Single sync + batch download
    cuda.synchronize()
    return [d.copy_to_host() for d in d_pyr1]


# ──────────────────────────────────────────────
# Legacy per-level functions (kept for backward compatibility)
# ──────────────────────────────────────────────

def compute_focusmap(array1, array2, kernel_size):
    """
    Compute focus map comparing two pyramid levels.
    Prefer fuse_pyramid_pair_gpu() for the full pipeline.
    """
    if not HAS_NUMBA_CUDA:
        from src.algorithms.stacking_algorithms import cpu as CPU
        return CPU.compute_focusmap(array1, array2, kernel_size)

    grid = _cuda_grid(array1.shape[:2])

    d_arr1 = _to_device(array1)
    d_arr2 = _to_device(array2)

    if array1.ndim == 3:
        d_gray1 = cuda.device_array(array1.shape[:2], dtype=np.float32)
        d_gray2 = cuda.device_array(array1.shape[:2], dtype=np.float32)
        _bgr2gray_kernel[grid, _BLOCK](d_arr1, d_gray1)
        _bgr2gray_kernel[grid, _BLOCK](d_arr2, d_gray2)
    else:
        d_gray1 = d_arr1
        d_gray2 = d_arr2

    d_focusmap = cuda.device_array(array1.shape[:2], dtype=np.uint8)
    _focusmap_kernel[grid, _BLOCK](d_gray1, d_gray2, kernel_size, d_focusmap)

    cuda.synchronize()
    return d_focusmap.copy_to_host()


def fuse_pyramid_levels_using_focusmap(pyr_level1, pyr_level2, focusmap):
    """Fuse two pyramid levels using focusmap. Returns modified pyr_level1."""
    if not HAS_NUMBA_CUDA:
        from src.algorithms.stacking_algorithms import cpu as CPU
        return CPU.fuse_pyramid_levels_using_focusmap(pyr_level1, pyr_level2, focusmap)

    grid = _cuda_grid(pyr_level1.shape[:2])

    d_pyr1 = _to_device(pyr_level1)
    d_pyr2 = _to_device(pyr_level2)

    if isinstance(focusmap, np.ndarray):
        d_fm = _to_device(focusmap)
    else:
        d_fm = focusmap

    _fuse_kernel[grid, _BLOCK](d_pyr1, d_pyr2, d_fm)
    cuda.synchronize()
    return d_pyr1.copy_to_host()
