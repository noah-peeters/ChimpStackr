"""
GPU-accelerated focus stacking algorithms.

Strategy: Use OpenCV's CUDA module (cv2.cuda) when available for pyrDown/pyrUp
and image operations. These keep data on the GPU without host↔device transfers.
Falls back to Numba CUDA kernels for the focus map computation (the main
bottleneck) which needs custom logic.

If cv2.cuda is not available, falls back to CPU OpenCV for pyramid ops
with only the focusmap/fuse kernels running on GPU via Numba.
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


# ──────────────────────────────────────────────
# Numba CUDA kernels (focusmap + fuse)
# ──────────────────────────────────────────────

if HAS_NUMBA_CUDA:
    @cuda.jit(device=True, fastmath=True)
    def _variance_patch(matrix, kernel_size):
        """Compute variance of a patch, accounting for virtual zero-padding."""
        y_shape = matrix.shape[0]
        x_shape = matrix.shape[1]
        num_elements = y_shape * x_shape

        y_pad = max(0, kernel_size - y_shape)
        x_pad = max(0, kernel_size - x_shape)
        total_elements = num_elements + y_pad * x_shape + x_pad * y_shape

        if total_elements == 0:
            return 0.0

        # Mean
        avg = 0.0
        for y in range(y_shape):
            for x in range(x_shape):
                avg += matrix[y, x]
        avg /= total_elements

        # Variance
        var = 0.0
        for y in range(y_shape):
            for x in range(x_shape):
                diff = matrix[y, x] - avg
                var += diff * diff
        # Zero-padded elements contribute avg² each
        var += (num_elements + y_pad * x_shape + x_pad * y_shape - num_elements) * avg * avg
        return var / total_elements

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

            v1 = _variance_patch(patch1, kernel_size)
            v2 = _variance_patch(patch2, kernel_size)

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


def _cuda_grid(shape_2d, block=(16, 16)):
    """Compute CUDA grid dimensions for a 2D array."""
    return (
        math.ceil(shape_2d[0] / block[0]),
        math.ceil(shape_2d[1] / block[1]),
    )


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
    """Gaussian pyramid using OpenCV CUDA — zero host↔device copies between levels."""
    gpu_mat = cv2.cuda_GpuMat()
    gpu_mat.upload(img.astype(np.float32))

    pyr = [img.astype(np.float32)]
    for _ in range(num_levels):
        gpu_mat = cv2.cuda.pyrDown(gpu_mat)
        pyr.append(gpu_mat.download().astype(np.float32))
    return pyr


def generate_laplacian_pyramid(img, num_levels):
    """Generate Laplacian pyramid."""
    gauss = gaussian_pyramid(img, num_levels)

    lap_pyr = [gauss[-1]]  # Smallest level (lowpass residual)
    for i in range(num_levels, 0, -1):
        size = (gauss[i - 1].shape[1], gauss[i - 1].shape[0])
        expanded = cv2.pyrUp(gauss[i], dstsize=size)
        lap = np.subtract(gauss[i - 1], expanded)
        lap_pyr.append(lap)
    return lap_pyr


def reconstruct_pyramid(laplacian_pyr):
    """Reconstruct image from Laplacian pyramid."""
    top = laplacian_pyr[0]
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        expanded = cv2.pyrUp(top, dstsize=size)
        top = cv2.add(laplacian_pyr[i + 1], expanded)
    return top


# ──────────────────────────────────────────────
# Exposed functions (match CPU API)
# ──────────────────────────────────────────────

def compute_focusmap(array1, array2, kernel_size):
    """
    Compute focus map comparing two grayscale pyramid levels.
    Keeps data on GPU — returns a Numba device array if possible,
    otherwise returns a numpy array.
    """
    if not HAS_NUMBA_CUDA:
        # Fallback to CPU
        from src.algorithms.stacking_algorithms import cpu as CPU
        return CPU.compute_focusmap(array1, array2, kernel_size)

    block = (16, 16)
    grid = _cuda_grid(array1.shape[:2], block)

    # Transfer to GPU
    d_arr1 = cuda.to_device(np.ascontiguousarray(array1))
    d_arr2 = cuda.to_device(np.ascontiguousarray(array2))

    # BGR to grayscale on GPU (skip if already 2D)
    if array1.ndim == 3:
        d_gray1 = cuda.device_array(array1.shape[:2], dtype=np.float32)
        d_gray2 = cuda.device_array(array1.shape[:2], dtype=np.float32)
        _bgr2gray_kernel[grid, block](d_arr1, d_gray1)
        _bgr2gray_kernel[grid, block](d_arr2, d_gray2)
    else:
        d_gray1 = d_arr1
        d_gray2 = d_arr2

    d_focusmap = cuda.device_array(array1.shape[:2], dtype=np.uint8)
    _focusmap_kernel[grid, block](d_gray1, d_gray2, kernel_size, d_focusmap)

    # Synchronize and return to host
    cuda.synchronize()
    return d_focusmap.copy_to_host()


def fuse_pyramid_levels_using_focusmap(pyr_level1, pyr_level2, focusmap):
    """Fuse two pyramid levels using focusmap. Returns modified pyr_level1."""
    if not HAS_NUMBA_CUDA:
        from src.algorithms.stacking_algorithms import cpu as CPU
        return CPU.fuse_pyramid_levels_using_focusmap(pyr_level1, pyr_level2, focusmap)

    block = (16, 16)
    grid = _cuda_grid(pyr_level1.shape[:2], block)

    d_pyr1 = cuda.to_device(np.ascontiguousarray(pyr_level1))
    d_pyr2 = cuda.to_device(np.ascontiguousarray(pyr_level2))

    # Focusmap may be a numpy array or already on device
    if isinstance(focusmap, np.ndarray):
        d_fm = cuda.to_device(np.ascontiguousarray(focusmap))
    else:
        d_fm = focusmap

    _fuse_kernel[grid, block](d_pyr1, d_pyr2, d_fm)
    cuda.synchronize()
    return d_pyr1.copy_to_host()
