"""
Focus stacking algorithms on CPU accelerated with Numba's njit.
"""
import numpy as np
import numba as nb
import cv2

### Internal functions ###

# Pad an array to be the kernel size (square). Only if needed
@nb.njit(
    nb.float32[:, :](nb.float32[:, :], nb.int64),
    fastmath=True,
    cache=True,
)
def pad_array(array, kernel_size):
    y_shape = array.shape[0]
    x_shape = array.shape[1]

    y_pad = kernel_size - y_shape
    x_pad = kernel_size - x_shape
    if y_pad > 0 or x_pad > 0:
        # Pad array (copy values into new; larger array)
        padded_array = np.zeros((y_shape + y_pad, x_shape + x_pad), dtype=array.dtype)
        padded_array[0:y_shape, 0:x_shape] = array
        return padded_array
    else:
        # Don't do anything
        return array


# Get deviation of a (grayscale image) matrix
@nb.njit(
    nb.float32(nb.float32[:, :]),
    fastmath=True,
    cache=True,
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
    lower = img.copy()
    gaussian_pyr = []
    gaussian_pyr.append(lower.astype(np.float32))  # Use same dtype
    # Compute all required pyramid levels
    for _ in range(num_levels):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(lower.astype(np.float32))  # convert_to_memmap
    return gaussian_pyr


### Exposed functions ###

# Compute focusmap for the same pyramid level in 2 different pyramids
@nb.njit(
    nb.uint8[:, :](nb.float32[:, :], nb.float32[:, :], nb.int64),
    fastmath=True,
    parallel=True,
    cache=True,
)
def compute_focusmap(pyr_level1, pyr_level2, kernel_size):
    y_range = pyr_level1.shape[0]
    x_range = pyr_level1.shape[1]

    # 2D focusmap (dtype=uint8); possible values:
    # 0 => pixel of pyr1
    # 1 => pixel of pyr2
    focusmap = np.empty((y_range, x_range), dtype=np.uint8)
    k = int(kernel_size / 2)

    # Loop through pixels of this pyramid level
    for y in nb.prange(y_range):  # Most images are wider (more values on x-axis)
        for x in nb.prange(x_range):
            # Get small patch (kernel_size) around this pixel
            patch = pyr_level1[y - k : y + k, x - k : x + k]
            # Padd array with zeros if needed (edges of image)
            padded_patch = pad_array(patch, kernel_size)
            dev1 = get_deviation(padded_patch)

            patch = pyr_level2[y - k : y + k, x - k : x + k]
            padded_patch = pad_array(patch, kernel_size)
            dev2 = get_deviation(padded_patch)

            value_to_insert = 0
            if dev2 > dev1:
                value_to_insert = 1

            # Write most in-focus pixel to output
            focusmap[y, x] = value_to_insert

    return focusmap


# Compute output pyramid_level from source arrays and focusmap
@nb.njit(
    nb.float32[:, :, :](nb.float32[:, :, :], nb.float32[:, :, :], nb.uint8[:, :]),
    fastmath=True,
    parallel=True,
    cache=True,
)
def fuse_pyramid_levels_using_focusmap(pyr_level1, pyr_level2, focusmap):
    # Copy directly in "pyr_level_1",
    # as creating a new array using ".copy()" takes longer
    for y in nb.prange(focusmap.shape[0]):
        for x in nb.prange(focusmap.shape[1]):
            if focusmap[y, x] == 0:
                pyr_level1[y, x, :] = pyr_level1[y, x, :]
            else:
                pyr_level1[y, x, :] = pyr_level2[y, x, :]
    return pyr_level1


def generate_laplacian_pyramid(img, num_levels):
    """Generate Laplacian pyramid (from Gaussian pyramid)"""
    # Create gaussian pyramid
    gaussian_pyr = gaussian_pyramid(img, num_levels)

    laplacian_top = gaussian_pyr[-1]

    laplacian_pyr = []
    # Insert smallest pyramid level (with color)
    laplacian_pyr.append(laplacian_top)
    # Loop through pyramid levels from smallest to largest shape
    for i in range(num_levels, 0, -1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)

        laplacian = np.subtract(gaussian_pyr[i - 1], gaussian_expanded)
        laplacian_pyr.append(laplacian)
    return laplacian_pyr


def reconstruct_pyramid(laplacian_pyr):
    """Reconstruct original image (highest resolution) from Laplacian pyramid."""
    laplacian_top = laplacian_pyr[0]
    laplacian_lst = [laplacian_top]
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
        laplacian_top = cv2.add(laplacian_pyr[i + 1], laplacian_expanded)

        laplacian_lst.append(laplacian_top)
    return laplacian_lst[num_levels]
