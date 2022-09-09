"""
Focus stacking algorithms on CPU accelerated with Numba's njit.
"""
import numpy as np
import numba as nb

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
