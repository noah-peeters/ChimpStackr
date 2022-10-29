"""
Focus stacking algorithms on GPU accelerated with Numba's cuda.
~100x speedup --from 5s execution time to 0.5s-- when using (6000x4000) arrays/images
on my hardware (intel i5-11400H, nvidia rtx 3060) when CPU runtime is compared to GPU runtime.

Arrays with relatively low amount of elements will be calculated on the CPU,
as they aren't worth the copy overhead to/from the GPU.
"""
import math
import numpy as np
import numba.cuda as cuda


### Internal functions ###

# Device function that can be called from within a kernel
# We are not able to pad an array on the gpu (np.zeros) like we use on the cpu,
# so we calculate how many zero values would be added to pad the array instead.
@cuda.jit(device=True, fastmath=True)
def get_deviation(matrix, kernel_size):
    # compute zeros to add
    y_shape = matrix.shape[0]
    x_shape = matrix.shape[1]
    num_elements = y_shape * matrix.shape[1]

    y_pad = kernel_size - y_shape
    x_pad = kernel_size - x_shape
    zeros_to_add = y_pad * y_shape + x_pad * x_shape
    # Total elements in padded matrix
    total_elements = num_elements + zeros_to_add

    # Get average value of matrix
    average_value = 0
    for y in range(y_shape):
        for x in range(x_shape):
            average_value += matrix[y, x]
    average_value = average_value / total_elements

    summed_deviation = float(0)
    for y in range(y_shape):
        for x in range(x_shape):
            summed_deviation += (matrix[y, x] - average_value) ** 2 / total_elements
    return summed_deviation


@cuda.jit(fastmath=True)
def compute_focusmap_gpu(array1, array2, kernel_size, focusmap):
    x, y = cuda.grid(2)
    # If grid index is larger than image shape, do nothing
    if x < array1.shape[0] and y < array1.shape[1]:
        value_to_insert = 0

        k = int(kernel_size / 2)
        patch1 = array1[x - k : x + k, y - k : y + k]
        patch2 = array2[x - k : x + k, y - k : y + k]

        # Compare 2 (standard) deviations
        if get_deviation(patch2, kernel_size) > get_deviation(patch1, kernel_size):
            value_to_insert = 1

        # Write most in-focus pixel to output
        focusmap[x, y] = value_to_insert


@cuda.jit(fastmath=True)
def fuse_pyramid_levels_using_focusmap_gpu(pyr_level1, pyr_level2, focusmap):
    x, y = cuda.grid(2)
    # If grid index is larger than image shape, do nothing
    if x < pyr_level1.shape[0] and y < pyr_level1.shape[1]:
        if focusmap[x, y] != 0:
            # Copy 3 color channels
            for i in range(3):
                pyr_level1[x, y, i] = pyr_level2[x, y, i]
        # if focusmap[y, x] == 0:
        #     pyr_level1[y, x, :] = pyr_level1[y, x, :]
        # else:
        #     pyr_level1[y, x, :] = pyr_level2[y, x, :]


@cuda.jit(fastmath=True)
def BGR2GRAY(array_in, array_out):
    """
    Convert a 3D BGR image array to a 2D grayscale image.
    According to this formula: 'Y = 0.299 R + 0.587 G + 0.114 B'.
    Has the same effect as cv2.cvtColor(array, cv2.COLOR_BGR2GRAY), but it's all on the GPU.
    """
    x, y = cuda.grid(2)
    # If grid index is larger than image shape, do nothing
    if x < array_in.shape[0] and y < array_in.shape[1]:
        array_out[x, y] = (
            0.299 * array_in[x, y, 2]  # Red
            + 0.587 * array_in[x, y, 1]  # Green
            + 0.114 * array_in[x, y, 0]  # Blue
        )


# TODO: Accelerate using gpu
def gaussian_pyramid(img, num_levels):
    """Calculate Gaussian pyramid."""
    import cv2

    lower = img.copy()
    gaussian_pyr = []
    gaussian_pyr.append(lower.astype(np.float32))  # Use same dtype
    # Compute all required pyramid levels
    for _ in range(num_levels):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(lower.astype(np.float32))  # convert_to_memmap
    return gaussian_pyr


### Exposed functions ###
# TODO: Don't recalculate cuda args?


def compute_focusmap(array1, array2, kernel_size):
    """
    Move arrays to device and call actual function next.
    Will not wait for result to be ready. (which is what we want)
    """
    threadsperblock = (16, 16)  # Should be a multiple of 32 (max 1024)
    blockspergrid_x = math.ceil(array1.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(array1.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Convert BGR to grayscale
    array1_gray = cuda.device_array(
        shape=(array1.shape[0], array1.shape[1]), dtype=np.float32
    )
    array2_gray = cuda.device_array_like(array1_gray)

    BGR2GRAY[blockspergrid, threadsperblock](array1, array1_gray)
    BGR2GRAY[blockspergrid, threadsperblock](array2, array2_gray)

    # Result will be stored here
    focusmap = cuda.device_array(shape=array1_gray.shape, dtype=np.uint8)

    # Start calculation
    compute_focusmap_gpu[blockspergrid, threadsperblock](
        array1_gray, array2_gray, kernel_size, focusmap
    )
    # Don't wait for completion
    # TODO: Check if that actually is the case
    return focusmap


def fuse_pyramid_levels_using_focusmap(pyr_level1, pyr_level2, focusmap):
    """Calculate cuda args and call kernel."""
    threadsperblock = (16, 16)  # Should be a multiple of 32 (max 1024)
    blockspergrid_x = math.ceil(pyr_level1.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(pyr_level1.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Start calculation
    fuse_pyramid_levels_using_focusmap_gpu[blockspergrid, threadsperblock](
        pyr_level1, pyr_level2, focusmap
    )
    return pyr_level1


@cuda.jit(fastmath=True)
def resize_image(array_in, array_out, width_out, height_out):
    """
    Algorithm src:
    https://eng.aurelienpierre.com/2020/03/bilinear-interpolation-on-images-stored-as-python-numpy-ndarray/
    A valid replacement for 'cv2.resize()',
    the only (visual) difference is that the cv2 result seems a little more blocky.
    """
    i, j = cuda.grid(2)
    height_in = array_in.shape[0]
    width_in = array_in.shape[1]
    if i < width_out and j < height_out:
        # Relative coordinates of the pixel in output space
        x_out = j / width_out
        y_out = i / height_out

        # Corresponding absolute coordinates of the pixel in input space
        x_in = x_out * width_in
        y_in = y_out * height_in

        # Nearest neighbours coordinates in input space
        x_prev = int(math.floor(x_in))
        x_next = x_prev + 1
        y_prev = int(math.floor(y_in))
        y_next = y_prev + 1

        # Sanitize bounds - no need to check for < 0
        x_prev = min(x_prev, width_in - 1)
        x_next = min(x_next, width_in - 1)
        y_prev = min(y_prev, height_in - 1)
        y_next = min(y_next, height_in - 1)

        # Distances between neighbour nodes in input space
        Dy_next = y_next - y_in
        Dy_prev = 1.0 - Dy_next
        # because next - prev = 1
        Dx_next = x_next - x_in
        Dx_prev = 1.0 - Dx_next
        # because next - prev = 1

        array_out[i][j] = Dy_prev * (
            array_in[y_next][x_prev] * Dx_next + array_in[y_next][x_next] * Dx_prev
        ) + Dy_next * (
            array_in[y_prev][x_prev] * Dx_next + array_in[y_prev][x_next] * Dx_prev
        )


# TODO: Accelerate using gpu
def generate_laplacian_pyramid(img, num_levels):
    """Generate Laplacian pyramid (from Gaussian pyramid)"""
    import cv2

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


# TODO: Accelerate using gpu?
def reconstruct_pyramid(laplacian_pyr):
    """Reconstruct original image (highest resolution) from Laplacian pyramid."""
    import cv2

    laplacian_top = laplacian_pyr[0]
    laplacian_lst = [laplacian_top]
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
        laplacian_top = cv2.add(laplacian_pyr[i + 1], laplacian_expanded)

        laplacian_lst.append(laplacian_top)
    return laplacian_lst[num_levels]
