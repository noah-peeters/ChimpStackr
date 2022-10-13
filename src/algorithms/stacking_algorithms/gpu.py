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


### Exposed functions ###

# TODO: Properly analyze if a further speedup could be achieved on the GPU
def compute_focusmap(array1, array2, kernel_size):
    """
    Move arrays to device and call actual function next.
    Will not wait for result to be ready. (which is what we want)
    """
    array1 = cuda.to_device(array1)
    array2 = cuda.to_device(array2)
    # Result will be stored here
    focusmap = cuda.to_device(np.zeros_like(array1).astype(np.uint8))

    threadsperblock = (16, 16)  # Should be a multiple of 32 (max 1024)
    blockspergrid_x = math.ceil(array1.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(array1.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Start calculation
    compute_focusmap_gpu[blockspergrid, threadsperblock](
        array1, array2, kernel_size, focusmap
    )
    # Wait for completion
    return focusmap.copy_to_host()
