"""
Algorithm(s) used by both the CPU and GPU.
They run on the CPU, as they don't take long to complete or running on the GPU wouldn't provide a significant speedup.
"""
import numba as nb

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
