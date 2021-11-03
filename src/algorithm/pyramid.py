"""
    Class that handles creating Laplacian (+Gaussian) pyramids for images.
    Can reconstruct image from pyramid.
"""

import cv2
import numpy as np
from numba.typed import List

# Create tempfile memmap from array (auto-deleted when GC)
# def convert_to_memmap(array):
#     tempfile_name = tempfile.NamedTemporaryFile().name
#     memmap = np.memmap(tempfile_name, mode="w+", shape=array.shape, dtype=array.dtype)
#     memmap[:] = array
#     return memmap


class Pyramid:
    def __init__(self):
        return

    # Calculate Gaussian pyramid
    def gaussian_pyramid(self, img, num_levels):
        lower = img.copy()
        gaussian_pyr = List()
        gaussian_pyr.append(lower.astype(np.float32))  # Use same dtype
        # Compute all required pyramid levels
        for i in range(num_levels):
            lower = cv2.pyrDown(lower)
            gaussian_pyr.append(lower.astype(np.float32))  # convert_to_memmap
        return gaussian_pyr

    # Calculate Laplacian pyramid (from Gaussian)
    def laplacian_pyramid(self, img, num_levels):
        # Create gaussian pyramid
        gaussian_pyr = self.gaussian_pyramid(img, num_levels)

        laplacian_top = gaussian_pyr[-1]

        laplacian_pyr = List()
        laplacian_pyr.append(laplacian_top)
        # Loop through pyramid levels from smallest to largest shape
        for i in range(num_levels, 0, -1):
            size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
            gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)

            laplacian = np.subtract(gaussian_pyr[i - 1], gaussian_expanded)
            laplacian_pyr.append(laplacian)  # convert_to_memmap
        return laplacian_pyr

    # Reconstruct original image (highest res) from Laplacian pyramid
    def reconstruct(self, laplacian_pyr):
        laplacian_top = laplacian_pyr[0]
        laplacian_lst = [laplacian_top]
        num_levels = len(laplacian_pyr) - 1
        for i in range(num_levels):
            size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
            laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
            laplacian_top = cv2.add(laplacian_pyr[i + 1], laplacian_expanded)
            laplacian_lst.append(laplacian_top)
        return laplacian_lst[num_levels]
