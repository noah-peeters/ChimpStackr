"""
    Main pyramid stacking algorithm(s) + image alignment algorithm.
"""
import os, tempfile, time
import cv2
import numpy as np
import numba as nb
from numba.typed import List

import src.algorithm.image_storage as image_storage
import src.algorithm.dft_imreg as dft_imreg
import src.algorithm.pyramid as pyramid_algorithm
import src.ImageLoadingHandler as ImageLoadingHandler


# Pad an array to be the kernel size (square). Only if needed
@nb.njit(nb.float32[:, :](nb.float32[:, :], nb.int64), fastmath=True, cache=True)
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
@nb.njit(nb.float32(nb.float32[:, :]), fastmath=True, cache=True)
def get_std_deviation(matrix):
    summed_deviation = float(0)
    average_value = np.mean(matrix)
    kernel_area = matrix.shape[0] * matrix.shape[1]

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            summed_deviation += (matrix[y, x] - average_value) ** 2 / kernel_area
    return np.sqrt(summed_deviation)


# Compute focusmap for the same pyramid level in 2 different pyramids
@nb.njit(
    nb.uint8[:, :](nb.float32[:, :], nb.float32[:, :], nb.int64),
    parallel=True,
    cache=True,
)
def compute_focusmap(pyr_level1, pyr_level2, kernel_size):
    y_range = pyr_level1.shape[0]
    x_range = pyr_level1.shape[1]

    # 2D focusmap (dtype=bool)
    # 0 => pixel of pyr1
    # 1 => pixel of pyr2
    focusmap = np.empty((y_range, x_range), dtype=np.uint8)

    # Loop through pixels of this pyramid level
    for y in nb.prange(y_range):  # Most images are wider (more values on x-axis)
        for x in nb.prange(x_range):
            highest_image_index = 0
            highest_value = float(0)
            for image_index in nb.prange(2):  # Loop through images
                current_pyramid = pyr_level1
                if image_index != 0:
                    current_pyramid = pyr_level2

                # Get small patch (kernel_size) around this pixel
                k = int(kernel_size / 2)
                patch = current_pyramid[y - k : y + k, x - k : x + k]

                # Padd array with zeros if needed (edges of image)
                padded_patch = pad_array(patch, kernel_size)

                # Get entropy of kernel
                # deviation = entropy(padded_patch, disk(10))
                # print(kernel_entropy)

                # Get deviation of kernel
                deviation = get_std_deviation(padded_patch)
                if deviation > highest_value:
                    highest_value = deviation
                    highest_image_index = image_index

            value_to_insert = 0
            if highest_image_index != 0:
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
    output = np.empty_like(pyr_level1)
    for y in nb.prange(focusmap.shape[0]):
        for x in nb.prange(focusmap.shape[1]):
            if focusmap[y, x] == 0:
                output[y, x, :] = pyr_level1[y, x, :]
            else:
                output[y, x, :] = pyr_level2[y, x, :]
    return output


class Algorithm:
    def __init__(self):
        self.ImageStorage = image_storage.ImageStorageHandler()
        self.ImageLoadingHandler = ImageLoadingHandler.ImageLoadingHandler()
        self.DFT_Imreg = dft_imreg.im_reg()

    # Fast Fourier Transform (FFT) image translational registration ((x, y)-shift only!)
    def align_image_pair(self, ref_im_path, im2_path, root_temp_dir):
        # Load images
        im0 = self.ImageLoadingHandler.read_image_from_path(ref_im_path)
        im1 = self.ImageLoadingHandler.read_image_from_path(im2_path)
        if ref_im_path == im2_path:
            output_image = im0  # Don't align if given 2 identical images
        else:
            # Calculate translational shift
            output_image = self.DFT_Imreg.register_image_translation(
                im0, im1, scale_factor=10
            )

        # Write aligned img to disk
        file_handle, tmp_file = tempfile.mkstemp(".npy", None, root_temp_dir.name)
        np.save(tmp_file, output_image, allow_pickle=False)

        os.close(file_handle)
        return tmp_file

    # Generate laplacian pyramids for every image (if not already created) and write to disk archive
    def generate_laplacian_pyramids(self, image_paths, root_dir, num_levels, signals):
        laplacian_pyramid_archive_names = []
        for i, path in enumerate(image_paths):
            start_time = time.time()

            # Load from src image
            image = self.ImageLoadingHandler.read_image_from_path(path)

            pyramid = pyramid_algorithm.laplacian_pyramid(image, num_levels)

            tmp_file = self.ImageStorage.write_laplacian_pyramid_to_disk(
                pyramid, root_dir
            )
            laplacian_pyramid_archive_names.append(tmp_file)
            del pyramid

            # Send progress signal
            signals.finished_inter_task.emit(
                [
                    "laplacian_pyramid_generation",
                    i + 1,
                    len(image_paths),
                    time.time() - start_time,
                ]
            )

        return laplacian_pyramid_archive_names

    # Fuse all images from their Laplacian pyramids
    def focus_fuse_pyramids(self, image_archive_names, kernel_size, signals):
        output_pyramid = List()
        for i, archive_name in enumerate(image_archive_names):
            start_time = time.time()

            if i == 0:
                # Directly "copy" first image's pyramid into output
                laplacian_pyramid = self.ImageStorage.load_laplacian_pyramid(
                    archive_name
                )
                output_pyramid = laplacian_pyramid
            else:
                # Focus fuse this pyramid to the output
                new_laplacian_pyramid = self.ImageStorage.load_laplacian_pyramid(
                    archive_name
                )

                # Upscale last/largest focusmap (faster than computation)
                threshold_index = len(new_laplacian_pyramid) - 1
                new_pyr = List()
                current_focusmap = None
                # Loop through pyramid levels from smallest to largest shape
                for pyramid_level in range(len(new_laplacian_pyramid)):
                    if pyramid_level < threshold_index:
                        # Regular computation (slow; accurate)
                        current_focusmap = compute_focusmap(
                            cv2.cvtColor(output_pyramid[pyramid_level], cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(new_laplacian_pyramid[pyramid_level], cv2.COLOR_BGR2GRAY),
                            kernel_size,
                        )
                    else:
                        # TODO: See if upscale really provides any benefit
                        # Upscale previous mask (faster; less accurate)
                        s = new_laplacian_pyramid[pyramid_level].shape
                        current_focusmap = cv2.resize(
                            current_focusmap, (s[1], s[0]), interpolation=cv2.INTER_AREA
                        )

                    # Write using focusmap
                    new_pyr_level = fuse_pyramid_levels_using_focusmap(
                        output_pyramid[pyramid_level],
                        new_laplacian_pyramid[pyramid_level],
                        current_focusmap,
                    )

                    new_pyr.append(new_pyr_level)
                # Set updated pyramid
                output_pyramid = new_pyr

            # Send progress signals
            signals.finished_inter_task.emit(
                [
                    "laplacian_pyramid_focus_fusion",
                    i + 1,
                    len(image_archive_names),
                    time.time() - start_time,
                ]
            )

        return output_pyramid
