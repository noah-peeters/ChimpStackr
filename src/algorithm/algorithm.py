"""
    Main pyramid stacking algorithm(s) + image alignment algorithm.
"""
import os, tempfile, time
import cv2
import numpy as np
import numba as nb
from numba.typed import List

import algorithm.image_storage as image_storage
import algorithm.pyramid as pyramid
import ImageLoadingHandler


@nb.njit(nb.float32[:, :, :](nb.float32[:, :, :], nb.int64, nb.int64), fastmath=True)
def pad_array(array, y_pad, x_pad):
    s = array.shape
    new_array = np.zeros(
        (s[0] + y_pad, s[1] + x_pad, array.shape[2]), dtype=array.dtype
    )
    # Copy old values into new (larger array)
    new_array[0 : s[0], 0 : s[1]] = array
    return new_array


# Get deviation of a (grayscale image) matrix
@nb.njit(nb.float64(nb.float64[:, :]), fastmath=True)
def get_deviation(matrix):
    summed_deviation = 0
    average_value = np.mean(matrix)
    kernel_area = matrix.shape[0] * matrix.shape[1]
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            summed_deviation += (matrix[y, x] - average_value) ** 2 / kernel_area
    return summed_deviation


# Compute focusmap for the same pyramid level in 2 different pyramids
@nb.njit(
    nb.uint8[:, :](nb.float32[:, :, :], nb.float32[:, :, :], nb.int64),
    parallel=True,
    fastmath=True,
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
        for x in range(x_range):
            highest_image_index = 0
            highest_value = float(0)
            for image_index in range(2):  # Loop through images
                if image_index == 0:
                    current_pyramid = pyr_level1
                else:
                    current_pyramid = pyr_level2

                # Get small patch (kernel_size) around this pixel
                k = int(kernel_size / 2)
                patch = current_pyramid[y - k : y + k, x - k : x + k]

                # Padd array with zeros if needed (edges of image)
                y_pad = kernel_size - patch.shape[0]
                x_pad = kernel_size - patch.shape[1]
                patch = pad_array(patch, y_pad, x_pad)

                # Convert BGR patch to grayscale
                grayscale_patch = (
                    0.2989 * patch[:, :, 2]
                    + 0.5870 * patch[:, :, 1]
                    + 0.1140 * patch[:, :, 0]
                )

                # Get entropy of kernel
                # deviation = entropy(grayscale_patch, disk(10))
                # print(kernel_entropy)

                # Get deviation of kernel
                deviation = get_deviation(grayscale_patch)
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
)
def fuse_pyramid_levels_using_focusmap(pyr_level1, pyr_level2, focusmap):
    output = np.empty_like(pyr_level1)
    for y in range(focusmap.shape[0]):
        for x in range(focusmap.shape[1]):
            if focusmap[y, x] == 0:
                output[y, x, :] = pyr_level1[y, x, :]
            else:
                output[y, x, :] = pyr_level2[y, x, :]
    return output


class Algorithm:
    def __init__(self):
        self.ImageStorage = image_storage.ImageStorageHandler()
        self.Pyramid = pyramid.Pyramid()
        self.ImageLoadingHandler = ImageLoadingHandler.ImageLoadingHandler()

    # ECC image alignment using pyramid approximation
    # src: https://stackoverflow.com/questions/45997891/cv2-motion-euclidean-for-the-warp-mode-in-ecc-image-alignment-method
    def align_image_pair(self, ref_im_path, im2_path, root_temp_dir):
        # Load images
        ref_im = self.ImageLoadingHandler.read_image_from_path(ref_im_path)
        im2 = self.ImageLoadingHandler.read_image_from_path(im2_path)

        # ECC params
        n_iters = 500
        e_thresh = 1e-5
        warp_mode = cv2.MOTION_EUCLIDEAN
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iters, e_thresh)

        num_levels = 4
        warp = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        warp = warp * np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32) ** (
            1 - num_levels
        )

        # Construct grayscale pyramid
        gray1 = cv2.cvtColor(ref_im, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        gray1_pyr = [gray1]
        gray2_pyr = [gray2]

        for level in range(num_levels):
            gray1_pyr.insert(
                0,
                cv2.resize(
                    gray1_pyr[0], None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA
                ),
            )
            gray2_pyr.insert(
                0,
                cv2.resize(
                    gray2_pyr[0], None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA
                ),
            )

        # run pyramid ECC
        for level in range(num_levels):
            _, warp = cv2.findTransformECC(
                gray1_pyr[level], gray2_pyr[level], warp, warp_mode, criteria, None, 1
            )

            if level != num_levels - 1:  # scale up for the next pyramid level
                warp = warp * np.array([[1, 1, 2], [1, 1, 2]], dtype=np.float32)

        # Align image
        aligned = cv2.warpAffine(
            im2,
            warp,
            (ref_im.shape[1], ref_im.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )

        # Write to disk
        file_handle, tmp_file = tempfile.mkstemp(".npy", None, root_temp_dir.name)
        np.save(tmp_file, aligned, allow_pickle=False)

        os.close(file_handle)
        return tmp_file

    # Generate laplacian pyramids for every image (if not already created) and write to disk archive
    def generate_laplacian_pyramids(
        self, image_paths, root_dir, num_levels, signals, load_from_tempfile=False
    ):
        laplacian_pyramid_archive_names = []
        for i, path in enumerate(image_paths):
            start_time = time.time()

            if not load_from_tempfile:
                # Load from src image
                image = self.ImageLoadingHandler.read_image_from_path(path)
            else:
                # Load from tempfile (aligned image)
                image = np.load(path, allow_pickle=False)

            pyramid = self.Pyramid.laplacian_pyramid(image, num_levels)

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

                # Upscale last focusmap (faster then computation)
                threshold_index = len(new_laplacian_pyramid) - 1
                new_pyr = List()
                current_focusmap = None
                # Loop through pyramid levels from smallest to largest shape
                for pyramid_level in range(len(new_laplacian_pyramid)):
                    if pyramid_level < threshold_index:
                        # Regular computation (slow; accurate)
                        current_focusmap = compute_focusmap(
                            output_pyramid[pyramid_level],
                            new_laplacian_pyramid[pyramid_level],
                            kernel_size,
                        )
                    else:
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
