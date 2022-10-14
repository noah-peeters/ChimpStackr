"""
    Main pyramid stacking + image alignment algorithm(s).
"""
import math
import cv2
import numba.cuda as cuda

import src.algorithms.dft_imreg as dft_imreg
import src.algorithms.pyramid as pyramid_algorithm
import src.ImageLoadingHandler as ImageLoadingHandler
import src.algorithms.stacking_algorithms.cpu as CPU
import src.algorithms.stacking_algorithms.gpu as GPU


class Algorithm:
    def __init__(self):
        self.ImageLoadingHandler = ImageLoadingHandler.ImageLoadingHandler()
        self.DFT_Imreg = dft_imreg.im_reg()
        self.useGpu = False

    def toggle_cpu_gpu(self, use_gpu, selected_gpu_id):
        if use_gpu:
            cuda.select_device(selected_gpu_id)
            self.useGpu = True
        else:
            self.useGpu = False

    def align_image_pair(self, ref_im, im_to_align):
        """
        Fast Fourier Transform (FFT) image translational registration ((x, y)-shift only!)
        'ref_im' and 'im_to_align' can be an image array (np.ndarray), or an image path (str).
        In the latter case, the images will be loaded into memory first.

        When both images are of type 'str', and they are the same,
        'im_to_align' will be loaded into memory and be returned without alignment.
        """
        if type(ref_im) == str and type(im_to_align) == str:
            return self.ImageLoadingHandler.read_image_from_path(im_to_align)

        if type(ref_im) == str:
            ref_im = self.ImageLoadingHandler.read_image_from_path(ref_im)
        if type(im_to_align) == str:
            im_to_align = self.ImageLoadingHandler.read_image_from_path(im_to_align)

        # Calculate translational shift
        # TODO: Allow adjusting "scale_factor"??
        return self.DFT_Imreg.register_image_translation(
            ref_im, im_to_align, scale_factor=10
        )

    def generate_laplacian_pyramid(self, im1, num_levels):
        """
        Generates a laplacian pyramid for each image.
        'im1' can be an image array (np.ndarray), or an image path (str).
        In the latter case, the image will be loaded into memory first.
        """
        if type(im1) == str:
            im1 = self.ImageLoadingHandler.read_image_from_path(im1)

        return pyramid_algorithm.laplacian_pyramid(im1, num_levels)

    def focus_fuse_pyramid_pair(self, pyr1, pyr2, kernel_size):
        """
        Fuse 2 image pyramids into one.
        Each pyramid level will be compared between the two pyramids,
        and the sharpest pixels/parts of each image will be placed in the output pyramid.
        """
        if self.useGpu:
            """Use GPU."""
            """
            Fuse 2 image pyramids into one.
            Each pyramid level will be compared between the two pyramids,
            and the sharpest pixels/parts of each image will be placed in the output pyramid.
            """
            # Upscale last/largest focusmap (faster than computation)
            threshold_index = len(pyr1) - 1
            new_pyr = []
            previous_focusmap = None

            # Copy first array to device (only first time, as it gets reused)
            if not cuda.is_cuda_array(pyr1[0]):
                inter_pyr = []
                for i in pyr1:
                    inter_pyr.append(cuda.to_device(i))
                pyr1 = inter_pyr.copy()

            # Copy second array to device (every time)
            inter_pyr = []
            for i in pyr2:
                inter_pyr.append(cuda.to_device(i))
            pyr2 = inter_pyr.copy()
            del inter_pyr

            # Loop through pyramid levels from smallest to largest shape, and fuse each level
            for pyramid_level in range(len(pyr1)):
                # First pyramid level is already stored on GPU memory
                if pyramid_level < threshold_index:
                    previous_focusmap = GPU.compute_focusmap(
                        pyr1[pyramid_level],
                        pyr2[pyramid_level],
                        kernel_size,
                    )
                    # Write output pyramid level using the calculated focusmap
                    new_pyr_level = GPU.fuse_pyramid_levels_using_focusmap(
                        pyr1[pyramid_level],
                        pyr2[pyramid_level],
                        previous_focusmap,
                    )
                else:
                    # TODO: Already really optimized (0.5ms runtime)
                    # Upscale previous mask (faster, but slightly less accurate)
                    s = pyr2[pyramid_level].shape
                    array_out = cuda.device_array(
                        # (
                        #     math.ceil(previous_focusmap.shape[0] * 2),
                        #     math.ceil(previous_focusmap.shape[1] * 2),
                        # ),
                        (s[0], s[1]),
                        previous_focusmap.dtype,
                    )

                    # TODO: Don't recalculate cuda args?
                    threadsperblock = (16, 16)  # Should be a multiple of 32 (max 1024)
                    blockspergrid_x = math.ceil(s[0] / threadsperblock[0])
                    blockspergrid_y = math.ceil(s[1] / threadsperblock[1])
                    blockspergrid = (blockspergrid_x, blockspergrid_y)

                    GPU.resize_image[blockspergrid, threadsperblock](
                        previous_focusmap,
                        array_out,
                        s[1],
                        s[0],
                    )

                    new_pyr_level = GPU.fuse_pyramid_levels_using_focusmap(
                        pyr1[pyramid_level],
                        pyr2[pyramid_level],
                        array_out,
                    )

                new_pyr.append(new_pyr_level)
            # Return GPU array
            return new_pyr
        else:
            """Use CPU."""
            # Upscale last/largest focusmap (faster than computation)
            threshold_index = len(pyr1) - 1
            new_pyr = []
            current_focusmap = None
            # Loop through pyramid levels from smallest to largest shape, and fuse each level
            for pyramid_level in range(len(pyr1)):
                # Calculate what parts are more/less in focus between the pyramids
                if pyramid_level < threshold_index:
                    # Regular computation (slow; accurate)
                    current_focusmap = CPU.compute_focusmap(
                        cv2.cvtColor(pyr1[pyramid_level], cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(pyr2[pyramid_level], cv2.COLOR_BGR2GRAY),
                        kernel_size,
                    )
                else:
                    # Upscale previous mask (about twice as fast, but slightly less accurate.)
                    s = pyr2[pyramid_level].shape
                    current_focusmap = cv2.resize(
                        current_focusmap, (s[1], s[0]), interpolation=cv2.INTER_AREA
                    )

                # Write output pyramid level using the calculated focusmap
                new_pyr_level = CPU.fuse_pyramid_levels_using_focusmap(
                    pyr1[pyramid_level],
                    pyr2[pyramid_level],
                    current_focusmap,
                )
                new_pyr.append(new_pyr_level)
            return new_pyr
