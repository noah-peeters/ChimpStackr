"""
    Main pyramid stacking + image alignment algorithm(s).
"""
import cv2

import src.algorithms.dft_imreg as dft_imreg
import src.algorithms.pyramid as pyramid_algorithm
import src.ImageLoadingHandler as ImageLoadingHandler
import src.algorithms.stacking_algorithms.cpu as CPU_Algos
import src.algorithms.stacking_algorithms.gpu as GPU_Algos
import src.algorithms.stacking_algorithms.shared as Shared_Algos


class Algorithm:
    def __init__(self):
        self.ImageLoadingHandler = ImageLoadingHandler.ImageLoadingHandler()
        self.DFT_Imreg = dft_imreg.im_reg()

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
        # Upscale last/largest focusmap (faster than computation)
        threshold_index = len(pyr2) - 1
        new_pyr = []
        current_focusmap = None
        # Loop through pyramid levels from smallest to largest shape, and fuse each level
        for pyramid_level in range(len(pyr2)):
            # Calculate what parts are more/less in focus between the pyramids
            if pyramid_level < threshold_index:
                # Regular computation (slow; accurate)
                current_focusmap = CPU_Algos.compute_focusmap(
                    cv2.cvtColor(pyr1[pyramid_level], cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(pyr2[pyramid_level], cv2.COLOR_BGR2GRAY),
                    kernel_size,
                )
            else:
                # TODO: See if upscale really provides any benefit
                # Upscale previous mask (faster; less accurate)
                s = pyr2[pyramid_level].shape
                current_focusmap = cv2.resize(
                    current_focusmap, (s[1], s[0]), interpolation=cv2.INTER_AREA
                )

            # Write output pyramid level using the calculated focusmap
            new_pyr_level = Shared_Algos.fuse_pyramid_levels_using_focusmap(
                pyr1[pyramid_level],
                pyr2[pyramid_level],
                current_focusmap,
            )
            new_pyr.append(new_pyr_level)
        return new_pyr
