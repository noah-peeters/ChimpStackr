"""
    Exposed API for easily aligning/stacking multiple images.
"""
import os, time

import src.utilities as utilities
import src.algorithms.pyramid as pyramid_algorithm
import src.algorithms as algorithms
import src.settings as settings


class LaplacianPyramid:
    def __init__(self, fusion_kernel_size=6, pyramid_num_levels=8):
        self.output_image = None
        self.image_paths = []
        # Archive names for Laplacian pyramids of src and aligned images
        self.laplacian_pyramid_archive_names = []
        self.laplacian_pyramid_archive_names_aligned = []

        self.Algorithm = algorithms.Algorithm()

        # Parameters
        self.fusion_kernel_size_pixels = fusion_kernel_size
        self.pyramid_num_levels = pyramid_num_levels

    # Set new image paths and sort them
    def update_image_paths(self, new_image_paths):
        new_image_paths = sorted(new_image_paths, key=utilities.int_string_sorting)
        if new_image_paths != self.image_paths:
            # Delete tempfiles
            for path in (
                self.laplacian_pyramid_archive_names
                + self.laplacian_pyramid_archive_names_aligned
            ):
                try:
                    os.remove(path)
                except:
                    print("Unable to remove tempfile")

            # Reset loaded Laplacian lists
            self.laplacian_pyramid_archive_names = []
            self.laplacian_pyramid_archive_names_aligned = []

        self.image_paths = new_image_paths

    # TODO: Rewrite for easy stopping of task (using signals??)
    def align_and_stack_images(self, signals):
        """
        Align and stack images.
        """
        # Align images to reference
        # TODO: Allow option to align image stack to start, end, middle or previous images
        # ref_index=round(len(self.image_paths)/2)

        # Will copy first image (ndarray) in this list without alignment
        aligned_images = [
            self.Algorithm.align_image_pair(self.image_paths[0], self.image_paths[0])
        ]
        fused_pyr = self.Algorithm.generate_laplacian_pyramid(
            aligned_images[0], self.pyramid_num_levels
        )

        for i, path in enumerate(self.image_paths):
            if i == 0:
                continue  # First image is already copied in list

            start_time = time.time()
            # Use previous *aligned* image instead of src image!
            aligned_images.append(
                self.Algorithm.align_image_pair(aligned_images[0], path)
            )

            # Generate pyramid for the (aligned) image
            new_pyr = self.Algorithm.generate_laplacian_pyramid(
                aligned_images[1], self.pyramid_num_levels
            )
            # Remove first aligned image array from list (lower memory usage)
            del aligned_images[0]
            # Fuse this new pyramid with the existing one
            fused_pyr = self.Algorithm.focus_fuse_pyramid_pair(
                fused_pyr, new_pyr, self.fusion_kernel_size_pixels
            )

            # Send progress signal
            signals.finished_inter_task.emit(
                [
                    "finished_image",
                    i + 1,
                    len(self.image_paths),
                    time.time() - start_time,
                ]
            )

        # Reconstruct image from Laplacian pyramid
        fused_image = pyramid_algorithm.reconstruct(fused_pyr)
        self.output_image = fused_image

    # TODO: Rewrite for easy stopping of task (using signals??)
    def stack_images(self, signals):
        """
        Stack images.
        """
        # Will just load first image from path
        im0 = self.Algorithm.align_image_pair(self.image_paths[0], self.image_paths[0])
        fused_pyr = self.Algorithm.generate_laplacian_pyramid(
            im0,
            self.pyramid_num_levels,
        )

        for i, path in enumerate(self.image_paths):
            if i == 0:
                continue  # First image is already copied in pyr

            start_time = time.time()

            # Load from path
            im1 = self.Algorithm.align_image_pair(path, path)
            # Generate pyramid for the image
            new_pyr = self.Algorithm.generate_laplacian_pyramid(
                im1, self.pyramid_num_levels
            )
            # Delete image (lower memory usage)
            del im1
            # Fuse this new pyramid with the existing one
            fused_pyr = self.Algorithm.focus_fuse_pyramid_pair(
                fused_pyr, new_pyr, self.fusion_kernel_size_pixels
            )

            # Send progress signal
            signals.finished_inter_task.emit(
                [
                    "finished_image",
                    i + 1,
                    len(self.image_paths),
                    time.time() - start_time,
                ]
            )

        # Reconstruct image from Laplacian pyramid
        fused_image = pyramid_algorithm.reconstruct(fused_pyr)
        self.output_image = fused_image
