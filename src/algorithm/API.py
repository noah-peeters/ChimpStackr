"""
    Exposed API for easily aligning/stacking multiple images.
"""
import os

import src.utilities as utilities
import src.algorithm.pyramid as pyramid
import src.algorithm.post_processing as post_processing
import src.algorithm.algorithm as algorithm


class LaplacianPyramid:
    def __init__(self, root_temp_directory, fusion_kernel_size=6, pyramid_num_levels=8):
        self.output_image = None
        self.image_paths = []
        # Archive names for Laplacian pyramids of src and aligned images
        self.laplacian_pyramid_archive_names = []
        self.laplacian_pyramid_archive_names_aligned = []
        self.root_temp_directory = root_temp_directory

        # Load classes
        self.Pyramid = pyramid.Pyramid()
        self.PostProcessing = post_processing.PostProcessing()
        self.Algorithm = algorithm.Algorithm()

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

    # Align and stack images
    def align_and_stack_images(self):
        # Align images to previous
        aligned_images_tmp_paths = []
        for i, path in enumerate(self.image_paths):
            if i != 0:
                aligned_images_tmp_paths.append(
                    self.Algorithm.align_image_pair(
                        self.image_paths[i - 1], path, self.root_temp_directory
                    )
                )

        self.laplacian_pyramid_archive_names_aligned = (
            self.Algorithm.generate_laplacian_pyramids(
                aligned_images_tmp_paths,
                self.root_temp_directory,
                self.pyramid_num_levels,
                True,
            )
        )

        stacked_pyramid = self.Algorithm.focus_fuse_pyramids(
            self.laplacian_pyramid_archive_names_aligned, self.fusion_kernel_size_pixels
        )

        # Reconstruct image from Laplacian pyramid
        fused_image = self.Pyramid.reconstruct(stacked_pyramid)

        self.output_image = self.PostProcessing.apply_brightness_contrast(
            fused_image, 8, 8
        )

    # Stack loaded images ( +create laplacian pyramids if not already created)
    def stack_images(self, signals):
        if len(self.laplacian_pyramid_archive_names) <= 0:
            # Compute Laplacian pyramids of src images
            self.laplacian_pyramid_archive_names = (
                self.Algorithm.generate_laplacian_pyramids(
                    self.image_paths,
                    self.root_temp_directory,
                    self.pyramid_num_levels,
                    signals,
                )
            )

        stacked_pyramid = self.Algorithm.focus_fuse_pyramids(
            self.laplacian_pyramid_archive_names,
            self.fusion_kernel_size_pixels,
            signals,
        )

        # Reconstruct image from Laplacian pyramid
        fused_image = self.Pyramid.reconstruct(stacked_pyramid)

        self.output_image = self.PostProcessing.apply_brightness_contrast(
            fused_image, 8, 8
        )
