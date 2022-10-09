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

    # Align and stack images
    def align_and_stack_images(self, signals):
        # Align images to reference
        # TODO: Allow option to align image stack to start, end, middle or previous images
        # ref_index=round(len(self.image_paths)/2)
        aligned_images_tmp_paths = []
        # TODO: Place every sub-step inside this loop and change emitted signal to "finished_image"
        for i, path in enumerate(self.image_paths):
            print("Aligning: " + path)
            start_time = time.time()
            if i == 0:
                # Will just return the image (no alignment)
                aligned_images_tmp_paths.append(
                    self.Algorithm.align_image_pair(path, path)
                )
            else:
                # Use previous *aligned* image instead of src image!
                aligned_images_tmp_paths.append(
                    self.Algorithm.align_image_pair(
                        aligned_images_tmp_paths[i - 1], path
                    )
                )
            # Send progress signal
            signals.finished_inter_task.emit(
                [
                    "align_images",
                    i + 1,
                    len(self.image_paths),
                    time.time() - start_time,
                ]
            )

        # Generate laplacian pyramids for every aligned image
        self.laplacian_pyramid_archive_names_aligned = (
            self.Algorithm.generate_laplacian_pyramids(
                aligned_images_tmp_paths,
                self.pyramid_num_levels,
                signals,
            )
        )

        # Stack aligned image pyramids
        stacked_pyramid = self.Algorithm.focus_fuse_pyramids(
            self.laplacian_pyramid_archive_names_aligned,
            self.fusion_kernel_size_pixels,
            signals,
        )

        # Reconstruct image from Laplacian pyramid
        fused_image = pyramid_algorithm.reconstruct(stacked_pyramid)
        self.output_image = fused_image

    # TODO: Rewrite for easy stopping of task (using signals??)
    # Stack loaded images ( +create laplacian pyramids if not already created)
    def stack_images(self, signals):
        if len(self.laplacian_pyramid_archive_names) <= 0:
            # Compute Laplacian pyramids of src images
            self.laplacian_pyramid_archive_names = (
                self.Algorithm.generate_laplacian_pyramids(
                    self.image_paths,
                    settings.globalVars["RootTempDir"],
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
        fused_image = pyramid_algorithm.reconstruct(stacked_pyramid)
        self.output_image = fused_image
