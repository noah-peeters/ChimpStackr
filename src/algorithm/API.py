"""
    Exposed API for easily aligning/stacking multiple images.
"""
import tempfile

import utilities
import algorithm.pyramid as pyramid
import algorithm.post_processing as post_processing
import algorithm.algorithm as algorithm


class LaplacianPyramid:
    def __init__(self, fusion_kernel_size=6, pyramid_num_levels=8):
        self.output_image = None
        self.image_paths = []
        # Archive names for Laplacian pyramids of src and aligned images
        self.laplacian_pyramid_archive_names = []
        self.laplacian_pyramid_archive_names_aligned = []
        self.root_temp_directory = tempfile.TemporaryDirectory(prefix="FocusStacking_")

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
            # Reset loaded Laplacians
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
        import time
        start_time = time.time()
        if len(self.laplacian_pyramid_archive_names) <= 0:
            # Compute Laplacian pyramids of src images
            self.laplacian_pyramid_archive_names = (
                self.Algorithm.generate_laplacian_pyramids(
                    self.image_paths, self.root_temp_directory, self.pyramid_num_levels
                )
            )

        laplacian_time = time.time()

        stacked_pyramid = self.Algorithm.focus_fuse_pyramids(
            self.laplacian_pyramid_archive_names, self.fusion_kernel_size_pixels
        )

        pyramid_fusion_time = time.time()

        # Reconstruct image from Laplacian pyramid
        fused_image = self.Pyramid.reconstruct(stacked_pyramid)

        pyramid_reconstruction_time = time.time()

        self.output_image = self.PostProcessing.apply_brightness_contrast(
            fused_image, 8, 8
        )

        post_processing_time = time.time()

        total_time = post_processing_time - start_time

        print("Total time: " + str(total_time) + "s")
        print(
            "Laplacian: " + str((laplacian_time - start_time) / total_time * 100) + "%"
        )
        print(
            "Pyramid fusion: "
            + str((pyramid_fusion_time - laplacian_time) / total_time * 100)
            + "%"
        )
        print(
            "Pyramid reconstruction: "
            + str(
                (pyramid_reconstruction_time - pyramid_fusion_time) / total_time * 100
            )
            + "%"
        )
        print(
            "Post processing: "
            + str(
                (post_processing_time - pyramid_reconstruction_time) / total_time * 100
            )
            + "%"
        )


# Test algorithm
if __name__ == "__main__":
    import glob, cv2, time
    import numba.cuda as cu

    cu.detect()
    print(cu.is_available())

    start_time = time.time()

    algo = LaplacianPyramid(fusion_kernel_size=6, pyramid_num_levels=9)

    algo.update_image_paths(glob.glob("stack_images/*.jpg"))

    print("Start stacking images...")
    algo.align_and_stack_images()

    out = algo.output_image
    cv2.imwrite("output_images/stack_output.jpg", out)

    print("--- Program execution took %s seconds ---" % (time.time() - start_time))
