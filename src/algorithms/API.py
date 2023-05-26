"""
    Exposed API for easily aligning/stacking multiple images.
"""
import time

import src.utilities as utilities
import src.algorithms as algorithms
import src.settings as settings


class LaplacianPyramid:
    def __init__(self, fusion_kernel_size=6, pyramid_num_levels=8, use_pyqt=True, use_gpu=False, selected_gpu_id = 0):
        self.output_image = None
        self.image_paths = []
        self.use_pyqt = use_pyqt
        self.use_gpu = use_gpu
        self.selected_gpu_id = selected_gpu_id

        self.Algorithm = algorithms.Algorithm()

        # Parameters
        self.fusion_kernel_size = fusion_kernel_size
        self.pyramid_num_levels = pyramid_num_levels

        self.UseGPU = False

    def toggle_cpu_gpu(self):
        """
        Use saved settings to toggle if GPU is used for some parts of the algorithm.
        This function will be called every time the algorithm starts (before start),
        so prevent changes mid-way are not possible.
        """
        if self.use_pyqt:
            self.Algorithm.toggle_cpu_gpu(
                bool(settings.globalVars["QSettings"].value("computing/use_gpu")),
                int(settings.globalVars["QSettings"].value("computing/selected_gpu_id")),
            )

        else:
            self.Algorithm.toggle_cpu_gpu(
                self.use_gpu,
                int(self.selected_gpu_id)
            )
            

    def update_image_paths(self, new_image_paths):
        """
        Set new image paths (sorted by name).
        """
        if self.use_pyqt:
            self.image_paths = sorted(new_image_paths, key=utilities.int_string_sorting)

        else:
            self.image_paths = new_image_paths

    # TODO: Rewrite for easy stopping of task (using signals??)
    def align_and_stack_images(self, signals):
        """
        Align and stack images.
        """
        self.toggle_cpu_gpu()
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
            # TODO: GPU speedup here
            time0 = time.perf_counter()
            aligned_images.append(
                self.Algorithm.align_image_pair(aligned_images[0], path)
            )
            print(f"Align image pair: {time.perf_counter() - time0}")
            # Generate pyramid for the (aligned) image
            # TODO: GPU speedup here
            time0 = time.perf_counter()
            new_pyr = self.Algorithm.generate_laplacian_pyramid(
                aligned_images[1], self.pyramid_num_levels
            )
            print(f"Generate 1 laplacian pyramid: {time.perf_counter() - time0}")
            # Remove first aligned image array from list (lower memory usage)
            del aligned_images[0]
            # Fuse this new pyramid with the existing one
            time0 = time.perf_counter()
            fused_pyr = self.Algorithm.focus_fuse_pyramid_pair(
                fused_pyr, new_pyr, self.fusion_kernel_size
            )
            print(f"Fuse pyramid pair: {time.perf_counter() - time0}")

            # Send progress signal
            if self.use_pyqt:
                # Send progress signal
                signals.finished_inter_task.emit(
                    [
                        "finished_image",
                        i + 1,
                        len(self.image_paths),
                        time.time() - start_time,
                    ]
                )            

        # Check the value used by algorithm, not the current value (might have been toggled during stacking operation)
        if self.Algorithm.useGpu:
            # Copy pyramid back to CPU
            inter_pyr = []
            for i in fused_pyr:
                inter_pyr.append(i.copy_to_host())
            fused_pyr = inter_pyr
            del inter_pyr
        # Reconstruct image from Laplacian pyramid
        fused_image = self.Algorithm.reconstruct_pyramid(fused_pyr)
        self.output_image = fused_image

        return self.output_image

    # TODO: Rewrite for easy stopping of task (using signals??)
    def stack_images(self, signals):
        """
        Stack images.
        """
        self.toggle_cpu_gpu()
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
                fused_pyr, new_pyr, self.fusion_kernel_size
            )

            # Send progress signal
            if self.use_pyqt:
                signals.finished_inter_task.emit(
                    [
                        "finished_image",
                        i + 1,
                        len(self.image_paths),
                        time.time() - start_time,
                    ]
                )
        # Reconstruct image from Laplacian pyramid
        fused_image = self.Algorithm.reconstruct_pyramid(fused_pyr)
        self.output_image = fused_image

        return self.output_image