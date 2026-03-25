"""
Test the algorithm API: pyramid generation, stacking, alignment, fusion, and configuration.
No PySide6/Qt dependency required.
"""
import os, sys, glob

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import cv2
import pytest

import src.settings as settings
settings.init()

from src.algorithms import Algorithm
from src.algorithms.API import LaplacianPyramid
from src.algorithms.stacking_algorithms import cpu as CPU
from src.config import AlgorithmConfig
from src.ImageLoadingHandler import ImageLoadingHandler


# -- Fixtures --

@pytest.fixture(scope="module")
def test_images():
    """Load all low-res test images."""
    loader = ImageLoadingHandler()
    image_dir = "tests/low_res_images"
    paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    images = []
    for p in paths:
        img = loader.read_image_from_path(p)
        assert img is not None
        images.append(img)
    return images


@pytest.fixture(scope="module")
def test_image_paths():
    """Return sorted paths to low-res test images."""
    image_dir = "tests/low_res_images"
    return sorted(glob.glob(os.path.join(image_dir, "*.jpg")))


@pytest.fixture
def algo():
    """Create a fresh Algorithm instance."""
    a = Algorithm()
    a.useGpu = False
    return a


# -- CPU Module Tests --

class TestCPUFunctions:
    def test_gaussian_pyramid(self, test_images):
        img = test_images[0]
        pyr = CPU.gaussian_pyramid(img, 8)
        assert len(pyr) == 9  # original + 8 levels
        assert pyr[0].shape == (500, 750, 3)
        assert pyr[0].dtype == np.float32
        # Each level should be roughly half the previous
        for i in range(1, len(pyr)):
            assert pyr[i].shape[0] <= pyr[i - 1].shape[0]
            assert pyr[i].shape[1] <= pyr[i - 1].shape[1]

    def test_laplacian_pyramid_generation(self, test_images):
        img = test_images[0]
        pyr = CPU.generate_laplacian_pyramid(img, 8)
        assert len(pyr) == 9
        # Smallest level first, largest last
        assert pyr[0].shape[0] < pyr[-1].shape[0]
        assert pyr[-1].shape == (500, 750, 3)
        assert pyr[0].dtype == np.float32

    def test_pyramid_roundtrip(self, test_images):
        """Generating a pyramid and reconstructing should approximate the original."""
        img = test_images[0].astype(np.float32)
        pyr = CPU.generate_laplacian_pyramid(img, 8)
        reconstructed = CPU.reconstruct_pyramid(pyr)

        assert reconstructed.shape == img.shape
        # Reconstruction should be close to original (within float precision)
        diff = np.abs(reconstructed - img).mean()
        assert diff < 1.0, f"Mean difference {diff} too large"

    def test_compute_focusmap(self, test_images):
        gray1 = cv2.cvtColor(test_images[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray2 = cv2.cvtColor(test_images[1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        focusmap = CPU.compute_focusmap(gray1, gray2, 6)

        assert focusmap.shape == gray1.shape
        assert focusmap.dtype == np.uint8
        assert set(np.unique(focusmap)).issubset({0, 1})

    def test_fuse_does_not_mutate_input(self, test_images):
        """Fusing should not modify the original input arrays."""
        img1 = test_images[0].astype(np.float32)
        img2 = test_images[1].astype(np.float32)

        pyr1 = CPU.generate_laplacian_pyramid(img1, 4)
        pyr2 = CPU.generate_laplacian_pyramid(img2, 4)

        # Copy originals for comparison
        pyr1_copy = [level.copy() for level in pyr1]

        gray1 = cv2.cvtColor(pyr1[0], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(pyr2[0], cv2.COLOR_BGR2GRAY)
        focusmap = CPU.compute_focusmap(gray1, gray2, 6)

        _ = CPU.fuse_pyramid_levels_using_focusmap(pyr1[0], pyr2[0], focusmap)

        # Original pyr1[0] should be unchanged
        np.testing.assert_array_equal(pyr1_copy[0], pyr1[0])


# -- Algorithm Class Tests --

class TestAlgorithm:
    def test_align_same_image(self, algo, test_images):
        """Aligning an image to itself via paths should just load it."""
        path = "tests/low_res_images/DSC_0356.jpg"
        result = algo.align_image_pair(path, path)
        assert isinstance(result, np.ndarray)
        assert result.shape == (500, 750, 3)

    def test_align_different_images(self, algo, test_images):
        """Alignment should produce an array of same shape."""
        result = algo.align_image_pair(test_images[0], test_images[1])
        assert isinstance(result, np.ndarray)
        assert result.shape == test_images[0].shape

    def test_generate_pyramid(self, algo, test_images):
        pyr = algo.generate_laplacian_pyramid(test_images[0], 8)
        assert len(pyr) == 9
        assert pyr[-1].shape == (500, 750, 3)

    def test_fuse_pair(self, algo, test_images):
        pyr1 = algo.generate_laplacian_pyramid(test_images[0], 4)
        pyr2 = algo.generate_laplacian_pyramid(test_images[1], 4)
        fused = algo.focus_fuse_pyramid_pair(pyr1, pyr2, 6)
        assert len(fused) == len(pyr1)
        for i in range(len(fused)):
            assert fused[i].shape == pyr1[i].shape

    def test_cancel_flag(self, algo):
        """Cancel mechanism should work correctly."""
        assert not algo.is_cancelled
        algo.cancel()
        assert algo.is_cancelled
        algo.reset_cancel()
        assert not algo.is_cancelled


# -- LaplacianPyramid API Tests --

class TestLaplacianPyramidAPI:
    def test_stack_only(self, test_image_paths):
        """Full stack-only pipeline should produce a valid output."""
        config = AlgorithmConfig(fusion_kernel_size=6, pyramid_num_levels=4)
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(test_image_paths)

        progress_log = []
        def log_progress(current, total, time_taken):
            progress_log.append((current, total))

        lp.stack_images(progress_callback=log_progress)

        assert lp.output_image is not None
        assert lp.output_image.shape == (500, 750, 3)
        assert lp.output_image.dtype == np.float32
        assert len(progress_log) == 9  # 10 images - 1 (first)

    def test_align_and_stack(self, test_image_paths):
        """Full align+stack pipeline should produce a valid output."""
        config = AlgorithmConfig(fusion_kernel_size=6, pyramid_num_levels=4)
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(test_image_paths)

        lp.align_and_stack_images()

        assert lp.output_image is not None
        assert lp.output_image.shape == (500, 750, 3)

    def test_stacked_output_sharper_than_inputs(self, test_image_paths):
        """The stacked output should be sharper than most individual inputs."""
        config = AlgorithmConfig(fusion_kernel_size=6, pyramid_num_levels=8)
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(test_image_paths)
        lp.stack_images()

        result = np.clip(lp.output_image, 0, 255).astype(np.uint8)
        output_sharpness = ImageLoadingHandler.compute_sharpness(result)

        loader = ImageLoadingHandler()
        input_sharpnesses = []
        for path in test_image_paths:
            img = loader.read_image_from_path(path)
            input_sharpnesses.append(ImageLoadingHandler.compute_sharpness(img))

        median_input = sorted(input_sharpnesses)[len(input_sharpnesses) // 2]
        assert output_sharpness > median_input, (
            f"Output sharpness {output_sharpness:.1f} should exceed median input {median_input:.1f}"
        )

    def test_alignment_reference_middle(self, test_image_paths):
        """Middle alignment reference should work."""
        config = AlgorithmConfig(
            fusion_kernel_size=6, pyramid_num_levels=4, alignment_reference="middle"
        )
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(test_image_paths)
        lp.align_and_stack_images()
        assert lp.output_image is not None

    def test_configure_method(self):
        """The configure method should update config attributes."""
        config = AlgorithmConfig()
        lp = LaplacianPyramid(config=config)
        lp.configure(fusion_kernel_size=10, pyramid_num_levels=4)
        assert lp.config.fusion_kernel_size == 10
        assert lp.config.pyramid_num_levels == 4

    def test_cancel_stops_processing(self, test_image_paths):
        """Cancelling during processing should leave output_image as None."""
        import threading

        config = AlgorithmConfig(fusion_kernel_size=6, pyramid_num_levels=4)
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(test_image_paths)

        # Cancel after first image via progress callback
        def cancel_on_first(current, total, time_taken):
            lp.cancel()

        lp.stack_images(progress_callback=cancel_on_first)

        # Output should be None since we cancelled during processing
        assert lp.output_image is None

    def test_update_image_paths_sorting(self):
        """Paths should be sorted numerically."""
        lp = LaplacianPyramid()
        paths = ["img_11.jpg", "img_2.jpg", "img_1.jpg"]
        lp.update_image_paths(paths)
        assert lp.image_paths == ["img_1.jpg", "img_2.jpg", "img_11.jpg"]


# -- Edge Cases --

class TestEdgeCases:
    def test_single_image_stack(self):
        """Stacking a single image should not crash (but also won't produce output via the loop)."""
        config = AlgorithmConfig(fusion_kernel_size=6, pyramid_num_levels=4)
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(["tests/low_res_images/DSC_0356.jpg"])
        # This should not crash - it just generates pyramid for the first image
        # but the loop body never executes so output stays None
        # (This is a known limitation - could be improved)
        lp.stack_images()

    def test_two_image_stack(self):
        """Stacking exactly 2 images should work."""
        paths = sorted(glob.glob("tests/low_res_images/*.jpg"))[:2]
        config = AlgorithmConfig(fusion_kernel_size=6, pyramid_num_levels=4)
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(paths)
        lp.stack_images()
        assert lp.output_image is not None
        assert lp.output_image.shape == (500, 750, 3)

    def test_different_kernel_sizes(self, test_image_paths):
        """Different kernel sizes should all produce valid output."""
        for kernel_size in [2, 4, 6, 8]:
            config = AlgorithmConfig(
                fusion_kernel_size=kernel_size, pyramid_num_levels=4
            )
            lp = LaplacianPyramid(config=config)
            lp.update_image_paths(test_image_paths[:3])
            lp.stack_images()
            assert lp.output_image is not None, f"Failed with kernel_size={kernel_size}"

    def test_different_pyramid_levels(self, test_image_paths):
        """Different pyramid levels should all produce valid output."""
        for levels in [2, 4, 6, 8]:
            config = AlgorithmConfig(
                fusion_kernel_size=6, pyramid_num_levels=levels
            )
            lp = LaplacianPyramid(config=config)
            lp.update_image_paths(test_image_paths[:3])
            lp.stack_images()
            assert lp.output_image is not None, f"Failed with levels={levels}"


# -- Config Tests --

class TestConfig:
    def test_default_config(self):
        config = AlgorithmConfig()
        assert config.fusion_kernel_size == 6
        assert config.pyramid_num_levels == 8
        assert config.alignment_scale_factor == 10
        assert config.use_gpu is False
        assert config.alignment_reference == "first"

    def test_custom_config(self):
        config = AlgorithmConfig(
            fusion_kernel_size=10,
            pyramid_num_levels=4,
            alignment_reference="middle",
        )
        assert config.fusion_kernel_size == 10
        assert config.pyramid_num_levels == 4
        assert config.alignment_reference == "middle"
