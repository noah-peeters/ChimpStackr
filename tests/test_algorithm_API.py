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

    def test_new_config_fields(self):
        config = AlgorithmConfig()
        assert config.stacking_method == "laplacian"
        assert config.align_rotation_scale is False

        config2 = AlgorithmConfig(stacking_method="depth_map", align_rotation_scale=True)
        assert config2.stacking_method == "depth_map"
        assert config2.align_rotation_scale is True


# -- Weighted Average Tests --

class TestWeightedAverage:
    def test_focus_weights(self):
        """Focus weights should be non-negative and highlight edges."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8).astype(np.float32)
        weights = CPU.compute_focus_weights(img)
        assert weights.shape == (100, 100)
        assert weights.min() >= 0

    def test_weighted_average_fuse_pair(self):
        """Fusing two images should produce same-shape output."""
        img1 = np.random.rand(50, 50, 3).astype(np.float32) * 255
        img2 = np.random.rand(50, 50, 3).astype(np.float32) * 255
        result = CPU.weighted_average_fuse_pair(img1, img2)
        assert result.shape == img1.shape
        assert result.dtype == np.float32

    def test_weighted_average_fuse_multi(self):
        """Fusing multiple images should produce valid output."""
        images = [np.random.rand(50, 50, 3).astype(np.float32) * 255 for _ in range(5)]
        result = CPU.weighted_average_fuse_multi(images)
        assert result.shape == images[0].shape
        assert result.dtype == np.float32

    def test_weighted_average_stack(self, test_image_paths):
        """Full weighted average stacking pipeline."""
        config = AlgorithmConfig(stacking_method="weighted_average", fusion_kernel_size=6)
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(test_image_paths[:3])
        lp.stack_images()
        assert lp.output_image is not None
        assert lp.output_image.dtype == np.float32

    def test_weighted_average_align_and_stack(self, test_image_paths):
        """Weighted average with alignment."""
        config = AlgorithmConfig(stacking_method="weighted_average", fusion_kernel_size=6)
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(test_image_paths[:3])
        lp.align_and_stack_images()
        assert lp.output_image is not None


# -- Depth Map Tests --

class TestDepthMap:
    def test_depthmap_index(self):
        """Depth map should return valid indices for each pixel."""
        images = [np.random.rand(50, 50, 3).astype(np.float32) * 255 for _ in range(4)]
        depth_idx = CPU.compute_depthmap_index(images)
        assert depth_idx.shape == (50, 50)
        assert depth_idx.min() >= 0
        assert depth_idx.max() < 4

    def test_depthmap_fuse_multi(self):
        """Depth map fusion should produce valid output with index map."""
        images = [np.random.rand(50, 50, 3).astype(np.float32) * 255 for _ in range(3)]
        result, depth_idx = CPU.depthmap_fuse_multi(images)
        assert result.shape == images[0].shape
        assert result.dtype == np.float32
        assert depth_idx.shape == (50, 50)

    def test_depthmap_stack(self, test_image_paths):
        """Full depth map stacking pipeline."""
        config = AlgorithmConfig(stacking_method="depth_map", fusion_kernel_size=11)
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(test_image_paths[:3])
        lp.stack_images()
        assert lp.output_image is not None

    def test_depthmap_align_and_stack(self, test_image_paths):
        """Depth map with alignment."""
        config = AlgorithmConfig(stacking_method="depth_map", fusion_kernel_size=11)
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(test_image_paths[:3])
        lp.align_and_stack_images()
        assert lp.output_image is not None


# -- 16-bit Pipeline Tests --

class TestFloat32Pipeline:
    def test_load_image_as_float32(self):
        """Loading via float32 path should return float32 array."""
        loader = ImageLoadingHandler()
        img = loader.read_image_as_float32("tests/low_res_images/DSC_0356.jpg")
        assert img is not None
        assert img.dtype == np.float32
        assert img.ndim == 3

    def test_alignment_preserves_float32(self):
        """Aligned images should stay float32."""
        algo = Algorithm()
        img = algo.load_image("tests/low_res_images/DSC_0356.jpg")
        assert img.dtype == np.float32
        # Align to self
        result = algo.align_image_pair(img, img.copy(), scale_factor=5)
        assert result.dtype == np.float32

    def test_stacking_output_float32(self, test_image_paths):
        """Stacking output should be float32."""
        config = AlgorithmConfig(fusion_kernel_size=6, pyramid_num_levels=4)
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(test_image_paths[:3])
        lp.stack_images()
        assert lp.output_image is not None
        assert lp.output_image.dtype == np.float32


# -- Rotation + Scale Alignment Tests --

class TestRSTAlignment:
    def test_rst_alignment_same_image(self):
        """RST aligning an image to itself should produce similar output."""
        algo = Algorithm()
        img = algo.load_image("tests/low_res_images/DSC_0356.jpg")
        result = algo.align_image_pair(img, img.copy(), scale_factor=5, use_rst=True)
        assert result is not None
        assert result.shape == img.shape
        assert result.dtype == np.float32

    def test_rst_config_flag(self, test_image_paths):
        """align_rotation_scale config should enable RST in the API."""
        config = AlgorithmConfig(
            fusion_kernel_size=6, pyramid_num_levels=4,
            align_rotation_scale=True
        )
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(test_image_paths[:2])
        lp.align_and_stack_images()
        assert lp.output_image is not None


# -- Laplacian Enhancement Tests --

class TestLaplacianEnhancements:
    def test_contrast_threshold(self):
        """Thresholded focusmap should mostly not switch in flat areas."""
        flat1 = np.full((40, 40), 128.0, dtype=np.float32)
        flat2 = np.full((40, 40), 130.0, dtype=np.float32)
        # With high threshold, both are below it → interior should be all 0
        fm = CPU.compute_focusmap_thresholded(flat1, flat2, 4, np.float32(100.0))
        # Interior (away from edges) should have no switching
        interior = fm[4:-4, 4:-4]
        assert interior.sum() == 0

    def test_feather_focusmap(self):
        """Feathered focusmap should be float in [0, 1]."""
        fm = np.zeros((50, 50), dtype=np.uint8)
        fm[20:30, 20:30] = 255
        soft = CPU.feather_focusmap(fm, radius=3)
        assert soft.dtype == np.float32
        assert soft.min() >= 0
        assert soft.max() <= 1.0
        # Should have values between 0 and 1 (smooth transitions)
        assert 0 < soft[19, 25] < 1.0  # Edge should be feathered

    def test_soft_fusion(self):
        """Soft fusion should blend, not hard-select."""
        img1 = np.full((20, 20, 3), 100.0, dtype=np.float32)
        img2 = np.full((20, 20, 3), 200.0, dtype=np.float32)
        soft_map = np.full((20, 20), 0.5, dtype=np.float32)
        result = CPU.fuse_pyramid_levels_soft(img1, img2, soft_map)
        # 50/50 blend should give ~150
        assert abs(result[10, 10, 0] - 150.0) < 1.0

    def test_laplacian_with_threshold_and_feather(self, test_image_paths):
        """Full pipeline with contrast threshold and feathering."""
        config = AlgorithmConfig(
            fusion_kernel_size=6, pyramid_num_levels=4,
            contrast_threshold=2.0, feather_radius=3,
        )
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(test_image_paths[:3])
        lp.stack_images()
        assert lp.output_image is not None

    def test_multires_sharpness(self):
        """Multi-res sharpness should produce valid output."""
        img = np.random.rand(50, 50, 3).astype(np.float32) * 255
        sharpness = CPU.compute_multires_sharpness(img)
        assert sharpness.shape == (50, 50)
        assert sharpness.min() >= 0

    def test_depthmap_smoothing(self):
        """Higher smoothing should produce smoother sharpness maps."""
        img = np.random.rand(50, 50, 3).astype(np.float32) * 255
        sharp_low = CPU.compute_multires_sharpness(img, smoothing=0)
        sharp_high = CPU.compute_multires_sharpness(img, smoothing=10)
        # Higher smoothing should have lower variance (smoother)
        assert sharp_high.var() < sharp_low.var()

    def test_depthmap_with_smoothing(self, test_image_paths):
        """Depth map with smoothing via API."""
        config = AlgorithmConfig(stacking_method="depth_map", depthmap_smoothing=8)
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(test_image_paths[:3])
        lp.stack_images()
        assert lp.output_image is not None


# -- Exposure Fusion Tests --

class TestExposureFusion:
    def test_mertens_fuse_batch(self):
        """Mertens fusion of a small batch should produce valid output."""
        images = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8).astype(np.float32)
                  for _ in range(3)]
        result = CPU.mertens_fuse_batch(images)
        assert result.shape == images[0].shape
        assert result.dtype == np.float32
        assert result.min() >= 0
        assert result.max() <= 255.0

    def test_mertens_single_image(self):
        """Mertens with a single image should still work."""
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8).astype(np.float32)
        result = CPU.mertens_fuse_batch([img])
        assert result is not None
        assert result.shape == img.shape

    def test_exposure_fusion_stack(self, test_image_paths):
        """Full exposure fusion stacking pipeline."""
        config = AlgorithmConfig(stacking_method="exposure_fusion")
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(test_image_paths[:4])
        lp.stack_images()
        assert lp.output_image is not None
        assert lp.output_image.dtype == np.float32


# -- Auto-crop Tests --

class TestAutoCrop:
    def test_crop_bounds_from_shifts(self, test_image_paths):
        """Auto-crop should compute bounds from alignment shifts."""
        config = AlgorithmConfig(fusion_kernel_size=6, pyramid_num_levels=4)
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(test_image_paths[:3])
        lp.align_and_stack_images()
        assert lp.output_image is not None

        bounds = lp.get_crop_bounds()
        if bounds is not None:
            top, bottom, left, right = bounds
            assert all(v >= 0 for v in bounds)

    def test_crop_bounds_no_alignment(self, test_image_paths):
        """Auto-crop with no alignment shifts should return None."""
        config = AlgorithmConfig(fusion_kernel_size=6, pyramid_num_levels=4)
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(test_image_paths[:3])
        lp.stack_images()  # stack without alignment
        bounds = lp.get_crop_bounds()
        assert bounds is None  # no shifts → nothing to crop

    def test_auto_crop_output(self, test_image_paths):
        """auto_crop_output should reduce image dimensions."""
        config = AlgorithmConfig(fusion_kernel_size=6, pyramid_num_levels=4)
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(test_image_paths[:3])
        lp.align_and_stack_images()
        assert lp.output_image is not None
        original_shape = lp.output_image.shape

        bounds = lp.auto_crop_output()
        if bounds is not None:
            # Cropped image should be smaller or equal
            assert lp.output_image.shape[0] <= original_shape[0]
            assert lp.output_image.shape[1] <= original_shape[1]


# -- Pause/Cancel Tests --

class TestPauseCancel:
    def test_pause_and_resume(self):
        """Pause/resume flags should toggle correctly."""
        algo = Algorithm()
        assert not algo.is_paused
        algo.pause()
        assert algo.is_paused
        algo.resume()
        assert not algo.is_paused

    def test_cancel_clears_on_reset(self):
        """reset_cancel should clear both cancel and pause states."""
        algo = Algorithm()
        algo.cancel()
        assert algo.is_cancelled
        algo.reset_cancel()
        assert not algo.is_cancelled
        assert not algo.is_paused

    def test_cancel_during_alignment(self, test_image_paths):
        """Cancel during align+stack should leave output None."""
        config = AlgorithmConfig(fusion_kernel_size=6, pyramid_num_levels=4)
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(test_image_paths)

        def cancel_early(current, total, time_taken):
            if current >= 2:
                lp.cancel()

        lp.align_and_stack_images(progress_callback=cancel_early)
        assert lp.output_image is None


# -- Memory Safety Tests --

class TestMemorySafety:
    def test_stack_cleanup_after_cancel(self, test_image_paths):
        """After cancel, the API should be reusable for a new stack."""
        config = AlgorithmConfig(fusion_kernel_size=6, pyramid_num_levels=4)
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(test_image_paths[:3])

        # Cancel immediately
        def cancel_now(current, total, time_taken):
            lp.cancel()
        lp.stack_images(progress_callback=cancel_now)
        assert lp.output_image is None

        # Now stack again — should succeed
        lp.stack_images()
        assert lp.output_image is not None

    def test_reuse_api_different_method(self, test_image_paths):
        """Switching stacking method and re-running should work."""
        config = AlgorithmConfig(fusion_kernel_size=6, pyramid_num_levels=4)
        lp = LaplacianPyramid(config=config)
        lp.update_image_paths(test_image_paths[:3])

        lp.stack_images()
        assert lp.output_image is not None
        first_output = lp.output_image.copy()

        lp.configure(stacking_method="weighted_average")
        lp.stack_images()
        assert lp.output_image is not None
        # Should be a different result
        assert not np.array_equal(first_output, lp.output_image)


# -- GPU Fallback Tests --

class TestGPUFallback:
    def test_gpu_module_imports(self):
        """GPU module should import without crashing even without CUDA."""
        from src.algorithms.stacking_algorithms import gpu as GPU
        assert hasattr(GPU, 'compute_focusmap')
        assert hasattr(GPU, 'fuse_pyramid_levels_using_focusmap')
        assert hasattr(GPU, 'generate_laplacian_pyramid')
        assert hasattr(GPU, 'reconstruct_pyramid')

    def test_gpu_focusmap_fallback(self):
        """GPU compute_focusmap should fall back to CPU when CUDA unavailable."""
        from src.algorithms.stacking_algorithms import gpu as GPU
        gray1 = np.random.rand(30, 30).astype(np.float32)
        gray2 = np.random.rand(30, 30).astype(np.float32)
        # This should work regardless of CUDA availability (falls back to CPU)
        fm = GPU.compute_focusmap(gray1, gray2, 4)
        assert fm.shape == gray1.shape
        assert fm.dtype == np.uint8


# -- DFT Alignment Regression Tests --

class TestDFTAlignment:
    def test_translation_detection(self):
        """A known translation should be detected approximately."""
        img = np.random.rand(100, 100).astype(np.float32) * 255
        # Create shifted version
        shifted = np.zeros_like(img)
        shifted[5:, 3:] = img[:-5, :-3]  # shift down 5, right 3

        from src.algorithms.dft_imreg import translation
        result = translation(img, shifted)
        assert 'tvec' in result
        assert 'success' in result
        # Translation vector should be approximately (-5, -3) or (5, 3)
        tvec = result['tvec']
        assert abs(abs(tvec[0]) - 5) < 2, f"Y shift {tvec[0]} not close to 5"
        assert abs(abs(tvec[1]) - 3) < 2, f"X shift {tvec[1]} not close to 3"

    def test_identical_images_no_shift(self):
        """Identical images should have near-zero translation."""
        img = np.random.rand(80, 80).astype(np.float32) * 255
        from src.algorithms.dft_imreg import translation
        result = translation(img, img.copy())
        tvec = result['tvec']
        assert abs(tvec[0]) < 1.0, f"Y shift {tvec[0]} should be ~0"
        assert abs(tvec[1]) < 1.0, f"X shift {tvec[1]} should be ~0"
