"""
Test loading of RAW and jpg images, including edge cases.
"""
import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import pytest

from src.ImageLoadingHandler import ImageLoadingHandler
import src.settings as settings

settings.init()
loader = ImageLoadingHandler()


def test_loading_of_jpg_image():
    jpg_path = "tests/ref_img.jpg"
    img = loader.read_image_from_path(jpg_path)

    assert isinstance(img, np.ndarray)
    assert img.shape == (4000, 6000, 3)
    assert img.dtype == np.uint8


def test_loading_of_raw_image():
    nef_path = "tests/ref_img.nef"
    img = loader.read_image_from_path(nef_path)

    assert isinstance(img, np.ndarray)
    assert img.shape == (4000, 6000, 3)
    assert img.dtype == np.uint8


def test_loading_nonexistent_file():
    """Loading a file that doesn't exist should return None."""
    img = loader.read_image_from_path("tests/nonexistent_file.jpg")
    assert img is None


def test_loading_unsupported_format():
    """Loading an unsupported format should return None."""
    img = loader.read_image_from_path("tests/test_ImageLoadingHandler.py")
    assert img is None


def test_loading_empty_path():
    """Loading with empty string should return None."""
    img = loader.read_image_from_path("")
    assert img is None


def test_sharpness_metric():
    """Sharpness metric should return a positive float for real images."""
    jpg_path = "tests/ref_img.jpg"
    img = loader.read_image_from_path(jpg_path)
    sharpness = ImageLoadingHandler.compute_sharpness(img)
    assert isinstance(sharpness, float)
    assert sharpness > 0


def test_sharpness_metric_none():
    """Sharpness of None should be 0."""
    assert ImageLoadingHandler.compute_sharpness(None) == 0.0


def test_low_res_images_load():
    """All 10 low-res test images should load successfully."""
    image_dir = "tests/low_res_images"
    images = sorted(os.listdir(image_dir))
    assert len(images) == 10

    for name in images:
        path = os.path.join(image_dir, name)
        img = loader.read_image_from_path(path)
        assert img is not None, f"Failed to load {name}"
        assert img.shape == (500, 750, 3)
        assert img.dtype == np.uint8


def test_custom_supported_formats():
    """ImageLoadingHandler accepts custom format lists."""
    custom_loader = ImageLoadingHandler(
        supported_formats=["jpg"],
        supported_raw=[],
    )
    # JPG should work
    img = custom_loader.read_image_from_path("tests/ref_img.jpg")
    assert img is not None

    # NEF should return None (not in custom supported_raw list)
    img = custom_loader.read_image_from_path("tests/ref_img.nef")
    # With custom empty raw list, NEF should not be recognized
    assert img is None, "NEF should not load when supported_raw is empty"


def test_float32_loading():
    """Float32 loading should return float32 array in 0-255 range."""
    img = loader.read_image_as_float32("tests/low_res_images/DSC_0356.jpg")
    assert img is not None
    assert img.dtype == np.float32
    assert img.max() <= 255.0
    assert img.min() >= 0.0


def test_float32_loading_nonexistent():
    """Float32 loading of nonexistent file should return None."""
    img = loader.read_image_as_float32("tests/nonexistent.jpg")
    assert img is None


def test_sharpness_flat_image():
    """Flat image should have very low sharpness."""
    flat = np.full((100, 100, 3), 128, dtype=np.uint8)
    sharpness = ImageLoadingHandler.compute_sharpness(flat)
    assert sharpness < 1.0


def test_sharpness_noisy_image():
    """Noisy image should have higher sharpness than flat."""
    flat = np.full((100, 100, 3), 128, dtype=np.uint8)
    noisy = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    flat_sharp = ImageLoadingHandler.compute_sharpness(flat)
    noisy_sharp = ImageLoadingHandler.compute_sharpness(noisy)
    assert noisy_sharp > flat_sharp


def test_is_supported_format():
    """Format checking should handle case insensitivity."""
    assert loader.is_supported("photo.JPG")
    assert loader.is_supported("photo.jpg")
    assert loader.is_supported("photo.PNG")
    assert loader.is_supported("photo.tiff")
    assert loader.is_supported("photo.NEF")
    assert not loader.is_supported("photo.txt")
    assert not loader.is_supported("photo.pdf")
    assert not loader.is_supported(".DS_Store")
