"""
Test loading of RAW and jpg images.
"""
import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np

from src.ImageLoadingHandler import ImageLoadingHandler
import src.settings as settings

settings.init()
loader = ImageLoadingHandler()


def test_loading_of_jpg_image():
    jpg_path = "tests/ref_img.jpg"
    img = loader.read_image_from_path(jpg_path)

    assert type(img) == np.ndarray
    assert img.shape == (4000, 6000, 3)
    assert img.dtype == np.uint8


def test_loading_of_raw_image():
    jpg_path = "tests/ref_img.nef"
    img = loader.read_image_from_path(jpg_path)

    assert type(img) == np.ndarray
    assert img.shape == (4000, 6000, 3)
    assert img.dtype == np.uint8
