import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.utilities import merge_image_paths


def test_merge_appends_new_paths_sorted():
    existing = ["/a/img1.png", "/a/img2.png"]
    new = ["/a/img10.png", "/a/img3.png"]
    merged = merge_image_paths(existing, new)
    # Numeric-aware sort: 1, 2, 3, 10 (not lexicographic 1,10,2,3)
    assert merged == ["/a/img1.png", "/a/img2.png", "/a/img3.png", "/a/img10.png"]


def test_merge_dedupes():
    existing = ["/a/img1.png"]
    new = ["/a/img1.png", "/a/img2.png"]
    merged = merge_image_paths(existing, new)
    assert merged == ["/a/img1.png", "/a/img2.png"]


def test_merge_into_empty():
    assert merge_image_paths([], ["/a/img1.png"]) == ["/a/img1.png"]


def test_merge_both_empty():
    assert merge_image_paths([], []) == []
