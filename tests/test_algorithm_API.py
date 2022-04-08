"""
Test multiple functions of algorithm API.
"""
import os, sys, tempfile
import numpy as np
import PySide6.QtCore as qtc

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import src.MainWindow.Threading as QThreading
import src.algorithm.API as API

ROOT_TEMP_DIRECTORY = tempfile.TemporaryDirectory(prefix="FocusStacking_")

laplacian_pyramid_algorithm = API.LaplacianPyramid(ROOT_TEMP_DIRECTORY)

# Test image loading (+clearing)
def test_image_paths_update():
    laplacian_pyramid_algorithm.update_image_paths(os.listdir("tests/low_res_images"))
    assert len(laplacian_pyramid_algorithm.image_paths) == 10


# TODO: Test aligning images

# Test stacking images
def test_image_stacking():
    worker = QThreading.Worker(laplacian_pyramid_algorithm.stack_images)
    threadpool = qtc.QThreadPool()

    def finished_stack():
        assert type(laplacian_pyramid_algorithm.output_image) == np.ndarray
        assert laplacian_pyramid_algorithm.output_image.shape == (4000, 6000, 3)
        assert laplacian_pyramid_algorithm.output_image.dtype == np.uint8

    worker.signals.finished.connect(finished_stack)
    threadpool.start(worker)
