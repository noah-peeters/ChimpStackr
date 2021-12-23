"""
Test multiple functions of algorithm API.
"""
# Hack to import modules from src
# src: https://codeolives.com/2020/01/10/python-reference-module-in-parent-directory/
import os, sys, tempfile

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from src.algorithm import API

ROOT_TEMP_DIRECTORY = tempfile.TemporaryDirectory(prefix="FocusStacking_")

laplacian_pyramid_algorithm = API.LaplacianPyramid(ROOT_TEMP_DIRECTORY)

# Test image loading (+clearing)

# TODO: Test aligning images

# Test stacking images

# Test output export write to disk


ROOT_TEMP_DIRECTORY.cleanup()
