"""
Test multiple functions of algorithm API.
"""
# Hack to import modules
# src: https://codeolives.com/2020/01/10/python-reference-module-in-parent-directory/
import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# TODO: resolve utilities import fail
from src.algorithm import image_storage


# Test saving and loading pyramid  to/from disk
