import argparse
import cv2
import numpy as np
import os
import sys
import tempfile

parser = argparse.ArgumentParser(
                    prog='ChimpStackr_cli',
                    description='Command line interface to chimpstackr',
                    epilog='Contributed by Ivo Blöchliger')

parser.add_argument('-o', '--output', default="stacked.jpg")
parser.add_argument('files', metavar='filename', type=str, nargs='+',
                    help='Input files')

args = parser.parse_args()

output  = args.output
inputs = args.files

print(output)
print(inputs)

# Allow imports from top level folder. Example: "src.algorithm.API"
# src: https://codeolives.com/2020/01/10/python-reference-module-in-parent-directory/
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)  # Insert at first place

import src.settings as settings
import src.MainWindow as MainWindow

# Directory for storing tempfiles. Automatically deletes on program exit.
ROOT_TEMP_DIRECTORY = tempfile.TemporaryDirectory(prefix="ChimpStackr_")
settings.init()
settings.globalVars["RootTempDir"] = ROOT_TEMP_DIRECTORY
class DummyQSettings:
    def value(self, key):
        return {"computing/use_gpu" : False, "computing/selected_gpu_id":0}[key]

settings.globalVars["QSettings"] = DummyQSettings()

import algorithms.API as algorithm_API

class DummySignals:
    class Emitter:
        def emit(self, params):
            print(params)

    def __init__(self):
        self.finished_inter_task = self.Emitter()

# Code collected from MainWindow/__init__.py
LaplacianAlgorithm = algorithm_API.LaplacianPyramid(6, 8)
LaplacianAlgorithm.update_image_paths(inputs)
signals = DummySignals()
LaplacianAlgorithm.align_and_stack_images(signals)

# Code collected from MainWindow/ImageSavingDialog.py
# Convert float32 image to uint8
imageArray = np.around(LaplacianAlgorithm.output_image)
imageArray[imageArray > 255] = 255
imageArray[imageArray < 0] = 0
imageArray = imageArray.astype(np.uint8)
cv2.imwrite(output, imageArray, [cv2.IMWRITE_JPEG_QUALITY, 95])

