import os
import sys
import cv2
import argparse
import glob

# Allow imports from top level folder. Example: "src.algorithm.API"
# src: https://codeolives.com/2020/01/10/python-reference-module-in-parent-directory/
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)  # Insert at first place

import src.algorithms
import src.settings
src.settings.init()
import src.utilities
from src.algorithms.API import LaplacianPyramid

def do_focus_stack(img_path, out_path):
    stacker = LaplacianPyramid(use_pyqt=False)

    z_stack = [] 
    files = glob.glob(os.path.join(img_path, '*'))

    for file in files:
        z_stack.append(cv2.imread(file)) 

    stacker.update_image_paths(z_stack)
    stacked_frame = stacker.align_and_stack_images(z_stack)

    if out_path is None:
        cv2.imwrite(os.path.join(img_path, "focus_stacked.jpg"), stacked_frame)
    else:
        cv2.imwrite(out_path, stacked_frame)    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-i','--img_dir', help='input images path', required=True)
    parser.add_argument('-o','--out_path', help='output path', required=False)
    args = vars(parser.parse_args())

    do_focus_stack(img_path=args["img_dir"], out_path=args["out_path"])