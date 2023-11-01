import cv2
import glob, os

import src.algorithms
import src.settings

src.settings.init()
import src.utilities
from src.algorithms.API import LaplacianPyramid


def do_focus_stack(img_path, out_path):
    stacker = LaplacianPyramid(use_pyqt=False)

    z_stack = []
    files = glob.glob(os.path.join(img_path, "*"))

    for file in files:
        z_stack.append(cv2.imread(file))

    stacker.update_image_paths(z_stack)
    stacked_frame = stacker.align_and_stack_images(z_stack)

    if out_path is None:
        cv2.imwrite(os.path.join(img_path, "focus_stacked.jpg"), stacked_frame)
    else:
        cv2.imwrite(out_path, stacked_frame)
