"""
    Class that handles loading/saving of RAW/other formats.
    If images are of a regular filetype (jpg, png, ...); they are opened using opencv.
    Else use rawpy to load RAW image.
"""
import os
from io import BytesIO
import rawpy
import cv2
import imageio
import numpy as np

import src.settings as settings


class ImageLoadingHandler:
    def __init__(self):
        return

    # TODO: Display error on image failed to load (image might have moved location)
    # Load src image to BGR 2D numpy array
    def read_image_from_path(self, path):
        # Get extension without dot at beginning
        _, extension = os.path.splitext(path)
        extension = extension[1:]
        if str.lower(extension) in settings.globalVars["SupportedImageReadFormats"]:
            # Regular imread (-1 loads the image "as is")
            try:
                return cv2.imread(path, -1)
            except Exception:
                return None
        elif str.upper(extension) in settings.globalVars["SupportedRAWFormats"]:
            # Load RAW image
            with rawpy.imread(path) as raw:
                processed = None
                try:
                    # Extract thumbnail or preview (faster)
                    thumb = raw.extract_thumb()
                except:
                    # If no thumb/preview, then postprocess RAW image (slower)
                    processed = raw.postprocess(use_camera_wb=True)
                else:
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        # Convert bytes object to ndarray
                        processed = imageio.imread(BytesIO(thumb.data))
                    elif thumb.format == rawpy.ThumbFormat.BITMAP:
                        # Ndarray
                        processed = thumb.data

                processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

            return processed
        elif str.lower(extension) == "npy":
            # Load data from ".npy" format
            return np.load(path, allow_pickle=False)

    # Get RAW image view from path (uses copy() to allow usage after closing raw file)
    def get_raw_view(self, path):
        raw = rawpy.imread(path)
        image = raw.raw_image_visible.copy()
        raw.close()
        return image
