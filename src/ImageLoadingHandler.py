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
    def read_image_from_path(self, path):
        """
        Load src image from path to BGR 2D numpy array.
        """
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
            r"""
            TODO: Error (when image was loaded from SD-card directly)
            File C:\Users\noahe\Documents\PythonFocusStackingGui\src\MainWindow\MainLayout\ImageViewer.py, line 67, in update_displayed_image
                image = self.ImageLoading.read_image_from_path(path)
            File C:\Users\noahe\Documents\PythonFocusStackingGui\src\ImageLoadingHandler.py, line 34, in read_image_from_path
                with rawpy.imread(path) as raw:
            File C:\Users\noahe\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.8_qbz5n2kfra8p0\LocalCache\local-packages\Python38\site-packages\rawpy\__init__.py, line 20, in imread
                d.open_file(pathOrFile)
            File rawpy\_rawpy.pyx, line 409, in rawpy._rawpy.RawPy.open_file
            File rawpy\_rawpy.pyx, line 936, in rawpy._rawpy.RawPy.handle_error
            rawpy._rawpy.LibRawIOError: b'Input/output error'
            """
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

    def get_raw_view(self, path):
        """Get RAW image view from path (uses copy() to allow usage after closing raw file)"""
        raw = rawpy.imread(path)
        image = raw.raw_image_visible.copy()
        raw.close()
        return image
