"""
    Class that handles loading/saving of RAW/other formats.
    If images are of a regular filetype (jpg, png, ...); they are opened using opencv.
    Else use rawpy to load RAW image.
"""
import os
import logging
from io import BytesIO
import rawpy
import cv2
import imageio.v2 as imageio
import numpy as np

from src.config import SUPPORTED_IMAGE_READ_FORMATS, SUPPORTED_RAW_FORMATS

logger = logging.getLogger(__name__)


class ImageLoadingHandler:
    def __init__(self, supported_formats=None, supported_raw=None):
        self.supported_formats = SUPPORTED_IMAGE_READ_FORMATS if supported_formats is None else supported_formats
        self.supported_raw = SUPPORTED_RAW_FORMATS if supported_raw is None else supported_raw

    def read_image_from_path(self, path):
        """
        Load src image from path to BGR 2D numpy array.
        Returns None if loading fails.
        """
        if not os.path.isfile(path):
            logger.error("File not found: %s", path)
            return None

        _, extension = os.path.splitext(path)
        extension = extension[1:]

        if extension.lower() in self.supported_formats:
            try:
                img = cv2.imread(path, -1)
                if img is None:
                    logger.error("cv2.imread returned None for: %s", path)
                return img
            except Exception as e:
                logger.error("Failed to load image %s: %s", path, e)
                return None
        elif extension.upper() in self.supported_raw:
            return self._load_raw(path)
        elif extension.lower() == "npy":
            try:
                return np.load(path, allow_pickle=False)
            except Exception as e:
                logger.error("Failed to load npy %s: %s", path, e)
                return None
        else:
            logger.warning("Unsupported format: %s", extension)
            return None

    def _load_raw(self, path):
        """Load a RAW image file."""
        try:
            with rawpy.imread(path) as raw:
                processed = None
                try:
                    thumb = raw.extract_thumb()
                except (rawpy.LibRawError, rawpy.LibRawNonFatalError):
                    processed = raw.postprocess(use_camera_wb=True)
                else:
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        processed = imageio.imread(BytesIO(thumb.data))
                    elif thumb.format == rawpy.ThumbFormat.BITMAP:
                        processed = thumb.data

                if processed is None:
                    logger.error("Failed to extract image data from RAW: %s", path)
                    return None

                processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                return processed
        except Exception as e:
            logger.error("Failed to load RAW image %s: %s", path, e)
            return None

    @staticmethod
    def compute_sharpness(image):
        """Compute Laplacian variance as a sharpness metric (higher = sharper)."""
        if image is None:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
