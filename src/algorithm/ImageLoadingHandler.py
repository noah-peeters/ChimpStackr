"""
    Class that handles loading/saving of RAW/other formats.
    If images are of a regular filetype (jpg, png, ...); they are opened using opencv.
    Else use rawpy to load RAW image.
"""
import os
import rawpy
import cv2

# All RAW formats; src: https://fileinfo.com/filetypes/camera_raw
supported_rawpy_formats = [
    "RWZ",
    "RW2",
    "CR2",
    "DNG",
    "ERF",
    "NRW",
    "RAF",
    "ARW",
    "NEF",
    "K25",
    "DNG",
    "SRF",
    "EIP",
    "DCR",
    "RAW",
    "CRW",
    "3FR",
    "BAY",
    "MEF",
    "CS1",
    "KDC",
    "ORF",
    "ARI",
    "SR2",
    "MOS",
    "MFW",
    "CR3",
    "FFF",
    "SRW",
    "J6I",
    "X3F",
    "KC2",
    "RWL",
    "MRW",
    "PEF",
    "IIQ",
    "CXI",
    "MDC",
]
# Open-cv imread supported formats; src: https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
supported_opencv_formats = [
    "bmp",
    "dib",
    "jpeg",
    "jpg",
    "jpe",
    "jp2",
    "png",
    "webp",
    "pbm",
    "pgm",
    "ppm",
    "pxm",
    "pnm",
    "pfm",
    "sr",
    "ras",
    "tiff",
    "tif",
    "exr",
    "hdr",
    "pic",
]


class ImageLoadingHandler:
    def __init__(self):
        return

    # Load src image to BGR 2D numpy array
    def read_image_from_path(self, path):
        # Get extension without dot at beginning
        _, extension = os.path.splitext(path)
        extension = extension[1:]
        print(extension)
        if extension in supported_opencv_formats:
            # Regular imread
            return cv2.imread(path)
        elif extension in supported_rawpy_formats:
            # Read RAW image
            raw = rawpy.imread(path)
            processed = raw.postprocess(use_camera_wb=True)
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

            raw.close()
            return processed
