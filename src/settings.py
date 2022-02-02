"""
Global variables shared between all scripts.
"""


def init():
    global globalVars
    globalVars = {}
    # Opencv imread/imwrite supported formats;
    # src: https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
    globalVars["SupportedExportFormats"] = [
        "jpg",
        "jpeg",
        "jpe",
        "jp2",
        "png",
        "bmp",
        "dib",
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
    # Supported RAW formats (rawpy); src: https://fileinfo.com/filetypes/camera_raw
    globalVars["SupportedRAWFormats"] = [
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
