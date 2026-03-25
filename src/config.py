"""
Application configuration using a dataclass instead of global dict.
Provides structured access to settings and supported formats.
"""
from dataclasses import dataclass, field
from typing import List, Optional


# Opencv imread/imwrite supported formats
SUPPORTED_IMAGE_READ_FORMATS = [
    "jpg", "jpeg", "jpe", "jp2", "png", "bmp", "dib", "webp",
    "pbm", "pgm", "ppm", "pxm", "pnm", "pfm", "sr", "ras",
    "tiff", "tif", "exr", "hdr", "pic",
]

# Supported RAW formats (rawpy)
SUPPORTED_RAW_FORMATS = [
    "RWZ", "RW2", "CR2", "DNG", "ERF", "NRW", "RAF", "ARW",
    "NEF", "K25", "SRF", "EIP", "DCR", "RAW", "CRW", "3FR",
    "BAY", "MEF", "CS1", "KDC", "ORF", "ARI", "SR2", "MOS",
    "MFW", "CR3", "FFF", "SRW", "J6I", "X3F", "KC2", "RWL",
    "MRW", "PEF", "IIQ", "CXI", "MDC",
]


STACKING_METHODS = ["laplacian", "weighted_average", "depth_map", "exposure_fusion"]


def auto_detect_params(image_shape, num_images):
    """
    Suggest algorithm parameters based on image dimensions and stack size.
    Returns a dict of suggested values.
    """
    h, w = image_shape[:2]
    megapixels = (h * w) / 1e6

    # Pyramid levels: more for larger images (log2 of shortest dimension, capped)
    import math
    max_levels = max(2, int(math.log2(min(h, w))) - 3)
    pyramid_levels = min(max_levels, 12)

    # Kernel size: larger for higher resolution (more detail to compare)
    if megapixels > 20:
        kernel_size = 8
    elif megapixels > 8:
        kernel_size = 6
    elif megapixels > 2:
        kernel_size = 4
    else:
        kernel_size = 3

    # Alignment scale: higher for larger images (more room for sub-pixel)
    if megapixels > 20:
        scale_factor = 15
    elif megapixels > 8:
        scale_factor = 10
    else:
        scale_factor = 6

    # Feather radius: scale with image size for smooth transitions
    if megapixels > 20:
        feather = 4
    elif megapixels > 8:
        feather = 3
    else:
        feather = 2

    # Contrast threshold: small default to reduce noise in flat areas
    contrast_threshold = 1.0

    return {
        "pyramid_num_levels": pyramid_levels,
        "fusion_kernel_size": kernel_size,
        "alignment_scale_factor": scale_factor,
        "feather_radius": feather,
        "contrast_threshold": contrast_threshold,
    }


@dataclass
class AlgorithmConfig:
    """Configuration for the stacking algorithm, independent of Qt."""
    stacking_method: str = "laplacian"  # "laplacian", "weighted_average", "depth_map"
    fusion_kernel_size: int = 6
    pyramid_num_levels: int = 8
    alignment_scale_factor: int = 10
    use_gpu: bool = False
    selected_gpu_id: int = 0
    alignment_reference: str = "first"  # "first", "middle", "previous"
    align_rotation_scale: bool = False  # Enable rotation + scale alignment
    contrast_threshold: float = 0.0  # Laplacian: min contrast to switch (0 = off)
    feather_radius: int = 2  # Laplacian: blur radius for soft focusmap edges (0 = hard)
    depthmap_smoothing: int = 5  # Depth map: smoothing radius for focus map (higher = smoother)


@dataclass
class AppConfig:
    """Top-level application configuration."""
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    supported_image_formats: List[str] = field(default_factory=lambda: SUPPORTED_IMAGE_READ_FORMATS)
    supported_raw_formats: List[str] = field(default_factory=lambda: SUPPORTED_RAW_FORMATS)
