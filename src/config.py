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


@dataclass
class AlgorithmConfig:
    """Configuration for the stacking algorithm, independent of Qt."""
    fusion_kernel_size: int = 6
    pyramid_num_levels: int = 8
    alignment_scale_factor: int = 10
    use_gpu: bool = False
    selected_gpu_id: int = 0
    alignment_reference: str = "first"  # "first", "middle", "previous"


@dataclass
class AppConfig:
    """Top-level application configuration."""
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    supported_image_formats: List[str] = field(default_factory=lambda: SUPPORTED_IMAGE_READ_FORMATS)
    supported_raw_formats: List[str] = field(default_factory=lambda: SUPPORTED_RAW_FORMATS)
