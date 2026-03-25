"""
    Command-line interface for ChimpStackr.
    Allows focus stacking without the GUI.

    Usage:
        python -m src.cli --input images/*.jpg --output result.tif
        python -m src.cli --input img1.jpg img2.jpg img3.jpg --output stacked.png --align
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

# Allow imports from top-level folder
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.config import AlgorithmConfig, AppConfig, auto_detect_params
import src.settings as settings
from src.algorithms.API import LaplacianPyramid
from src.ImageLoadingHandler import ImageLoadingHandler


def parse_args():
    parser = argparse.ArgumentParser(
        prog="chimpstackr",
        description="ChimpStackr - Focus stacking from the command line",
    )
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        help="Input image files or glob patterns",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output file path (supports .jpg, .png, .tif)",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        default=False,
        help="Align images before stacking (default: stack only)",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=6,
        help="Fusion kernel size (default: 6)",
    )
    parser.add_argument(
        "--pyramid-levels",
        type=int,
        default=8,
        help="Number of pyramid levels (default: 8)",
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        default=10,
        help="DFT alignment scale factor (default: 10)",
    )
    parser.add_argument(
        "--alignment-ref",
        choices=["first", "middle", "previous"],
        default="first",
        help="Alignment reference strategy (default: first)",
    )
    parser.add_argument(
        "--method",
        choices=["laplacian", "weighted_average", "depth_map"],
        default="laplacian",
        help="Stacking method (default: laplacian)",
    )
    parser.add_argument(
        "--rotation-scale",
        action="store_true",
        default=False,
        help="Enable rotation + scale alignment (focus breathing correction)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Use GPU acceleration (requires CUDA)",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="CUDA GPU device ID (default: 0)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality 0-100 (default: 95)",
    )
    parser.add_argument(
        "--quality-report",
        action="store_true",
        default=False,
        help="Print sharpness metrics for input and output images",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        default=False,
        help="Auto-detect optimal parameters based on image size",
    )
    parser.add_argument(
        "--auto-crop",
        action="store_true",
        default=False,
        help="Auto-crop black edges from alignment shifts",
    )
    return parser.parse_args()


def expand_input_paths(patterns):
    """Expand glob patterns and return list of valid file paths."""
    paths = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if expanded:
            paths.extend(expanded)
        elif os.path.isfile(pattern):
            paths.append(pattern)
        else:
            print(f"Warning: No files matched '{pattern}'", file=sys.stderr)
    return sorted(set(paths))


class ProgressTracker:
    """CLI progress display with bar, ETA, and speed."""
    def __init__(self):
        self.times = []
        self.start_time = time.time()

    def __call__(self, current, total, time_taken):
        self.times.append(time_taken)
        pct = current / total * 100
        remaining = total - current
        avg_time = sum(self.times) / len(self.times)
        eta = avg_time * remaining

        # Progress bar
        bar_width = 30
        filled = int(bar_width * current / total)
        bar = "█" * filled + "░" * (bar_width - filled)

        elapsed = time.time() - self.start_time
        speed = f"{time_taken:.1f}s/img"
        eta_str = f"ETA {eta:.0f}s" if remaining > 0 else "done"

        print(f"\r  {bar} {pct:5.1f}% [{current}/{total}] {speed} {eta_str}   ",
              end="", file=sys.stderr)
        if current == total:
            print(f"\n  Completed in {elapsed:.1f}s (avg {avg_time:.2f}s/image)", file=sys.stderr)


def main():
    args = parse_args()

    # Initialize settings for backward compatibility
    settings.init()

    # Expand input paths
    input_paths = expand_input_paths(args.input)
    if len(input_paths) < 2:
        print(f"Error: Need at least 2 images, got {len(input_paths)}", file=sys.stderr)
        sys.exit(1)

    print(f"ChimpStackr CLI - Focus Stacking")
    print(f"  Input: {len(input_paths)} images")
    print(f"  Output: {args.output}")
    print(f"  Mode: {'Align + Stack' if args.align else 'Stack only'}")
    # Auto-detect parameters from first image if requested
    kernel_size = args.kernel_size
    pyramid_levels = args.pyramid_levels
    scale_factor = args.scale_factor

    if args.auto:
        loader = ImageLoadingHandler()
        sample = loader.read_image_from_path(input_paths[0])
        if sample is not None:
            auto = auto_detect_params(sample.shape, len(input_paths))
            kernel_size = auto["fusion_kernel_size"]
            pyramid_levels = auto["pyramid_num_levels"]
            scale_factor = auto["alignment_scale_factor"]
            print(f"  Auto-detected: kernel={kernel_size}, levels={pyramid_levels}, scale={scale_factor}")
            del sample

    print(f"  Method: {args.method}")
    print(f"  Kernel size: {kernel_size}, Pyramid levels: {pyramid_levels}")
    if args.align:
        print(f"  Alignment: ref={args.alignment_ref}, scale={scale_factor}")
        if args.rotation_scale:
            print(f"  Rotation + Scale correction: enabled")

    # Configure algorithm
    config = AlgorithmConfig(
        stacking_method=args.method,
        fusion_kernel_size=kernel_size,
        pyramid_num_levels=pyramid_levels,
        alignment_scale_factor=scale_factor,
        use_gpu=args.gpu,
        selected_gpu_id=args.gpu_id,
        alignment_reference=args.alignment_ref,
        align_rotation_scale=args.rotation_scale,
    )

    algo = LaplacianPyramid(config=config)
    algo.update_image_paths(input_paths)

    # Quality report on inputs
    if args.quality_report:
        loader = ImageLoadingHandler()
        print("\n  Input image sharpness:")
        for path in input_paths:
            img = loader.read_image_from_path(path)
            if img is not None:
                sharpness = ImageLoadingHandler.compute_sharpness(img)
                print(f"    {os.path.basename(path)}: {sharpness:.1f}")

    # Run stacking
    print()
    progress = ProgressTracker()
    total_start = time.time()

    if args.align:
        algo.align_and_stack_images(progress_callback=progress)
    else:
        algo.stack_images(progress_callback=progress)

    total_elapsed = time.time() - total_start

    if algo.output_image is None:
        print("Error: Stacking failed or was cancelled", file=sys.stderr)
        sys.exit(1)

    # Auto-crop if requested
    if args.auto_crop and args.align:
        bounds = algo.auto_crop_output()
        if bounds:
            top, bottom, left, right = bounds
            print(f"  Auto-cropped: {top}px top, {bottom}px bottom, {left}px left, {right}px right")

    # Save output
    result = np.clip(np.around(algo.output_image), 0, 255).astype(np.uint8)

    # Determine compression params
    ext = os.path.splitext(args.output)[1].lower()
    params = None
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, args.quality]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 4]

    success = cv2.imwrite(args.output, result, params)
    if not success:
        print(f"Error: Failed to write output to {args.output}", file=sys.stderr)
        sys.exit(1)

    file_size = os.path.getsize(args.output)
    print(f"  Output saved: {args.output} ({file_size / 1024:.0f} KB)")
    print(f"  Total time: {total_elapsed:.2f}s")
    print(f"  Output shape: {result.shape}")

    # Quality report on output
    if args.quality_report:
        sharpness = ImageLoadingHandler.compute_sharpness(result)
        print(f"  Output sharpness: {sharpness:.1f}")


if __name__ == "__main__":
    main()
