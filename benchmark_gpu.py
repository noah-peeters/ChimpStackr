"""
GPU pipeline benchmark -- profiles every stage of the stacking algorithm.
Compares old per-pixel kernel approach vs new vectorized approach.
"""
import os
import time
import sys
import glob
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.algorithms.stacking_algorithms.gpu as GPU
import src.algorithms.stacking_algorithms.cpu as CPU


def fmt_ms(seconds):
    return f"{seconds * 1000:.1f} ms"


def fmt_s(seconds):
    return f"{seconds:.3f} s"


def benchmark():
    image_dir = r"C:\Users\noahe\Pictures\MierenkoninginHoofd"
    paths = sorted(glob.glob(os.path.join(image_dir, "*.JPG")))[:10]
    print(f"=== GPU Pipeline Benchmark ===")
    print(f"Images: {len(paths)} files from {os.path.basename(image_dir)}")

    img0 = cv2.imread(paths[0]).astype(np.float32)
    h, w = img0.shape[:2]
    print(f"Resolution: {w}x{h} ({h * w / 1e6:.1f} MP)")
    print(f"cv2.cuda available: {GPU.HAS_CV_CUDA}")
    print()

    from src.config import auto_detect_params
    params = auto_detect_params(img0.shape, len(paths))
    num_levels = params['pyramid_num_levels']
    kernel_size = params['fusion_kernel_size']
    print(f"Config: pyramid_levels={num_levels}, kernel_size={kernel_size}")
    print()

    # -- Load images --
    print("--- Image Loading ---")
    t0 = time.perf_counter()
    images = [cv2.imread(p).astype(np.float32) for p in paths]
    load_time = time.perf_counter() - t0
    print(f"  {len(images)} images: {fmt_s(load_time)}  ({fmt_ms(load_time/len(images))} each)")
    print()

    # -- Build pyramids --
    print("--- Laplacian Pyramid Generation ---")
    t0 = time.perf_counter()
    pyramids = [GPU.generate_laplacian_pyramid(img, num_levels) for img in images]
    pyr_time = time.perf_counter() - t0
    print(f"  {len(images)} pyramids: {fmt_s(pyr_time)}  ({fmt_ms(pyr_time/len(images))} each)")
    print(f"  Levels: ", end="")
    for i, level in enumerate(pyramids[0]):
        print(f"L{i}={level.shape[1]}x{level.shape[0]}", end="  ")
    print()
    print()

    # -- Focusmap: OLD (Numba prange) vs NEW (cv2.blur) --
    print("--- Focusmap: Old (Numba per-pixel) vs New (cv2.blur vectorized) ---")
    pyr1 = pyramids[0]
    pyr2 = pyramids[1]

    for level_idx in [len(pyr1) - 1, len(pyr1) - 2, len(pyr1) // 2, 0]:
        level = pyr1[level_idx]
        lh, lw = level.shape[:2]
        label = f"L{level_idx} ({lw}x{lh})"

        if level.ndim == 3:
            gray1 = cv2.cvtColor(pyr1[level_idx], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(pyr2[level_idx], cv2.COLOR_BGR2GRAY)
        else:
            gray1 = pyr1[level_idx]
            gray2 = pyr2[level_idx]

        # Old: Numba per-pixel
        t0 = time.perf_counter()
        fm_old = CPU.compute_focusmap(gray1, gray2, kernel_size)
        old_time = time.perf_counter() - t0

        # New: cv2.blur vectorized
        t0 = time.perf_counter()
        fm_new = GPU.compute_focusmap_fast(pyr1[level_idx], pyr2[level_idx], kernel_size)
        new_time = time.perf_counter() - t0

        speedup = old_time / new_time if new_time > 0 else float('inf')
        # Check agreement
        agreement = np.mean(fm_old == fm_new) * 100
        print(f"  {label}:  old={fmt_ms(old_time)}  new={fmt_ms(new_time)}  speedup={speedup:.1f}x  agreement={agreement:.1f}%")
    print()

    # -- Fuse: OLD (Numba prange) vs NEW (np.where) --
    print("--- Fuse: Old (Numba parallel) vs New (np.where vectorized) ---")
    for level_idx in [len(pyr1) - 1, len(pyr1) - 2, 0]:
        level = pyr1[level_idx]
        lh, lw = level.shape[:2]
        label = f"L{level_idx} ({lw}x{lh})"

        fm = GPU.compute_focusmap_fast(pyr1[level_idx], pyr2[level_idx], kernel_size)

        # Old: Numba
        t0 = time.perf_counter()
        fused_old = CPU.fuse_pyramid_levels_using_focusmap(
            pyr1[level_idx], pyr2[level_idx], fm
        )
        old_time = time.perf_counter() - t0

        # New: np.where
        t0 = time.perf_counter()
        fused_new = GPU.fuse_levels_fast(pyr1[level_idx], pyr2[level_idx], fm)
        new_time = time.perf_counter() - t0

        speedup = old_time / new_time if new_time > 0 else float('inf')
        print(f"  {label}:  old={fmt_ms(old_time)}  new={fmt_ms(new_time)}  speedup={speedup:.1f}x")
    print()

    # -- Full fusion pipeline: OLD vs NEW --
    print("--- Full Pyramid Pair Fusion ---")

    # Old: CPU Numba per-pixel
    t0 = time.perf_counter()
    fm = None
    threshold = len(pyr1) - 1
    fused_old = []
    for lvl in range(len(pyr1)):
        if lvl < threshold:
            g1 = cv2.cvtColor(pyr1[lvl], cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(pyr2[lvl], cv2.COLOR_BGR2GRAY)
            fm = CPU.compute_focusmap(g1, g2, kernel_size)
        else:
            s = pyr2[lvl].shape
            fm = cv2.resize(fm, (s[1], s[0]), interpolation=cv2.INTER_AREA)
        fused_old.append(CPU.fuse_pyramid_levels_using_focusmap(pyr1[lvl], pyr2[lvl], fm))
    old_fuse_time = time.perf_counter() - t0
    print(f"  Old (Numba per-pixel):    {fmt_s(old_fuse_time)}")

    # New: vectorized
    t0 = time.perf_counter()
    fused_new = GPU.fuse_pyramid_pair_gpu(pyr1, pyr2, kernel_size)
    new_fuse_time = time.perf_counter() - t0
    print(f"  New (cv2.blur+np.where):  {fmt_s(new_fuse_time)}")
    print(f"  Speedup:                  {old_fuse_time / new_fuse_time:.1f}x")
    print()

    # -- End-to-End: 5 images --
    print("--- End-to-End Stack (first 5 images) ---")
    n_e2e = min(5, len(images))

    # Old CPU
    t0 = time.perf_counter()
    fused = CPU.generate_laplacian_pyramid(images[0], num_levels)
    for i in range(1, n_e2e):
        new_pyr = CPU.generate_laplacian_pyramid(images[i], num_levels)
        fm = None
        threshold = len(fused) - 1
        result = []
        for lvl in range(len(fused)):
            if lvl < threshold:
                g1 = cv2.cvtColor(fused[lvl], cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(new_pyr[lvl], cv2.COLOR_BGR2GRAY)
                fm = CPU.compute_focusmap(g1, g2, kernel_size)
            else:
                s = new_pyr[lvl].shape
                fm = cv2.resize(fm, (s[1], s[0]), interpolation=cv2.INTER_AREA)
            result.append(CPU.fuse_pyramid_levels_using_focusmap(fused[lvl], new_pyr[lvl], fm))
        fused = result
    CPU.reconstruct_pyramid(fused)
    old_e2e = time.perf_counter() - t0
    print(f"  Old (Numba per-pixel):    {fmt_s(old_e2e)}  ({fmt_s(old_e2e / (n_e2e - 1))} per pair)")

    # New vectorized
    t0 = time.perf_counter()
    fused = GPU.generate_laplacian_pyramid(images[0], num_levels)
    for i in range(1, n_e2e):
        new_pyr = GPU.generate_laplacian_pyramid(images[i], num_levels)
        fused = GPU.fuse_pyramid_pair_gpu(fused, new_pyr, kernel_size)
    GPU.reconstruct_pyramid(fused)
    new_e2e = time.perf_counter() - t0
    print(f"  New (cv2.blur+np.where):  {fmt_s(new_e2e)}  ({fmt_s(new_e2e / (n_e2e - 1))} per pair)")
    print(f"  Speedup:                  {old_e2e / new_e2e:.1f}x")
    print()

    # -- Breakdown: new pipeline internals --
    print("--- New Pipeline Internals (1 pair, per-stage) ---")
    pyr1 = pyramids[0]
    pyr2 = pyramids[1]
    threshold_index = len(pyr1) - 1

    total_focusmap = 0
    total_fuse = 0

    for level in range(len(pyr1)):
        lw, lh = pyr1[level].shape[1], pyr1[level].shape[0]

        if level < threshold_index:
            t0 = time.perf_counter()
            fm = GPU.compute_focusmap_fast(pyr1[level], pyr2[level], kernel_size)
            fm_time = time.perf_counter() - t0
            total_focusmap += fm_time
        else:
            s = pyr2[level].shape
            fm = cv2.resize(fm, (s[1], s[0]), interpolation=cv2.INTER_AREA)
            fm_time = 0

        t0 = time.perf_counter()
        fused = GPU.fuse_levels_fast(pyr1[level], pyr2[level], fm)
        fuse_time = time.perf_counter() - t0
        total_fuse += fuse_time

        if level >= len(pyr1) - 3 or level == 0:  # only print largest + smallest
            print(f"    L{level} ({lw}x{lh}): focusmap={fmt_ms(fm_time)}  fuse={fmt_ms(fuse_time)}")

    print(f"  Total focusmap: {fmt_ms(total_focusmap)}")
    print(f"  Total fuse:     {fmt_ms(total_fuse)}")
    print(f"  Total compute:  {fmt_ms(total_focusmap + total_fuse)}")
    print()

    # -- Full 10-image stack --
    print(f"--- Full Stack ({len(images)} images) ---")
    t0 = time.perf_counter()
    fused = GPU.generate_laplacian_pyramid(images[0], num_levels)
    for i in range(1, len(images)):
        new_pyr = GPU.generate_laplacian_pyramid(images[i], num_levels)
        fused = GPU.fuse_pyramid_pair_gpu(fused, new_pyr, kernel_size)
    output = GPU.reconstruct_pyramid(fused)
    full_time = time.perf_counter() - t0
    print(f"  Total:          {fmt_s(full_time)}")
    print(f"  Per pair:       {fmt_s(full_time / (len(images) - 1))}")
    print(f"  Output shape:   {output.shape}")


if __name__ == "__main__":
    benchmark()
