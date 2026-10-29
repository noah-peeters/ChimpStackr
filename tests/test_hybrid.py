"""
Tests for hybrid stacking and slabbing.

No PySide6/Qt dependency required.
"""
import os
import subprocess
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import glob

import cv2
import numpy as np
import pytest

import src.settings as settings
settings.init()

import src.algorithms.stacking_algorithms.cpu as CPU
from src.algorithms.API import LaplacianPyramid
from src.config import AlgorithmConfig


LOW_RES = sorted(glob.glob(os.path.join(parentdir, "tests", "low_res_images", "*.jpg")))


def two_object_scene(noise=0.0, seed=0, h=240, w=320):
    """Two textured discs at different depths: each frame has one in focus and
    the other blurred (spilling into the background). Returns frames, the
    all-in-focus truth, and a mask of the background ring around the discs."""
    rs = np.random.RandomState(seed)
    tex = cv2.GaussianBlur(rs.rand(h, w).astype(np.float32), (0, 0), 1.0) * 120
    yy, xx = np.mgrid[0:h, 0:w]
    a = ((yy - 120) ** 2 + (xx - 90) ** 2 < 45 ** 2).astype(np.float32)
    b = ((yy - 120) ** 2 + (xx - 230) ** 2 < 45 ** 2).astype(np.float32)
    bg = np.full((h, w), 30, np.float32)
    sharp_a = bg * (1 - a) + (110 + tex) * a
    sharp_b = bg * (1 - b) + (110 + tex) * b

    def blur(x):
        return cv2.GaussianBlur(x, (0, 0), 7.0)

    frames = [sharp_a + blur(sharp_b) - bg, blur(sharp_a) + sharp_b - bg]
    frames = [f + rs.normal(0, noise, f.shape).astype(np.float32) for f in frames]
    truth = sharp_a + sharp_b - bg
    ring = (cv2.dilate((a + b > 0).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0) & (a + b == 0)

    def to_bgr(x):
        return cv2.cvtColor(np.clip(x, 0, 255).astype(np.float32), cv2.COLOR_GRAY2BGR)

    return [to_bgr(f) for f in frames], to_bgr(truth), ring


def write_npy(tmp_path, frames):
    paths = []
    for i, f in enumerate(frames):
        p = str(tmp_path / f"frame_{i:02d}.npy")
        np.save(p, f)
        paths.append(p)
    return paths


def stack(paths, method="hybrid", align=False, levels=5, **kw):
    cfg = AlgorithmConfig(stacking_method=method, fusion_kernel_size=6,
                          pyramid_num_levels=levels, **kw)
    algo = LaplacianPyramid(config=cfg)
    algo.update_image_paths(paths)
    progress = []
    run = algo.align_and_stack_images if align else algo.stack_images
    run(progress_callback=lambda c, t, d: progress.append((c, t)))
    return algo, progress


class TestHybridFuser:
    def test_fine_level_count_scales_with_resolution(self):
        assert CPU.hybrid_fine_level_count((2000, 3000), 8) == 3
        assert CPU.hybrid_fine_level_count((4000, 6000), 8) == 4
        assert CPU.hybrid_fine_level_count((500, 750), 8) == 2   # lower bound
        assert CPU.hybrid_fine_level_count((8000, 12000), 3) == 3  # capped by levels

    def test_single_frame_roundtrip(self):
        frames, _, _ = two_object_scene()
        fuser = CPU.HybridFuser(5)
        fuser.add(*fuser.prepare(frames[0]))
        assert np.abs(fuser.result() - frames[0]).max() < 1e-2

    def test_identical_frames_unchanged(self):
        frames, _, _ = two_object_scene(noise=2.0)
        fuser = CPU.HybridFuser(5)
        for _ in range(4):
            fuser.add(*fuser.prepare(frames[0]))
        assert np.abs(fuser.result() - frames[0]).max() < 1e-2

    def test_empty_result_is_none(self):
        assert CPU.HybridFuser(5).result() is None

    def test_grayscale_frames(self):
        frames, _, _ = two_object_scene()
        fuser = CPU.HybridFuser(5)
        for f in frames:
            fuser.add(*fuser.prepare(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)))
        out = fuser.result()
        assert out.ndim == 2 and out.shape == frames[0].shape[:2]


class TestHybridQuality:
    def test_recovers_both_focus_planes_without_halos(self, tmp_path):
        frames, truth, ring = two_object_scene()
        algo, _ = stack(write_npy(tmp_path, frames))
        err = np.abs(algo.output_image - truth).mean(axis=2)
        assert err.mean() < 0.5, f"overall error {err.mean():.2f}"
        assert err[ring].mean() < 1.0, f"halo (background ring) error {err[ring].mean():.2f}"

    def test_halos_not_worse_than_laplacian_with_noise(self, tmp_path):
        frames, truth, ring = two_object_scene(noise=3.0)
        paths = write_npy(tmp_path, frames)
        hyb, _ = stack(paths, "hybrid")
        lap, _ = stack(paths, "laplacian")
        ring_err = {
            name: np.abs(a.output_image - truth).mean(axis=2)[ring].mean()
            for name, a in (("hybrid", hyb), ("laplacian", lap))
        }
        assert ring_err["hybrid"] <= ring_err["laplacian"], ring_err

    def test_flat_areas_are_blended_not_noise_picked(self, tmp_path):
        """Max-contrast selection picks the noisiest frame in featureless
        areas; hybrid blends there, so noise drops below a single frame's."""
        rs = np.random.RandomState(4)
        sigma = 5.0
        frames = [np.clip(100 + rs.normal(0, sigma, (160, 200, 3)), 0, 255).astype(np.float32)
                  for _ in range(8)]
        paths = write_npy(tmp_path, frames)
        hyb, _ = stack(paths, "hybrid", levels=4)
        lap, _ = stack(paths, "laplacian", levels=4)
        inner = (slice(20, -20), slice(20, -20))
        assert hyb.output_image[inner].std() < 0.6 * sigma
        assert hyb.output_image[inner].std() < lap.output_image[inner].std()


class TestHybridPipeline:
    def test_align_and_stack(self):
        algo, progress = stack(LOW_RES, "hybrid", align=True, levels=4)
        out = algo.output_image
        assert out is not None and out.shape == (500, 750, 3)
        assert np.isfinite(out).all() and out.mean() > 5
        assert progress[-1][0] == progress[-1][1]

    def test_stack_only(self):
        algo, _ = stack(LOW_RES, "hybrid", levels=4)
        assert algo.output_image is not None and algo.output_image.shape == (500, 750, 3)

    def test_gpu_flag_falls_back_to_cpu(self):
        algo, _ = stack(LOW_RES[:3], "hybrid", levels=4, use_gpu=True)
        assert algo.output_image is not None

    def test_stack_only_progress_ignores_previous_alignment_offset(self):
        algo = LaplacianPyramid(config=AlgorithmConfig(stacking_method="hybrid",
                                                       pyramid_num_levels=4))
        algo.update_image_paths(LOW_RES[:3])
        algo._progress_offset, algo._progress_total = 99, 200   # left by an earlier run
        progress = []
        algo.stack_images(progress_callback=lambda c, t, d: progress.append((c, t)))
        assert progress[-1] == (3, 3)


class TestSlabbing:
    def test_slab_groups(self):
        groups = LaplacianPyramid.slab_groups(50, 10, 2)
        assert [(g[0], g[-1]) for g in groups] == [
            (0, 9), (8, 17), (16, 25), (24, 33), (32, 41), (40, 49)]
        covered = sorted({i for g in groups for i in g})
        assert covered == list(range(50))
        assert LaplacianPyramid.slab_groups(10, 10, 2) == [list(range(10))]
        # overlap is clamped to size - 1, so slabs advance one frame at a time
        assert LaplacianPyramid.slab_groups(5, 3, 5) == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]

    @pytest.mark.parametrize("method", ["hybrid", "laplacian", "weighted_average", "depth_map"])
    def test_slabbed_stack_matches_single_pass(self, method):
        single, _ = stack(LOW_RES, method, align=True, levels=4)
        slabbed, progress = stack(LOW_RES, method, align=True, levels=4,
                                  slab_size=4, slab_overlap=1)
        assert slabbed.output_image.shape == single.output_image.shape
        diff = np.abs(slabbed.output_image - single.output_image).mean()
        assert diff < 3.0, f"{method}: slabbed differs from single pass by {diff:.2f}"
        steps = [c for c, _ in progress]
        assert steps == sorted(steps) and len({t for _, t in progress}) == 1
        assert progress[-1][0] == progress[-1][1]

    def test_progress_monotonic_with_breathing_alignment(self):
        """The breathing estimation phase must already count the slab steps."""
        algo, progress = stack(LOW_RES, "hybrid", align=True, levels=4, slab_size=4,
                               alignment_mode="breathing")
        assert algo.output_image is not None
        steps = [c for c, _ in progress]
        assert steps == sorted(steps), progress
        assert len({t for _, t in progress}) == 1, progress
        assert progress[-1][0] == progress[-1][1]

    def test_slabbing_off_for_short_stacks_and_hdr(self):
        algo = LaplacianPyramid(config=AlgorithmConfig(slab_size=20))
        algo.update_image_paths(LOW_RES)
        assert not algo._slabbing_active()                     # fewer frames than slab
        algo.config = AlgorithmConfig(slab_size=4, stacking_method="exposure_fusion")
        assert not algo._slabbing_active()

    def test_cancel_during_slabs(self):
        algo = LaplacianPyramid(config=AlgorithmConfig(stacking_method="hybrid",
                                                       pyramid_num_levels=4, slab_size=4))
        algo.update_image_paths(LOW_RES)
        algo.stack_images(progress_callback=lambda c, t, d: c >= 3 and algo.cancel())
        assert algo.output_image is None


class TestCLI:
    def _run(self, *extra):
        return subprocess.run(
            [sys.executable, "-m", "src.cli", "--input", "tests/low_res_images/*.jpg",
             "--output", os.devnull, "--pyramid-levels", "4", *extra],
            capture_output=True, text=True, cwd=parentdir, timeout=180,
        )

    def test_hybrid_with_slabs(self, tmp_path):
        out = str(tmp_path / "out.png")
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "--input", "tests/low_res_images/*.jpg",
             "--output", out, "--pyramid-levels", "4", "--align",
             "--method", "hybrid", "--slab-size", "4"],
            capture_output=True, text=True, cwd=parentdir, timeout=180,
        )
        assert result.returncode == 0, result.stderr
        assert cv2.imread(out) is not None

    @pytest.mark.parametrize("args", [("--slab-size", "1"), ("--slab-size", "-3"),
                                      ("--slab-size", "4", "--slab-overlap", "4")])
    def test_rejects_invalid_slab_settings(self, args):
        result = self._run(*args)
        assert result.returncode != 0
        assert "--slab" in result.stderr
