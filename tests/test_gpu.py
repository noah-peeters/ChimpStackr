"""
GPU (OpenCL) stacking: kernels match OpenCV, and the GPU pipeline gives the
same result as the CPU pipeline. Skipped when no OpenCL device is available;
CI runs them on a CPU OpenCL implementation (pocl) via
CHIMPSTACKR_OPENCL_ANY_DEVICE=1.
"""
import glob
import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import cv2
import numpy as np
import pytest

import src.settings as settings
settings.init()

import src.algorithms.stacking_algorithms.cpu as CPU
import src.algorithms.stacking_algorithms.ocl as OCL
from src.algorithms.API import LaplacianPyramid
from src.config import AlgorithmConfig

LOW_RES = sorted(glob.glob(os.path.join(parentdir, "tests", "low_res_images", "*.jpg")))
needs_gpu = pytest.mark.skipif(not OCL.available(), reason="no OpenCL device")


def stack(paths, method, gpu, align, mode="similarity", **kw):
    algo = LaplacianPyramid(config=AlgorithmConfig(
        stacking_method=method, use_gpu=gpu, alignment_mode=mode, pyramid_num_levels=5, **kw))
    algo.update_image_paths(paths)
    progress = []
    run = algo.align_and_stack_images if align else algo.stack_images
    run(progress_callback=lambda c, t, d: progress.append((c, t)))
    return algo, progress


@needs_gpu
class TestKernels:
    @pytest.fixture(scope="class")
    def gpu(self):
        return OCL.get()

    @pytest.mark.parametrize("shape", [(37, 53, 3), (120, 161, 3), (64, 80)])
    def test_pyramid_ops_match_opencv(self, gpu, shape):
        img = (np.random.RandomState(0).rand(*shape) * 255).astype(np.float32)
        h, w = shape[:2]
        assert np.abs(gpu.pyr_down(gpu.upload(img)).get() - cv2.pyrDown(img)).max() < 1e-3
        small = cv2.pyrDown(img)
        up = gpu.pyr_up_combine(gpu.upload(small), gpu.upload(np.zeros_like(img)), 1.0).get()
        assert np.abs(up - cv2.pyrUp(small, dstsize=(w, h))).max() < 1e-3
        lap = gpu.laplacian_pyramid(gpu.upload(img), 4)
        assert np.abs(gpu.reconstruct(lap).get() - img).max() < 1e-3

    def test_filters_match_opencv(self, gpu):
        g = (np.random.RandomState(1).rand(90, 130) * 255).astype(np.float32)
        d = gpu.upload(g)
        assert np.abs(gpu.box(d, 6).get() - cv2.blur(g, (6, 6))).max() < 1e-3
        assert np.abs(gpu.box(d, 6, square=True).get()
                      - cv2.sqrBoxFilter(g, cv2.CV_32F, (6, 6))).max() < 1e-1
        assert np.abs(gpu.gauss(d, 2.0).get() - cv2.GaussianBlur(g, (0, 0), 2.0)).max() < 1e-3
        color = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        assert np.abs(gpu.fine_sharpness(gpu.upload(color)).get()
                      - CPU.fine_sharpness(color)).max() < 1e-2

    @pytest.mark.parametrize("lanczos,flags,border", [
        (False, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT),
        (True, cv2.INTER_LANCZOS4, cv2.BORDER_REPLICATE)])
    def test_warp_matches_opencv(self, gpu, lanczos, flags, border):
        img = cv2.imread(LOW_RES[0])
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.4, 1.01)
        M[:, 2] += (2.3, -1.7)
        out = gpu.warp(img, M, lanczos=lanczos, replicate=lanczos).get()
        ref = cv2.warpAffine(img.astype(np.float32), M, (w, h), flags=flags | cv2.WARP_INVERSE_MAP,
                             borderMode=border)
        # OpenCV interpolates with 1/32-pixel fixed-point positions
        assert np.abs(out - ref)[4:-4, 4:-4].mean() < 0.1


@needs_gpu
class TestPipelineMatchesCPU:
    @pytest.mark.parametrize("method", ["hybrid", "laplacian"])
    @pytest.mark.parametrize("align,mode", [(False, "similarity"), (True, "similarity"),
                                            (True, "translation"), (True, "breathing")])
    def test_same_result_as_cpu(self, method, align, mode):
        cpu, _ = stack(LOW_RES, method, False, align, mode)
        gpu, progress = stack(LOW_RES, method, True, align, mode)
        diff = np.abs(cpu.output_image - gpu.output_image)
        assert diff.mean() < 0.3, f"mean difference {diff.mean():.3f}"
        assert (diff.max(axis=2) > 2).mean() < 0.01
        assert gpu.get_crop_bounds() == cpu.get_crop_bounds()
        assert progress[-1][0] == progress[-1][1]

    def test_16bit_frames(self, tmp_path):
        paths = []
        for i, p in enumerate(LOW_RES[:4]):
            img16 = cv2.imread(p).astype(np.uint16) * 257
            out = str(tmp_path / f"f{i}.png")
            cv2.imwrite(out, img16)
            paths.append(out)
        cpu, _ = stack(paths, "hybrid", False, False)
        gpu, _ = stack(paths, "hybrid", True, False)
        assert np.abs(cpu.output_image - gpu.output_image).mean() < 0.05

    def test_contrast_threshold_uses_cpu(self, monkeypatch):
        monkeypatch.setattr(OCL, "get", lambda *a, **k: pytest.fail("GPU used"))
        algo, _ = stack(LOW_RES[:3], "laplacian", True, False, contrast_threshold=1.0)
        assert algo.output_image is not None


class TestFallback:
    def test_no_gpu_runs_on_cpu(self, monkeypatch):
        monkeypatch.setattr(OCL, "available", lambda: False)
        cpu, _ = stack(LOW_RES[:3], "hybrid", False, False)
        gpu_requested, _ = stack(LOW_RES[:3], "hybrid", True, False)
        assert np.array_equal(cpu.output_image, gpu_requested.output_image)

    def test_gpu_error_falls_back_to_cpu(self, monkeypatch):
        monkeypatch.setattr(OCL, "available", lambda: True)

        def broken(*a, **k):
            raise RuntimeError("simulated OpenCL failure")

        monkeypatch.setattr(OCL, "get", broken)
        algo, _ = stack(LOW_RES[:3], "laplacian", True, True)
        assert algo.output_image is not None
