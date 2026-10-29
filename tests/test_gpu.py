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

METHODS = ["laplacian", "hybrid", "weighted_average", "depth_map", "exposure_fusion"]
LOW_RES = sorted(glob.glob(os.path.join(parentdir, "tests", "low_res_images", "*.jpg")))


def _gpu_works():
    """An OpenCL device that can build the kernels (a CPU implementation's
    compiler may not support the machine it runs on)."""
    if not OCL.available():
        return False
    try:
        OCL.get()
        return True
    except Exception:
        return False


needs_gpu = pytest.mark.skipif(not _gpu_works(), reason="no working OpenCL device")


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

    @pytest.mark.parametrize("shape", [(64, 80), (65, 80), (64, 81), (121, 160)])
    def test_area_upscale_matches_opencv(self, gpu, shape):
        h, w = shape
        small = (np.random.RandomState(2).rand((h + 1) // 2, (w + 1) // 2) > 0.5).astype(np.uint8)
        ref = cv2.resize(small, (w, h), interpolation=cv2.INTER_AREA)
        assert np.array_equal(gpu.area_up(gpu.upload(small.astype(np.float32)), shape).get(), ref)

    @pytest.mark.parametrize("s", [512 / 750, 0.3413, 0.17])
    def test_estimation_levels_match_opencv(self, gpu, s):
        img = cv2.imread(LOW_RES[1], cv2.IMREAD_GRAYSCALE)
        est = gpu.estimation()
        with est.session():
            (ref, same), = est.ecc_levels(img, img, None, [s])
            ref, same = ref.get(), same.get()
        expected = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        assert np.array_equal(ref, expected)
        assert np.array_equal(same, expected)

    @pytest.mark.parametrize("motion", [cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE])
    def test_ecc_matches_opencv(self, gpu, motion):
        ref = cv2.imread(LOW_RES[2], cv2.IMREAD_GRAYSCALE)
        h, w = ref.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.3, 1.0)
        M[:, 2] += (1.7, -0.9)
        img = cv2.warpAffine(ref, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
        rho, expected = cv2.findTransformECC(ref, img, np.eye(2, 3, dtype=np.float32), motion,
                                             criteria, None, 5)
        est = gpu.estimation()
        with est.session():
            rho_gpu, warp = est.find_transform_ecc(
                est._upload(ref.astype(np.float32)), est._upload(img.astype(np.float32)),
                np.eye(2, 3, dtype=np.float32), motion, criteria, 5)
        corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64).T
        assert np.abs((warp - expected) @ corners).max() < 0.01
        assert abs(rho_gpu - rho) < 1e-5

    def test_thresholded_focusmap_matches_cpu(self, gpu):
        rs = np.random.RandomState(3)
        a, b = (rs.rand(2, 41, 57) * 4).astype(np.float32)
        fuser = OCL.PyramidGPU(gpu, 1, 6, 0, contrast_threshold=1.0)
        fm = fuser._focusmap(gpu.upload(a), gpu.upload(b)).get()
        ref = CPU.compute_focusmap_thresholded(a, b, 6, np.float32(1.0))
        assert (fm != ref).mean() < 0.002

    def test_depth_sharpness_matches_cpu(self, gpu):
        img = cv2.imread(LOW_RES[0]).astype(np.float32)
        out = OCL.DepthMapGPU(gpu, 6, 5).sharpness(gpu.upload(img)).get()
        ref = CPU.compute_multires_sharpness(img, scales=(5, 7, 19), smoothing=5)
        assert np.abs(out - ref).max() < 1e-2 * ref.max()

    def test_exposure_merge_matches_opencv(self, gpu):
        imgs = [cv2.imread(p) for p in LOW_RES[:3]]
        out = OCL.ExposureGPU(gpu).merge([lambda im=im: gpu.upload(im) for im in imgs]).get()
        ref = np.clip(cv2.createMergeMertens(1.0, 1.0, 0.0).process(imgs) * 255, 0, 255)
        assert np.abs(out - ref).mean() < 0.01


@needs_gpu
class TestPipelineMatchesCPU:
    @pytest.mark.parametrize("method", METHODS)
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
        assert all(p[0] <= q[0] for p, q in zip(progress, progress[1:]))

    def test_breathing_middle_reference(self):
        # stacking starts once the chain below the reference frame is known
        cpu, _ = stack(LOW_RES, "hybrid", False, True, "breathing", alignment_reference="middle")
        gpu, progress = stack(LOW_RES, "hybrid", True, True, "breathing",
                              alignment_reference="middle")
        assert np.abs(cpu.output_image - gpu.output_image).mean() < 0.3
        assert gpu.sequential_log is not None and len(gpu.sequential_log) == len(LOW_RES)
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

    @pytest.mark.parametrize("method", ["laplacian", "hybrid", "weighted_average", "depth_map"])
    def test_slabbing(self, method):
        kw = dict(slab_size=4, slab_overlap=1)
        cpu, cpu_progress = stack(LOW_RES, method, False, True, "breathing", **kw)
        gpu, progress = stack(LOW_RES, method, True, True, "breathing", **kw)
        diff = np.abs(cpu.output_image - gpu.output_image)
        assert diff.mean() < 0.3
        assert progress[-1] == cpu_progress[-1]
        assert all(a[0] <= b[0] for a, b in zip(progress, progress[1:]))

    def test_contrast_threshold(self):
        cpu, _ = stack(LOW_RES[:4], "laplacian", False, False, contrast_threshold=1.0)
        gpu, _ = stack(LOW_RES[:4], "laplacian", True, False, contrast_threshold=1.0)
        assert np.abs(cpu.output_image - gpu.output_image).mean() < 0.3

    def test_tone_map(self):
        img = cv2.imread(LOW_RES[0]).astype(np.float32)
        gpu = OCL.get()
        out = gpu.tone_map(gpu.upload(img), strength=0.3).get()
        # cv2's float LAB conversion goes through a fixed-point table (its own
        # round trip is off by up to ~1 level); the GPU uses the exact curves
        assert np.abs(out - CPU.local_tone_map(img, strength=0.3)).mean() < 0.3
        assert np.abs(gpu.tone_map(gpu.upload(img), strength=0.0).get() - img).max() < 0.02


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
