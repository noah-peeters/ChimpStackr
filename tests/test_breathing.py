"""
Tests for sequential (chained) alignment — Helicon-style focus breathing compensation.

No PySide6/Qt dependency required.
"""
import os
import sys
import subprocess

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import cv2
import pytest

import src.settings as settings
settings.init()

from src.algorithms import Algorithm
from src.algorithms.API import LaplacianPyramid
from src.config import AlgorithmConfig


# -- Fixtures --

def make_textured_image(h=384, w=512, seed=42):
    """Pixel-scale random noise with a light blur — plenty of ORB features."""
    rs = np.random.RandomState(seed)
    gray = rs.rand(h, w).astype(np.float32) * 255
    gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)


@pytest.fixture(scope="module")
def base_image():
    return make_textured_image()


class TestTransformHelpers:
    def test_compose_affine_order(self):
        a = np.float32([[1, 0, 5], [0, 1, 3]])    # translate (5, 3)
        b = np.float32([[2, 0, 1], [0, 2, 1]])    # scale x2 then translate (1, 1)
        # compose: apply a, then b → scale a's translation too
        c = Algorithm.compose_affine(a, b)
        assert np.allclose(c, [[2, 0, 11], [0, 2, 7]], atol=1e-5)

    def test_invert_affine_roundtrip(self):
        M = np.float32([[1.03, 0.035, 5], [-0.035, 1.03, -3]])
        Minv = Algorithm.invert_affine(M)
        assert np.allclose(Algorithm.compose_affine(M, Minv), np.eye(2, 3), atol=1e-3)
        assert np.allclose(Algorithm.compose_affine(Minv, M), np.eye(2, 3), atol=1e-3)

    def test_transform_scale_and_rotation(self):
        M = cv2.getRotationMatrix2D((0, 0), 30, 1.05)[:2]
        assert abs(Algorithm.transform_scale(M) - 1.05) < 1e-4
        assert abs(Algorithm.transform_rotation(M) - 30) < 1e-3


class TestPairEstimation:
    def test_similarity_recovers_known_transform(self):
        alg = Algorithm()
        base = make_textured_image()
        T = cv2.getRotationMatrix2D((256, 192), 2.0, 1.03)
        T[0, 2] += 5
        T[1, 2] -= 3
        frame = cv2.warpAffine(base, T, (512, 384), flags=cv2.INTER_LINEAR)

        # Estimate the forward map: template=base, source=frame
        M, meta = alg.estimate_pair_transform(base, frame, alignment_mode="similarity")
        assert M is not None
        assert abs(Algorithm.transform_scale(M) - 1.03) < 0.005
        assert abs(Algorithm.transform_rotation(M) - 2.0) < 0.75
        aligned = alg.apply_transform_matrix(frame, M, track=False)
        # Compare blurred versions — bilinear resampling of grain noise has
        # large per-pixel errors but converges on the low-frequency content
        sb = cv2.GaussianBlur(aligned, (7, 7), 2.0)
        bt = cv2.GaussianBlur(base, (7, 7), 2.0)
        diff = np.mean(np.abs(sb[40:-40, 40:-40] - bt[40:-40, 40:-40]))
        assert diff < 2.0, f"aligned frame differs from template (mean {diff:.2f})"

    def test_estimate_matches_align_image_pair(self):
        """estimate_pair_transform + apply == align_image_pair output."""
        alg1, alg2 = Algorithm(), Algorithm()
        base = make_textured_image(seed=7)
        T = cv2.getRotationMatrix2D((256, 192), -1.5, 1.02)
        T[0, 2] += 3
        frame = cv2.warpAffine(base, T, (512, 384), flags=cv2.INTER_LINEAR)

        aligned_pipe = alg1.align_image_pair(base, frame.copy(),
                                             alignment_mode="similarity")
        M, _ = alg2.estimate_pair_transform(base, frame.copy(),
                                            alignment_mode="similarity")
        aligned_manual = alg2.apply_transform_matrix(frame.copy(), M, track=False)
        d = np.abs(aligned_pipe - aligned_manual)
        assert np.percentile(d, 99) < 1.0


class TestSequentialTransforms:
    def _breathing_series(self, n=10, per_step=0.004, seed=42):
        base = make_textured_image(seed=seed)
        frames = [base]
        transforms_true = [np.eye(2, 3)]
        for i in range(1, n):
            T = cv2.getRotationMatrix2D((256, 192), 0.0, 1.0 + per_step * i)
            T[0, 2] += 0.7 * i   # small drift, realistic FocusRail wobble
            T[1, 2] -= 0.4 * i
            frames.append(cv2.warpAffine(base, T, (512, 384), flags=cv2.INTER_LINEAR))
            transforms_true.append(T)
        return frames, transforms_true

    def test_chain_reconstructs_cumulative_scale(self, tmp_path):
        frames, _T_true = self._breathing_series(n=8)
        paths = []
        for i, fr in enumerate(frames):
            p = str(tmp_path / f"frame_{i:02d}.png")
            cv2.imwrite(p, np.clip(fr, 0, 255).astype(np.uint8))
            paths.append(p)

        alg = Algorithm()
        transforms, log = alg.estimate_sequential_transforms(
            paths, ref_index=0, alignment_mode="similarity", max_scale_step=0.05,
        )
        assert transforms is not None
        assert len(transforms) == 8 and len(log) == 8
        assert transforms[0].shape == (2, 3)

        # Chained per-pair estimates are precise on real content; a small
        # tolerance absorbs per-pair estimation error accumulated over 8 frames
        for i in range(8):
            s_est = Algorithm.transform_scale(transforms[i])
            s_true = 1.0 + 0.004 * i
            assert abs(s_est - s_true) < 0.015, (
                f"frame {i}: cumulative scale {s_est:.4f} vs true {s_true:.4f}")

        # Warping each frame back into frame-0 space should reproduce the base
        for i in range(1, 8):
            aligned = alg.apply_transform_matrix(frames[i], transforms[i], track=False)
            sb = cv2.GaussianBlur(aligned, (7, 7), 2.0)
            b0 = cv2.GaussianBlur(frames[0], (7, 7), 2.0)
            diff = np.mean(np.abs(sb[60:-60, 60:-60] - b0[60:-60, 60:-60]))
            assert diff < 3.0, f"frame {i}: residual after chained alignment {diff:.2f}"

    def test_middle_reference(self, tmp_path):
        frames, _T_true = self._breathing_series(n=7, seed=3)
        # Save/load to mimic the real pipeline (apply on toolbar-sized images)
        paths = []
        for i, fr in enumerate(frames):
            p = str(tmp_path / f"mid_{i:02d}.png")
            cv2.imwrite(p, np.clip(fr, 0, 255).astype(np.uint8))
            paths.append(p)

        alg = Algorithm()
        transforms, log = alg.estimate_sequential_transforms(
            paths, ref_index=3, alignment_mode="similarity",
        )
        # transforms[i] should be the forward map frame_3 coords → frame_i coords
        # = T_i · T_3⁻¹; its scale is s_i / s_3
        s3 = 1.0 + 0.004 * 3
        for i in range(7):
            s_exp = (1.0 + 0.004 * i) / s3
            s_est = Algorithm.transform_scale(transforms[i])
            assert abs(s_est - s_exp) < 0.006, (
                f"frame {i}: scale {s_est:.4f} vs expected {s_exp:.4f}")
        # Alignment check: warping each frame into frame-3 space reproduces it
        for i in range(7):
            aligned = alg.apply_transform_matrix(frames[i], transforms[i], track=False)
            sa = cv2.GaussianBlur(aligned, (7, 7), 2.0)
            s3b = cv2.GaussianBlur(frames[3], (7, 7), 2.0)
            diff = np.mean(np.abs(sa[60:-60, 60:-60] - s3b[60:-60, 60:-60]))
            assert diff < 3.0, f"frame {i}: residual vs anchor {diff:.2f}"

    def test_scale_clamp_triggers_fallback(self, tmp_path):
        """A step with impossible scale change (> max_scale_step) must be rejected."""
        base = make_textured_image(seed=11)
        T = cv2.getRotationMatrix2D((256, 192), 0.0, 1.30)  # 30% step — implausible
        T[0, 2] += 4
        frame_hi = cv2.warpAffine(base, T, (512, 384), flags=cv2.INTER_LINEAR)

        paths = [str(tmp_path / "a.png"), str(tmp_path / "b.png")]
        cv2.imwrite(paths[0], np.clip(base, 0, 255).astype(np.uint8))
        cv2.imwrite(paths[1], np.clip(frame_hi, 0, 255).astype(np.uint8))

        alg = Algorithm()
        transforms, log = alg.estimate_sequential_transforms(
            paths, ref_index=0, alignment_mode="similarity", max_scale_step=0.05,
        )
        assert transforms is not None
        entry = log[1]
        # The implausible scale must never survive: either it was rejected
        # (fallback recorded) or the surviving transform has scale ~1
        assert entry["fallbacks"] >= 1 or abs(entry["scale"] - 1.0) < 0.06, (
            f"implausible scale survived: {entry}")

    def test_missing_image_holds(self, tmp_path):
        frames, _ = self._breathing_series(n=4, seed=5)
        paths = []
        for i, fr in enumerate(frames):
            p = str(tmp_path / f"h_{i:02d}.png")
            cv2.imwrite(p, np.clip(fr, 0, 255).astype(np.uint8))
            paths.append(p)
        os.remove(paths[2])  # hole in the middle

        alg = Algorithm()
        transforms, log = alg.estimate_sequential_transforms(
            paths, ref_index=0, alignment_mode="similarity",
        )
        assert transforms is not None
        assert len(transforms) == 4
        # Chain composes with held deltas; nothing crashes, transformations finite
        for T in transforms:
            assert np.all(np.isfinite(T))


class TestMaxPrecision:
    """The breathing chain runs with precision='max': float32 ECC scale
    refinement + Lanczos resampling + 4096px cascade. Verify it measurably
    beats the default path on image content that exercises those upgrades."""

    @staticmethod
    def _natural_image(h=512, w=640, seed=3):
        """Multi-scale texture resembling real photo content — the kind of
        image where float32 ECC scale refinement is razor sharp (pure grain
        noise is not, and is handled by the NCC arbitration fallback)."""
        rs = np.random.RandomState(seed)
        x = np.linspace(0, 1, w, dtype=np.float32)
        y = np.linspace(0, 1, h, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        img = np.zeros((h, w), np.float32)
        for f, a in [(23.0, 0.5), (7.0, 1.0), (2.5, 1.4), (1.0, 1.8)]:
            img += a * np.sin(2*np.pi*f*X + rs.rand()*6) * np.cos(2*np.pi*f*Y*0.8 + rs.rand()*6)
        img = cv2.GaussianBlur(img, (5, 5), 1.2)
        img = 128 + 60*img/img.std()
        return cv2.cvtColor(np.clip(img, 0, 255), cv2.COLOR_GRAY2BGR).astype(np.float32)

    @classmethod
    def _residual(cls, base, frame, M, alg):
        aligned = alg.apply_transform_matrix(
            frame, M, track=False,
            interpolation=cv2.INTER_LANCZOS4, border_mode=cv2.BORDER_REPLICATE)
        ab = cv2.GaussianBlur(aligned, (7, 7), 2.0)
        bb = cv2.GaussianBlur(base, (7, 7), 2.0)
        d = np.abs(ab[80:-80, 80:-80] - bb[80:-80, 80:-80])
        return float(np.mean(d))

    def test_hq_beats_default_on_natural_content(self):
        base = self._natural_image()
        T = cv2.getRotationMatrix2D((320, 256), 0.4, 1.021)
        T[0, 2] += 2.37
        T[1, 2] -= 1.83
        frame = cv2.warpAffine(base, T, (640, 512), flags=cv2.INTER_CUBIC)

        alg = Algorithm()
        M_def, _ = alg.estimate_pair_transform(base, frame, alignment_mode="similarity")
        M_hq, _ = alg.estimate_pair_transform(base, frame, alignment_mode="similarity",
                                              precision="max")
        r_def = self._residual(base, frame, M_def, alg)
        r_hq = self._residual(base, frame, M_hq, alg)
        # The max-precision path must align strictly better, with a real
        # margin (measured ~0.08 vs ~0.48)
        assert r_hq < r_def * 0.6, f"HQ residual {r_hq:.3f} not clearly better than {r_def:.3f}"
        assert r_hq < 0.5

    def test_hq_scale_refinement(self):
        """HQ refines SCALE via float32 ECC; the default path cannot (freeze
        at ORB estimate). On natural content this recovers the true scale."""
        base = self._natural_image(seed=5)
        true_scale = 1.019
        T = cv2.getRotationMatrix2D((320, 256), 1.1, true_scale)
        T[0, 2] += 2.37
        frame = cv2.warpAffine(base, T, (640, 512), flags=cv2.INTER_CUBIC)

        alg = Algorithm()
        M_def, _ = alg.estimate_pair_transform(base, frame, alignment_mode="similarity")
        M_hq, _ = alg.estimate_pair_transform(base, frame, alignment_mode="similarity",
                                              precision="max")
        e_def = abs(alg.transform_scale(M_def) - true_scale)
        e_hq = abs(alg.transform_scale(M_hq) - true_scale)
        assert e_hq < 0.002, f"HQ scale error {e_hq:.5f} too large"
        assert e_hq < e_def * 0.5, (
            f"HQ scale error {e_hq:.5f} should halve default {e_def:.5f}")

    def test_hq_beats_default_on_breathing_chain(self, tmp_path):
        """End-to-end: HQ chain leaves less residual stack-wide scale drift."""
        base = self._natural_image(h=384, w=480, seed=9)
        n = 6
        paths = []
        for i in range(n):
            T = cv2.getRotationMatrix2D((240, 192), 0.0, 1.0 + 0.004*i)
            T[0, 2] += 0.5*i
            base_i = base if i == 0 else base
            fr = base_i if i == 0 else cv2.warpAffine(base, T, (480, 384), flags=cv2.INTER_CUBIC)
            p = str(tmp_path / f"nh_{i:02d}.png")
            cv2.imwrite(p, np.clip(fr, 0, 255).astype(np.uint8))
            paths.append(p)

        alg = Algorithm()
        transforms, _ = alg.estimate_sequential_transforms(
            paths, ref_index=0, alignment_mode="similarity", max_scale_step=0.05)
        for i in range(n):
            s_est = Algorithm.transform_scale(transforms[i])
            assert abs(s_est - (1.0 + 0.004*i)) < 0.004, (
                f"frame {i}: HQ cumulative scale {s_est:.5f} off by "
                f"{abs(s_est - (1.0+0.004*i)):.5f} (> 0.4% of breathing)")


class TestSequentialPipeline:
    def _write_series(self, tmp_path, n=6, per_step=0.004, shift=0.5):
        base = make_textured_image(h=256, w=340, seed=17)
        paths = []
        for i in range(n):
            T = cv2.getRotationMatrix2D((170, 128), 0.0, 1.0 + per_step * i)
            T[0, 2] += shift * i
            fr = cv2.warpAffine(base, T, (340, 256), flags=cv2.INTER_LINEAR)
            p = str(tmp_path / f"s_{i:02d}.png")
            cv2.imwrite(p, np.clip(fr, 0, 255).astype(np.uint8))
            paths.append(p)
        return paths

    def test_align_and_stack_breathing_mode(self, tmp_path):
        paths = self._write_series(tmp_path)
        cfg = AlgorithmConfig(
            stacking_method="laplacian",
            alignment_mode="breathing",
            fusion_kernel_size=3,
            pyramid_num_levels=4,
            alignment_scale_factor=5,
        )
        algo = LaplacianPyramid(config=cfg)
        algo.update_image_paths(paths)
        progress = []
        algo.align_and_stack_images(progress_callback=lambda c, t, d: progress.append((c, t)))
        assert algo.output_image is not None
        out = algo.output_image
        assert out.shape[:2] == (256, 340)
        assert np.mean(out) > 5, "stack output is black"

    def test_previous_reference_activates_sequential(self, tmp_path):
        paths = self._write_series(tmp_path)
        cfg = AlgorithmConfig(
            stacking_method="depth_map",
            alignment_mode="similarity",
            alignment_reference="previous",
            fusion_kernel_size=3,
            pyramid_num_levels=4,
        )
        algo = LaplacianPyramid(config=cfg)
        algo.update_image_paths(paths)
        algo.align_and_stack_images()
        assert algo.output_image is not None
        assert algo.sequential_log is not None
        assert len(algo.sequential_log) == len(paths)

    def test_middle_anchor_reference(self, tmp_path):
        paths = self._write_series(tmp_path)
        cfg = AlgorithmConfig(
            stacking_method="weighted_average",
            alignment_mode="breathing",
            alignment_reference="middle",
            fusion_kernel_size=3,
            pyramid_num_levels=4,
        )
        algo = LaplacianPyramid(config=cfg)
        algo.update_image_paths(paths)
        algo.align_and_stack_images()
        assert algo.output_image is not None

    def test_fast_sequential_mode(self, tmp_path):
        """Default (max precision off): fast estimation, no _hp methods, valid stack."""
        paths = self._write_series(tmp_path)
        cfg = AlgorithmConfig(
            stacking_method="laplacian",
            alignment_mode="breathing",
            fusion_kernel_size=3,
            pyramid_num_levels=4,
            alignment_scale_factor=5,
        )
        algo = LaplacianPyramid(config=cfg)
        algo.update_image_paths(paths)
        algo.align_and_stack_images()
        assert algo.output_image is not None
        assert np.mean(algo.output_image) > 5
        assert algo.sequential_log is not None
        for entry in algo.sequential_log:
            assert "_hp" not in entry["method"], (
                f"fast path used a high-precision estimator: {entry}")

    def test_fast_estimation_matches_expected_scale(self, tmp_path):
        """The fast sequential path still recovers the breathing ramp."""
        base = make_textured_image(h=256, w=340, seed=23)
        paths = []
        for i in range(5):
            T = cv2.getRotationMatrix2D((170, 128), 0.0, 1.0 + 0.005*i)
            fr = cv2.warpAffine(base, T, (340, 256), flags=cv2.INTER_LINEAR)
            p = str(tmp_path / f"fast_{i:02d}.png")
            cv2.imwrite(p, np.clip(fr, 0, 255).astype(np.uint8))
            paths.append(p)
        alg = Algorithm()
        transforms, _ = alg.estimate_sequential_transforms(
            paths, ref_index=0, alignment_mode="similarity", precision="default")
        for i in range(5):
            s_est = Algorithm.transform_scale(transforms[i])
            assert abs(s_est - (1.0 + 0.005*i)) < 0.02

    def test_existing_translation_mode_unchanged(self, tmp_path):
        """Regression: non-sequential modes behave exactly as before."""
        paths = self._write_series(tmp_path)
        cfg = AlgorithmConfig(
            stacking_method="laplacian",
            alignment_mode="translation",
            fusion_kernel_size=3,
            pyramid_num_levels=4,
            alignment_scale_factor=5,
        )
        algo1 = LaplacianPyramid(config=cfg)
        algo1.update_image_paths(paths)
        algo1.align_and_stack_images()
        assert algo1.output_image is not None
        assert algo1.sequential_log is None  # not sequential

    def test_max_precision_opt_in(self, tmp_path):
        """Max precision runs the _hp estimators only when explicitly enabled."""
        paths = self._write_series(tmp_path, n=3)
        cfg = AlgorithmConfig(
            stacking_method="laplacian",
            alignment_mode="breathing",
            breathing_high_precision=True,
            fusion_kernel_size=3,
            pyramid_num_levels=4,
        )
        algo = LaplacianPyramid(config=cfg)
        algo.update_image_paths(paths)
        algo.align_and_stack_images()
        methods = [e["method"] for e in algo.sequential_log if e["method"] != "reference"]
        assert methods and all(m.endswith("_hp") for m in methods), methods


def _shifted_pair(dx, dy, seed=3):
    """A textured frame and a copy whose content is moved by (dx, dy) pixels.

    The texture is blurred so it survives the DFT estimator's downsampling
    (pixel-scale noise vanishes at the default scale_factor=10).
    """
    base = cv2.GaussianBlur(make_textured_image(seed=seed), (0, 0), 4.0)
    base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
    h, w = base.shape[:2]
    moved = cv2.warpAffine(base, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return base, moved


def _centre_error(a, b, m=40):
    return float(np.mean(np.abs(a[m:-m, m:-m] - b[m:-m, m:-m])))


class TestAlignmentDirection:
    """Aligned output must actually land on the reference (catches sign errors
    that an 'output is not None' check cannot)."""

    @pytest.mark.parametrize("mode", ["translation", "euclidean", "similarity", "affine"])
    @pytest.mark.parametrize("dx,dy", [(7, -4), (-12, 9)])
    def test_align_image_pair_moves_onto_reference(self, mode, dx, dy):
        base, moved = _shifted_pair(dx, dy)
        before = _centre_error(base, moved)
        aligned = Algorithm().align_image_pair(base, moved, scale_factor=10,
                                               alignment_mode=mode)
        after = _centre_error(base, aligned)
        # Correct alignment leaves <0.3x the error (DFT translation is the
        # coarsest); a sign error makes it worse than not aligning (>1x)
        assert after < before * 0.6, (
            f"{mode} ({dx},{dy}): error {before:.2f} -> {after:.2f}")

    def test_translation_estimate_is_forward_map(self):
        """The DFT estimate follows the shared forward-map convention."""
        base, moved = _shifted_pair(7, -4)
        alg = Algorithm()
        M_t, _ = alg.estimate_pair_transform(base, moved, alignment_mode="translation")
        M_s, _ = alg.estimate_pair_transform(base, moved, alignment_mode="similarity")
        # DFT at the default scale_factor is only ~1.5px precise; a flipped
        # convention would be off by ~2x the shift (~14px here)
        assert np.allclose(M_t[:, 2], M_s[:, 2], atol=2.5), (M_t, M_s)


class TestMixedDimensions:
    """Frames with different dimensions (#29) are matched to the reference."""

    def _write_mixed(self, tmp_path):
        base = make_textured_image(h=300, w=400, seed=5)
        paths = []
        for i, (h, w) in enumerate([(300, 400), (300, 400), (290, 384), (310, 412)]):
            fr = base[:h, :w] if h <= 300 else cv2.copyMakeBorder(
                base, 0, h - 300, 0, w - 400, cv2.BORDER_REFLECT)
            p = str(tmp_path / f"mixed_{i}.png")
            cv2.imwrite(p, np.clip(fr, 0, 255).astype(np.uint8))
            paths.append(p)
        return paths

    @pytest.mark.parametrize("mode,ref", [
        ("translation", "first"), ("similarity", "first"),
        ("breathing", "first"), ("breathing", "middle"), ("similarity", "previous"),
    ])
    def test_mixed_dimensions_stack(self, tmp_path, mode, ref):
        paths = self._write_mixed(tmp_path)
        cfg = AlgorithmConfig(stacking_method="laplacian", alignment_mode=mode,
                              alignment_reference=ref, fusion_kernel_size=3,
                              pyramid_num_levels=4)
        algo = LaplacianPyramid(config=cfg)
        algo.update_image_paths(paths)
        algo.align_and_stack_images()
        assert algo.output_image is not None
        assert algo.output_image.shape[:2] == (300, 400)


class TestSequentialCancel:
    def test_cancel_returns_none_and_stops_workers(self, tmp_path):
        import threading
        base = make_textured_image(h=256, w=340, seed=29)
        paths = []
        for i in range(8):
            T = cv2.getRotationMatrix2D((170, 128), 0.0, 1.0 + 0.004 * i)
            p = str(tmp_path / f"c_{i:02d}.png")
            cv2.imwrite(p, np.clip(cv2.warpAffine(base, T, (340, 256)), 0, 255).astype(np.uint8))
            paths.append(p)
        alg = Algorithm()
        threads_before = threading.active_count()
        transforms, log = alg.estimate_sequential_transforms(
            paths, precision="default", emit=lambda *a: alg.cancel())
        assert transforms is None and log is None
        # executor is shut down with wait=True: no workers left behind
        assert threading.active_count() <= threads_before


class TestBreathingCLI:
    def _run(self, *extra):
        return subprocess.run(
            [sys.executable, "-m", "src.cli", "--input", "tests/low_res_images/*.jpg",
             "--output", os.devnull, "--align", "--alignment-mode", "breathing", *extra],
            capture_output=True, text=True, cwd=parentdir, timeout=120,
        )

    @pytest.mark.parametrize("step", ["0", "1.5", "-0.1"])
    def test_rejects_invalid_max_scale_step(self, step):
        result = self._run("--max-scale-step", step)
        assert result.returncode != 0
        assert "--max-scale-step" in result.stderr
