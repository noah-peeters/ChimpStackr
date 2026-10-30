"""Automatic GPU/CPU choice (src.compute): rules, model, cache, API and CLI."""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.compute as compute  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
LOW_RES = os.path.join(ROOT, "tests", "low_res_images")
GB = 1 << 30
INFO = {"name": "Test GPU", "driver": "1.0", "version": "OpenCL 1.2", "memory": 8 * GB,
        "max_alloc": 2 * GB}


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CHIMPSTACKR_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


def entry(gpu, cpu, frames=(2, 5), mp=(1.0, 6.0)):
    """Cache entry from (setup, per_frame_per_mp) per device: total time of
    n frames of m megapixels = setup + n * per_mp * m."""
    def times(setup, per_mp):
        return [[setup + n * per_mp * m for n in frames] for m in mp]
    return {"mp": list(mp), "frames": list(frames), "gpu": times(*gpu), "cpu": times(*cpu),
            "gpu_error": ""}


@pytest.fixture
def gpu(monkeypatch):
    """A fake usable GPU with a given measurement."""
    state = {"entry": entry((1.0, 0.02), (0.0, 0.08)), "measured": 0, "info": dict(INFO)}

    def fake_measure(method, gpu_index=0, report=None, check_load=False):
        state["measured"] += 1
        if report:
            report("Measuring GPU speed (1/4)…")
        return state["entry"]
    monkeypatch.setattr(compute, "gpu_info", lambda index=0: state["info"])
    monkeypatch.setattr(compute, "measure", fake_measure)
    monkeypatch.setattr(compute, "system_busy", lambda: state.get("busy", False))
    return state


class TestModel:
    def test_setup_and_per_frame_are_separated(self):
        e = entry((2.0, 0.01), (0.0, 0.05))        # GPU: slow to start, fast per frame
        g, c = compute.predict(e, 24.0, 50)
        assert g == pytest.approx(2.0 + 50 * 0.24) and c == pytest.approx(50 * 1.2)
        g2, c2 = compute.predict(e, 24.0, 1)
        assert g2 > c2                             # one frame: setup dominates, CPU wins

    def test_gpu_failure_has_no_prediction(self):
        e = entry((0, 0.01), (0, 0.05))
        e["gpu_error"] = "boom"
        assert compute.predict(e, 6, 10) is None

    def test_modes(self):
        assert compute.normalize_mode(True) == "gpu"
        assert compute.normalize_mode(False) == "cpu"
        assert compute.normalize_mode("") == "auto"
        assert compute.normalize_mode("nonsense") == "auto"


class TestDecide:
    def test_forced_cpu_never_touches_the_gpu(self, monkeypatch):
        monkeypatch.setattr(compute, "gpu_info", lambda index=0: pytest.fail("GPU queried"))
        d = compute.decide("cpu", 6000, 4000, "hybrid")
        assert not d.use_gpu and d.summary() == "CPU"

    def test_no_usable_gpu(self, monkeypatch):
        monkeypatch.setattr(compute, "gpu_info", lambda index=0: None)
        for mode in ("auto", "gpu"):
            d = compute.decide(mode, 6000, 4000, "hybrid")
            assert not d.use_gpu and d.reason == "no usable GPU"

    def test_forced_gpu(self, gpu):
        d = compute.decide("gpu", 400, 300, "hybrid")
        assert d.use_gpu and d.device == "Test GPU" and gpu["measured"] == 0

    def test_small_frames_use_the_cpu_without_measuring(self, gpu):
        d = compute.decide("auto", 750, 500, "hybrid")
        assert not d.use_gpu and "small frames" in d.reason and gpu["measured"] == 0

    def test_small_frames_follow_an_existing_measurement(self, gpu):
        gpu["entry"] = entry((0.0, 0.02), (0.0, 0.08))     # GPU faster, no setup cost
        compute.decide("auto", 6000, 4000, "depth_map")      # measured on a big stack
        d = compute.decide("auto", 750, 500, "depth_map", frames=10)
        assert d.use_gpu and d.reason.startswith("measured") and gpu["measured"] == 1

    def test_not_enough_gpu_memory(self, gpu):
        gpu["info"]["memory"] = 2 * GB     # hybrid on 24 MP needs ~4.5 GB
        d = compute.decide("auto", 6000, 4000, "hybrid")
        assert not d.use_gpu and "memory" in d.reason
        assert compute.decide("auto", 3000, 2000, "hybrid").use_gpu      # 6 MP fits

    def test_frame_larger_than_one_gpu_buffer(self, gpu):
        gpu["info"]["max_alloc"] = 100 << 20
        assert not compute.decide("auto", 6000, 4000, "laplacian").use_gpu

    def test_measured_speed_decides(self, gpu):
        d = compute.decide("auto", 6000, 4000, "hybrid", frames=50)
        assert d.use_gpu and d.speedup > 3 and d.reason.startswith("measured")
        gpu["entry"] = entry((0.0, 0.08), (0.0, 0.05))       # CPU faster
        compute.clear_measurements()
        d = compute.decide("auto", 6000, 4000, "hybrid", frames=50)
        assert not d.use_gpu and d.reason.startswith("CPU measured")

    def test_near_ties_stay_on_the_gpu(self, gpu):
        gpu["entry"] = entry((0.0, 0.06), (0.0, 0.05))     # CPU 1.2x faster: not clearly
        d = compute.decide("auto", 6000, 4000, "hybrid", frames=50)
        assert d.use_gpu and d.reason == "about as fast as the CPU"

    def test_frame_count_matters(self, gpu):
        # GPU: 1 s to set up, then 4x cheaper per frame
        assert compute.decide("auto", 2000, 1500, "laplacian", frames=50).use_gpu
        assert not compute.decide("auto", 2000, 1500, "laplacian", frames=2).use_gpu

    def test_gpu_failing_in_the_test(self, gpu):
        gpu["entry"] = dict(gpu["entry"], gpu_error="CL_OUT_OF_RESOURCES")
        d = compute.decide("auto", 6000, 4000, "depth_map")
        assert not d.use_gpu and "failed" in d.reason

    def test_without_measuring(self, gpu):
        d = compute.decide("auto", 6000, 4000, "hybrid", calibrate=False)
        assert d.use_gpu and d.reason == "not measured yet" and gpu["measured"] == 0


class TestBackground:
    def test_measures_missing_methods(self, gpu):
        assert set(compute.missing_methods()) == set(compute.GPU_FRAMES_NEEDED)
        done = compute.measure_in_background(["hybrid", "laplacian"])
        assert done == ["hybrid", "laplacian"] and gpu["measured"] == 2
        assert "hybrid" not in compute.missing_methods()

    def test_a_stack_stops_measuring(self, gpu, monkeypatch):
        import threading
        started, release = threading.Event(), threading.Event()

        def slow_measure(method, gpu_index=0, report=None, check_load=False):
            started.set()
            release.wait(5)
            if compute.STOP.is_set():          # measure() checks STOP between steps
                raise compute.Cancelled()
            return gpu["entry"]
        monkeypatch.setattr(compute, "measure", slow_measure)
        result = {}
        t = threading.Thread(target=lambda: result.update(
            done=compute.measure_in_background(["hybrid", "laplacian"])))
        t.start()
        assert started.wait(5)
        got_lock = []
        stacker = threading.Thread(target=lambda: got_lock.append(compute.yield_to_stack().__enter__()))
        stacker.start()
        release.set()                     # the measuring step ends, sees STOP, gives up
        stacker.join(5)
        t.join(5)
        assert result["done"] == [] and got_lock
        assert compute.missing_methods() == list(compute.GPU_FRAMES_NEEDED)   # nothing half-saved
        got_lock[0].__exit__(None, None, None)
        assert not compute.STOP.is_set()

    def test_busy_computer_postpones_measuring(self, gpu):
        gpu["busy"] = True
        assert compute.measure_in_background(["hybrid"]) == [] and gpu["measured"] == 0

    def test_load_check_cancels_a_real_measurement(self, monkeypatch):
        monkeypatch.setattr(compute, "system_busy", lambda: True)
        with pytest.raises(compute.Cancelled):
            compute.measure("laplacian", check_load=True)

    def test_others_cpu(self):
        cores = compute.others_cpu_cores()
        assert cores is None or cores >= 0

    def test_busy_device_isnt_waited_for(self, gpu):
        with compute.yield_to_stack():
            assert compute.measure_in_background(["hybrid"]) == []
        assert gpu["measured"] == 0


class TestCache:
    def test_measured_once_then_cached(self, gpu, cache):
        reports = []
        compute.decide("auto", 6000, 4000, "hybrid", report=reports.append)
        compute.decide("auto", 6000, 4000, "hybrid", report=reports.append)
        assert gpu["measured"] == 1 and len(reports) == 1
        assert (cache / "device-speed.json").exists()
        compute.decide("auto", 6000, 4000, "laplacian")      # per method
        assert gpu["measured"] == 2

    def test_new_driver_or_app_version_measures_again(self, gpu, monkeypatch):
        compute.decide("auto", 6000, 4000, "hybrid")
        gpu["info"]["driver"] = "2.0"
        compute.decide("auto", 6000, 4000, "hybrid")
        assert gpu["measured"] == 2
        monkeypatch.setattr(compute, "_app_version", lambda: "v9.9.9")
        compute.decide("auto", 6000, 4000, "hybrid")
        assert gpu["measured"] == 3

    def test_clear_and_list(self, gpu):
        compute.decide("auto", 6000, 4000, "hybrid")
        assert len(compute.measurements()) == 1
        compute.clear_measurements()
        assert compute.measurements() == {}

    def test_corrupt_cache_is_ignored(self, gpu, cache):
        cache.mkdir(parents=True)
        (cache / "device-speed.json").write_text("{not json")
        assert compute.decide("auto", 6000, 4000, "hybrid").use_gpu


class TestAPI:
    def _algo(self, mode, paths):
        import src.settings as settings
        settings.init()
        from src.algorithms.API import LaplacianPyramid
        from src.config import AlgorithmConfig
        a = LaplacianPyramid(config=AlgorithmConfig(stacking_method="hybrid", gpu_mode=mode))
        a.update_image_paths(paths)
        return a

    def test_resolve_passes_frame_size_and_count(self, monkeypatch):
        seen = []
        monkeypatch.setattr(compute, "decide", lambda *a, **k: seen.append((a, k)) or
                            compute.Decision(False, "auto", reason="test"))
        paths = sorted(os.path.join(LOW_RES, f) for f in os.listdir(LOW_RES))
        a = self._algo("auto", paths)
        a.resolve_device()
        a.resolve_device()                          # remembered
        (args, kwargs), = seen
        assert args[5] is False                     # stacks never wait for a measurement
        assert args[:4] == ("auto", 750, 500, "hybrid") and kwargs["frames"] == len(paths)
        a.config.stacking_method = "laplacian"      # settings changed: decided again
        a.resolve_device()
        assert len(seen) == 2

    def test_default_config_stays_on_the_cpu(self):
        from src.config import AlgorithmConfig
        from src.algorithms.API import LaplacianPyramid
        assert LaplacianPyramid(config=AlgorithmConfig()).gpu_mode() == "cpu"
        assert LaplacianPyramid(config=AlgorithmConfig(use_gpu=True)).gpu_mode() == "gpu"


class TestCLI:
    def _run(self, *args, env_cache):
        env = {k: v for k, v in os.environ.items() if k != "CHIMPSTACKR_OPENCL_ANY_DEVICE"}
        env["CHIMPSTACKR_CACHE_DIR"] = str(env_cache)
        return subprocess.run([sys.executable, "-m", "src.cli", "--input", "tests/low_res_images/*.jpg",
                               "--pyramid-levels", "3", *args], capture_output=True, text=True,
                              cwd=ROOT, env=env, timeout=180)

    def test_cpu_flag(self, tmp_path, cache):
        r = self._run("--output", str(tmp_path / "o.jpg"), "--cpu", env_cache=cache)
        assert r.returncode == 0, r.stderr
        assert "Device: CPU" in r.stdout

    def test_auto_is_the_default(self, tmp_path, cache):
        r = self._run("--output", str(tmp_path / "o.jpg"), env_cache=cache)
        assert r.returncode == 0, r.stderr
        line = next(ln for ln in r.stdout.splitlines() if "Device:" in ln)
        # 0.4 MP test frames: the CPU, without measuring anything
        assert "Auto: CPU" in line and ("small frames" in line or "no usable GPU" in line)
        assert "Measuring" not in r.stdout

    def test_measure_flag(self, tmp_path, cache):
        # tiny frames never start a measurement, even with --measure
        r = self._run("--output", str(tmp_path / "o.jpg"), "--measure", env_cache=cache)
        assert r.returncode == 0, r.stderr
        assert "Measuring" not in r.stdout
