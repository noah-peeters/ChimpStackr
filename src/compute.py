"""
GPU or CPU? The device choice shared by the app and the command line.

Modes: "gpu" and "cpu" force a device ("gpu" still falls back to the CPU
when no GPU works). "auto" (the default) picks per stack:

1. No usable OpenCL GPU (none found, or its driver can't build the
   kernels): CPU.
2. The stack wouldn't fit in GPU memory (measured per method): CPU.
3. Small frames (under SMALL_MP megapixels) before anything was measured:
   CPU, without measuring (the test would take longer than the stack).
   Larger frames before a measurement exists: GPU, which wins most real
   stacks.
4. Otherwise a one-time measurement on this computer decides: short
   stacks of synthetic frames (decoded, aligned and stacked like a real
   stack) at two sizes and two frame counts, on the GPU and on the CPU, for
   the chosen method. Stack time is modelled as
   setup + frames * per_frame (both linear in megapixels) for each device
   and compared for the real frame size and count. Results are cached in
   a small JSON file shared by the app and the CLI, keyed by GPU, driver
   and app version, so it is measured again after any of them changes.

Similar to how Capture One ("Auto" tests the card) and darktable (startup
benchmark, re-run when the GPU setup changes) decide.
"""
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MODES = ("auto", "gpu", "cpu")
SMALL_MP = 1.0
MEMORY_HEADROOM = 0.8          # use at most this share of the GPU's memory
# Peak GPU memory per method in float RGB frames (w * h * 12 bytes), measured
# on 24 MP stacks with alignment, plus 25 % headroom
GPU_FRAMES_NEEDED = {"laplacian": 10.0, "hybrid": 15.5, "weighted_average": 8.0,
                     "depth_map": 9.5, "exposure_fusion": 14.5}
CALIBRATION_SIZES = ((1224, 816), (3000, 2000))     # 1.0 and 6.0 MP
CALIBRATION_FRAMES = (2, 5)       # frame counts per size
CACHE_VERSION = 3

_lock = threading.Lock()          # the cache file
# One heavy job at a time: background measuring holds WORK_LOCK and stops
# between steps once STOP is set; a stack sets STOP and takes the lock (the
# GPU's upload buffers aren't shared between jobs)
WORK_LOCK = threading.Lock()
STOP = threading.Event()


class Cancelled(Exception):
    pass


def others_cpu_cores():
    """CPU cores other processes are using right now (from `ps`), or None
    where that isn't available (Windows)."""
    if sys.platform == "win32":
        return None
    import subprocess
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,%cpu="], capture_output=True, text=True,
                             timeout=5).stdout
    except Exception:
        return None
    me = os.getpid()
    total = 0.0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) != me:
            try:
                total += float(parts[1].replace(",", "."))
            except ValueError:
                pass
    return total / 100.0


def system_busy():
    """True when other programs use enough CPU to skew a speed measurement."""
    cores = others_cpu_cores()
    return cores is not None and cores > max(1.5, 0.3 * (os.cpu_count() or 4))


def yield_to_stack():
    """Called by a stack before it starts: stops background measuring and
    waits for its current step. Use as `with yield_to_stack(): ...`."""
    STOP.set()
    return _Held()


class _Held:
    def __enter__(self):
        WORK_LOCK.acquire()
        return self

    def __exit__(self, *exc):
        STOP.clear()
        WORK_LOCK.release()


@dataclass
class Decision:
    use_gpu: bool
    mode: str                 # the requested mode
    device: str = ""          # GPU name when use_gpu
    reason: str = ""          # short, for people: "measured 2.2× faster"
    speedup: float = 0.0      # CPU time / GPU time at this size, when measured

    def summary(self):
        where = f"GPU · {self.device}" if self.use_gpu else "CPU"
        if self.mode == "auto":
            return f"Auto: {where}" + (f" ({self.reason})" if self.reason else "")
        return where + (f" ({self.reason})" if self.reason else "")


def normalize_mode(mode):
    if mode is True:
        return "gpu"
    if mode is False:
        return "cpu"
    mode = str(mode or "auto").lower()
    return mode if mode in MODES else "auto"


# ── cache ──
def cache_dir():
    override = os.environ.get("CHIMPSTACKR_CACHE_DIR")
    if override:
        return override
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Caches/ChimpStackr")
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return os.path.join(base, "ChimpStackr", "Cache")
    return os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "chimpstackr")


def cache_path():
    return os.path.join(cache_dir(), "device-speed.json")


def _load_cache():
    try:
        with open(cache_path()) as f:
            data = json.load(f)
        return data if data.get("version") == CACHE_VERSION else {"version": CACHE_VERSION}
    except (OSError, ValueError):
        return {"version": CACHE_VERSION}


def _save_cache(data):
    try:
        os.makedirs(cache_dir(), exist_ok=True)
        tmp = cache_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1, sort_keys=True)
        os.replace(tmp, cache_path())
    except OSError as exc:
        logger.warning(f"Couldn't save the device speed cache: {exc}")


def clear_measurements():
    with _lock:
        try:
            os.remove(cache_path())
        except OSError:
            pass


def _app_version():
    try:
        import src
        return str(src.__version__)
    except Exception:
        return "?"


def _device_key(info, method):
    return "|".join((info["name"], info["driver"], info["version"], _app_version(), method))


def measurements():
    """Cached measurements for this app version: {device|method: entry}."""
    data = _load_cache()
    return {k: v for k, v in data.get("entries", {}).items() if k.split("|")[3:4] == [_app_version()]}


# ── GPU facts ──
def gpu_info(index=0):
    """Name, driver and memory of GPU `index` if it is usable, else None."""
    import src.algorithms.stacking_algorithms.ocl as OCL
    if not OCL.available():
        return None
    try:
        gpu = OCL.get(index)                 # builds the kernels; raises if the driver can't
    except Exception as exc:
        logger.info(f"GPU not usable: {exc}")
        return None
    dev = gpu.ctx.devices[0]
    return {"name": gpu.name, "driver": str(dev.driver_version).strip(),
            "version": str(dev.version).strip(), "memory": int(dev.global_mem_size),
            "max_alloc": int(dev.max_mem_alloc_size)}


def fits_in_gpu(info, width, height, method):
    frame = width * height * 3 * 4
    need = frame * GPU_FRAMES_NEEDED.get(method, 15.5)
    return need <= info["memory"] * MEMORY_HEADROOM and frame <= info["max_alloc"]


# ── measuring ──
def _synthetic_frames(folder, size, n):
    """A small, alignable focus stack: a textured scene, every frame moved
    slightly (shift, rotation, scale like a real stack) and sharp only in
    its own band, saved as JPEG like most camera stacks."""
    import cv2
    import numpy as np
    w, h = size
    rng = np.random.RandomState(7)
    scene = np.zeros((h, w, 3), np.float32)
    for cell, weight in ((2, 0.25), (9, 0.35), (37, 0.4)):      # detail at several scales
        noise = rng.rand(max(2, h // cell), max(2, w // cell), 3).astype(np.float32)
        scene += weight * cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    scene = (np.clip(scene, 0, 1) * 255).astype(np.uint8)
    for _ in range(max(20, w * h // 40000)):                  # edges and corners to match
        c = tuple(int(v) for v in rng.randint(0, 255, 3))
        p = (int(rng.randint(0, w)), int(rng.randint(0, h)))
        cv2.circle(scene, p, int(rng.randint(4, max(6, w // 40))), c, int(rng.choice([-1, 2])))
    paths = []
    for i in range(n):
        angle, scale = rng.uniform(-0.15, 0.15), 1 + rng.uniform(-0.003, 0.003)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[:, 2] += rng.uniform(-4, 4, 2)
        moved = cv2.warpAffine(scene, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img = cv2.GaussianBlur(moved, (0, 0), 2.5)
        y0, y1 = h * i // n, h * (i + 1) // n
        img[y0:y1] = moved[y0:y1]
        path = os.path.join(folder, f"frame_{i:02d}.jpg")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths.append(path)
    return paths


def _time_stack(paths, method, gpu, gpu_index, align=True):
    """Seconds per frame to align and stack `paths` the way a default stack
    does (Standard alignment)."""
    from src.algorithms.API import LaplacianPyramid
    from src.config import AlgorithmConfig, auto_detect_params
    import cv2
    h, w = cv2.imread(paths[0], cv2.IMREAD_REDUCED_GRAYSCALE_2).shape[:2]
    params = auto_detect_params((h * 2, w * 2), len(paths))
    algo = LaplacianPyramid(config=AlgorithmConfig(
        stacking_method=method, use_gpu=gpu, gpu_mode="gpu" if gpu else "cpu",
        selected_gpu_id=gpu_index, fusion_kernel_size=params["fusion_kernel_size"],
        pyramid_num_levels=params["pyramid_num_levels"], alignment_mode="similarity",
        alignment_scale_factor=params["alignment_scale_factor"]))
    algo.update_image_paths(paths)
    fell_back = []

    class Watch(logging.Handler):
        def emit(self, record):
            if "falling back" in record.getMessage():
                fell_back.append(record.getMessage())
    watch = Watch()
    logging.getLogger("src.algorithms.API").addHandler(watch)
    try:
        t = time.perf_counter()
        (algo.align_and_stack_images if align else algo.stack_images)()
        elapsed = time.perf_counter() - t
    finally:
        logging.getLogger("src.algorithms.API").removeHandler(watch)
    if algo.output_image is None:
        raise RuntimeError("calibration stack produced no image")
    if gpu and fell_back:
        raise RuntimeError(fell_back[0])
    return elapsed / len(paths)


def measure(method, gpu_index=0, report=None, check_load=False):
    """Time short stacks of synthetic frames on the GPU and the CPU, at two
    frame sizes and two frame counts, so the fixed cost of a stack (setup,
    the final read-back) and the cost per frame can be told apart. Returns
    the cache entry. `report(text)` gets progress. With check_load, gives up
    (Cancelled) when other programs keep the CPU busy."""
    tmp = tempfile.mkdtemp(prefix="chimpstackr-calibrate-")
    try:
        os.makedirs(os.path.join(tmp, "warm"))
        warm = _synthetic_frames(os.path.join(tmp, "warm"), (400, 300), 2)
        for gpu in (True, False):       # JIT / kernel warm-up, not timed
            try:
                _time_stack(warm, method, gpu, gpu_index)
            except Exception:
                pass
        n_small, n_large = CALIBRATION_FRAMES
        result = {"mp": [], "frames": [n_small, n_large], "gpu": [], "cpu": [], "gpu_error": ""}
        steps = len(CALIBRATION_SIZES) * 2
        step = 0
        for k, size in enumerate(CALIBRATION_SIZES):
            folder = os.path.join(tmp, str(k))
            os.makedirs(folder)
            paths = _synthetic_frames(folder, size, n_large)
            result["mp"].append(size[0] * size[1] / 1e6)
            times = {"gpu": [], "cpu": []}
            for n in (n_small, n_large):
                step += 1
                if report:
                    report(f"step {step} of {steps}…")
                # GPU and CPU back to back, so changing load hits both alike
                for dev in ("gpu", "cpu"):
                    if dev == "gpu" and result["gpu_error"]:
                        continue
                    if STOP.is_set() or (check_load and system_busy()):
                        raise Cancelled()
                    try:
                        t = _time_stack(paths[:n], method, dev == "gpu", gpu_index) * n
                        if t < 1.0:      # short runs are noisy: best of two
                            t = min(t, _time_stack(paths[:n], method, dev == "gpu", gpu_index) * n)
                    except Exception as exc:
                        if dev == "cpu":
                            raise
                        result["gpu_error"] = str(exc)[:200]
                        continue
                    times[dev].append(t)
            for dev in ("gpu", "cpu"):
                if len(times[dev]) == 2:
                    result[dev].append(times[dev])
        result["measured"] = time.time()
        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _line(x0, x1, y0, y1):
    """a + b * x through two points (b >= 0, a >= 0)."""
    b = max(0.0, (y1 - y0) / (x1 - x0)) if x1 != x0 else 0.0
    return max(0.0, y0 - b * x0), b


def predict(entry, mp, frames):
    """Predicted total seconds (GPU, CPU) to stack `frames` frames of `mp`
    megapixels, or None when the GPU couldn't be measured. Per size the
    stack time is setup + frames * per_frame; both grow linearly with the
    frame size."""
    if entry.get("gpu_error") or len(entry.get("gpu", [])) != 2 or len(entry.get("cpu", [])) != 2:
        return None
    n0, n1 = entry["frames"]
    m0, m1 = entry["mp"]
    out = []
    for dev in ("gpu", "cpu"):
        setups, per_frame = [], []
        for t_small, t_large in entry[dev]:
            setup, each = _line(n0, n1, t_small, t_large)
            setups.append(setup)
            per_frame.append(each)
        sa, sb = _line(m0, m1, *setups)
        pa, pb = _line(m0, m1, *per_frame)
        out.append(sa + sb * mp + frames * (pa + pb * mp))
    return tuple(out)


def measurement(method, gpu_index=0, info=None, calibrate=True, report=None, check_load=False):
    """The cached measurement for this GPU/driver/app version and method,
    measuring it first if needed (and allowed)."""
    info = info or gpu_info(gpu_index)
    if info is None:
        return None
    key = _device_key(info, method)
    entry = _load_cache().get("entries", {}).get(key)
    if entry is None and calibrate:
        logger.info(f"Measuring GPU vs CPU speed for {method} on {info['name']}")
        entry = measure(method, gpu_index, report, check_load)
        with _lock:
            data = _load_cache()
            data.setdefault("entries", {})[key] = entry
            _save_cache(data)
    return entry


def missing_methods(gpu_index=0, methods=None):
    """Methods without a measurement for the current GPU/driver/app version."""
    info = gpu_info(gpu_index)
    if info is None:
        return []
    have = _load_cache().get("entries", {})
    return [m for m in (methods or GPU_FRAMES_NEEDED) if _device_key(info, m) not in have]


def measure_in_background(methods, gpu_index=0, report=None):
    """Measure the given methods while nothing else runs; returns the
    methods measured. Stops early (losing only the method in progress) when
    a stack starts (yield_to_stack) or other programs keep the CPU busy (a
    measurement then would be skewed). Doesn't wait for a busy device."""
    if not WORK_LOCK.acquire(blocking=False):
        return []
    done = []
    try:
        STOP.clear()
        info = gpu_info(gpu_index)
        if info is None:
            return []
        for method in methods:
            if system_busy():
                break
            try:
                measurement(method, gpu_index, info, True, report, check_load=True)
            except Cancelled:
                break
            done.append(method)
    finally:
        WORK_LOCK.release()
    return done


def decide(mode="auto", width=0, height=0, method="laplacian", gpu_index=0,
           calibrate=True, report=None, frames=50):
    """Which device stacks `frames` frames of width x height with `method`."""
    mode = normalize_mode(mode)
    if mode == "cpu":
        return Decision(False, mode, reason="")
    info = gpu_info(gpu_index)
    if info is None:
        return Decision(False, mode, reason="no usable GPU")
    if mode == "gpu":
        return Decision(True, mode, info["name"])
    mp = width * height / 1e6
    if width and height and not fits_in_gpu(info, width, height, method):
        return Decision(False, mode, reason="not enough GPU memory for these frames")
    small = bool(width and height) and mp < SMALL_MP
    # a stack of small frames is quick either way: don't spend the test's
    # ~15 s on it, but use a measurement that already exists
    entry = measurement(method, gpu_index, info, calibrate and not small, report)
    if entry is None and small:
        return Decision(False, mode, reason=f"small frames ({mp:.1f} MP)")
    if entry is None:
        # not measured yet: the GPU, which wins most real stacks
        return Decision(True, mode, info["name"], "not measured yet")
    if entry.get("gpu_error"):
        return Decision(False, mode, reason="the GPU failed during the speed test")
    times = predict(entry, mp or 24.0, max(2, frames))
    if times is None:
        return Decision(True, mode, info["name"], "not measured yet")
    gpu_t, cpu_t = times
    speedup = cpu_t / gpu_t if gpu_t > 0 else float("inf")
    if gpu_t <= cpu_t:
        return Decision(True, mode, info["name"], f"measured {speedup:.1f}× faster", speedup)
    return Decision(False, mode, reason=f"CPU measured {1 / speedup:.1f}× faster", speedup=speedup)
