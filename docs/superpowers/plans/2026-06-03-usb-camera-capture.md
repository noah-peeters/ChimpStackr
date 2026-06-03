# USB Camera / Microscope Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users capture frames directly from a connected USB camera / microscope inside ChimpStackr (live preview, single capture, and timed focus-bracketing), feeding captured images straight into the existing focus-stacking pipeline.

**Architecture:** A pure, Qt-free OpenCV capture backend (`src/CameraCaptureHandler.py`) does device enumeration, frame grabbing, and lossless saving — fully unit-testable with no hardware via a dependency-injected capture factory. A PySide6 widget (`src/MainWindow/MainLayout/CameraCaptureWidget.py`) provides the live-preview UI as a new "Capture" tab and drives the backend. Captured frames are written as lossless PNGs to the session temp dir and routed into the existing loaded-images list through a new additive `MainWindow.add_captured_image_files()` method, so the alignment/stacking pipeline needs **zero** changes.

**Tech Stack:** Python 3.9–3.13, OpenCV (`opencv-python-headless`, already a dependency — `cv2.VideoCapture`), PySide6/Qt6 (already a dependency), pytest 9 (already a dev dependency). **No new runtime dependencies.**

**Scope — explicitly OUT:** Native "still-image pin" / full-sensor-resolution capture (DirectShow `PIN_CATEGORY_STILL`, AVFoundation `AVCapturePhotoOutput`, patched Linux `uvcvideo`). Research (see `docs` / issue #166 discussion) confirmed this is a per-device, firmware-gated UVC capability (Method 2/3) that most cameras don't expose and that Linux mainline doesn't support at all. v1 captures from the video stream at the highest resolution the device accepts. Full-res still-pin support may be revisited later as an optional per-platform backend behind the same `CameraCaptureHandler` interface.

---

## Environment Notes (read before starting)

- **Project venv:** `.venv` at repo root. It has `cv2` (4.13) and `pytest` (9) but **does NOT have PySide6 installed**. The existing test suite (`tests/`) therefore only imports Qt-free modules. **Keep `src/CameraCaptureHandler.py` and `src/utilities.py` free of any PySide6 import** so their tests run in `.venv`. GUI modules import PySide6 and are verified manually, not by automated tests in this environment.
- **Run tests with:** `.venv/bin/python -m pytest` from the repo root.
- **Branch:** Work happens on `feature/usb-camera-capture` (already created and checked out).
- **Existing test pattern** (see `tests/test_ImageLoadingHandler.py`): tests call `import src.settings as settings; settings.init()` at module top, then import the class under test. Follow this pattern.
- **Pipeline contract:** Loaded images are tracked as **file paths** in `settings.globalVars["LoadedImagePaths"]`. The three calls that register a path list are: `self._main_content.set_loaded_images(paths)`, `self.LaplacianAlgorithm.update_image_paths(paths)`, and assigning `settings.globalVars["LoadedImagePaths"] = paths`. See `src/MainWindow/__init__.py:278-286`.

---

## File Structure

| File | Responsibility | New/Modify |
|---|---|---|
| `src/CameraCaptureHandler.py` | Qt-free OpenCV backend: enumerate devices, open/read/release, resolution control, lossless save. DI capture factory for tests. | **Create** |
| `tests/test_CameraCaptureHandler.py` | Unit tests for the backend using a fake capture object (no hardware). | **Create** |
| `src/utilities.py` | Add pure `merge_image_paths(existing, new)` helper. | Modify |
| `tests/test_utilities.py` | Unit tests for `merge_image_paths`. | **Create** |
| `src/MainWindow/MainLayout/CameraCaptureWidget.py` | PySide6 widget: device/resolution selectors, live preview, capture + bracketing controls. Module-level `bgr_to_qimage()` helper. | **Create** |
| `src/MainWindow/MainLayout/__init__.py` | Add the "Capture" tab to `CenterWidget.tabWidget`; wire captured-frames signal. | Modify |
| `src/MainWindow/__init__.py` | Add additive `add_captured_image_files()` method. | Modify |
| `packaging/flatpak/io.github.noah_peeters.ChimpStackr.yml` | Add `--device=all` (or `--device=/dev/video*`) so the Flatpak sandbox can see USB cameras. | Modify |
| `packaging/entitlements.plist` + PyInstaller spec | macOS camera permission (entitlement + `NSCameraUsageDescription`). | Modify |

---

## Task 1: Backend — device enumeration

**Files:**
- Create: `src/CameraCaptureHandler.py`
- Test: `tests/test_CameraCaptureHandler.py`

Enumeration uses a dependency-injected `capture_factory` (defaults to `cv2.VideoCapture`) so tests inject a fake. It probes indices `0..max_probe-1`, and an index is "present" if the factory returns an object whose `.isOpened()` is `True`. Each probed capture is released immediately.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_CameraCaptureHandler.py
"""Tests for the Qt-free OpenCV camera capture backend (no hardware)."""
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import numpy as np
import pytest

from src.CameraCaptureHandler import CameraCaptureHandler


class FakeCapture:
    """Stand-in for cv2.VideoCapture. `present` controls isOpened()."""
    def __init__(self, index, present=True, frame=None, width=640, height=480):
        self.index = index
        self._present = present
        self._frame = frame
        self._released = False
        self._props = {3: float(width), 4: float(height)}  # CAP_PROP_FRAME_WIDTH/HEIGHT

    def isOpened(self):
        return self._present and not self._released

    def read(self):
        if not self.isOpened() or self._frame is None:
            return False, None
        return True, self._frame.copy()

    def set(self, prop_id, value):
        self._props[prop_id] = float(value)
        return True

    def get(self, prop_id):
        return self._props.get(prop_id, 0.0)

    def release(self):
        self._released = True


def make_factory(present_indices):
    """Return a factory where only `present_indices` produce opened captures."""
    def factory(index):
        return FakeCapture(index, present=index in present_indices)
    return factory


def test_enumerate_finds_present_devices():
    handler = CameraCaptureHandler(capture_factory=make_factory({0, 2}))
    devices = handler.enumerate_devices(max_probe=4)
    indices = [d["index"] for d in devices]
    assert indices == [0, 2]
    # Each device has a human-readable name
    assert all(isinstance(d["name"], str) and d["name"] for d in devices)


def test_enumerate_empty_when_no_devices():
    handler = CameraCaptureHandler(capture_factory=make_factory(set()))
    assert handler.enumerate_devices(max_probe=4) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_CameraCaptureHandler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.CameraCaptureHandler'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/CameraCaptureHandler.py
"""
Qt-free OpenCV-based USB camera / microscope capture backend.

Captures from the camera's video stream (UVC) at the highest resolution the
device accepts. Native still-image-pin full-resolution capture is intentionally
out of scope (see docs/superpowers/plans/2026-06-03-usb-camera-capture.md).

No PySide6 imports here — keeps the module unit-testable in the headless venv.
"""
import logging
import cv2

logger = logging.getLogger(__name__)

# OpenCV CAP_PROP ids (avoid importing constants for clarity)
_CAP_PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
_CAP_PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT


class CameraCaptureHandler:
    def __init__(self, capture_factory=cv2.VideoCapture):
        """capture_factory(index) -> object with isOpened/read/set/get/release.
        Defaults to cv2.VideoCapture; tests inject a fake."""
        self._factory = capture_factory
        self._cap = None
        self._index = None

    def enumerate_devices(self, max_probe=5):
        """Probe indices 0..max_probe-1; return [{"index": int, "name": str}]."""
        found = []
        for index in range(max_probe):
            cap = self._factory(index)
            try:
                if cap.isOpened():
                    found.append({"index": index, "name": f"Camera {index}"})
            finally:
                cap.release()
        return found
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_CameraCaptureHandler.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/CameraCaptureHandler.py tests/test_CameraCaptureHandler.py
git commit -m "feat(camera): add capture backend with device enumeration"
```

---

## Task 2: Backend — open, read frame, capture still, release

**Files:**
- Modify: `src/CameraCaptureHandler.py`
- Test: `tests/test_CameraCaptureHandler.py`

`capture_still()` discards a few warm-up frames (cameras need auto-exposure settle time) then returns a frame. Warm-up count is a parameter so tests can set it to 0.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_CameraCaptureHandler.py

def _synthetic_frame(width=640, height=480):
    # Deterministic non-flat BGR uint8 frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, : width // 2] = (10, 20, 30)
    frame[:, width // 2 :] = (200, 150, 100)
    return frame


def test_open_read_and_release():
    frame = _synthetic_frame()

    def factory(index):
        return FakeCapture(index, present=True, frame=frame)

    handler = CameraCaptureHandler(capture_factory=factory)
    assert handler.open(0) is True
    assert handler.is_opened is True

    ok, got = handler.read_frame()
    assert ok is True
    assert isinstance(got, np.ndarray)
    assert got.shape == (480, 640, 3)
    assert got.dtype == np.uint8

    handler.release()
    assert handler.is_opened is False


def test_open_failure_returns_false():
    handler = CameraCaptureHandler(capture_factory=make_factory(set()))
    assert handler.open(0) is False
    assert handler.is_opened is False


def test_capture_still_returns_frame():
    frame = _synthetic_frame()

    def factory(index):
        return FakeCapture(index, present=True, frame=frame)

    handler = CameraCaptureHandler(capture_factory=factory)
    handler.open(0)
    still = handler.capture_still(warmup_frames=0)
    assert isinstance(still, np.ndarray)
    assert still.shape == (480, 640, 3)


def test_capture_still_without_open_returns_none():
    handler = CameraCaptureHandler(capture_factory=make_factory(set()))
    assert handler.capture_still(warmup_frames=0) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_CameraCaptureHandler.py -v`
Expected: FAIL with `AttributeError: 'CameraCaptureHandler' object has no attribute 'open'`

- [ ] **Step 3: Write minimal implementation**

```python
# add these methods inside class CameraCaptureHandler in src/CameraCaptureHandler.py

    @property
    def is_opened(self):
        return self._cap is not None and self._cap.isOpened()

    def open(self, index):
        """Open the camera at `index`. Returns True on success."""
        self.release()
        cap = self._factory(index)
        if not cap.isOpened():
            cap.release()
            logger.error("Failed to open camera index %s", index)
            return False
        self._cap = cap
        self._index = index
        return True

    def read_frame(self):
        """Grab one frame. Returns (ok: bool, frame: ndarray|None) in BGR uint8."""
        if not self.is_opened:
            return False, None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    def capture_still(self, warmup_frames=5):
        """Discard `warmup_frames` frames (exposure settle) then return one BGR frame."""
        if not self.is_opened:
            return None
        for _ in range(max(0, warmup_frames)):
            self._cap.read()
        ok, frame = self.read_frame()
        return frame if ok else None

    def release(self):
        """Release the underlying capture device."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None
        self._index = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_CameraCaptureHandler.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/CameraCaptureHandler.py tests/test_CameraCaptureHandler.py
git commit -m "feat(camera): add open/read/capture_still/release to backend"
```

---

## Task 3: Backend — resolution control

**Files:**
- Modify: `src/CameraCaptureHandler.py`
- Test: `tests/test_CameraCaptureHandler.py`

`set_resolution(w, h)` requests a resolution and returns the **actual** `(w, h)` the device accepted (cameras silently clamp to nearest supported). `request_max_resolution()` requests an absurdly large size so the driver clamps to its real maximum — a common OpenCV trick to grab the highest stream resolution.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_CameraCaptureHandler.py

class ClampingCapture(FakeCapture):
    """Simulates a camera that clamps requested resolution to a max."""
    MAX_W, MAX_H = 1920, 1080

    def set(self, prop_id, value):
        if prop_id == 3:    # width
            self._props[3] = float(min(value, self.MAX_W))
        elif prop_id == 4:  # height
            self._props[4] = float(min(value, self.MAX_H))
        else:
            self._props[prop_id] = float(value)
        return True


def test_set_resolution_returns_actual():
    handler = CameraCaptureHandler(capture_factory=lambda i: ClampingCapture(i))
    handler.open(0)
    actual = handler.set_resolution(1280, 720)
    assert actual == (1280, 720)


def test_request_max_resolution_clamps_to_device_max():
    handler = CameraCaptureHandler(capture_factory=lambda i: ClampingCapture(i))
    handler.open(0)
    actual = handler.request_max_resolution()
    assert actual == (1920, 1080)


def test_get_resolution_reflects_current():
    handler = CameraCaptureHandler(capture_factory=lambda i: ClampingCapture(i))
    handler.open(0)
    handler.set_resolution(640, 480)
    assert handler.get_resolution() == (640, 480)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_CameraCaptureHandler.py -k resolution -v`
Expected: FAIL with `AttributeError: ... has no attribute 'set_resolution'`

- [ ] **Step 3: Write minimal implementation**

```python
# add these methods inside class CameraCaptureHandler in src/CameraCaptureHandler.py

    def set_resolution(self, width, height):
        """Request a capture resolution; return the (w, h) actually applied."""
        if not self.is_opened:
            return None
        self._cap.set(_CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(_CAP_PROP_FRAME_HEIGHT, height)
        return self.get_resolution()

    def request_max_resolution(self):
        """Request an oversized resolution so the driver clamps to its real max."""
        return self.set_resolution(100000, 100000)

    def get_resolution(self):
        """Return the current (width, height) as ints."""
        if not self.is_opened:
            return None
        w = int(round(self._cap.get(_CAP_PROP_FRAME_WIDTH)))
        h = int(round(self._cap.get(_CAP_PROP_FRAME_HEIGHT)))
        return (w, h)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_CameraCaptureHandler.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add src/CameraCaptureHandler.py tests/test_CameraCaptureHandler.py
git commit -m "feat(camera): add resolution control to backend"
```

---

## Task 4: Backend — lossless still save

**Files:**
- Modify: `src/CameraCaptureHandler.py`
- Test: `tests/test_CameraCaptureHandler.py`

Saving must be **lossless** (PNG) so stacking detail isn't lost to JPEG compression — the core motivation in issue #166. This is a real `cv2.imwrite`/`cv2.imread` round-trip test (no fake), verifying pixels are byte-identical.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_CameraCaptureHandler.py
import cv2 as _cv2  # local alias for the roundtrip test

def test_save_still_lossless_roundtrip(tmp_path):
    frame = _synthetic_frame(64, 48)
    out_path = str(tmp_path / "cap_0001.png")

    saved = CameraCaptureHandler.save_still_lossless(frame, out_path)
    assert saved is True
    assert os.path.isfile(out_path)

    reloaded = _cv2.imread(out_path, _cv2.IMREAD_UNCHANGED)
    assert reloaded is not None
    # PNG is lossless: pixels must be byte-identical
    assert np.array_equal(reloaded, frame)


def test_save_still_lossless_none_frame(tmp_path):
    out_path = str(tmp_path / "none.png")
    assert CameraCaptureHandler.save_still_lossless(None, out_path) is False
    assert not os.path.exists(out_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_CameraCaptureHandler.py -k lossless -v`
Expected: FAIL with `AttributeError: ... has no attribute 'save_still_lossless'`

- [ ] **Step 3: Write minimal implementation**

```python
# add this staticmethod inside class CameraCaptureHandler in src/CameraCaptureHandler.py

    @staticmethod
    def save_still_lossless(frame, path):
        """Write a BGR frame to `path` losslessly (PNG). Returns True on success."""
        if frame is None:
            return False
        try:
            # PNG compression level 1 = fast, still lossless
            return bool(cv2.imwrite(path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]))
        except Exception as e:
            logger.error("Failed to save still %s: %s", path, e)
            return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_CameraCaptureHandler.py -v`
Expected: PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
git add src/CameraCaptureHandler.py tests/test_CameraCaptureHandler.py
git commit -m "feat(camera): add lossless PNG still save"
```

---

## Task 5: Utility — merge captured paths into existing list

**Files:**
- Modify: `src/utilities.py`
- Test: `tests/test_utilities.py`

Captured frames must be **added** to the loaded-images list (not replace it, and not trigger the "clear all?" confirmation dialog). The merge logic is pure and testable: combine existing + new, drop duplicates (preserving first occurrence), and sort with the existing numeric-aware `int_string_sorting`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_utilities.py
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.utilities import merge_image_paths


def test_merge_appends_new_paths_sorted():
    existing = ["/a/img1.png", "/a/img2.png"]
    new = ["/a/img10.png", "/a/img3.png"]
    merged = merge_image_paths(existing, new)
    # Numeric-aware sort: 1, 2, 3, 10 (not lexicographic 1,10,2,3)
    assert merged == ["/a/img1.png", "/a/img2.png", "/a/img3.png", "/a/img10.png"]


def test_merge_dedupes():
    existing = ["/a/img1.png"]
    new = ["/a/img1.png", "/a/img2.png"]
    merged = merge_image_paths(existing, new)
    assert merged == ["/a/img1.png", "/a/img2.png"]


def test_merge_into_empty():
    assert merge_image_paths([], ["/a/img1.png"]) == ["/a/img1.png"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_utilities.py -v`
Expected: FAIL with `ImportError: cannot import name 'merge_image_paths'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/utilities.py

def merge_image_paths(existing, new):
    """Combine two path lists, drop duplicates (keep first), numeric-aware sort.

    Used to append camera-captured frames to the already-loaded image list.
    """
    seen = set()
    combined = []
    for path in list(existing) + list(new):
        if path not in seen:
            seen.add(path)
            combined.append(path)
    return sorted(combined, key=int_string_sorting)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_utilities.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/utilities.py tests/test_utilities.py
git commit -m "feat(camera): add merge_image_paths utility for additive loading"
```

---

## Task 6: MainWindow — additive `add_captured_image_files`

**Files:**
- Modify: `src/MainWindow/__init__.py` (add method after `remove_some_images`, around line 233)

This wires captured PNG paths into the pipeline **without** clearing existing images or prompting. It mirrors the three registration calls in `set_new_loaded_image_files` (`src/MainWindow/__init__.py:280-286`) but uses `merge_image_paths` instead of replacing. No automated test (requires PySide6, absent in `.venv`); verified manually in Task 11.

- [ ] **Step 1: Add the import**

In `src/MainWindow/__init__.py`, find the existing imports and add (next to the other `from src...` imports):

```python
from src.utilities import merge_image_paths
```

- [ ] **Step 2: Add the method**

Insert immediately after the `remove_some_images` method (after line 232):

```python
    # Add camera-captured frames to the loaded list WITHOUT clearing/prompting.
    def add_captured_image_files(self, new_paths):
        if self.is_stacking:
            self.statusBar().showMessage(
                "Cannot add images while stacking", self.statusbar_msg_display_time
            )
            return
        if not new_paths:
            return

        existing = settings.globalVars.get("LoadedImagePaths", [])
        merged = merge_image_paths(existing, new_paths)

        self.current_image_directory = os.path.dirname(merged[0])
        self._main_content.set_loaded_images(merged)
        self.LaplacianAlgorithm.update_image_paths(merged)
        settings.globalVars["LoadedImagePaths"] = merged
        self._output_exported = False
        self.SettingsWidget._auto_detect_params()
        self.statusBar().showMessage(
            f"Added {len(new_paths)} captured image{'s' if len(new_paths) > 1 else ''}",
            self.statusbar_msg_display_time,
        )
```

- [ ] **Step 3: Verify it imports cleanly (syntax check)**

Run: `.venv/bin/python -c "import ast; ast.parse(open('src/MainWindow/__init__.py').read()); print('OK')"`
Expected: `OK` (full import needs PySide6; syntax check is sufficient here)

- [ ] **Step 4: Commit**

```bash
git add src/MainWindow/__init__.py
git commit -m "feat(camera): add additive add_captured_image_files to MainWindow"
```

---

## Task 7: GUI — `bgr_to_qimage` preview helper

**Files:**
- Create: `src/MainWindow/MainLayout/CameraCaptureWidget.py` (start the file with the pure helper)
- Test: manual (PySide6 absent in `.venv`)

The preview converts OpenCV BGR frames to a `QImage` for display. Isolate the conversion in a module-level function. We use `Format_BGR888`, matching the existing pattern in `src/MainWindow/MainLayout/__init__.py:122-123`.

- [ ] **Step 1: Create the file with the helper**

```python
# src/MainWindow/MainLayout/CameraCaptureWidget.py
"""
Live camera-capture tab: device/resolution selection, preview, single capture,
and timed focus-bracketing. Drives the Qt-free CameraCaptureHandler backend.
"""
import os
import tempfile

import numpy as np
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw
import PySide6.QtGui as qtg

from src.CameraCaptureHandler import CameraCaptureHandler
import src.settings as settings


def bgr_to_qimage(frame):
    """Convert an OpenCV BGR uint8 ndarray to a QImage (Format_BGR888).

    Copies so the QImage owns its buffer (the numpy frame may be reused).
    """
    if frame is None:
        return qtg.QImage()
    h, w = frame.shape[:2]
    contiguous = np.ascontiguousarray(frame)
    qimg = qtg.QImage(contiguous.data, w, h, w * 3, qtg.QImage.Format_BGR888)
    return qimg.copy()
```

- [ ] **Step 2: Manual smoke check (only if PySide6 is available in your env)**

Run:
```bash
QT_QPA_PLATFORM=offscreen python -c "
import numpy as np
from src.MainWindow.MainLayout.CameraCaptureWidget import bgr_to_qimage
img = bgr_to_qimage(np.zeros((48,64,3), dtype=np.uint8))
assert img.width()==64 and img.height()==48, (img.width(), img.height())
print('bgr_to_qimage OK')
"
```
Expected: `bgr_to_qimage OK`. If PySide6 is not installed, skip and rely on Task 11 manual verification.

- [ ] **Step 3: Commit**

```bash
git add src/MainWindow/MainLayout/CameraCaptureWidget.py
git commit -m "feat(camera): add CameraCaptureWidget module with bgr_to_qimage helper"
```

---

## Task 8: GUI — `CameraCaptureWidget` UI and live preview

**Files:**
- Modify: `src/MainWindow/MainLayout/CameraCaptureWidget.py`

Build the widget: device dropdown, "Refresh" button, resolution dropdown, live preview label driven by a `QTimer` (~15 fps), and a "Capture frame" button. Capture writes a lossless PNG to the session temp dir and emits `framesCaptured(list[str])`. Bracketing is added in Task 9.

- [ ] **Step 1: Append the widget class**

```python
# append to src/MainWindow/MainLayout/CameraCaptureWidget.py

# Common video-stream resolutions to offer (descending). The device clamps
# unsupported sizes; we keep only those it actually accepts.
_CANDIDATE_RESOLUTIONS = [
    (3840, 2160), (2592, 1944), (1920, 1080),
    (1280, 720), (1024, 768), (640, 480),
]

_PREVIEW_INTERVAL_MS = 66  # ~15 fps


class CameraCaptureWidget(qtw.QWidget):
    framesCaptured = qtc.Signal(list)  # list of saved PNG paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self.handler = CameraCaptureHandler()
        self._capture_counter = 0

        # --- Controls ---
        self.device_combo = qtw.QComboBox()
        self.refresh_btn = qtw.QToolButton()
        self.refresh_btn.setText("⟳")
        self.refresh_btn.setToolTip("Refresh device list")
        self.resolution_combo = qtw.QComboBox()
        self.start_btn = qtw.QPushButton("Start preview")
        self.start_btn.setCheckable(True)
        self.capture_btn = qtw.QPushButton("Capture frame")
        self.capture_btn.setEnabled(False)

        controls = qtw.QHBoxLayout()
        controls.addWidget(qtw.QLabel("Camera:"))
        controls.addWidget(self.device_combo, 1)
        controls.addWidget(self.refresh_btn)
        controls.addWidget(qtw.QLabel("Resolution:"))
        controls.addWidget(self.resolution_combo)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.capture_btn)

        # --- Preview area ---
        self.preview_label = qtw.QLabel("No preview")
        self.preview_label.setAlignment(qtc.Qt.AlignCenter)
        self.preview_label.setMinimumSize(320, 240)
        self.preview_label.setStyleSheet("background:#111; color:#777;")

        layout = qtw.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.preview_label, 1)
        self.setLayout(layout)

        # --- Timer ---
        self._timer = qtc.QTimer(self)
        self._timer.setInterval(_PREVIEW_INTERVAL_MS)
        self._timer.timeout.connect(self._update_preview)

        # --- Signals ---
        self.refresh_btn.clicked.connect(self.refresh_devices)
        self.start_btn.toggled.connect(self._on_toggle_preview)
        self.capture_btn.clicked.connect(self._on_capture_clicked)
        self.resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)

        self.refresh_devices()

    # ---- device / resolution ----
    def refresh_devices(self):
        self.device_combo.clear()
        for dev in self.handler.enumerate_devices(max_probe=5):
            self.device_combo.addItem(dev["name"], dev["index"])
        if self.device_combo.count() == 0:
            self.device_combo.addItem("No cameras found", -1)

    def _populate_resolutions(self):
        """Probe candidate resolutions on the open device; keep accepted ones."""
        self.resolution_combo.blockSignals(True)
        self.resolution_combo.clear()
        accepted = []
        for (w, h) in _CANDIDATE_RESOLUTIONS:
            actual = self.handler.set_resolution(w, h)
            if actual and actual not in accepted:
                accepted.append(actual)
        for (w, h) in accepted:
            self.resolution_combo.addItem(f"{w} × {h}", (w, h))
        self.resolution_combo.blockSignals(False)
        # Default to the highest accepted resolution
        if accepted:
            self.handler.set_resolution(*accepted[0])

    def _on_resolution_changed(self, _index):
        data = self.resolution_combo.currentData()
        if data:
            self.handler.set_resolution(*data)

    # ---- preview ----
    def _on_toggle_preview(self, checked):
        if checked:
            index = self.device_combo.currentData()
            if index is None or index < 0 or not self.handler.open(index):
                self.start_btn.setChecked(False)
                self.preview_label.setText("Could not open camera")
                return
            self._populate_resolutions()
            self.start_btn.setText("Stop preview")
            self.capture_btn.setEnabled(True)
            self._timer.start()
        else:
            self._timer.stop()
            self.handler.release()
            self.start_btn.setText("Start preview")
            self.capture_btn.setEnabled(False)
            self.preview_label.setText("No preview")

    def _update_preview(self):
        ok, frame = self.handler.read_frame()
        if not ok:
            return
        qimg = bgr_to_qimage(frame)
        pix = qtg.QPixmap.fromImage(qimg).scaled(
            self.preview_label.size(),
            qtc.Qt.KeepAspectRatio,
            qtc.Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(pix)

    # ---- capture ----
    def _temp_dir(self):
        root = settings.globalVars.get("RootTempDir")
        return root.name if root is not None else tempfile.gettempdir()

    def _save_frame(self, frame):
        self._capture_counter += 1
        name = f"capture_{self._capture_counter:04d}.png"
        path = os.path.join(self._temp_dir(), name)
        if CameraCaptureHandler.save_still_lossless(frame, path):
            return path
        return None

    def _on_capture_clicked(self):
        frame = self.handler.capture_still(warmup_frames=2)
        if frame is None:
            return
        path = self._save_frame(frame)
        if path:
            self.framesCaptured.emit([path])

    def closeEvent(self, event):
        self._timer.stop()
        self.handler.release()
        super().closeEvent(event)
```

- [ ] **Step 2: Syntax check**

Run: `.venv/bin/python -c "import ast; ast.parse(open('src/MainWindow/MainLayout/CameraCaptureWidget.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/MainWindow/MainLayout/CameraCaptureWidget.py
git commit -m "feat(camera): live preview, device/resolution selection, single capture"
```

---

## Task 9: GUI — timed focus bracketing

**Files:**
- Modify: `src/MainWindow/MainLayout/CameraCaptureWidget.py`

Add "Capture N frames" with an adjustable delay so the user refocuses between shots. Implemented as a non-blocking `QTimer` sequence (no `time.sleep`, which would freeze the UI). All N paths are emitted together when the sequence finishes.

- [ ] **Step 1: Add bracketing controls to `__init__`**

In `CameraCaptureWidget.__init__`, after the `self.capture_btn` line, add the controls:

```python
        self.bracket_count = qtw.QSpinBox()
        self.bracket_count.setRange(2, 200)
        self.bracket_count.setValue(10)
        self.bracket_count.setToolTip("Number of frames to capture")
        self.bracket_delay = qtw.QDoubleSpinBox()
        self.bracket_delay.setRange(0.0, 30.0)
        self.bracket_delay.setSingleStep(0.5)
        self.bracket_delay.setValue(2.0)
        self.bracket_delay.setSuffix(" s")
        self.bracket_delay.setToolTip("Delay between frames (time to refocus)")
        self.bracket_btn = qtw.QPushButton("Capture N frames")
        self.bracket_btn.setEnabled(False)
```

Then add them to the `controls` layout (after `controls.addWidget(self.capture_btn)`):

```python
        controls.addWidget(qtw.QLabel("N:"))
        controls.addWidget(self.bracket_count)
        controls.addWidget(qtw.QLabel("Delay:"))
        controls.addWidget(self.bracket_delay)
        controls.addWidget(self.bracket_btn)
```

And initialize bracketing state + signal at the end of `__init__` (before `self.refresh_devices()`):

```python
        self._bracket_remaining = 0
        self._bracket_paths = []
        self._bracket_timer = qtc.QTimer(self)
        self._bracket_timer.setSingleShot(True)
        self._bracket_timer.timeout.connect(self._bracket_step)
        self.bracket_btn.clicked.connect(self._start_bracketing)
```

- [ ] **Step 2: Enable/disable bracket button with preview**

In `_on_toggle_preview`, set `self.bracket_btn.setEnabled(True)` where `self.capture_btn.setEnabled(True)` is, and `self.bracket_btn.setEnabled(False)` where it is disabled.

- [ ] **Step 3: Add bracketing methods**

```python
# add inside CameraCaptureWidget in src/MainWindow/MainLayout/CameraCaptureWidget.py

    def _start_bracketing(self):
        if self._bracket_remaining > 0 or not self.handler.is_opened:
            return
        self._bracket_remaining = self.bracket_count.value()
        self._bracket_paths = []
        self._set_bracket_ui_running(True)
        self._bracket_step()  # first frame immediately

    def _bracket_step(self):
        frame = self.handler.capture_still(warmup_frames=2)
        path = self._save_frame(frame) if frame is not None else None
        if path:
            self._bracket_paths.append(path)
        self._bracket_remaining -= 1

        if self._bracket_remaining > 0:
            captured = len(self._bracket_paths)
            total = self.bracket_count.value()
            self.bracket_btn.setText(f"Capturing {captured}/{total}…")
            self._bracket_timer.start(int(self.bracket_delay.value() * 1000))
        else:
            self._finish_bracketing()

    def _finish_bracketing(self):
        self._set_bracket_ui_running(False)
        if self._bracket_paths:
            self.framesCaptured.emit(list(self._bracket_paths))
        self._bracket_paths = []

    def _set_bracket_ui_running(self, running):
        self.bracket_btn.setText("Capture N frames" if not running else "Capturing…")
        self.bracket_btn.setEnabled(not running)
        self.capture_btn.setEnabled(not running and self.handler.is_opened)
        self.bracket_count.setEnabled(not running)
        self.bracket_delay.setEnabled(not running)
```

- [ ] **Step 4: Stop bracketing on close**

In `closeEvent`, add before `self.handler.release()`:

```python
        self._bracket_timer.stop()
        self._bracket_remaining = 0
```

- [ ] **Step 5: Syntax check**

Run: `.venv/bin/python -c "import ast; ast.parse(open('src/MainWindow/MainLayout/CameraCaptureWidget.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add src/MainWindow/MainLayout/CameraCaptureWidget.py
git commit -m "feat(camera): add non-blocking timed focus bracketing"
```

---

## Task 10: Integrate the Capture tab into CenterWidget

**Files:**
- Modify: `src/MainWindow/MainLayout/__init__.py`

Add the "Capture" tab next to View/Compare and route `framesCaptured` to `MainWindow.add_captured_image_files`.

- [ ] **Step 1: Import the widget**

At the top of `src/MainWindow/MainLayout/__init__.py`, with the other `from src.MainWindow.MainLayout...` imports (after line 16):

```python
from src.MainWindow.MainLayout.CameraCaptureWidget import CameraCaptureWidget
```

- [ ] **Step 2: Create the tab in `CenterWidget.__init__`**

In `CenterWidget.__init__`, after `self.ComparisonViewer = ComparisonWidget()` (line 46), add:

```python
        self.CameraCaptureViewer = CameraCaptureWidget()
```

Then after the existing `self.tabWidget.addTab(self.ComparisonViewer, "Compare")` (line 64), add:

```python
        self.tabWidget.addTab(self.CameraCaptureViewer, "Capture")
        self.CameraCaptureViewer.framesCaptured.connect(self._on_frames_captured)
```

- [ ] **Step 3: Add the routing slot to `CenterWidget`**

Add this method to the `CenterWidget` class (e.g. after `set_loaded_images`):

```python
    def _on_frames_captured(self, paths):
        """Route camera-captured frame paths into the main loaded-image list."""
        main_window = settings.globalVars.get("MainWindow")
        if main_window is not None and paths:
            main_window.add_captured_image_files(paths)
```

- [ ] **Step 4: Syntax check**

Run: `.venv/bin/python -c "import ast; ast.parse(open('src/MainWindow/MainLayout/__init__.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 5: Run the full headless test suite (no regressions)**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass (existing + new backend/utility tests; GUI modules are not imported by tests).

- [ ] **Step 6: Commit**

```bash
git add src/MainWindow/MainLayout/__init__.py
git commit -m "feat(camera): add Capture tab and wire frames into loaded images"
```

---

## Task 11: Manual verification (real hardware)

**Files:** none (verification only)

Requires a machine with PySide6 installed and a USB camera/webcam connected.

- [ ] **Step 1: Launch the app**

Run: `python -m src.run` (or the installed `chimpstackr` entry point) on a machine with PySide6 + a camera.

- [ ] **Step 2: Verify the capture flow**

- Open the **Capture** tab.
- Confirm the camera appears in the **Camera** dropdown (use ⟳ if not).
- Click **Start preview** → live video appears.
- Confirm the **Resolution** dropdown lists sizes and changing it changes the stream.
- Click **Capture frame** → a `capture_0001.png` appears in the **Source images** list and is selectable in the View tab.
- Set **N = 5**, **Delay = 1.0 s**, click **Capture N frames** → button shows progress; after completion 5 frames are added to the list.
- Run **Align & Stack** on the captured frames → confirm a stacked output is produced.

- [ ] **Step 3: Verify cleanup**

- Switch away from the Capture tab / close the app → no crash, camera light turns off (device released).

- [ ] **Step 4: Record results**

Note the camera model, OS, and observed max resolution in the PR description.

---

## Task 12: Flatpak — grant camera device access

**Files:**
- Modify: `packaging/flatpak/io.github.noah_peeters.ChimpStackr.yml`

The Flatpak sandbox currently grants only `--device=dri`; without video-device access the camera feature can't see any camera inside the Flatpak build.

- [ ] **Step 1: Add the device permission**

In the `finish-args` list, alongside the existing `--device=dri`, add:

```yaml
  - --device=all
```

(Rationale: `--device=all` covers USB cameras across kernels portably. `--device=/dev/video*` is narrower but less reliable across portals; since this is a self-hosted artifact, not a Flathub submission, `--device=all` is acceptable. Adjust to taste.)

- [ ] **Step 2: Verify YAML parses**

Run: `.venv/bin/python -c "import yaml; yaml.safe_load(open('packaging/flatpak/io.github.noah_peeters.ChimpStackr.yml')); print('YAML OK')"`
Expected: `YAML OK` (if `pyyaml` isn't present, visually confirm indentation instead).

- [ ] **Step 3: Commit**

```bash
git add packaging/flatpak/io.github.noah_peeters.ChimpStackr.yml
git commit -m "build(flatpak): grant device access for USB camera capture"
```

---

## Task 13: macOS — camera permission

**Files:**
- Modify: `packaging/entitlements.plist`
- Modify: `chimpstackr.spec` (PyInstaller `BUNDLE` `info_plist`)

macOS refuses camera access without a usage-description string and (for sandboxed/hardened builds) the camera entitlement. Without these the app crashes or silently gets no frames on first capture.

- [ ] **Step 1: Add the camera entitlement**

In `packaging/entitlements.plist`, inside the top-level `<dict>`, add:

```xml
    <key>com.apple.security.device.camera</key>
    <true/>
```

- [ ] **Step 2: Add the usage description to the app bundle**

In `chimpstackr.spec`, locate the macOS `BUNDLE(...)` call and add/extend its `info_plist` dict with:

```python
    info_plist={
        'NSCameraUsageDescription':
            'ChimpStackr uses your camera to capture images for focus stacking.',
    },
```

(If `BUNDLE` already has an `info_plist`, merge this key in rather than replacing it.)

- [ ] **Step 3: Verify the plist parses**

Run: `plutil -lint packaging/entitlements.plist`
Expected: `packaging/entitlements.plist: OK` (macOS only; skip on other platforms).

- [ ] **Step 4: Commit**

```bash
git add packaging/entitlements.plist chimpstackr.spec
git commit -m "build(macos): add camera entitlement and usage description"
```

---

## Final verification

- [ ] Run the complete headless suite once more:

Run: `.venv/bin/python -m pytest -q`
Expected: all green.

- [ ] Confirm no PySide6 import leaked into `src/CameraCaptureHandler.py` or `src/utilities.py`:

Run: `grep -L PySide6 src/CameraCaptureHandler.py src/utilities.py && echo "clean"`
Expected: both files listed, then `clean`.

- [ ] Manual hardware verification (Task 11) completed and results recorded.

---

## Self-Review notes (author)

- **Spec coverage:** UVC capture ✅ (Tasks 1-3), live preview ✅ (Task 8), lossless save ✅ (Task 4), focus bracketing ✅ (Task 9), GUI integration ✅ (Task 10), cross-platform packaging ✅ (Tasks 12-13). Full-res still-pin **deliberately deferred** per research — documented in Scope.
- **Type consistency:** `enumerate_devices` → `[{"index", "name"}]` consumed via `currentData()` (index) in Task 8; `set_resolution`/`get_resolution`/`request_max_resolution` → `(w, h)` tuples; `save_still_lossless` is a staticmethod used in both single (Task 8) and bracket (Task 9) capture; `framesCaptured` always emits `list[str]`; `add_captured_image_files(list)` ← matches.
- **No-hardware testability:** All backend tests use `FakeCapture`/`ClampingCapture` via the injected `capture_factory`. The only real I/O test is the PNG lossless round-trip (Task 4), which needs no camera.
