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
