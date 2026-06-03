"""
Live camera-capture tab: device/resolution selection, preview, single capture,
and timed focus-bracketing. Drives the Qt-free CameraCaptureHandler backend.

Note: frame capture (capture_still + lossless save) runs synchronously on the UI
thread. A single capture is a one-shot ~100 ms stall on an explicit click, and in
bracketing it is dwarfed by the user-set inter-frame delay, so this is acceptable
for v1. If capture is ever moved to a worker thread, CameraCaptureHandler must
gain a lock guarding _cap access (the preview timer reads the same device).
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
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        # Format_BGR888 assumes a 3-channel uint8 buffer with stride w*3.
        return qtg.QImage()
    h, w = frame.shape[:2]
    contiguous = np.ascontiguousarray(frame)
    qimg = qtg.QImage(contiguous.data, w, h, w * 3, qtg.QImage.Format_BGR888)
    return qimg.copy()


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
        self._enumerated_once = False

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

        controls = qtw.QHBoxLayout()
        controls.addWidget(qtw.QLabel("Camera:"))
        controls.addWidget(self.device_combo, 1)
        controls.addWidget(self.refresh_btn)
        controls.addWidget(qtw.QLabel("Resolution:"))
        controls.addWidget(self.resolution_combo)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.capture_btn)
        controls.addWidget(qtw.QLabel("N:"))
        controls.addWidget(self.bracket_count)
        controls.addWidget(qtw.QLabel("Delay:"))
        controls.addWidget(self.bracket_delay)
        controls.addWidget(self.bracket_btn)

        # --- Preview area ---
        self.preview_label = qtw.QLabel("No preview")
        self.preview_label.setAlignment(qtc.Qt.AlignCenter)
        self.preview_label.setMinimumSize(320, 240)
        self.preview_label.setStyleSheet("background:#111; color:#777;")

        layout = qtw.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.preview_label, 1)

        # --- Timer ---
        self._timer = qtc.QTimer(self)
        self._timer.setInterval(_PREVIEW_INTERVAL_MS)
        self._timer.timeout.connect(self._update_preview)

        # --- Bracketing state ---
        self._bracket_remaining = 0
        self._bracket_paths = []
        self._bracket_timer = qtc.QTimer(self)
        self._bracket_timer.setSingleShot(True)
        self._bracket_timer.timeout.connect(self._bracket_step)

        # --- Signals ---
        self.refresh_btn.clicked.connect(self.refresh_devices)
        self.start_btn.toggled.connect(self._on_toggle_preview)
        self.capture_btn.clicked.connect(self._on_capture_clicked)
        self.resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.bracket_btn.clicked.connect(self._start_bracketing)

        # Enumeration is deferred to first show (probing camera indices can block
        # for seconds, so we must not do it during app startup).
        self.device_combo.addItem("Click ⟳ to detect cameras", -1)

    def showEvent(self, event):
        if not self._enumerated_once:
            self._enumerated_once = True
            self.refresh_devices()
        super().showEvent(event)

    # ---- device / resolution ----
    def refresh_devices(self):
        # Stop any running preview before re-enumerating (frees the open device).
        if self.handler.is_opened:
            self.start_btn.setChecked(False)
        self.device_combo.clear()
        for dev in self.handler.enumerate_devices(max_probe=5):
            self.device_combo.addItem(dev["name"], dev["index"])
        if self.device_combo.count() == 0:
            self.device_combo.addItem("No cameras found", -1)

    def _on_device_changed(self, _index):
        # Switching device while previewing: stop so the user restarts cleanly.
        if self.handler.is_opened:
            self.start_btn.setChecked(False)

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
        # Default to the highest accepted resolution; if probing yielded nothing
        # usable, fall back to the device's native maximum.
        if accepted:
            self.handler.set_resolution(*accepted[0])
        else:
            self.handler.request_max_resolution()

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
            self.bracket_btn.setEnabled(True)
            self._timer.start()
        else:
            self._timer.stop()
            self.handler.release()
            self.start_btn.setText("Start preview")
            self.capture_btn.setEnabled(False)
            self.bracket_btn.setEnabled(False)
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
        # Unique temp file per capture (mirrors add_processed_image); avoids
        # stale-counter collisions when the temp root is recreated.
        file_handle, path = tempfile.mkstemp(suffix=".png", dir=self._temp_dir())
        os.close(file_handle)
        if CameraCaptureHandler.save_still_lossless(frame, path):
            return path
        try:
            os.remove(path)  # clean up the empty placeholder on failure
        except OSError:
            pass
        return None

    def _on_capture_clicked(self):
        frame = self.handler.capture_still(warmup_frames=2)
        if frame is None:
            return
        path = self._save_frame(frame)
        if path:
            self.framesCaptured.emit([path])

    # ---- bracketing ----
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
            total = self.bracket_count.value()
            done = total - self._bracket_remaining  # frames attempted so far
            self.bracket_btn.setText(f"Capturing {done}/{total}…")
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

    def closeEvent(self, event):
        self._bracket_timer.stop()
        self._bracket_remaining = 0
        self._timer.stop()
        self.handler.release()
        super().closeEvent(event)
