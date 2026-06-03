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

# Cache cv2 CAP_PROP constants at module level
_CAP_PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
_CAP_PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT

# Oversized sentinel: drivers clamp a too-large request to their real maximum
_OVERSIZED_RESOLUTION = 100000


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
            except Exception:
                logger.debug("Error probing camera index %s", index, exc_info=True)
            finally:
                cap.release()
        return found

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
                logger.warning("Error releasing camera device", exc_info=True)
        self._cap = None
        self._index = None

    def set_resolution(self, width, height):
        """Request a capture resolution; return the (w, h) actually applied."""
        if not self.is_opened:
            return None
        self._cap.set(_CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(_CAP_PROP_FRAME_HEIGHT, height)
        return self.get_resolution()

    def request_max_resolution(self):
        """Request an oversized resolution so the driver clamps to its real max."""
        return self.set_resolution(_OVERSIZED_RESOLUTION, _OVERSIZED_RESOLUTION)

    def get_resolution(self):
        """Return the current (width, height) as ints."""
        if not self.is_opened:
            return None
        w = int(round(self._cap.get(_CAP_PROP_FRAME_WIDTH)))
        h = int(round(self._cap.get(_CAP_PROP_FRAME_HEIGHT)))
        return (w, h)

    @staticmethod
    def save_still_lossless(frame, path):
        """Write a BGR frame to `path` losslessly (PNG). Returns True on success.

        `path` may be a str or os.PathLike; it is coerced to str for cv2.
        """
        if frame is None:
            return False
        try:
            # PNG compression level 1 = fast, still lossless
            return bool(cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]))
        except Exception as e:
            logger.error("Failed to save still %s: %s", path, e)
            return False
