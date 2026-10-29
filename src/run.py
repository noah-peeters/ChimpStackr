import os
import sys
import tempfile
import logging
import multiprocessing

# Frozen builds: multiprocessing helpers (e.g. the resource tracker Python 3.14
# starts for numba's parallel lock) re-run this executable. Let them do their
# job and exit instead of opening another copy of the app (issue #179).
if __name__ == "__main__":
    multiprocessing.freeze_support()

# Enable OpenEXR support in OpenCV (must be set before cv2 import)
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

# Only show warnings+ in production; set CHIMPSTACKR_DEBUG=1 for full logs
_log_level = logging.DEBUG if os.environ.get("CHIMPSTACKR_DEBUG") else logging.WARNING
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw
import PySide6.QtGui as qtg

# Allow imports from top level folder
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import src.settings as settings
import src.MainWindow as MainWindow

# Directory for storing tempfiles. Automatically deletes on program exit.
ROOT_TEMP_DIRECTORY = tempfile.TemporaryDirectory(prefix="ChimpStackr_")
settings.init()
settings.globalVars["RootTempDir"] = ROOT_TEMP_DIRECTORY

APP_ID = "noah.peeters.chimpstackr"
# Basename of the .desktop entry shipped in the AppImage (chimpstackr.desktop)
DESKTOP_FILE_NAME = "chimpstackr"


def _setup_platform_icon():
    """Platform-specific taskbar/dock icon registration. Must be called before QApplication."""
    if sys.platform == "win32":
        # Windows: set AppUserModelID so taskbar groups under our icon, not Python's
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except Exception:
            pass
    elif sys.platform == "linux":
        # Linux/X11/Wayland: set WM_CLASS and desktop entry name
        # Qt uses applicationName for WM_CLASS; also set desktopFileName
        # so Wayland compositors match the .desktop entry
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")  # hint, not forced


_setup_platform_icon()


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        base_path = sys._MEIPASS  # PyInstaller bundle
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def _find_icon_path():
    """Find the best icon file for the current platform."""
    icons_dir = resource_path("packaging/icons")

    # Windows: prefer .ico (multi-size, includes 16/32/48/256)
    if sys.platform == "win32":
        ico = os.path.join(icons_dir, "icon.ico")
        if os.path.isfile(ico):
            return ico

    # All platforms: prefer largest PNG for best quality
    for name in ["chimpstackr_icon.png", "icon_512x512.png", "icon_256x256.png", "icon_128x128.png"]:
        p = os.path.join(icons_dir, name)
        if os.path.isfile(p):
            return p

    # AppImage fallback
    fallback = "icon_128x128.png"
    if os.path.isfile(fallback):
        return fallback

    return None


def _apply_app_icon(qApp, icon_path):
    """Set the application icon on all platforms."""
    if not icon_path or not os.path.isfile(icon_path):
        return
    icons_dir = os.path.dirname(icon_path)
    settings.globalVars["AppIconPath"] = icon_path

    if sys.platform == "darwin":
        # The .app's .icns (on Apple's icon grid) is the Dock icon; setting a
        # window icon would replace it at runtime. From source, use the
        # macOS-shaped PNG so the Dock doesn't show Python's icon.
        if getattr(sys, "frozen", False):
            return
        mac_png = os.path.join(icons_dir, "chimpstackr_icon_macos.png")
        qApp.setWindowIcon(qtg.QIcon(mac_png if os.path.isfile(mac_png) else icon_path))
        return

    # Build QIcon with all available sizes for sharpest rendering everywhere
    icon = qtg.QIcon()
    # On Windows, add .ico first (contains all sizes for taskbar/titlebar)
    if sys.platform == "win32":
        ico = os.path.join(icons_dir, "icon.ico")
        if os.path.isfile(ico):
            icon.addFile(ico)
    for name in ["icon_128x128.png", "icon_256x256.png", "icon_512x512.png", "chimpstackr_icon.png"]:
        p = os.path.join(icons_dir, name)
        if os.path.isfile(p):
            icon.addFile(p)
    if icon.isNull():
        icon = qtg.QIcon(icon_path)
    qApp.setWindowIcon(icon)

    # Linux/Wayland: set desktopFileName so compositors match the .desktop entry
    if sys.platform == "linux":
        qApp.setDesktopFileName(DESKTOP_FILE_NAME)


class _App(qtw.QApplication):
    """Also takes files opened with the app (Finder "Open With", drops on
    the Dock icon), which macOS sends as FileOpen events."""

    def __init__(self, argv):
        super().__init__(argv)
        self.window = None
        self.pending = []

    def event(self, ev):
        if ev.type() == qtc.QEvent.FileOpen and ev.file():
            if self.window is not None:
                self.window.add_paths([ev.file()])
            else:
                self.pending.append(ev.file())
            return True
        return super().event(ev)


def main():
    qApp = _App(sys.argv[:1])
    qApp.setApplicationName("chimpstackr")
    qApp.setOrganizationName("noah.peeters")
    qApp.setOrganizationDomain("noah.peeters")
    qApp.setApplicationDisplayName("ChimpStackr")
    settings.globalVars["QSettings"] = qtc.QSettings()
    settings.globalVars["MainApplication"] = qApp

    import src.MainWindow.theme as theme
    theme.install(qApp, str(settings.globalVars["QSettings"].value("ui/theme") or "system"))

    _apply_app_icon(qApp, _find_icon_path())

    window = MainWindow.Window()
    # Also set on the window directly (Windows taskbar needs this)
    if not qApp.windowIcon().isNull():
        window.setWindowIcon(qApp.windowIcon())
    window.show()

    # Files passed on the command line or opened with the app ("Open With")
    qApp.window = window
    window.updates.start_automatic()
    window.measurer.schedule()      # GPU vs CPU speed, in the background when idle
    args = [a for a in sys.argv[1:] if os.path.exists(a)] + qApp.pending
    if args:
        window.add_paths(args)

    # Warm up Numba JIT in background (prevents lag on first stack)
    def _warmup_jit():
        try:
            import numpy as np
            from src.algorithms.stacking_algorithms import cpu as CPU
            dummy = np.zeros((8, 8, 3), dtype=np.float32)
            pyr = CPU.generate_laplacian_pyramid(dummy, 2)
            gray = np.zeros((4, 4), dtype=np.float32)
            CPU.compute_focusmap(gray, gray, 2)
            CPU.fuse_pyramid_levels_using_focusmap(pyr[0], pyr[0].copy(), np.zeros((2, 3), dtype=np.uint8))
        except Exception:
            pass

    import threading
    threading.Thread(target=_warmup_jit, daemon=True).start()

    qApp.exec()


if __name__ == "__main__":
    main()
