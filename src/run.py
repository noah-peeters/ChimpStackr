import os, sys, tempfile, logging

# Enable logging so GPU timing info is visible
logging.basicConfig(
    level=logging.INFO,
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

    # Build QIcon with all available sizes for sharpest rendering everywhere
    icon = qtg.QIcon()
    icons_dir = os.path.dirname(icon_path)

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

    # macOS: override the dock icon (Python scripts show the generic Python icon)
    if sys.platform == "darwin":
        try:
            from AppKit import NSImage, NSApplication
            ns_image = NSImage.alloc().initWithContentsOfFile_(icon_path)
            if ns_image:
                NSApplication.sharedApplication().setApplicationIconImage_(ns_image)
        except ImportError:
            pass  # pyobjc not installed

    # Linux/Wayland: set desktopFileName so compositors match the .desktop entry
    if sys.platform == "linux":
        qApp.setDesktopFileName(APP_ID)


def main():
    qApp = qtw.QApplication([])
    qApp.setApplicationName("chimpstackr")
    qApp.setOrganizationName("noah.peeters")
    qApp.setOrganizationDomain("noah.peeters")
    qApp.setApplicationDisplayName("ChimpStackr")
    settings.globalVars["QSettings"] = qtc.QSettings()
    settings.globalVars["MainApplication"] = qApp

    icon_path = _find_icon_path()

    window = MainWindow.Window()

    from src.MainWindow.style import get_stylesheet
    qApp.setStyleSheet(get_stylesheet())

    _apply_app_icon(qApp, icon_path)
    # Also set on the window directly (Windows taskbar needs this)
    if icon_path and os.path.isfile(icon_path):
        window.setWindowIcon(qApp.windowIcon())

    window.showMaximized()

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
