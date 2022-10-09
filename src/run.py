import os, sys, tempfile
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw
import PySide6.QtGui as qtg

# Allow imports from top level folder. Example: "src.algorithm.API"
# src: https://codeolives.com/2020/01/10/python-reference-module-in-parent-directory/
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)  # Insert at first place

import src.settings as settings
import src.MainWindow as MainWindow

# Directory for storing tempfiles. Automatically deletes on program exit.
ROOT_TEMP_DIRECTORY = tempfile.TemporaryDirectory(prefix="ChimpStackr_")
settings.init()
settings.globalVars["RootTempDir"] = ROOT_TEMP_DIRECTORY


# Taskbar icon fix for Windows
# Src: https://stackoverflow.com/questions/1551605/how-to-set-applications-taskbar-icon-in-windows-7S
if os.name == "nt":
    import ctypes

    myappid = "test.application"  # arbitrary string
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass  # Platform older than Windows 7


class CustomSplashScreen(qtw.QSplashScreen):
    def __init__(self, my_pixmap):
        super().__init__(my_pixmap)
        self.setWindowFlags(qtc.Qt.WindowStaysOnTopHint | qtc.Qt.FramelessWindowHint)

    def mousePressEvent(self, event: qtg.QMouseEvent) -> None:
        # Disable default "click-to-dismiss" behaviour
        event.accept()
        return


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def main():
    qApp = qtw.QApplication([])
    # Needed for saving QSettings
    qApp.setApplicationName("ChimpStackr")
    qApp.setOrganizationName("noah.peeters")
    # Uses names of QApplication (above)
    settings.globalVars["QSettings"] = qtc.QSettings()
    settings.globalVars["MainApplication"] = qApp

    # Get icon for Windows/Mac (PyInstaller) or source code run
    icon_path = resource_path("snap/gui/icon.png")
    if not os.path.isfile(icon_path):
        # Path to icon inside snap package
        icon_path = "meta/gui/icon.png"

    # Setup splashscreen
    splash = CustomSplashScreen(qtg.QPixmap(icon_path))
    splash.show()

    window = MainWindow.Window()

    icon = qtg.QIcon(icon_path)
    qApp.setWindowIcon(icon)

    window.showMaximized()
    splash.finish(window)

    try:
        qApp.exec()
        sys.exit()
    finally:
        # Cleanup temporary diretory, even if an exception occured in the app
        ROOT_TEMP_DIRECTORY.cleanup()


if __name__ == "__main__":
    main()
