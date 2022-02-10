import os, sys, tempfile
import PySide6.QtWidgets as qtw
import PySide6.QtGui as qtg

# Hack to allow imports from src/ example: "src.algorithm.API"
# src: https://codeolives.com/2020/01/10/python-reference-module-in-parent-directory/
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from src.MainWindow.MainWindow import Window
import src.settings as settings

# Directory for storing tempfiles. Automatically deletes on program exit.
ROOT_TEMP_DIRECTORY = tempfile.TemporaryDirectory(prefix="FocusStacking_")
settings.init()
settings.globalVars["RootTempDir"] = ROOT_TEMP_DIRECTORY


# Taskbar icon fix for Windows 7
# Src: https://stackoverflow.com/questions/1551605/how-to-set-applications-taskbar-icon-in-windows-7S
try:
    import ctypes
    myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except Exception:
    print("Don't fix Windows 7 icon")

if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    # Needed for saving QSettings
    app.setApplicationName("ChimpStackr")
    app.setOrganizationName("noah.peeters")
    app.setWindowIcon(qtg.QIcon("snap/gui/icon.png"))

    settings.globalVars["MainApplication"] = app

    window = Window()
    window.showMaximized()

    sys.exit(app.exec())

ROOT_TEMP_DIRECTORY.cleanup()
