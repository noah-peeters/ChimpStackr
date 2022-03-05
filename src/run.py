import os, sys, tempfile
import PySide6.QtWidgets as qtw
import PySide6.QtGui as qtg

# Allow imports from top level folder. Example: "src.algorithm.API"
# src: https://codeolives.com/2020/01/10/python-reference-module-in-parent-directory/
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)  # Insert at first place

import src.settings as settings
import src.MainWindow.MainWindow as MainWindow

# Directory for storing tempfiles. Automatically deletes on program exit.
ROOT_TEMP_DIRECTORY = tempfile.TemporaryDirectory(prefix="ChimpStackr_")
settings.init()
settings.globalVars["RootTempDir"] = ROOT_TEMP_DIRECTORY


# Taskbar icon fix for Windows 7
# Src: https://stackoverflow.com/questions/1551605/how-to-set-applications-taskbar-icon-in-windows-7S
if os.name == "nt":
    print("Fix Windows taskbar icon")
    import ctypes

    myappid = "test.application"  # arbitrary string
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass  # Platform older than Windows 7


def main():
    app = qtw.QApplication([])
    # Needed for saving QSettings
    app.setApplicationName("ChimpStackr")
    app.setOrganizationName("noah.peeters")

    settings.globalVars["MainApplication"] = app

    window = MainWindow.Window()
    window.showMaximized()

    print(os.path.expanduser("~"))
    icon_path = "snap/gui/icon.png"
    if not os.path.isfile(icon_path):
        # Path to icon inside snap package
        icon_path = "meta/gui/icon.png"
    
    icon = qtg.QIcon(icon_path)
    app.setWindowIcon(icon)
    window.setWindowIcon(icon)
    app.exec()
    # sys.exit(app.exec())
    ROOT_TEMP_DIRECTORY.cleanup()


if __name__ == "__main__":
    main()
