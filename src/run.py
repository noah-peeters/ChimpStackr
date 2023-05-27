import os, sys, tempfile, argparse
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw
import PySide6.QtGui as qtg

# Allow imports from top level folder. Example: "src.algorithm.API"
# src: https://codeolives.com/2020/01/10/python-reference-module-in-parent-directory/
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)  # Insert at first place


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def main_ui():
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

    qApp = qtw.QApplication([])
    # Needed for saving QSettings
    qApp.setApplicationName("ChimpStackr")
    qApp.setOrganizationName("noah.peeters")
    # Uses names of QApplication (above)
    settings.globalVars["QSettings"] = qtc.QSettings()
    settings.globalVars["MainApplication"] = qApp

    # Get icon for Windows/Mac (PyInstaller) or source code run
    icon_path = resource_path("packaging/icons/icon_512x512.png")
    if not os.path.isfile(icon_path):
        # Path to icon inside AppImage
        icon_path = "icon_128x128.png"

    window = MainWindow.Window()

    icon = qtg.QIcon(icon_path)
    qApp.setWindowIcon(icon)

    window.showMaximized()
    qApp.exec()


def main_cmd_line():
    import src.command_line as cmd_line

    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("-i", "--img_dir", help="input images path", required=True)
    parser.add_argument("-o", "--out_path", help="output path", required=False)
    args = vars(parser.parse_args())

    cmd_line.do_focus_stack(img_path=args["img_dir"], out_path=args["out_path"])


if __name__ == "__main__":
    if len(sys.argv) - 1 <= 0:  # First parameter is this file name
        main_ui()  # Show UI when no cmdline arguments passed
    else:
        main_cmd_line()  # Don't show UI when no cmdline arguments passed
