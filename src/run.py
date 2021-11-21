import sys, tempfile
import PySide6.QtWidgets as qtw

from MainWindow.MainWindow import Window

# Directory for storing tempfiles. Automatically deletes on program exit.
ROOT_TEMP_DIRECTORY = tempfile.TemporaryDirectory(prefix="FocusStacking_")

if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    window = Window(ROOT_TEMP_DIRECTORY)

    window.show()
    sys.exit(app.exec())

ROOT_TEMP_DIRECTORY.cleanup()
