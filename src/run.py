import sys
import PySide6.QtWidgets as qtw

from MainWindow.MainWindow import Window


if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    window = Window()

    window.show()
    sys.exit(app.exec())
