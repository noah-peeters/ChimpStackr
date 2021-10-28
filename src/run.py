import sys
import PySide6.QtWidgets as qtw
from qt_material import apply_stylesheet

from MainWindow.MainWindow import Window


if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    window = Window()

    # TODO: Make setting toggle that saves stylesheet
    # Setup stylesheet
    apply_stylesheet(app, "dark_amber.xml")

    window.show()
    sys.exit(app.exec())