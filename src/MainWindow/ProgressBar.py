"""
    Class that houses a QWidget displaying a progressbar + progress text.
"""
import PySide6.QtWidgets as qtw
import PySide6.QtGui as qtg

class ProgressBar(qtw.QWidget):
    def __init__(self):
        super().__init__()

        # Setup layout
        layout = qtw.QHBoxLayout(self)

        self.progressbar = qtw.QProgressBar(self)
        self.progressbar.setVisible(True)
        layout.addWidget(self.progressbar)

        self.progress_label = qtw.QLabel("This is a progress text: 4/5")
        layout.addWidget(self.progress_label)

        self.setLayout(layout)