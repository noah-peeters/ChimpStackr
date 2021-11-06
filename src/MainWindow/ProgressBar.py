"""
    Class that houses a QWidget displaying a progressbar + progress text.
    Progressbar is used to display progress on a task;
    Progress text is used to display "time left until completed" messages.
"""
import PySide6.QtWidgets as qtw

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