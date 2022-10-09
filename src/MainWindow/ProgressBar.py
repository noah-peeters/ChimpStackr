"""
    Class that houses a QWidget displaying a progressbar + progress text.
    Progressbar is used to display progress on a task;
    Progress text is used to display "time left until completed" messages.
"""
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw


class ProgressBar(qtw.QWidget):
    anim_duration = 500  # In ms

    def __init__(self):
        super().__init__()

        # Setup layout
        layout = qtw.QHBoxLayout(self)

        self.progressbar = qtw.QProgressBar(self)
        self.progressbar.setFormat("%p%")
        self.progressbar.setTextVisible(True)
        self.progressbar.setAlignment(qtc.Qt.AlignCenter)
        layout.addWidget(self.progressbar)

        self.progress_label = qtw.QLabel("Please wait while the task is beginning...")
        layout.addWidget(self.progress_label)

        self.setVisible(False)
        self.setLayout(layout)

    # Update progressbar value and/or label text
    def update_value(self, value=None, text=None):
        """
        Update progressbar value and/or label text.
        Will smoothly animate progressbar slider.

        If both values are none (default), this widget will be hidden.
        """
        if value:
            # Smoothly animate progressbar movement
            if hasattr(self, "animation"):
                self.animation.stop()
            else:
                self.animation = qtc.QPropertyAnimation(
                    targetObject=self.progressbar, propertyName=b"value"
                )
                self.animation.setDuration(self.anim_duration)
                self.animation.setEasingCurve(qtc.QEasingCurve.OutQuad)
            self.animation.setEndValue(value)
            self.animation.start()
        if text:
            self.progress_label.setText(text)

        if not value and not text:
            # Hide and reset
            self.setVisible(False)
            self.progressbar.reset()
            self.progress_label.setText("Please wait while the task is beginning...")
