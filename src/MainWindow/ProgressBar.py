"""
    Progress bar widget with task description, animated bar, time remaining, and cancel button.
"""
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw

import src.settings as settings


class ProgressBar(qtw.QWidget):
    anim_duration = 500  # In ms

    def __init__(self):
        super().__init__()

        main_layout = qtw.QVBoxLayout(self)
        main_layout.setContentsMargins(8, 4, 8, 4)
        main_layout.setSpacing(4)

        # Task description
        self.task_label = qtw.QLabel("Processing...")
        self.task_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        main_layout.addWidget(self.task_label)

        # Progress bar
        self.progressbar = qtw.QProgressBar(self)
        self.progressbar.setFormat("%p%")
        self.progressbar.setTextVisible(True)
        self.progressbar.setAlignment(qtc.Qt.AlignCenter)
        self.progressbar.setFixedHeight(22)
        main_layout.addWidget(self.progressbar)

        # Bottom row: time remaining + cancel button
        bottom_layout = qtw.QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        self.progress_label = qtw.QLabel("Starting...")
        self.progress_label.setStyleSheet("color: #999999; font-size: 11px;")
        bottom_layout.addWidget(self.progress_label)

        bottom_layout.addStretch()

        self.cancel_btn = qtw.QPushButton("Cancel")
        self.cancel_btn.setFixedWidth(70)
        self.cancel_btn.setStyleSheet(
            "QPushButton { background: #ff6961; color: white; border: none; "
            "border-radius: 4px; padding: 3px 8px; font-size: 11px; font-weight: 600; }"
            "QPushButton:hover { background: #ff8580; }"
        )
        self.cancel_btn.clicked.connect(self._on_cancel)
        bottom_layout.addWidget(self.cancel_btn)

        main_layout.addLayout(bottom_layout)

        self.setVisible(False)
        self.setLayout(main_layout)

    def _on_cancel(self):
        try:
            settings.globalVars["MainWindow"].cancel_stacking()
        except (KeyError, AttributeError):
            pass

    def update_value(self, value=None, text=None):
        """
        Update progressbar value and/or label text.
        Smoothly animates the progressbar slider.
        If both values are None (default), this widget will be hidden.
        """
        if value:
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
            self.setVisible(False)
            self.progressbar.reset()
            self.progress_label.setText("Starting...")
            self.task_label.setText("Processing...")

    def set_task_description(self, text):
        self.task_label.setText(text)
