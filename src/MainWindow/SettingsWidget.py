"""
Settings widget that handles user changing settings.
"""
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw

# Settings under "View" tab
class ViewWidget(qtw.QWidget):
    def __init__(self):
        super().__init__()
        # TODO: Allow change of theme
        label = qtw.QLabel("This is a text label", self)


class SettingsWidget(qtw.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Edit settings")

        stackedLayout = qtw.QStackedLayout()
        stackedLayout.addWidget(ViewWidget())

        mainLayout = qtw.QHBoxLayout()
        mainLayout.addLayout(stackedLayout)
        self.setLayout(mainLayout)
