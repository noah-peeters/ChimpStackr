"""
Settings widget that handles user changing settings.
"""
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw

import modules.pyqtconfig as pyqtconfig
import src.settings as settings

# Settings under "View" tab
class ViewWidget(qtw.QWidget):
    default_settings = {
        "theme": 2,
    }
    themes_map_dict = {
        "dark_amber": 1,
        "dark_blue": 2,
        "dark_cyan": 3,
        "dark_lightgreen": 4,
        "dark_pink": 5,
        "dark_purple": 6,
        "dark_red": 7,
        "dark_teal": 8,
        "dark_yellow": 9,
        "light_amber": 10,
        "light_blue": 11,
        "light_cyan": 12,
        "light_cyan_500": 13,
        "light_lightgreen": 14,
        "light_pink": 15,
        "light_purple": 16,
        "light_red": 17,
        "light_teal": 18,
        "light_yellow": 19,
    }

    # TODO: Allow change of theme
    def __init__(self):
        super().__init__()

        combobox = qtw.QComboBox(self)
        combobox.addItems(self.themes_map_dict)

        self.current_config = qtw.QTextEdit()

        v_layout = qtw.QVBoxLayout()
        v_layout.addWidget(combobox)
        v_layout.addWidget(self.current_config)

        self.setLayout(v_layout)

        # Settings config
        self.config = pyqtconfig.QSettingsManager()
        self.config.set_defaults(self.default_settings)
        self.config.updated.connect(self.config_updated)

        # Settings handlers
        self.config.add_handler("theme", combobox, mapper=self.themes_map_dict)

        # First set
        self.config_updated()

    def config_updated(self):
        # Get dict index (theme name) from value
        newThemeName = list(self.themes_map_dict.keys())[
            list(self.themes_map_dict.values()).index(self.config.get("theme"))
        ]
        settings.globalVars["MainWindow"].apply_stylesheet(
            settings.globalVars["MainWindow"],
            newThemeName + ".xml",
        )

        self.current_config.setText(str(self.config.as_dict()))


class SettingsWidget(qtw.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Edit settings")

        stackedLayout = qtw.QStackedLayout()
        stackedLayout.addWidget(ViewWidget())

        mainLayout = qtw.QHBoxLayout()
        mainLayout.addLayout(stackedLayout)
        self.setLayout(mainLayout)
