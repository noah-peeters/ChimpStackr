"""
Settings widget that handles user changing settings.
"__init__" functions are called on application startup,
to initialize the saved QSettings object.
"""
import PySide6.QtWidgets as qtw
import PySide6.QtCore as qtc
import qt_material

import src.settings as settings

# Settings under "User interface" tab
class UserInterfaceWidget(qtw.QWidget):
    themes_map_dict = {
        "dark_amber": 0,
        "dark_blue": 1,
        "dark_cyan": 2,
        "dark_lightgreen": 3,
        "dark_pink": 4,
        "dark_purple": 5,
        "dark_red": 6,
        "dark_teal": 7,
        "dark_yellow": 8,
        "light_amber": 9,
        "light_blue": 10,
        "light_cyan": 11,
        "light_cyan_500": 12,
        "light_lightgreen": 13,
        "light_pink": 14,
        "light_purple": 15,
        "light_red": 16,
        "light_teal": 17,
        "light_yellow": 18,
    }

    def __init__(self, settings_widget):
        super().__init__()
        self.settings_widget = settings_widget

        self.hide()
        self.combobox = qtw.QComboBox(self)
        self.combobox.addItems(self.themes_map_dict)
        # First set
        self.combo_box_changed(
            settings.globalVars["QSettings"].value("user_interface/theme")
        )
        self.combobox.currentIndexChanged.connect(self.combo_box_changed)

        h_layout = qtw.QHBoxLayout()
        h_layout.addWidget(qtw.QLabel("Application theme:"))
        h_layout.addWidget(self.combobox)

        self.current_config = qtw.QTextEdit()

        v_layout = qtw.QVBoxLayout()
        v_layout.addLayout(h_layout)
        v_layout.addWidget(self.current_config)

        self.setLayout(v_layout)

    def combo_box_changed(self, newIndex):
        # Get and set new theme.
        newTheme = (
            list(self.themes_map_dict.keys())[
                list(self.themes_map_dict.values()).index(newIndex)
            ]
            + ".xml"
        )
        qt_material.apply_stylesheet(
            settings.globalVars["MainApplication"], theme=newTheme
        )
        self.combobox.setCurrentIndex(newIndex)
        # Save new theme
        self.settings_widget.change_setting("user_interface/theme", newIndex)


class ComputingWidget(qtw.QWidget):
    def __init__(self, settings_widget):
        super().__init__()
        self.settings_widget = settings_widget


class SettingsWidget(qtw.QTabWidget):
    default_settings = {
        "user_interface": {
            "theme": 2,
        }
    }
    setting_updated = qtc.Signal(tuple)

    def __init__(self):
        super().__init__()

        # Use default settings if not yet saved
        for key, dict in self.default_settings.items():
            for name, value in dict.items():
                str1 = f"{key}/{name}"
                if settings.globalVars["QSettings"].contains(str1):
                    value = settings.globalVars["QSettings"].value(str1)
                self.change_setting(str1, value)

        self.setWindowTitle("Edit settings")
        self.addTab(UserInterfaceWidget(self), "User interface")
        self.addTab(ComputingWidget(self), "Processing")

    def change_setting(self, key, value):
        """
        Change setting to new value and notify of change with a signal.
        """
        settings.globalVars["QSettings"].setValue(key, value)
        self.setting_updated.emit((key, value))
