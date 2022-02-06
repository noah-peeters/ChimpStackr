# -*- coding: utf-8 -*-
"""
This module is a header-like file for pyqtconfig. Handles compatiblity
with both PyQt5 and PySide2.
"""
import sys

if "PyQt5" in sys.modules:
    from PyQt5 import QtWidgets
    from PyQt5.QtCore import (
        QMutex,
        QMutexLocker,
        QObject,
        QSettings,
        Qt,
        QVariant,
        pyqtSignal,
    )
    from PyQt5.QtWidgets import (
        QAction,
        QActionGroup,
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QGridLayout,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QPlainTextEdit,
        QPushButton,
        QSlider,
        QSpinBox,
        QTabWidget,
        QTextEdit,
        QWidget,
    )

    Signal = pyqtSignal

elif "PySide2" in sys.modules:
    from PySide2 import QtWidgets
    from PySide2.QtCore import QMutex, QMutexLocker, QObject, QSettings, Qt, Signal
    from PySide2.QtWidgets import (
        QAction,
        QActionGroup,
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QGridLayout,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QPlainTextEdit,
        QPushButton,
        QSlider,
        QSpinBox,
        QTabWidget,
        QTextEdit,
        QWidget,
    )

    QVariant = None

elif "PyQt6" in sys.modules:
    from PyQt6 import QtWidgets
    from PyQt6.QtCore import (
        QMutex,
        QMutexLocker,
        QObject,
        QSettings,
        Qt,
        QVariant,
        pyqtSignal,
    )
    from PyQt6.QtGui import QAction, QActionGroup
    from PyQt6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QGridLayout,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QPlainTextEdit,
        QPushButton,
        QSlider,
        QSpinBox,
        QTabWidget,
        QTextEdit,
        QWidget,
    )

    Signal = pyqtSignal

elif "PySide6" in sys.modules:
    from PySide6 import QtWidgets
    from PySide6.QtCore import QMutex, QMutexLocker, QObject, QSettings, Qt, Signal
    from PySide6.QtGui import QAction, QActionGroup
    from PySide6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QGridLayout,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QPlainTextEdit,
        QPushButton,
        QSlider,
        QSpinBox,
        QTabWidget,
        QTextEdit,
        QWidget,
    )

    QVariant = None

else:
    raise ImportError("Qt library must be imported before pyqtconfig.")
