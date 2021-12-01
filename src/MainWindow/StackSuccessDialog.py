"""
    Dialog displayed at end of stacking.
    Shows stats about the stack.
"""
import PySide6.QtWidgets as qtw

class Message(qtw.QMessageBox):
    def __init__(self):
        super().__init__()

