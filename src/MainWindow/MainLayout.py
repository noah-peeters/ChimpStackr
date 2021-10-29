"""
    Center widget layout for MainWindow.
"""
import PySide6.QtWidgets as qtw


class CenterWidget(qtw.QWidget):
    def __init__(self, main_window):
        super().__init__()
        loaded_images_list = qtw.QListWidget()
        # loaded_images_list.setAlternatingRowColors(True)
        loaded_images_list.setSelectionMode(qtw.QAbstractItemView.ExtendedSelection)
        loaded_images_list.addItems(
            [
                "Loaded images will appear here.",
                "Please load them in from the 'file' menu.",
            ]
        )

        # Create splitter
        splitter = qtw.QSplitter()
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(loaded_images_list)

        # Set horizontal layout
        layout = qtw.QHBoxLayout(self)
        layout.addWidget(splitter)
        self.setLayout(layout)
