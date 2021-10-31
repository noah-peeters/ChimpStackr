"""
    Center widget layout for MainWindow.
    Parent of loaded images widget, ImageViewer and splitter layout.
"""
import os
import PySide6.QtWidgets as qtw

from MainWindow.ImageViewer import ImageViewer
from UI_utilities import int_string_sorting


class CenterWidget(qtw.QWidget):
    default_loaded_images_items = [
        "Loaded images will appear here.",
        "Please load them in from the 'file' menu.",
    ]

    def __init__(self, main_window):
        super().__init__()

        # Loaded images list widget
        self.loaded_images_list = qtw.QListWidget()
        # loaded_images_list.setAlternatingRowColors(True)
        self.loaded_images_list.setSelectionMode(
            qtw.QAbstractItemView.ExtendedSelection
        )
        self.loaded_images_list.addItems(self.default_loaded_images_items)

        # Image display widget
        image_display = ImageViewer(self.loaded_images_list)

        # Create splitter
        splitter = qtw.QSplitter()
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.loaded_images_list)
        splitter.addWidget(image_display)

        # Set splitter default size
        width = main_window.screen().availableGeometry().width()
        splitter.setSizes([int(width / 5), width])

        # Set horizontal layout
        layout = qtw.QHBoxLayout(self)
        layout.addWidget(splitter)
        self.setLayout(layout)

    # Called from parent MainWindow. Process image path names and display new loaded images inside list widget.
    # QListWidget data is set to the full image path for quick retrieval later.
    def set_loaded_images(self, new_image_files):
        if len(new_image_files) <= 0:
            # No images selected, set default
            self.loaded_images_list.addItems(self.default_loaded_images_items)
            return

        # Clear and set new files
        self.loaded_images_list.clear()
        for path in sorted(new_image_files, int_string_sorting):
            name = os.path.basename(path)
            item = qtw.QListWidgetItem()
            item.setData(path)  # Set data to full image path
            item.setText(name)  # Set text to image name
            self.loaded_images_list.addItem(item)