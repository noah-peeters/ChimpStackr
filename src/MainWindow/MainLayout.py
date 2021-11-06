"""
    Center widget layout for MainWindow.
    Parent of loaded images widget, ImageViewer and splitter layout.
"""
import os
import PySide6.QtWidgets as qtw
import PySide6.QtCore as qtc
import PySide6.QtGui as qtg

from MainWindow.ImageViewer import ImageViewer
from utilities import int_string_sorting


class LoadedImagesWidget(qtw.QWidget):
    default_loaded_images_items = [
        "Loaded images will appear here.",
        "Please load them in from the 'file' menu.",
    ]
    default_header_text = "Source images"

    def __init__(self):
        super().__init__()
        self.header_label = qtw.QLabel(self.default_header_text)

        self.list = qtw.QListWidget()
        self.list.setSelectionMode(qtw.QAbstractItemView.ExtendedSelection)
        self.list.addItems(self.default_loaded_images_items)

        v_layout = qtw.QVBoxLayout()
        v_layout.addWidget(self.header_label)
        v_layout.addWidget(self.list)
        self.setLayout(v_layout)

    # Reset list and header label to default values
    def reset_to_default(self):
        self.list.clear()
        self.list.addItems(self.default_loaded_images_items)

        self.header_label.setText(self.default_header_text)


class CenterWidget(qtw.QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.loaded_images_widget = LoadedImagesWidget()

        # Image display widget
        image_display = ImageViewer(self.loaded_images_widget.list)

        # Create splitter
        splitter = qtw.QSplitter()
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.loaded_images_widget)
        splitter.addWidget(image_display)

        # Set splitter default size
        width = main_window.screen().availableGeometry().width()
        splitter.setSizes([int(width / 4), width])

        # Set horizontal layout
        layout = qtw.QHBoxLayout(self)
        layout.addWidget(splitter)
        self.setLayout(layout)

    # Called from parent MainWindow. Process image path names and display new loaded images inside list widget.
    # QListWidget data is set to the full image path for quick retrieval later.
    def set_loaded_images(self, new_image_files):
        self.loaded_images_widget.reset_to_default()
        if len(new_image_files) <= 0:
            # No images selected, use default
            return

        # New images selected: update list and header text
        self.loaded_images_widget.header_label.setText(
            "Source images (" + str(len(new_image_files)) + ")"
        )

        # Set new files
        self.loaded_images_widget.list.clear()
        for path in sorted(new_image_files, key=int_string_sorting):
            name = os.path.basename(path)
            item = qtw.QListWidgetItem()
            item.setData(qtc.Qt.UserRole, path)  # Set data to full image path
            item.setText(name)  # Set text to image name
            self.loaded_images_widget.list.addItem(item)
