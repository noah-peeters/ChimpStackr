"""
    Center widget layout for MainWindow.
    Parent of loaded images widget, ImageViewer and splitter layout(s).
"""
import os
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw

import MainWindow.MainLayout.ImageWidgets as ImageWidgets
import MainWindow.MainLayout.ImageViewer as ImageViewer
from utilities import int_string_sorting


class CenterWidget(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.ImageWidgets = ImageWidgets.ImageWidgets()

        # Image display widget
        image_display = ImageViewer.ImageViewer(self.ImageWidgets.loaded_images_widget.list)

        # Create vertical splitter (QListWidgets/ImageViewer)
        v_splitter = qtw.QSplitter()
        v_splitter.setChildrenCollapsible(False)
        v_splitter.addWidget(self.ImageWidgets)
        v_splitter.addWidget(image_display)

        # Set splitter default size
        width = self.screen().availableGeometry().width()
        v_splitter.setSizes([int(width / 4), width])

        # Set horizontal layout
        layout = qtw.QHBoxLayout(self)
        layout.addWidget(v_splitter)
        self.setLayout(layout)

    # Called from parent MainWindow. Process image path names and display new loaded images inside list widget.
    # QListWidget data is set to the full image path for quick retrieval later.
    def set_loaded_images(self, new_image_files):
        self.ImageWidgets.loaded_images_widget.reset_to_default()
        if len(new_image_files) <= 0:
            # No images selected, use default
            return

        # New images selected: update list and header text
        self.ImageWidgets.loaded_images_widget.header_label.setText(
            "Source images (" + str(len(new_image_files)) + ")"
        )

        # Set new files
        self.ImageWidgets.loaded_images_widget.list.clear()
        for path in sorted(new_image_files, key=int_string_sorting):
            name = os.path.basename(path)
            item = qtw.QListWidgetItem()
            item.setData(qtc.Qt.UserRole, path)  # Set data to full image path
            item.setText(name)  # Set text to image name
            self.ImageWidgets.loaded_images_widget.list.addItem(item)
