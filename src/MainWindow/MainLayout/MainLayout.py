"""
    Center widget layout for MainWindow.
    Parent of loaded images widget, ImageViewer and splitter layout(s).
"""
import os, tempfile
import cv2
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw

import src.MainWindow.MainLayout.ImageWidgets as ImageWidgets
import src.MainWindow.MainLayout.ImageViewer as ImageViewer
from src.utilities import int_string_sorting
import src.settings as settings


class CenterWidget(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.root_temp_directory = settings.globalVars["RootTempDir"]
        self.ImageWidgets = ImageWidgets.ImageWidgets()
        self.image_display = ImageViewer.ImageViewer()

        # Connect to "selected item change" signals
        self.ImageWidgets.loaded_images_widget.list.currentItemChanged.connect(
            self.image_display.update_displayed_image
        )
        self.ImageWidgets.processed_images_widget.list.currentItemChanged.connect(
            self.image_display.update_displayed_image
        )

        # Create vertical splitter (QListWidgets/ImageViewer)
        v_splitter = qtw.QSplitter()
        v_splitter.setChildrenCollapsible(False)
        v_splitter.addWidget(self.ImageWidgets)
        v_splitter.addWidget(self.image_display)

        # Set splitter default size
        width = self.screen().availableGeometry().width()
        v_splitter.setSizes([int(width / 4), width])

        # Set horizontal layout
        layout = qtw.QHBoxLayout(self)
        layout.addWidget(v_splitter)
        self.setLayout(layout)

    # Update currently loaded images + relevant UI
    def set_loaded_images(self, new_image_files):
        # Clear currently displaying image
        self.ImageWidgets.loaded_images_widget.reset_to_default()
        self.image_display.update_displayed_image(None)

        if len(new_image_files) <= 0:
            # No images selected, use default
            return

        # Update header text
        settings.globalVars["LoadedImagesWidget"].headerText.setText(
            "Source images (" + str(len(new_image_files)) + ")"
        )

        # Set new files
        settings.globalVars["LoadedImagesWidget"].list.clear()
        for path in sorted(new_image_files, key=int_string_sorting):
            name = os.path.basename(path)
            item = qtw.QListWidgetItem()
            item.setData(qtc.Qt.UserRole, path)  # Set data to full image path
            item.setText(name)  # Set text to image name
            settings.globalVars["LoadedImagesWidget"].list.addItem(item)

    # Called from parent MainWindow. Display generated name inside ProcessedImagesWidget
    def add_processed_image(self, new_image_array):
        if new_image_array is None:
            self.ImageWidgets.processed_images_widget.list.clear()
            self.ImageWidgets.processed_images_widget.setVisible(False)
        else:
            self.ImageWidgets.processed_images_widget.setVisible(True)
            # Add image to list & store data file
            # TODO: Why is tempfile stored in this folder??
            file_handle, tmp_file = tempfile.mkstemp(
                suffix=".jpg", dir=self.root_temp_directory.name
            )
            item = qtw.QListWidgetItem()
            item.setText("lap_pyr_stacked")
            item.setData(qtc.Qt.UserRole, tmp_file)

            cv2.imwrite(tmp_file, new_image_array)

            os.close(file_handle)
            settings.globalVars["ProcessedImagesWidget"].list.addItem(item)
