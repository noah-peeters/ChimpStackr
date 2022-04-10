"""
    Center widget layout for MainWindow.
    Parent of loaded images widget, ImageViewer and splitter layout(s).
"""
import os, tempfile
from datetime import datetime
import cv2
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw

import src.ImageLoadingHandler as ImageLoadingHandler
import src.MainWindow.MainLayout.ImageWidgets as ImageWidgets
import src.MainWindow.MainLayout.ImageViewers as ImageViewers
from src.utilities import int_string_sorting
import src.settings as settings


class CenterWidget(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.root_temp_directory = settings.globalVars["RootTempDir"]
        self.ImageWidgets = ImageWidgets.ImageWidgets()
        self.ImageViewer = ImageViewers.ImageViewer()
        self.RetouchingViewer = ImageViewers.ImageRetouchingWidget()
        self.ImageLoading = ImageLoadingHandler.ImageLoadingHandler()

        # Connect to "selected item change" signals
        # "currentItemChanged" for keyboard key presses + "itemClicked" for mouseclicks
        self.ImageWidgets.loaded_images_widget.list.currentItemChanged.connect(
            self.display_new_image
        )
        self.ImageWidgets.loaded_images_widget.list.itemClicked.connect(
            self.display_new_image
        )

        self.ImageWidgets.processed_images_widget.list.currentItemChanged.connect(
            self.display_new_image
        )
        self.ImageWidgets.processed_images_widget.list.itemClicked.connect(
            self.display_new_image
        )

        # QTabWidget above ImageViewer (toggle View/Retouch modes)
        tabWidget = qtw.QTabWidget()
        tabWidget.addTab(self.ImageViewer, "View")
        tabWidget.addTab(self.RetouchingViewer, "Retouch")

        # Create vertical splitter (QListWidgets/ImageViewer)
        v_splitter = qtw.QSplitter()
        v_splitter.setChildrenCollapsible(True)  # TODO: 3 dots on collapse
        v_splitter.addWidget(self.ImageWidgets)
        v_splitter.addWidget(tabWidget)

        # Set splitter default size
        width = self.screen().availableGeometry().width()
        v_splitter.setSizes([int(width / 4), width])

        # Set horizontal layout
        layout = qtw.QHBoxLayout(self)
        layout.addWidget(v_splitter)
        self.setLayout(layout)

    # Handles image update listwidget item clicks
    def display_new_image(self, list_widget_item: qtw.QListWidgetItem) -> None:
        if list_widget_item:
            # Display selected image
            path = list_widget_item.data(qtc.Qt.UserRole)
            if path:
                image = self.ImageLoading.read_image_from_path(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Always update regular viewer's image
                self.ImageViewer.set_image(image)

                # Update retouching image depending on if image is source or output
                if (
                    list_widget_item.listWidget()
                    == self.ImageWidgets.loaded_images_widget.list
                ):
                    self.RetouchingViewer.set_retouch_image(image)
                else:
                    self.RetouchingViewer.set_output_image(image)
                return

        # Clear all (if no image was changed)
        self.ImageViewer.set_image(None)
        self.RetouchingViewer.set_retouch_image(None)
        self.RetouchingViewer.set_output_image(None)

    # Update currently loaded images + relevant UI
    def set_loaded_images(self, new_image_files):
        # Clear currently displaying image
        self.ImageWidgets.loaded_images_widget.reset_to_default()
        self.ImageViewer.set_image(None)

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
            item.setText(datetime.today().strftime("%Y%m%d") + "_LaplacianStacked")
            item.setData(qtc.Qt.UserRole, tmp_file)

            cv2.imwrite(tmp_file, new_image_array)

            os.close(file_handle)
            settings.globalVars["ProcessedImagesWidget"].list.addItem(item)
