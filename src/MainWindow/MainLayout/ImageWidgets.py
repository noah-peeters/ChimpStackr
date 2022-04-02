"""
    Class containing clickable image QListWidgets for loaded images and processed images.
    Clicking on an item will trigger displaying it.
"""
import os
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw
import PySide6.QtGui as qtg

import src.settings as settings

# Helper class with infinite scrolling
class InfiniteQListWidget(qtw.QListWidget):
    def __init__(self):
        super().__init__()
        self.setSelectionMode(qtw.QAbstractItemView.ExtendedSelection)

    # Setup infinite scrolling
    def keyPressEvent(self, event):
        if event.key() == qtc.Qt.Key_Down:
            if self.currentRow() == self.count()-1:
                self.setCurrentRow(0)
                return
        elif event.key() == qtc.Qt.Key_Up:
            if self.currentRow() == 0:
                self.setCurrentRow(self.count()-1)
                return

        # Otherwise, parent behavior
        super().keyPressEvent(event)

# QListWidget for displaying loaded images (allow drag & drop)
class LoadedImagesList(InfiniteQListWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(qtc.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    # Try setting loaded images on drag
    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(qtc.Qt.CopyAction)
            event.accept()
            image_paths = []
            for url in event.mimeData().urls():
                image_paths.append(str(url.toLocalFile()))

            # If dragged path is a folder; use it's content (absolute paths)
            if len(image_paths) == 1:
                directory = image_paths[0]
                if os.path.isdir(directory):
                    image_paths = [
                        os.path.abspath(os.path.join(directory, p))
                        for p in os.listdir(directory)
                    ]

            settings.globalVars["MainWindow"].set_new_loaded_image_files(image_paths)

        else:
            event.ignore()


# Widget for displaying loaded images
class LoadedImagesWidget(qtw.QWidget):
    default_header_text = "Source images"
    default_loaded_images_items = [
        "Load images from the 'file' menu,",
        "or drop them here.",
    ]

    def __init__(self):
        super().__init__()
        settings.globalVars["LoadedImagesWidget"] = self

        self.headerText = qtw.QLabel(self.default_header_text)
        self.list = LoadedImagesList()
        self.list.addItems(self.default_loaded_images_items)

        v_layout = qtw.QVBoxLayout()
        v_layout.addWidget(self.headerText)
        v_layout.addWidget(self.list)
        self.setLayout(v_layout)

    # Reset list and header label to default values
    def reset_to_default(self):
        self.list.clear()
        self.list.addItems(self.default_loaded_images_items)
        self.headerText.setText(self.default_header_text)

    # Needed to prevent calling from other thread
    def setHeaderText(self, msg):
        self.headerText.setText(msg)

    """
        Overridden signals
    """

    # Remove selected images on right-click
    def contextMenuEvent(self, event: qtg.QContextMenuEvent) -> None:
        menu = qtw.QMenu()
        reset_zoom_action = qtg.QAction("Remove selected images")
        menu.addAction(reset_zoom_action)
        selected_action = menu.exec(event.globalPos())

        if selected_action == reset_zoom_action:
            self.reset_zoom = reset_zoom_action.isChecked()
            paths_to_remove=[]
            for listItem in self.list.selectedItems():
                paths_to_remove.append(listItem.data(qtc.Qt.UserRole))
            settings.globalVars["MainWindow"].remove_some_images(paths_to_remove)

# Widget for displaying processed/stacked images
class ProcessedImagesWidget(qtw.QWidget):
    def __init__(self):
        super().__init__()
        settings.globalVars["ProcessedImagesWidget"] = self

        self.headerText = qtw.QLabel("Output image(s)")
        self.list = InfiniteQListWidget()

        v_layout = qtw.QVBoxLayout()
        v_layout.addWidget(self.headerText)
        v_layout.addWidget(self.list)
        self.setLayout(v_layout)

        self.setVisible(False)


# Widget bringing both QListWidgets together in a horizontal splitter layout
class ImageWidgets(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.loaded_images_widget = LoadedImagesWidget()
        self.processed_images_widget = ProcessedImagesWidget()

        # Create Horizontal splitter (LoadedImagesWidget/ProcessedImagesWidget)
        h_splitter = qtw.QSplitter(qtc.Qt.Vertical)
        h_splitter.setChildrenCollapsible(False) # TODO: Change to "True", but division by 0 in "ImageViewer" class
        h_splitter.addWidget(self.loaded_images_widget)
        h_splitter.addWidget(self.processed_images_widget)

        height = self.screen().availableGeometry().height()
        h_splitter.setSizes([height, int(height / 3.75)])

        layout = qtw.QHBoxLayout(self)
        layout.addWidget(h_splitter)
        self.setLayout(layout)
