"""
    Class containing clickable image QListWidgets for loaded images and processed images.
    Clicking on an item will trigger displaying it.
"""
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw
import PySide6.QtGui as qtg


class LoadedImagesList(qtg.QListWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        # self.list.setDragDropMode(qtw.QAbstractItemView.DragDrop)

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

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(qtc.Qt.CopyAction)
            event.accept()
            images = []
            for url in event.mimeData().urls():
                images.append(str(url.toLocalFile()))
            print(images)
        else:
            event.ignore()


# Widget for displaying loaded images
class LoadedImagesWidget(qtw.QWidget):
    default_header_text = "Source images"
    default_loaded_images_items = [
        "Loaded images will appear here.",
        "Please load them in from the 'file' menu,",
        "or drag and drop them here.",
    ]

    def __init__(self):
        super().__init__()
        self.header_label = qtw.QLabel(self.default_header_text)
        self.list = LoadedImagesList()
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

    """
        Overridden signals
    """

    # TODO: Implement method of removing selected images on rightclick
    # Display image options on right click
    # def contextMenuEvent(self, event: qtg.QContextMenuEvent) -> None:
    #     menu = qtw.QMenu()

    #     reset_zoom_action = qtg.QAction("Reset zoom")
    #     reset_zoom_action.setStatusTip("Reset zoom in between image selections.")
    #     reset_zoom_action.setCheckable(True)
    #     reset_zoom_action.setChecked(False)

    #     menu.addAction(reset_zoom_action)
    #     selected_action = menu.exec(event.globalPos())

    #     if selected_action == reset_zoom_action:
    #         self.reset_zoom = reset_zoom_action.isChecked()


# Widget for displaying processed/stacked images
class ProcessedImagesWidget(qtw.QWidget):
    def __init__(self):
        super().__init__()

        self.header_label = qtw.QLabel("Output image(s)")

        self.list = qtw.QListWidget()
        self.list.setSelectionMode(qtw.QAbstractItemView.ExtendedSelection)

        v_layout = qtw.QVBoxLayout()
        v_layout.addWidget(self.header_label)
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
        h_splitter.setChildrenCollapsible(False)
        h_splitter.addWidget(self.loaded_images_widget)
        h_splitter.addWidget(self.processed_images_widget)

        height = self.screen().availableGeometry().height()
        h_splitter.setSizes([height, int(height / 3.75)])

        layout = qtw.QHBoxLayout(self)
        layout.addWidget(h_splitter)
        self.setLayout(layout)
