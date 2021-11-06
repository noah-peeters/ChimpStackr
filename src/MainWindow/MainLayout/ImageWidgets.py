"""
    Class containing clickable image QListWidgets for loaded images and processed images.
    Clicking on an item will trigger displaying it.
"""
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw

# Widget for displaying loaded images
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
