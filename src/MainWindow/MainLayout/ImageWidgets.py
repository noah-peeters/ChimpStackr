"""
    Class containing clickable image QListWidgets for loaded images and processed images.
    Clicking on an item will trigger displaying it.
"""
import os
import cv2
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw
import PySide6.QtGui as qtg

import src.settings as settings
from src.MainWindow.style import ACCENT, ACCENT_DIM, TEXT_MUTED, BG_SIDEBAR


# Background thumbnail loader
class ThumbnailWorker(qtc.QRunnable):
    class Signals(qtc.QObject):
        thumbnail_ready = qtc.Signal(str, qtg.QIcon)

    def __init__(self, path, size=48):
        super().__init__()
        self.path = path
        self.size = size
        self.signals = self.Signals()

    def run(self):
        try:
            # Load at 1/8 resolution for speed (IMREAD_REDUCED_COLOR_8)
            img = cv2.imread(self.path, cv2.IMREAD_REDUCED_COLOR_8)
            if img is None:
                # Fallback to full load for formats that don't support reduced
                img = cv2.imread(self.path)
            if img is not None:
                h, w = img.shape[:2]
                if max(h, w) > self.size:
                    scale = self.size / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                     interpolation=cv2.INTER_AREA)
                thumb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                qimg = qtg.QImage(
                    thumb.data, thumb.shape[1], thumb.shape[0],
                    thumb.shape[1] * 3, qtg.QImage.Format_RGB888
                )
                icon = qtg.QIcon(qtg.QPixmap.fromImage(qimg))
                self.signals.thumbnail_ready.emit(self.path, icon)
        except Exception:
            pass


# Helper class with infinite scrolling
class InfiniteQListWidget(qtw.QListWidget):
    def __init__(self):
        super().__init__()
        self.setSelectionMode(qtw.QAbstractItemView.ExtendedSelection)

    def keyPressEvent(self, event):
        if event.key() == qtc.Qt.Key_Down:
            if self.currentRow() == self.count() - 1:
                self.setCurrentRow(0)
                return
        elif event.key() == qtc.Qt.Key_Up:
            if self.currentRow() == 0:
                self.setCurrentRow(self.count() - 1)
                return
        elif event.key() == qtc.Qt.Key_Space:
            # Toggle between source and output lists
            try:
                main = settings.globalVars["MainWindow"]
                proc_widget = settings.globalVars.get("ProcessedImagesWidget")
                if proc_widget and proc_widget.isVisible() and proc_widget.list.count() > 0:
                    if self == main._main_content.ImageWidgets.loaded_images_widget.list:
                        proc_widget.list.setCurrentRow(0)
                    else:
                        main._main_content.ImageWidgets.loaded_images_widget.list.setCurrentRow(0)
            except (KeyError, AttributeError):
                pass
            return
        super().keyPressEvent(event)


# QListWidget for displaying loaded images (allow drag & drop)
class LoadedImagesList(InfiniteQListWidget):
    def __init__(self):
        super().__init__()
        self.setIconSize(qtc.QSize(48, 48))


# Widget for displaying loaded images — handles drag-drop from Finder/file managers
class LoadedImagesWidget(qtw.QWidget):
    default_header_text = "Source images"

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        settings.globalVars["LoadedImagesWidget"] = self

        self.headerText = qtw.QLabel(self.default_header_text)
        self.headerText.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px 0;")
        self.list = LoadedImagesList()

        # Empty state placeholder
        self.empty_label = qtw.QLabel("Drop images here\nor use File > Load Images")
        self.empty_label.setAlignment(qtc.Qt.AlignCenter)
        self.empty_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px; padding: 24px;")

        self.stack = qtw.QStackedWidget()
        self.stack.addWidget(self.list)
        self.stack.addWidget(self.empty_label)
        self.stack.setCurrentIndex(1)  # Show empty state by default

        v_layout = qtw.QVBoxLayout()
        v_layout.setContentsMargins(4, 4, 4, 4)
        v_layout.addWidget(self.headerText)
        v_layout.addWidget(self.stack)
        self.setLayout(v_layout)

        self._threadpool = qtc.QThreadPool()
        self._threadpool.setMaxThreadCount(4)
        self._thumbnail_map = {}
        self._default_style = self.styleSheet()

    # --- Drag-drop from Finder ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            # Only highlight the list/empty area, not the header
            self.stack.setStyleSheet(
                f"QStackedWidget {{ border: 2px dashed {ACCENT}; "
                f"border-radius: 6px; background: {ACCENT_DIM}; }}"
            )
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.stack.setStyleSheet("")
        event.accept()

    def dropEvent(self, event):
        self.stack.setStyleSheet("")
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            raw_paths = []
            for url in event.mimeData().urls():
                local = url.toLocalFile()
                if local:
                    raw_paths.append(local)

            # Expand folders recursively
            image_paths = []
            for path in raw_paths:
                if os.path.isdir(path):
                    for root, dirs, files in os.walk(path):
                        for f in files:
                            image_paths.append(os.path.abspath(os.path.join(root, f)))
                elif os.path.isfile(path):
                    image_paths.append(os.path.abspath(path))

            if image_paths:
                settings.globalVars["MainWindow"].set_new_loaded_image_files(image_paths)
        else:
            event.ignore()

    def reset_to_default(self):
        self.list.clear()
        self.headerText.setText(self.default_header_text)
        self.stack.setCurrentIndex(1)
        self._thumbnail_map.clear()

    def show_list(self):
        self.stack.setCurrentIndex(0)

    def load_thumbnails(self, paths):
        """Kick off background thumbnail loading for all paths."""
        self._thumbnail_map.clear()
        for path in paths:
            worker = ThumbnailWorker(path)
            worker.signals.thumbnail_ready.connect(self._on_thumbnail_ready)
            self._threadpool.start(worker)

    def _on_thumbnail_ready(self, path, icon):
        """Set the thumbnail icon on the matching list item."""
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.data(qtc.Qt.UserRole) == path:
                item.setIcon(icon)
                break

    def setHeaderText(self, msg):
        self.headerText.setText(msg)

    def contextMenuEvent(self, event: qtg.QContextMenuEvent) -> None:
        menu = qtw.QMenu()
        remove_action = qtg.QAction("Remove selected images")
        menu.addAction(remove_action)
        selected_action = menu.exec(event.globalPos())

        if selected_action == remove_action:
            paths_to_remove = []
            for listItem in self.list.selectedItems():
                paths_to_remove.append(listItem.data(qtc.Qt.UserRole))
            settings.globalVars["MainWindow"].remove_some_images(paths_to_remove)


# Widget for displaying processed/stacked images
class ProcessedImagesWidget(qtw.QWidget):
    def __init__(self):
        super().__init__()
        settings.globalVars["ProcessedImagesWidget"] = self

        self.headerText = qtw.QLabel("Output images")
        self.headerText.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px 0;")
        self.list = InfiniteQListWidget()

        v_layout = qtw.QVBoxLayout()
        v_layout.setContentsMargins(4, 4, 4, 4)
        v_layout.addWidget(self.headerText)
        v_layout.addWidget(self.list)
        self.setLayout(v_layout)

        self.setVisible(False)


# Widget bringing both QListWidgets together
class ImageWidgets(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.loaded_images_widget = LoadedImagesWidget()
        self.processed_images_widget = ProcessedImagesWidget()

        h_splitter = qtw.QSplitter(qtc.Qt.Vertical)
        h_splitter.setChildrenCollapsible(True)
        h_splitter.addWidget(self.loaded_images_widget)
        h_splitter.addWidget(self.processed_images_widget)

        height = self.screen().availableGeometry().height()
        h_splitter.setSizes([height, int(height / 3.75)])

        layout = qtw.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(h_splitter)
        self.setLayout(layout)
