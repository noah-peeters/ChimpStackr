"""
    Center widget layout for MainWindow.
    Parent of loaded images widget, ImageViewer and splitter layout(s).
"""
import os, tempfile
from datetime import datetime
import cv2
import numpy as np
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw
import PySide6.QtGui as qtg

import src.ImageLoadingHandler as ImageLoadingHandler
import src.MainWindow.MainLayout.ImageWidgets as ImageWidgets
import src.MainWindow.MainLayout.ImageViewers as ImageViewers
from src.MainWindow.MainLayout.ImageViewers.ComparisonViewer import ComparisonWidget
from src.utilities import int_string_sorting
import src.settings as settings


class PreviewWorker(qtc.QRunnable):
    """Loads a full-resolution image in the background for preview display."""
    class Signals(qtc.QObject):
        ready = qtc.Signal(str, object)  # path, bgr_image

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.signals = self.Signals()

    def run(self):
        try:
            bgr = cv2.imread(self.path, cv2.IMREAD_COLOR)
            if bgr is not None:
                self.signals.ready.emit(self.path, bgr)
        except Exception:
            pass


class CenterWidget(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.ImageWidgets = ImageWidgets.ImageWidgets()
        self.ImageViewer = ImageViewers.ImageViewer()
        self.RetouchingViewer = ImageViewers.ImageRetouchingWidget()
        self.ComparisonViewer = ComparisonWidget()
        self.ImageLoading = ImageLoadingHandler.ImageLoadingHandler()
        self._recent_dirs = []
        self._preview_pool = qtc.QThreadPool()
        self._preview_pool.setMaxThreadCount(2)
        self._pending_preview_path = None

        # Connect list selection signals
        for widget_list in [
            self.ImageWidgets.loaded_images_widget.list,
            self.ImageWidgets.processed_images_widget.list,
        ]:
            widget_list.currentItemChanged.connect(self.display_new_image)
            widget_list.itemClicked.connect(self.display_new_image)

        # Tab widget
        self.tabWidget = qtw.QTabWidget()
        self.tabWidget.addTab(self.ImageViewer, "View")
        self.tabWidget.addTab(self.ComparisonViewer, "Compare")

        # Splitter: sidebar | content
        v_splitter = qtw.QSplitter()
        v_splitter.setChildrenCollapsible(True)
        v_splitter.addWidget(self.ImageWidgets)
        v_splitter.addWidget(self.tabWidget)

        # Narrower sidebar: ~200px fixed, rest to content
        v_splitter.setSizes([200, 1000])
        v_splitter.setStretchFactor(0, 0)  # Sidebar doesn't stretch
        v_splitter.setStretchFactor(1, 1)  # Content stretches

        layout = qtw.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(v_splitter)
        self.setLayout(layout)

    def display_new_image(self, list_widget_item: qtw.QListWidgetItem) -> None:
        if list_widget_item:
            path = list_widget_item.data(qtc.Qt.UserRole)
            if path and os.path.isfile(path):
                self._pending_preview_path = path
                self._pending_is_source = (
                    list_widget_item.listWidget()
                    == self.ImageWidgets.loaded_images_widget.list
                )

                # Show immediate status update (non-blocking)
                try:
                    name = os.path.basename(path)
                    size_kb = os.path.getsize(path) / 1024
                    size_str = f"{size_kb/1024:.1f} MB" if size_kb > 1024 else f"{size_kb:.0f} KB"
                    ext = os.path.splitext(path)[1].upper()[1:]
                    settings.globalVars["MainWindow"].image_info_label.setText(
                        f"{name}  —  {ext}  |  {size_str}  |  Loading..."
                    )
                except Exception:
                    pass

                # Load full-res in background thread (UI stays responsive)
                worker = PreviewWorker(path)
                worker.signals.ready.connect(self._on_preview_ready)
                self._preview_pool.start(worker)
                return

        self.ImageViewer.set_image(None)
        self._pending_preview_path = None
        try:
            settings.globalVars["MainWindow"].image_info_label.setText("No image loaded")
        except (KeyError, AttributeError):
            pass

    def _on_preview_ready(self, path, bgr_image):
        """Called on main thread when background image load completes."""
        if path != self._pending_preview_path:
            return

        # Use BGR direct path — skips cvtColor, uses Format_BGR888 → ARGB32_Premultiplied
        self.ImageViewer.viewerScene.set_image_bgr(bgr_image)
        self.ImageViewer.empty_state.setVisible(False)

        # Update status bar
        try:
            h, w = bgr_image.shape[:2]
            ext = os.path.splitext(path)[1].upper()[1:]
            size_kb = os.path.getsize(path) / 1024
            size_str = f"{size_kb/1024:.1f} MB" if size_kb > 1024 else f"{size_kb:.0f} KB"
            name = os.path.basename(path)
            info = f"{name}  —  {w} x {h}  |  {ext}  |  {size_str}"
            settings.globalVars["MainWindow"].image_info_label.setText(info)
        except Exception:
            pass

    def set_loaded_images(self, new_image_files):
        loaded_widget = self.ImageWidgets.loaded_images_widget
        loaded_widget.reset_to_default()
        self.ImageViewer.set_image(None)

        if len(new_image_files) <= 0:
            return

        # Track recent directories
        directory = os.path.dirname(new_image_files[0])
        if directory not in self._recent_dirs:
            self._recent_dirs.insert(0, directory)
            self._recent_dirs = self._recent_dirs[:10]

        loaded_widget.setHeaderText(f"Source images ({len(new_image_files)})")
        loaded_widget.show_list()

        sorted_paths = sorted(new_image_files, key=int_string_sorting)
        loaded_widget.list.clear()
        for path in sorted_paths:
            name = os.path.basename(path)
            item = qtw.QListWidgetItem()
            item.setData(qtc.Qt.UserRole, path)
            item.setText(name)
            loaded_widget.list.addItem(item)

        loaded_widget.load_thumbnails(sorted_paths)

        # Auto-select first image for preview
        if loaded_widget.list.count() > 0:
            loaded_widget.list.setCurrentRow(0)

    def add_processed_image(self, new_image_array):
        if new_image_array is None:
            self.ImageWidgets.processed_images_widget.list.clear()
            self.ImageWidgets.processed_images_widget.setVisible(False)
            self.ComparisonViewer.set_after_image(None)
        else:
            self.ImageWidgets.processed_images_widget.setVisible(True)
            file_handle, tmp_file = tempfile.mkstemp(
                suffix=".jpg", dir=settings.globalVars["RootTempDir"].name
            )
            item = qtw.QListWidgetItem()
            item.setText(datetime.today().strftime("%Y%m%d") + "_LaplacianStacked")
            item.setData(qtc.Qt.UserRole, tmp_file)

            cv2.imwrite(tmp_file, new_image_array)
            os.close(file_handle)

            # Add thumbnail to output item
            result_u8 = np.clip(np.around(new_image_array), 0, 255).astype(np.uint8)
            result_rgb = cv2.cvtColor(result_u8, cv2.COLOR_BGR2RGB)
            thumb_size = 48
            h, w = result_rgb.shape[:2]
            scale = thumb_size / max(h, w)
            thumb = cv2.resize(result_rgb, (int(w * scale), int(h * scale)))
            qimg = qtg.QImage(thumb.data, thumb.shape[1], thumb.shape[0],
                             thumb.shape[1] * 3, qtg.QImage.Format_RGB888)
            item.setIcon(qtg.QIcon(qtg.QPixmap.fromImage(qimg)))

            settings.globalVars["ProcessedImagesWidget"].list.addItem(item)

            # Update comparison viewer
            self.ComparisonViewer.set_after_image(result_rgb)

            loaded_paths = settings.globalVars.get("LoadedImagePaths", [])
            if loaded_paths:
                before_img = self.ImageLoading.read_image_from_path(loaded_paths[0])
                if before_img is not None:
                    before_rgb = cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB)
                    self.ComparisonViewer.set_before_image(before_rgb)
