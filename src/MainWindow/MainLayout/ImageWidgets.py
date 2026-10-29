"""
Sidebar: the loaded frames and the stacked results, with thumbnails.

Frames: multi-select, Delete/Backspace removes, right-click for Remove and
Show in Finder/Explorer, drop files or folders to add. Results: one entry
per stack of this session; selecting one shows it.
"""
import os
import subprocess
import sys

import cv2
import numpy as np
import PySide6.QtCore as qtc
import PySide6.QtGui as qtg
import PySide6.QtWidgets as qtw

import src.ImageLoadingHandler as ImageLoadingHandler
from src.MainWindow.icons import icon

THUMB = 44        # thumbnail size in points (rendered at 2x for HiDPI)
PATH_ROLE = qtc.Qt.UserRole
INFO_ROLE = qtc.Qt.UserRole + 1


def reveal_in_file_manager(path):
    """Show a file in Finder / Explorer / the desktop's file manager."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
        else:
            qtg.QDesktopServices.openUrl(qtc.QUrl.fromLocalFile(os.path.dirname(path)))
    except Exception:
        qtg.QDesktopServices.openUrl(qtc.QUrl.fromLocalFile(os.path.dirname(path)))


def file_manager_name():
    return {"darwin": "Finder", "win32": "Explorer"}.get(sys.platform, "File Manager")


def thumbnail_image(bgr, size=THUMB, dpr=2.0):
    """Square, rounded HiDPI thumbnail QImage from a BGR image. QImage (not
    QPixmap), so it can be made on a worker thread."""
    if bgr is None:
        return None
    if bgr.dtype == np.uint16:
        bgr = (bgr >> 8).astype(np.uint8)
    elif bgr.dtype != np.uint8:
        bgr = np.clip(np.around(bgr), 0, 255).astype(np.uint8)
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    px = int(size * dpr)
    h, w = bgr.shape[:2]
    s = px / max(h, w)
    small = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    src = qtg.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3,
                     qtg.QImage.Format_RGB888).copy()
    canvas = qtg.QImage(px, px, qtg.QImage.Format_ARGB32_Premultiplied)
    canvas.fill(qtc.Qt.transparent)
    p = qtg.QPainter(canvas)
    p.setRenderHint(qtg.QPainter.Antialiasing)
    p.setRenderHint(qtg.QPainter.SmoothPixmapTransform)
    target = qtc.QRectF((px - src.width()) / 2, (px - src.height()) / 2, src.width(), src.height())
    path = qtg.QPainterPath()
    path.addRoundedRect(target, 4 * dpr, 4 * dpr)
    p.setClipPath(path)
    p.drawImage(target, src)
    p.end()
    canvas.setDevicePixelRatio(dpr)
    return canvas


def thumbnail_icon(bgr, size=THUMB, dpr=2.0):
    img = thumbnail_image(bgr, size, dpr)
    return qtg.QIcon(qtg.QPixmap.fromImage(img)) if img is not None else qtg.QIcon()


class ThumbnailWorker(qtc.QRunnable):
    class Signals(qtc.QObject):
        ready = qtc.Signal(str, object, str, int)   # path, QImage, info text, generation

    def __init__(self, path, generation):
        super().__init__()
        self.path = path
        self.generation = generation
        self.signals = self.Signals()

    def run(self):
        img, info = None, ""
        try:
            img = cv2.imread(self.path, cv2.IMREAD_REDUCED_COLOR_8)
            if img is None:
                img = ImageLoadingHandler.ImageLoadingHandler().read_image_from_path(self.path)
            size = os.path.getsize(self.path)
            info = f"{size / 1e6:.1f} MB" if size >= 1e6 else f"{size / 1e3:.0f} KB"
        except Exception:
            pass
        if img is not None:
            thumb = thumbnail_image(img)
            if thumb is not None:
                self.signals.ready.emit(self.path, thumb, info, self.generation)


class ThumbList(qtw.QListWidget):
    """List with thumbnails, keyboard deletion and wrap-around navigation."""
    delete_requested = qtc.Signal()

    def __init__(self, deletable=True):
        super().__init__()
        self.deletable = deletable
        self.setIconSize(qtc.QSize(THUMB, THUMB))
        self.setUniformItemSizes(True)
        self.setSelectionMode(qtw.QAbstractItemView.ExtendedSelection)
        self.setVerticalScrollMode(qtw.QAbstractItemView.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        self.setTextElideMode(qtc.Qt.ElideMiddle)
        self.setSpacing(1)
        self.setFrameShape(qtw.QFrame.NoFrame)
        qtw.QScroller.grabGesture(self.viewport(), qtw.QScroller.TouchGesture)   # touch flick scrolling

    def keyPressEvent(self, event):
        if self.deletable and event.key() in (qtc.Qt.Key_Delete, qtc.Qt.Key_Backspace) \
                and self.selectedItems():
            self.delete_requested.emit()
            return
        super().keyPressEvent(event)

    def selected_paths(self):
        return [i.data(PATH_ROLE) for i in self.selectedItems()]


class SectionHeader(qtw.QWidget):
    def __init__(self, title):
        super().__init__()
        layout = qtw.QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 8, 4)
        layout.setSpacing(6)
        self.title = qtw.QLabel(title.upper())
        self.title.setObjectName("SectionTitle")
        self.badge = qtw.QLabel("")
        self.badge.setObjectName("Badge")
        self.badge.setVisible(False)
        layout.addWidget(self.title)
        layout.addWidget(self.badge)
        layout.addStretch()
        self.buttons = layout

    def add_button(self, button):
        self.buttons.addWidget(button)

    def set_count(self, n):
        self.badge.setText(str(n))
        self.badge.setVisible(n > 0)


def icon_button(name, tooltip, slot):
    b = qtw.QToolButton()
    b.setObjectName("IconButton")
    b.setIcon(icon(name))
    b.setIconSize(qtc.QSize(16, 16))
    b.setToolTip(tooltip)
    b.setAccessibleName(tooltip)
    b.setCursor(qtc.Qt.PointingHandCursor)
    b.clicked.connect(slot)
    return b


class Sidebar(qtw.QWidget):
    """Frames and results. Emits what the user picked; the main window owns
    the actual image list."""
    frame_selected = qtc.Signal(str)          # path
    result_selected = qtc.Signal(int)         # index in results
    remove_requested = qtc.Signal(list)       # paths
    add_requested = qtc.Signal()
    files_dropped = qtc.Signal(list)

    def __init__(self):
        super().__init__()
        self.setObjectName("Sidebar")
        self.setAttribute(qtc.Qt.WA_StyledBackground, True)
        self.setAcceptDrops(True)
        self.setMinimumWidth(180)
        self._pool = qtc.QThreadPool(self)
        self._pool.setMaxThreadCount(max(2, min(6, (os.cpu_count() or 4) // 2)))
        self._generation = 0
        self._rows = {}

        layout = qtw.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Frames
        self.frames_header = SectionHeader("Frames")
        self.frames_header.add_button(icon_button("add", "Add images", self.add_requested.emit))
        self.frames = ThumbList()
        self.frames.setAccessibleName("Frames")
        self.frames.delete_requested.connect(lambda: self.remove_requested.emit(self.frames.selected_paths()))
        self.frames.currentItemChanged.connect(self._on_frame_changed)
        self.frames.itemClicked.connect(lambda it: self._on_frame_changed(it, None))
        self.frames.setContextMenuPolicy(qtc.Qt.CustomContextMenu)
        self.frames.customContextMenuRequested.connect(self._frames_menu)

        self.frames_empty = qtw.QLabel("No frames yet.\nDrop images or a folder here.")
        self.frames_empty.setObjectName("Hint")
        self.frames_empty.setAlignment(qtc.Qt.AlignCenter)
        self.frames_empty.setWordWrap(True)
        self.frames_stack = qtw.QStackedWidget()
        self.frames_stack.addWidget(self.frames_empty)
        self.frames_stack.addWidget(self.frames)

        # Results
        self.results_header = SectionHeader("Results")
        self.results = ThumbList(deletable=False)
        self.results.setSelectionMode(qtw.QAbstractItemView.SingleSelection)
        self.results.setAccessibleName("Results")
        self.results.currentRowChanged.connect(lambda r: r >= 0 and self.result_selected.emit(r))
        self.results.itemClicked.connect(lambda it: self.result_selected.emit(self.results.row(it)))
        self.results_box = qtw.QWidget()
        rb = qtw.QVBoxLayout(self.results_box)
        rb.setContentsMargins(0, 0, 0, 0)
        rb.setSpacing(0)
        rb.addWidget(self.results_header)
        rb.addWidget(self.results)
        self.results_box.setVisible(False)

        frames_box = qtw.QWidget()
        fb = qtw.QVBoxLayout(frames_box)
        fb.setContentsMargins(0, 0, 0, 0)
        fb.setSpacing(0)
        fb.addWidget(self.frames_header)
        fb.addWidget(self.frames_stack)

        split = qtw.QSplitter(qtc.Qt.Vertical)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(1)
        split.addWidget(frames_box)
        split.addWidget(self.results_box)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        self.splitter = split
        layout.addWidget(split)

        for lst in (self.frames, self.results):
            lst.setStyleSheet("QListWidget { padding: 0 6px; }")

    # ── frames ──
    def set_frames(self, paths):
        self._generation += 1
        self.frames.blockSignals(True)
        self.frames.clear()
        for path in paths:
            item = qtw.QListWidgetItem(os.path.basename(path))
            item.setData(PATH_ROLE, path)
            item.setToolTip(path)
            item.setIcon(icon("image", "secondary"))
            self.frames.addItem(item)
        self.frames.blockSignals(False)
        self.frames_header.set_count(len(paths))
        self.frames_stack.setCurrentIndex(1 if paths else 0)
        self._rows = {path: i for i, path in enumerate(paths)}
        for path in paths:
            w = ThumbnailWorker(path, self._generation)
            w.signals.ready.connect(self._on_thumb)
            self._pool.start(w)

    def _on_thumb(self, path, image, info, generation):
        if generation != self._generation:
            return
        row = self._rows.get(path)
        item = self.frames.item(row) if row is not None else None
        if item is not None and item.data(PATH_ROLE) == path:
            item.setIcon(qtg.QIcon(qtg.QPixmap.fromImage(image)))
            item.setToolTip(f"{path}\n{info}")

    def select_frame(self, row):
        if 0 <= row < self.frames.count():
            self.frames.setCurrentRow(row)

    def current_frame_row(self):
        return self.frames.currentRow()

    def _on_frame_changed(self, item, _prev):
        if item is not None:
            self.results.blockSignals(True)
            self.results.clearSelection()
            self.results.setCurrentRow(-1)
            self.results.blockSignals(False)
            self.frame_selected.emit(item.data(PATH_ROLE))

    def _frames_menu(self, pos):
        item = self.frames.itemAt(pos)
        menu = qtw.QMenu(self)
        if item is not None:
            if not item.isSelected():
                self.frames.setCurrentItem(item)
            paths = self.frames.selected_paths()
            n = len(paths)
            remove = menu.addAction(icon("trash"), f"Remove {n} frame{'s' if n != 1 else ''}")
            remove.triggered.connect(lambda: self.remove_requested.emit(paths))
            reveal = menu.addAction(icon("reveal"), f"Show in {file_manager_name()}")
            reveal.triggered.connect(lambda: reveal_in_file_manager(item.data(PATH_ROLE)))
            menu.addSeparator()
        menu.addAction(icon("add"), "Add Images…").triggered.connect(self.add_requested.emit)
        menu.exec(self.frames.viewport().mapToGlobal(pos))

    # ── results ──
    def add_result(self, title, subtitle, bgr):
        item = qtw.QListWidgetItem(f"{title}\n{subtitle}")
        item.setIcon(thumbnail_icon(bgr))
        item.setToolTip(f"{title} · {subtitle}")
        self.results.insertItem(0, item)
        self.results_box.setVisible(True)
        self.results_header.set_count(self.results.count())
        self.frames.blockSignals(True)
        self.frames.clearSelection()
        self.frames.blockSignals(False)
        self.results.setCurrentRow(0)

    def clear_results(self):
        self.results.clear()
        self.results_box.setVisible(False)
        self.results_header.set_count(0)

    # ── drag & drop ──
    @staticmethod
    def _urls(event):
        md = event.mimeData()
        return [u.toLocalFile() for u in md.urls() if u.isLocalFile()] if md.hasUrls() else []

    def dragEnterEvent(self, event):
        if self._urls(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        paths = self._urls(event)
        if paths:
            event.acceptProposedAction()
            self.files_dropped.emit(paths)
