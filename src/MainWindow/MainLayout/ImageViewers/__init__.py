"""
Exposes viewer objects.
These should be used by external scripts.
"""
import PySide6.QtWidgets as qtw
import PySide6.QtCore as qtc
import PySide6.QtGui as qtg
from sqlalchemy import true

import src.MainWindow.MainLayout.ImageViewers.ImageScene as image_scene
import src.MainWindow.MainLayout.ImageViewers.ImageRetouchScene as image_retouch_scene

# Regular viewer
class ImageViewer(qtw.QGraphicsView):
    sendWheelEvent = qtc.Signal(qtg.QWheelEvent)
    current_zoom = 1
    zoom_in_factor = 1.15  # zoom out is derived as: 1/zoom_in
    max_zoom_in = 10  # x100 to get percentage
    tooltip_displaytime_ms = 750

    def __init__(self, viewerScene=None):
        super().__init__()
        # Scene setup
        if not viewerScene:
            self.viewerScene = image_scene.ImageScene(self)
        else:
            self.viewerScene = viewerScene

        self.setScene(self.viewerScene)
        self.setTransformationAnchor(qtw.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(qtw.QGraphicsView.AnchorUnderMouse)

        self.setVerticalScrollBarPolicy(qtc.Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAsNeeded)

        self.setBackgroundBrush(qtg.QBrush(qtg.QColor(30, 30, 30)))
        self.setFrameShape(qtw.QFrame.NoFrame)
        self.setDragMode(qtw.QGraphicsView.ScrollHandDrag)

        # Tooltip hovering next to mouse (used for info like zoom percentage)
        self.mouse_tooltip = qtw.QToolTip()
        self.zoom_out_factor = 1 / self.zoom_in_factor

    # Convenience for parent script (no need to call through "viewerScene")
    def set_image(self, image):
        self.viewerScene.set_image(image)

    # Fit image to view (internal use; uses parent QGraphicsViewer)
    def fitInView(self):
        rect = qtc.QRectF(self.viewerScene.pixmapPicture.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.viewerScene.hasImage:
                unity = self.transform().mapRect(qtc.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(
                    viewrect.width() / scenerect.width(),
                    viewrect.height() / scenerect.height(),
                )
                self.scale(factor, factor)
                self.current_zoom = 1

    # Separate to prevent infinite recursion signal call
    def handleWheelEvent(self, event: qtg.QWheelEvent) -> None:
        # Only zoom when Ctrl is pressed
        if event.modifiers() & qtc.Qt.ControlModifier:
            if self.viewerScene.hasImage:
                if event.angleDelta().y() > 0:
                    # Zoom in
                    if self.current_zoom * self.zoom_in_factor <= self.max_zoom_in + 1:
                        self.current_zoom *= self.zoom_in_factor
                        self.scale(self.zoom_in_factor, self.zoom_in_factor)
                    else:
                        new_zoom = (self.max_zoom_in + 1) / self.current_zoom
                        self.current_zoom *= new_zoom
                        self.scale(new_zoom, new_zoom)
                else:
                    # Zoom out
                    if self.current_zoom * self.zoom_out_factor >= 1:
                        self.current_zoom *= self.zoom_out_factor
                        self.scale(self.zoom_out_factor, self.zoom_out_factor)
                    else:
                        self.fitInView()

                self.mouse_tooltip.showText(
                    qtg.QCursor.pos(),
                    str(round((self.current_zoom - 1) * 100, 2)) + "% zoom",
                    msecShowTime=self.tooltip_displaytime_ms,
                )
                return True

    """
        Overridden signals
    """

    # Zoom in/out on mousewheel scroll
    def wheelEvent(self, event: qtg.QWheelEvent) -> None:
        self.sendWheelEvent.emit(event)
        if self.handleWheelEvent(event) == True:
            event.accept()
            return
        super().wheelEvent(event)


# Retouching viewer
class ImageRetouchViewer(ImageViewer):
    def __init__(self):
        # Scene setup
        viewerScene = image_retouch_scene.ImageRetouchScene(self)
        super().__init__(viewerScene)


# Widget that displays a retouching viewer, and a regular image viewer
class ImageRetouchingWidget(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.retouch_viewer = ImageRetouchViewer()
        self.image_viewer = ImageViewer()

        vSplitter = qtw.QSplitter()
        vSplitter.setChildrenCollapsible(False)
        vSplitter.addWidget(self.retouch_viewer)
        vSplitter.addWidget(self.image_viewer)

        width = int(self.size().width() / 2)
        vSplitter.setSizes([width, width])

        hLayout = qtw.QHBoxLayout()
        hLayout.addWidget(vSplitter)
        self.setLayout(hLayout)

        # Sync movement between viewers
        self.retouch_viewer.verticalScrollBar().valueChanged.connect(
            self.image_viewer.verticalScrollBar().setValue
        )
        self.retouch_viewer.horizontalScrollBar().valueChanged.connect(
            self.image_viewer.horizontalScrollBar().setValue
        )
        self.image_viewer.verticalScrollBar().valueChanged.connect(
            self.retouch_viewer.verticalScrollBar().setValue
        )
        self.image_viewer.horizontalScrollBar().valueChanged.connect(
            self.retouch_viewer.horizontalScrollBar().setValue
        )
        # Sync zoom between viewers
        self.retouch_viewer.sendWheelEvent.connect(self.image_viewer.handleWheelEvent)
        self.image_viewer.sendWheelEvent.connect(self.retouch_viewer.handleWheelEvent)

    def set_retouch_image(self, image):
        self.retouch_viewer.set_image(image)

    def set_image(self, image):
        self.image_viewer.set_image(image)