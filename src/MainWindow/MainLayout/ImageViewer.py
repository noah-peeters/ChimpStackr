"""
    Image viewer usinng QGraphicsView.
    Allows for zooming in/out on image and panning on mouse drag.
"""
import PySide6.QtCore as qtc
import PySide6.QtGui as qtg
import PySide6.QtWidgets as qtw

import cv2

import ImageLoadingHandler


# TODO: Display current zoom factor (%)
class ImageViewer(qtw.QGraphicsView):
    photoClicked = qtc.Signal(qtc.QPoint)
    hasImage = False
    zoom_factor = 1
    zoom_increment = 0.25
    reset_zoom = True

    def __init__(self):
        super().__init__()

        # Tooltip displaying current zoom percentage
        self.zoom_percentage_tooltip = qtw.QToolTip()

        self.ImageLoading = ImageLoadingHandler.ImageLoadingHandler()

        # Scene setup
        self._scene = qtw.QGraphicsScene(self)
        self._photo = qtw.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)

        self.setScene(self._scene)
        self.setTransformationAnchor(qtw.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(qtw.QGraphicsView.AnchorUnderMouse)

        self.setVerticalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)

        self.setBackgroundBrush(qtg.QBrush(qtg.QColor(30, 30, 30)))
        self.setFrameShape(qtw.QFrame.NoFrame)

    # Fit image to view
    def fitInView(self):
        rect = qtc.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasImage:
                unity = self.transform().mapRect(qtc.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(
                    viewrect.width() / scenerect.width(),
                    viewrect.height() / scenerect.height(),
                )
                self.scale(factor, factor)
                self.zoom_factor = 1

    # Change displayed image
    def update_displayed_image(self, selected_widget_item):
        if selected_widget_item:
            # Display selected image
            path = selected_widget_item.data(qtc.Qt.UserRole)
            if path != None:
                image = self.ImageLoading.read_image_from_path(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Convert np RGB array to QImage
                # src: https://stackoverflow.com/questions/34232632/convert-python-opencv-image-numpy-array-to-pyqt-qpixmap-image
                qimage = qtg.QImage(
                    image,
                    image.shape[1],
                    image.shape[0],
                    image.shape[1] * 3,
                    qtg.QImage.Format_RGB888,
                )

                self.setDragMode(qtw.QGraphicsView.ScrollHandDrag)
                self._photo.setPixmap(qtg.QPixmap.fromImage(qimage))
                self.hasImage = True
        else:
            # Clear image view
            self.setDragMode(qtw.QGraphicsView.NoDrag)
            self._photo.setPixmap(qtg.QPixmap())
            self.hasImage = False

        # Reset zoom if enabled
        if self.reset_zoom:
            self.fitInView()

    """
        Overridden signals
    """

    # Fit image on window resize
    def resizeEvent(self, event: qtg.QResizeEvent) -> None:
        self.fitInView()
        return super().resizeEvent(event)

    # Zoom in/out on mousewheel scroll
    def wheelEvent(self, event):
        if self.hasImage:
            if event.angleDelta().y() > 0:
                # Zoom in (25%)
                if self.zoom_factor < 3:
                    self.zoom_factor += self.zoom_increment
                    self.scale(self.zoom_factor, self.zoom_factor)
            else:
                # Zoom out (25%)
                if self.zoom_factor > 1:
                    self.zoom_factor -= self.zoom_increment
                    factor = 1 / (self.zoom_factor + self.zoom_increment)
                    self.scale(factor, factor)
                else:
                    self.fitInView()
            self.zoom_percentage_tooltip.showText(
                qtg.QCursor.pos(),
                str((self.zoom_factor - 1) * 100) + "%",
                msecShowTime=750,
            )

    def toggleDragMode(self):
        if self.dragMode() == qtw.QGraphicsView.ScrollHandDrag:
            self.setDragMode(qtw.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(qtw.QGraphicsView.ScrollHandDrag)

    # Move (zoomed) image around on left mouse drag
    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super().mousePressEvent(event)

    # Display image viewer options on right click
    def contextMenuEvent(self, event: qtg.QContextMenuEvent) -> None:
        menu = qtw.QMenu()

        reset_zoom_action = qtg.QAction("Reset zoom")
        reset_zoom_action.setStatusTip("Reset zoom in between image selections.")
        reset_zoom_action.setCheckable(True)
        reset_zoom_action.setChecked(self.reset_zoom)

        menu.addAction(reset_zoom_action)
        selected_action = menu.exec(event.globalPos())

        if selected_action == reset_zoom_action:
            self.reset_zoom = reset_zoom_action.isChecked()
