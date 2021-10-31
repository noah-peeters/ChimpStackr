"""
    Image viewer usinng QGraphicsView.
    Allows for zooming in/out on image and panning on mouse drag.
"""
import PySide6.QtCore as qtc
import PySide6.QtGui as qtg
import PySide6.QtWidgets as qtw

from PIL import Image, ImageQt
import cv2


class ImageViewer(qtw.QGraphicsView):
    photoClicked = qtc.Signal(qtc.QPoint)

    def __init__(self, loaded_images_list):
        super().__init__()
        self.current_zoom_level = 0
        self.reset_zoom = True
        self._scene = qtw.QGraphicsScene(self)
        self._photo = qtw.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)

        # Scene setup
        self.setScene(self._scene)
        self.setTransformationAnchor(qtw.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(qtw.QGraphicsView.AnchorUnderMouse)

        self.setVerticalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)

        self.setBackgroundBrush(qtg.QBrush(qtg.QColor(30, 30, 30)))
        self.setFrameShape(qtw.QFrame.NoFrame)

        # Display selected image from list (connection to signal)
        loaded_images_list.currentItemChanged.connect(self.update_displayed_image)

    # Fit image to view
    def fitInView(self, scale=True):  # TODO: Implement/remove self.scale
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
            self.current_zoom_level = 0

    # Set image
    # def setImage(self, pixmap=None):
    #     if pixmap and not pixmap.isNull():
    #         self.image_loaded = False
    #         self.setDragMode(qtw.QGraphicsView.ScrollHandDrag)
    #         self._photo.setPixmap(pixmap)
    #     else:
    #         self.image_loaded = True
    #         self.setDragMode(qtw.QGraphicsView.NoDrag)
    #         self._photo.setPixmap(qtg.QPixmap())

    #     # Reset zoom
    #     if self.reset_zoom:
    #         self.current_zoom_level = 0
    #         self.fitInView()

    # Change displayed image
    def update_displayed_image(self, new_widget_item, prev_widget_item):
        print("Change clicked image")
        # TODO: Gets called many times on loaded images widget item change
        if new_widget_item:
            # Display selected image
            path = new_widget_item.data(qtc.Qt.UserRole)
            if path != None:
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                qimage = ImageQt.ImageQt(Image.fromarray(image))

                self.setDragMode(qtw.QGraphicsView.ScrollHandDrag)
                self._photo.setPixmap(qtg.QPixmap.fromImage(qimage))

                self.hasImage = True
        else:
            # Clear image view
            self.setDragMode(qtw.QGraphicsView.NoDrag)
            self._photo.setPixmap(qtg.QPixmap())
            self.hasImage = False

    """
        Overridden signals
    """

    def wheelEvent(self, event):
        if self.hasImage:
            if event.angleDelta().y() > 0:
                factor = 1.25
                self.current_zoom_level += 1
            else:
                factor = 0.8
                self.current_zoom_level -= 1
            if self.current_zoom_level > 0:
                self.scale(factor, factor)
            elif self.current_zoom_level == 0:
                self.fitInView()
            else:
                self.current_zoom_level = 0

    def toggleDragMode(self):
        if self.dragMode() == qtw.QGraphicsView.ScrollHandDrag:
            self.setDragMode(qtw.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(qtw.QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super().mousePressEvent(event)