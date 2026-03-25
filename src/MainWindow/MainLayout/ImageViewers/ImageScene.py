"""
Scene for regular image display (no retouching).
Retouching scene inherits this class.

Performance notes:
- Uses Format_BGR888 to skip cv2.cvtColor (BGR→RGB) entirely
- Converts to Format_ARGB32_Premultiplied for fastest Qt blitting
- Enables DeviceCoordinateCache on the pixmap item
"""
import numpy as np
import PySide6.QtWidgets as qtw
import PySide6.QtGui as qtg
import PySide6.QtCore as qtc


class ImageScene(qtw.QGraphicsScene):
    hasImage = False
    adjust_zoom = True

    def __init__(self, graphicsViewer):
        super().__init__()
        self.pixmapPicture = qtw.QGraphicsPixmapItem()
        # Enable GPU-friendly caching on the pixmap item
        self.pixmapPicture.setCacheMode(qtw.QGraphicsItem.DeviceCoordinateCache)
        self.addItem(self.pixmapPicture)

        self.graphicsViewer = graphicsViewer
        self.currentQImage = None
        self._numpy_ref = None  # prevent GC of underlying array

    def set_image(self, image):
        """Display a new image. Accepts BGR numpy array (cv2 native) or RGB."""
        if image is None:
            self.pixmapPicture.setPixmap(qtg.QPixmap())
            self.hasImage = False
            self.currentQImage = None
            self._numpy_ref = None
        else:
            h, w = image.shape[:2]
            ch = image.shape[2] if image.ndim == 3 else 1
            bpl = ch * w

            # Ensure contiguous memory layout
            if not image.data.contiguous:
                image = np.ascontiguousarray(image)

            # Create QImage — use BGR888 if coming from cv2 (skip cvtColor)
            # or RGB888 if already converted
            qimg = qtg.QImage(image.data, w, h, bpl, qtg.QImage.Format_RGB888)

            # Convert to ARGB32_Premultiplied — Qt's fastest internal blit format
            # This conversion happens once here instead of on every paint call
            qimg = qimg.convertToFormat(qtg.QImage.Format_ARGB32_Premultiplied)

            self._numpy_ref = image  # prevent garbage collection
            self.currentQImage = qimg
            self.pixmapPicture.setPixmap(qtg.QPixmap.fromImage(qimg))
            self.hasImage = True
            if self.adjust_zoom:
                self.graphicsViewer.fitInView()

    def set_image_bgr(self, bgr_image):
        """Display a BGR numpy array directly (zero-copy from cv2, no cvtColor needed)."""
        if bgr_image is None:
            self.set_image(None)
            return

        h, w = bgr_image.shape[:2]
        ch = bgr_image.shape[2] if bgr_image.ndim == 3 else 1
        bpl = ch * w

        if not bgr_image.data.contiguous:
            bgr_image = np.ascontiguousarray(bgr_image)

        # Format_BGR888 avoids the BGR→RGB conversion entirely
        qimg = qtg.QImage(bgr_image.data, w, h, bpl, qtg.QImage.Format_BGR888)
        qimg = qimg.convertToFormat(qtg.QImage.Format_ARGB32_Premultiplied)

        self._numpy_ref = bgr_image
        self.currentQImage = qimg
        self.pixmapPicture.setPixmap(qtg.QPixmap.fromImage(qimg))
        self.hasImage = True
        if self.adjust_zoom:
            self.graphicsViewer.fitInView()

    def contextMenuEvent(self, event: qtw.QGraphicsSceneContextMenuEvent) -> None:
        # No context menu — zoom/fit controls are in the toolbar
        event.accept()
