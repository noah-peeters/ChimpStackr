"""
Scene for regular image display (no retouching).
Retouching scene inherits this class.
"""
import PySide6.QtWidgets as qtw
import PySide6.QtGui as qtg


class ImageScene(qtw.QGraphicsScene):
    hasImage = False
    adjust_zoom = True

    def __init__(self, graphicsViewer):
        super().__init__()
        self.pixmapPicture = qtw.QGraphicsPixmapItem()
        self.addItem(self.pixmapPicture)

        self.graphicsViewer = graphicsViewer
        self.currentQImage = None

    # Display new image (RGB)
    def set_image(self, image):
        if image is None:
            # Clear pixmap image
            self.pixmapPicture.setPixmap(qtg.QPixmap())
            self.hasImage = False
            self.currentQImage = None
        else:
            # Convert np RGB array to QImage
            # src: https://stackoverflow.com/questions/34232632/convert-python-opencv-image-numpy-array-to-pyqt-qpixmap-image
            self.currentQImage = qtg.QImage(
                image,
                image.shape[1],
                image.shape[0],
                image.shape[1] * 3,
                qtg.QImage.Format_RGB888,
            )
            self.pixmapPicture.setPixmap(qtg.QPixmap.fromImage(self.currentQImage))
            self.hasImage = True
            if self.adjust_zoom:
                self.graphicsViewer.fitInView()

    """
        Overridden signals
    """

    # Display image viewer options on right click
    def contextMenuEvent(self, event: qtw.QGraphicsSceneContextMenuEvent) -> None:
        menu = qtw.QMenu()
        menu.setToolTipsVisible(True)
        zoom_action = qtg.QAction("Adjust zoom")
        zoom_action.setToolTip("Adjust zoom on change (for better image fit).")
        zoom_action.setCheckable(True)
        zoom_action.setChecked(self.adjust_zoom)

        fit_action = qtg.QAction("Fit to view")
        fit_action.setToolTip("Fit image to current view.")

        menu.addAction(zoom_action)
        menu.addAction(fit_action)
        selected_action = menu.exec(event.screenPos())

        if selected_action == zoom_action:
            self.adjust_zoom = zoom_action.isChecked()
        elif selected_action == fit_action:
            self.graphicsViewer.fitInView()
