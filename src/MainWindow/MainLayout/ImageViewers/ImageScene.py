"""
Scene for regular image display (no retouching).
Retouching scene inherits this class.
"""
import PySide6.QtWidgets as qtw
import PySide6.QtCore as qtc
import PySide6.QtGui as qtg


class ImageScene(qtw.QGraphicsScene):
    hasImage = False
    current_zoom = 1
    zoom_in_factor = 1.15  # zoom out is derived as: 1/zoom_in
    max_zoom_in = 10  # x100 to get percentage
    adjust_zoom = True
    tooltip_displaytime_ms = 750

    def __init__(self, graphicsViewer):
        super().__init__()
        self.pixmapPicture = qtw.QGraphicsPixmapItem()
        self.addItem(self.pixmapPicture)

        self.graphicsViewer = graphicsViewer

        # Tooltip hovering next to mouse (used for info like zoom percentage)
        self.mouse_tooltip = qtw.QToolTip()

        self.zoom_out_factor = 1 / self.zoom_in_factor

    # Display new image (RGB)
    def set_image(self, image):
        if image is None:
            # Clear pixmap image
            self.pixmapPicture.setPixmap(qtg.QPixmap())
            self.hasImage = False
        else:
            # Convert np RGB array to QImage
            # src: https://stackoverflow.com/questions/34232632/convert-python-opencv-image-numpy-array-to-pyqt-qpixmap-image
            qimage = qtg.QImage(
                image,
                image.shape[1],
                image.shape[0],
                image.shape[1] * 3,
                qtg.QImage.Format_RGB888,
            )
            self.pixmapPicture.setPixmap(qtg.QPixmap.fromImage(qimage))
            self.hasImage = True
            if self.adjust_zoom:
                self.fitInView()

    # Fit image to view (internal use; uses parent QGraphicsViewer)
    def fitInView(self):
        rect = qtc.QRectF(self.pixmapPicture.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasImage:
                unity = self.graphicsViewer.transform().mapRect(qtc.QRectF(0, 0, 1, 1))
                self.graphicsViewer.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.graphicsViewer.viewport().rect()
                scenerect = self.graphicsViewer.transform().mapRect(rect)
                factor = min(
                    viewrect.width() / scenerect.width(),
                    viewrect.height() / scenerect.height(),
                )
                self.graphicsViewer.scale(factor, factor)
                self.current_zoom = 1

    """
        Overridden signals
    """

    # Zoom in/out on mousewheel scroll
    def wheelEvent(self, event):
        # Only zoom when Ctrl is pressed
        if not event.modifiers() & qtc.Qt.ControlModifier:
            return

        if self.hasImage:
            if event.delta() > 0:
                # Zoom in
                if self.current_zoom * self.zoom_in_factor <= self.max_zoom_in + 1:
                    self.current_zoom *= self.zoom_in_factor
                    self.graphicsViewer.scale(self.zoom_in_factor, self.zoom_in_factor)
                else:
                    new_zoom = (self.max_zoom_in + 1) / self.current_zoom
                    self.current_zoom *= new_zoom
                    self.graphicsViewer.scale(new_zoom, new_zoom)
            else:
                # Zoom out
                if self.current_zoom * self.zoom_out_factor >= 1:
                    self.current_zoom *= self.zoom_out_factor
                    self.graphicsViewer.scale(
                        self.zoom_out_factor, self.zoom_out_factor
                    )
                else:
                    self.fitInView()

            self.mouse_tooltip.showText(
                qtg.QCursor.pos(),
                str(round((self.current_zoom - 1) * 100, 2)) + "% zoom",
                msecShowTime=self.tooltip_displaytime_ms,
            )
            event.accept()

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
            self.fitInView()
