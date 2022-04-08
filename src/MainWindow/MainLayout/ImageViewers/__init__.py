"""
Exposes viewer objects.
These should be used by external scripts.
"""
import PySide6.QtWidgets as qtw
import PySide6.QtCore as qtc
import PySide6.QtGui as qtg

import src.MainWindow.MainLayout.ImageViewers.ImageScene as image_scene
import src.MainWindow.MainLayout.ImageViewers.ImageRetouchScene as image_retouch_scene

# Regular viewer
class ImageViewer(qtw.QGraphicsView):
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

    # Convenience for parent script (no need to call through "viewerScene")
    def set_image(self, image):
        self.viewerScene.set_image(image)


# Retouching viewer
class ImageRetouchViewer(ImageViewer):
    def __init__(self):
        # Scene setup
        viewerScene = image_retouch_scene.ImageRetouchScene(self)
        super().__init__(viewerScene)