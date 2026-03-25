"""
Image viewer with smooth zoom (pinch + scroll), pan (trackpad + mouse), and touch support.
Uses QNativeGestureEvent for macOS trackpad pinch-to-zoom (QPinchGesture is broken on macOS).
"""
import PySide6.QtWidgets as qtw
import PySide6.QtCore as qtc
import PySide6.QtGui as qtg

import src.settings as settings
import src.MainWindow.MainLayout.ImageViewers.ImageScene as image_scene
import src.MainWindow.MainLayout.ImageViewers.ImageRetouchScene as image_retouch_scene


class ImageViewer(qtw.QGraphicsView):
    sendWheelEvent = qtc.Signal(qtg.QWheelEvent)
    current_zoom = 1.0
    min_zoom = 0.1
    max_zoom = 20.0

    def __init__(self, viewerScene=None):
        super().__init__()
        if not viewerScene:
            self.viewerScene = image_scene.ImageScene(self)
        else:
            self.viewerScene = viewerScene

        self.setScene(self.viewerScene)
        self.setTransformationAnchor(qtw.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(qtw.QGraphicsView.AnchorUnderMouse)

        self.setVerticalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)

        self.setBackgroundBrush(qtg.QBrush(qtg.QColor(30, 30, 30)))
        self.setFrameShape(qtw.QFrame.NoFrame)
        self.setDragMode(qtw.QGraphicsView.ScrollHandDrag)
        self.setCacheMode(qtw.QGraphicsView.CacheBackground)
        self.setViewportUpdateMode(qtw.QGraphicsView.SmartViewportUpdate)
        self.setOptimizationFlag(qtw.QGraphicsView.DontAdjustForAntialiasing, True)
        self.setRenderHint(qtg.QPainter.SmoothPixmapTransform, True)

        # Empty state overlay
        self.empty_state = qtw.QWidget(self)
        self.empty_state.setAttribute(qtc.Qt.WA_TransparentForMouseEvents)
        empty_layout = qtw.QVBoxLayout(self.empty_state)
        empty_layout.setAlignment(qtc.Qt.AlignCenter)

        title = qtw.QLabel("No Image Selected")
        title.setStyleSheet("font-size: 18px; font-weight: 600; color: #999999; background: transparent;")
        title.setAlignment(qtc.Qt.AlignCenter)

        subtitle = qtw.QLabel("Load images from File menu or drag and drop")
        subtitle.setStyleSheet("font-size: 12px; color: #666666; background: transparent;")
        subtitle.setAlignment(qtc.Qt.AlignCenter)

        empty_layout.addWidget(title)
        empty_layout.addWidget(subtitle)
        self.empty_state.setVisible(True)

    def set_image(self, image):
        self.viewerScene.set_image(image)
        self.empty_state.setVisible(image is None)

    def _apply_zoom(self, factor):
        """Apply a zoom factor. Cannot zoom out past fit-to-view (1.0)."""
        new_zoom = self.current_zoom * factor
        new_zoom = max(1.0, min(self.max_zoom, new_zoom))
        actual = new_zoom / self.current_zoom
        if abs(actual - 1.0) < 0.001:
            return
        self.current_zoom = new_zoom
        self.scale(actual, actual)
        self._update_zoom_label()

    def zoom_in(self):
        if self.viewerScene.hasImage:
            self._apply_zoom(1.25)

    def zoom_out(self):
        if self.viewerScene.hasImage:
            self._apply_zoom(1 / 1.25)

    def _update_zoom_label(self):
        label = settings.globalVars.get("ZoomLabel")
        if label:
            label.setText(f" {round(self.current_zoom * 100)}% ")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.empty_state.setGeometry(self.viewport().geometry())

    def fitInView(self):
        rect = qtc.QRectF(self.viewerScene.pixmapPicture.pixmap().rect())
        if not rect.isNull() and self.viewerScene.hasImage:
            self.setSceneRect(rect)
            unity = self.transform().mapRect(qtc.QRectF(0, 0, 1, 1))
            self.scale(1 / unity.width(), 1 / unity.height())
            viewrect = self.viewport().rect()
            scenerect = self.transform().mapRect(rect)
            factor = min(
                viewrect.width() / scenerect.width(),
                viewrect.height() / scenerect.height(),
            )
            self.scale(factor, factor)
            self.current_zoom = 1.0
            self._update_zoom_label()

    # --- Input: macOS native gestures (pinch-to-zoom) ---

    def event(self, ev):
        """Catch QNativeGestureEvent for macOS trackpad pinch-to-zoom."""
        if ev.type() == qtc.QEvent.Type.NativeGesture:
            return self._handle_native_gesture(ev)
        return super().event(ev)

    def _handle_native_gesture(self, ev):
        """Handle macOS native gesture events."""
        if not self.viewerScene.hasImage:
            return False

        gesture_type = ev.gestureType()

        if gesture_type == qtc.Qt.NativeGestureType.ZoomNativeGesture:
            # ev.value() is an incremental factor (small, e.g. 0.02)
            # Formula: scale *= (1 + value)
            factor = 1.0 + ev.value()
            self._apply_zoom(factor)
            ev.accept()
            return True

        elif gesture_type == qtc.Qt.NativeGestureType.SmartZoomNativeGesture:
            # Double-tap with two fingers — toggle fit/100%
            if self.current_zoom < 1.5:
                self._apply_zoom(2.0 / self.current_zoom)
            else:
                self.fitInView()
            ev.accept()
            return True

        return False

    # --- Input: mouse wheel + trackpad scroll ---

    def wheelEvent(self, event: qtg.QWheelEvent):
        self.sendWheelEvent.emit(event)

        if not self.viewerScene.hasImage:
            return

        # Detect trackpad vs mouse:
        # Trackpad sends pixelDelta, mouse sends only angleDelta
        has_pixel = event.pixelDelta() != qtc.QPoint(0, 0)

        if has_pixel:
            # Trackpad two-finger scroll = pan
            delta = event.pixelDelta()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
        else:
            # Mouse wheel = zoom
            angle = event.angleDelta().y()
            if angle != 0:
                factor = 1.0 + angle / 480.0
                factor = max(0.8, min(1.25, factor))
                self._apply_zoom(factor)

        event.accept()

    # --- Input: mouse ---

    def mouseDoubleClickEvent(self, event):
        """Double-click to fit image to view."""
        if self.viewerScene.hasImage:
            self.fitInView()
        super().mouseDoubleClickEvent(event)

    def handleWheelEvent(self, event: qtg.QWheelEvent):
        """External wheel event handler (for synced viewers)."""
        if not self.viewerScene.hasImage:
            return
        angle = event.angleDelta().y()
        if angle != 0:
            factor = 1.0 + angle / 480.0
            self._apply_zoom(max(0.8, min(1.25, factor)))
            return True


# Retouching viewer
class ImageRetouchViewer(ImageViewer):
    def __init__(self):
        viewerScene = image_retouch_scene.ImageRetouchScene(self)
        super().__init__(viewerScene)


class RetouchingTopWidget(qtw.QWidget):
    def __init__(self):
        super().__init__()
        combobox = qtw.QComboBox()
        combobox.addItems({"Direct copy": 0, "Lighten": 1, "Darken": 2})
        button = qtw.QPushButton("Save output image.")
        self.setMaximumHeight(50)
        hLayout = qtw.QHBoxLayout()
        hLayout.addWidget(combobox)
        hLayout.addWidget(button)
        self.setLayout(hLayout)


class ImageRetouchingWidget(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.retouch_viewer = ImageRetouchViewer()
        self.image_viewer = ImageViewer()

        vSplitter = qtw.QSplitter()
        vSplitter.setChildrenCollapsible(False)
        vSplitter.addWidget(self.retouch_viewer)
        vSplitter.addWidget(self.image_viewer)
        vSplitter.setSizes([400, 400])

        vLayout = qtw.QVBoxLayout()
        vLayout.addWidget(RetouchingTopWidget())
        vLayout.addWidget(vSplitter)
        self.setLayout(vLayout)

        self.retouch_viewer.verticalScrollBar().valueChanged.connect(
            self.image_viewer.verticalScrollBar().setValue)
        self.retouch_viewer.horizontalScrollBar().valueChanged.connect(
            self.image_viewer.horizontalScrollBar().setValue)
        self.image_viewer.verticalScrollBar().valueChanged.connect(
            self.retouch_viewer.verticalScrollBar().setValue)
        self.image_viewer.horizontalScrollBar().valueChanged.connect(
            self.retouch_viewer.horizontalScrollBar().setValue)
        self.retouch_viewer.sendWheelEvent.connect(self.image_viewer.handleWheelEvent)
        self.image_viewer.sendWheelEvent.connect(self.retouch_viewer.handleWheelEvent)

    def set_retouch_image(self, image):
        self.retouch_viewer.set_image(image)

    def set_output_image(self, image):
        if image is None:
            self.image_viewer.set_image(None)
            return
        qImage = qtg.QImage(
            image, image.shape[1], image.shape[0],
            image.shape[1] * 3, qtg.QImage.Format_RGB888,
        )
        if (self.image_viewer.viewerScene.hasImage
                and qImage != self.image_viewer.viewerScene.currentQImage):
            reply = qtw.QMessageBox.question(
                self, "Change output?",
                "Select a new output image for retouching?",
                qtw.QMessageBox.Cancel, qtw.QMessageBox.Ok,
            )
            if reply == qtw.QMessageBox.Ok:
                self.image_viewer.set_image(image)
        else:
            self.image_viewer.set_image(image)
