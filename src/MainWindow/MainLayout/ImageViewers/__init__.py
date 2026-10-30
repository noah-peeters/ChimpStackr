"""
Image viewer: zoom, pan and before/after comparison for every input device.

- Mouse wheel zooms at the cursor; a trackpad's two-finger scroll pans, and
  pinch zooms (macOS native gesture, Windows precision touchpads, which send
  Ctrl+wheel, and touchscreens via QPinchGesture). Ctrl/Cmd+scroll zooms
  with any device.
- Arrow keys step through the frames at Fit, and pan when zoomed in.
- Drag (mouse, pen or one finger) pans; double-click / double-tap toggles
  Fit and 100 %; a two-finger double-tap on a Mac trackpad does the same.
- Zoom is the real pixel scale (100 % = one image pixel per screen point),
  and pixels are shown crisp above 200 %.
- Compare mode shows the "before" image left of a draggable divider, at the
  same zoom and position as the result.
"""
import numpy as np
import PySide6.QtCore as qtc
import PySide6.QtGui as qtg
import PySide6.QtWidgets as qtw

import src.MainWindow.theme as theme
from src.MainWindow.icons import icon


def _qimage_from_array(image, bgr=True):
    """QImage (ARGB32 premultiplied, owning its data) from an 8-bit HxW(x3)
    array, or None."""
    if image is None:
        return None
    image = np.ascontiguousarray(image)
    if image.dtype == np.uint16:
        image = (image >> 8).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(np.around(image), 0, 255).astype(np.uint8)
    image = np.ascontiguousarray(image)
    h, w = image.shape[:2]
    if image.ndim == 2:
        qimg = qtg.QImage(image.data, w, h, w, qtg.QImage.Format.Format_Grayscale8)
    else:
        fmt = qtg.QImage.Format.Format_BGR888 if bgr else qtg.QImage.Format.Format_RGB888
        qimg = qtg.QImage(image.data, w, h, w * 3, fmt)
    # convertToFormat copies, so the QImage no longer references `image`
    return qimg.convertToFormat(qtg.QImage.Format.Format_ARGB32_Premultiplied)


class _ClippedPixmapItem(qtw.QGraphicsPixmapItem):
    """Pixmap drawn only left of the viewer's comparison divider."""

    def __init__(self, viewer):
        super().__init__()
        self._viewer = viewer

    def paint(self, painter, option, widget=None):
        x = self._viewer.split_scene_x()
        if x is None:
            return
        rect = self.boundingRect()
        rect.setRight(min(rect.right(), self.mapFromScene(x, 0).x()))
        if rect.width() <= 0:
            return
        painter.save()
        painter.setClipRect(rect)
        super().paint(painter, option, widget)
        painter.restore()


class EmptyState(qtw.QWidget):
    """Drop target shown while nothing is loaded."""
    add_clicked = qtc.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dragging = False
        layout = qtw.QVBoxLayout(self)
        layout.setAlignment(qtc.Qt.AlignCenter)
        layout.setSpacing(10)

        self.icon_label = qtw.QLabel()
        self.icon_label.setAlignment(qtc.Qt.AlignCenter)
        self.title = qtw.QLabel("Drop your focus stack here")
        self.title.setAlignment(qtc.Qt.AlignCenter)
        f = self.title.font()
        f.setPointSizeF(f.pointSizeF() * 1.35)
        f.setWeight(qtg.QFont.DemiBold)
        self.title.setFont(f)
        self.subtitle = qtw.QLabel("Images or a whole folder: JPEG, TIFF, PNG and camera RAW")
        self.subtitle.setObjectName("Secondary")
        self.subtitle.setAlignment(qtc.Qt.AlignCenter)
        self.button = qtw.QPushButton("Add Images…")
        self.button.setObjectName("PrimaryButton")
        self.button.setCursor(qtc.Qt.PointingHandCursor)
        self.button.clicked.connect(self.add_clicked)
        self.shortcut_hint = qtw.QLabel(
            f"or press {qtg.QKeySequence(qtg.QKeySequence.Open).toString(qtg.QKeySequence.NativeText)}")
        self.shortcut_hint.setObjectName("Hint")
        self.shortcut_hint.setAlignment(qtc.Qt.AlignCenter)

        layout.addWidget(self.icon_label)
        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)
        layout.addSpacing(6)
        layout.addWidget(self.button, 0, qtc.Qt.AlignCenter)
        layout.addWidget(self.shortcut_hint)
        self.refresh_theme()

    def refresh_theme(self):
        self.icon_label.setPixmap(icon("drop", "secondary").pixmap(56, 56))
        self.update()

    def set_dragging(self, on):
        self._dragging = on
        self.update()

    def paintEvent(self, event):
        t = theme.tokens()
        p = qtg.QPainter(self)
        p.setRenderHint(qtg.QPainter.Antialiasing)
        rect = qtc.QRectF(self.rect()).adjusted(24, 24, -24, -24)
        color = qtg.QColor(t.accent if self._dragging else t.border)
        if self._dragging:
            fill = qtg.QColor(t.accent)
            fill.setAlpha(28)
            p.setBrush(fill)
        else:
            p.setBrush(qtc.Qt.NoBrush)
        pen = qtg.QPen(color, 2 if self._dragging else 1.5, qtc.Qt.DashLine)
        pen.setDashPattern([6, 5])
        p.setPen(pen)
        p.drawRoundedRect(rect, theme.RADIUS_LG, theme.RADIUS_LG)


class ImageViewer(qtw.QGraphicsView):
    zoom_changed = qtc.Signal(float)          # pixel scale (1.0 = 100 %)
    files_dropped = qtc.Signal(list)
    add_requested = qtc.Signal()
    step_requested = qtc.Signal(int)         # arrow keys at fit: previous / next frame

    MAX_ZOOM = 32.0
    ZOOM_STEP = 1.25
    SPLIT_GRAB = 14          # px either side of the divider that grab it

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = qtw.QGraphicsScene(self)
        self.setScene(self._scene)
        self.after_item = qtw.QGraphicsPixmapItem()
        self.after_item.setTransformationMode(qtc.Qt.SmoothTransformation)
        self.before_item = _ClippedPixmapItem(self)
        self.before_item.setTransformationMode(qtc.Qt.SmoothTransformation)
        self.before_item.setZValue(1)
        self.before_item.setVisible(False)
        self._scene.addItem(self.after_item)
        self._scene.addItem(self.before_item)

        self.has_image = False
        self.compare = False
        self._has_before = False
        self.split = 0.5               # divider position, fraction of the viewport width
        self._dragging_split = False
        self._fit_mode = True          # keep fitting on resize until the user zooms
        self._zoom = 1.0
        self._last_pinch = 1.0

        self.setTransformationAnchor(qtw.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(qtw.QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        self.setFrameShape(qtw.QFrame.NoFrame)
        self.setDragMode(qtw.QGraphicsView.ScrollHandDrag)
        self.setViewportUpdateMode(qtw.QGraphicsView.FullViewportUpdate)
        self.setRenderHint(qtg.QPainter.SmoothPixmapTransform, True)
        self.setFocusPolicy(qtc.Qt.StrongFocus)
        self.setAcceptDrops(True)
        self.viewport().setAttribute(qtc.Qt.WA_AcceptTouchEvents, True)
        self.viewport().grabGesture(qtc.Qt.PinchGesture)
        self.setAccessibleName("Image viewer")

        self.empty_state = EmptyState(self)
        self.empty_state.add_clicked.connect(self.add_requested)

        self.loading_label = qtw.QLabel("Loading…", self)
        self.loading_label.setObjectName("Badge")
        self.loading_label.setVisible(False)

        tm = theme.manager()
        if tm is not None:
            tm.changed.connect(self._refresh_theme)
        self._refresh_theme()

    # ── content ──
    def _refresh_theme(self):
        self.setBackgroundBrush(qtg.QColor(theme.tokens().canvas))
        self.empty_state.refresh_theme()
        self.viewport().update()

    def set_image(self, image, bgr=True):
        """Show an image (8-bit or float 0-255 array, BGR by default), or
        clear the viewer with None. Keeps the zoom when the size matches."""
        qimg = _qimage_from_array(image, bgr)
        old = self.after_item.pixmap().size()
        if qimg is None:
            self.after_item.setPixmap(qtg.QPixmap())
            self.has_image = False
        else:
            self.after_item.setPixmap(qtg.QPixmap.fromImage(qimg))
            self.has_image = True
        self.loading_label.setVisible(False)
        self.empty_state.setVisible(False)
        self._update_compare_visibility()
        new = self.after_item.pixmap().size()
        self._scene.setSceneRect(qtc.QRectF(self.after_item.pixmap().rect()))
        if self.has_image and (self._fit_mode or new != old):
            self.fit()
        self.viewport().update()

    def set_before_image(self, image, bgr=True):
        qimg = _qimage_from_array(image, bgr)
        self._has_before = qimg is not None
        self.before_item.setPixmap(qtg.QPixmap.fromImage(qimg) if qimg is not None else qtg.QPixmap())
        if self._has_before and self.has_image:
            # scale the before image onto the result (auto-crop can change its size)
            a, b = self.after_item.pixmap().size(), self.before_item.pixmap().size()
            if b.width() and b.height():
                self.before_item.setTransform(qtg.QTransform.fromScale(
                    a.width() / b.width(), a.height() / b.height()))
        self._update_compare_visibility()

    def show_empty_state(self, on=True):
        self.empty_state.setVisible(on)
        if on:
            self.set_image(None)
            self.empty_state.setVisible(True)

    def show_loading(self, on=True):
        self.loading_label.setVisible(on)
        self._place_overlays()

    def set_compare(self, on):
        self.compare = bool(on)
        self._update_compare_visibility()

    def can_compare(self):
        return self.has_image and self._has_before

    def _update_compare_visibility(self):
        self.before_item.setVisible(self.compare and self.can_compare())
        self.viewport().update()

    def split_scene_x(self):
        if not self.before_item.isVisible():
            return None
        x = self.viewport().width() * self.split
        return self.mapToScene(int(round(x)), 0).x()

    # ── zoom ──
    def zoom(self):
        return self._zoom

    def _fit_scale(self):
        size = self.after_item.pixmap().size()
        if size.isEmpty():
            return 1.0
        vp = self.viewport().size()
        margin = 2 * 12
        return max(1e-3, min((vp.width() - margin) / size.width(), (vp.height() - margin) / size.height()))

    def _min_zoom(self):
        return min(self._fit_scale(), 1.0) * 0.5

    def _set_zoom(self, z, anchor=None):
        """Set the pixel scale, keeping the scene point under `anchor`
        (viewport position) in place."""
        if not self.has_image:
            return
        z = max(self._min_zoom(), min(self.MAX_ZOOM, z))
        if anchor is None:
            anchor = self.viewport().rect().center()
        anchor = qtc.QPointF(anchor).toPoint()
        target = self.mapToScene(anchor)
        self.setTransformationAnchor(qtw.QGraphicsView.NoAnchor)
        self.setTransform(qtg.QTransform.fromScale(z, z))
        self.setTransformationAnchor(qtw.QGraphicsView.AnchorUnderMouse)
        moved = self.mapFromScene(target) - anchor
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + moved.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + moved.y())
        self._zoom = z
        # crisp pixels once they are clearly visible
        mode = qtc.Qt.FastTransformation if z >= 2.0 else qtc.Qt.SmoothTransformation
        self.after_item.setTransformationMode(mode)
        self.before_item.setTransformationMode(mode)
        self.setRenderHint(qtg.QPainter.SmoothPixmapTransform, z < 2.0)
        self.zoom_changed.emit(z)

    def zoom_by(self, factor, anchor=None):
        self._fit_mode = False
        self._set_zoom(self._zoom * factor, anchor)

    def zoom_in(self):
        self.zoom_by(self.ZOOM_STEP)

    def zoom_out(self):
        self.zoom_by(1 / self.ZOOM_STEP)

    def fit(self):
        if not self.has_image:
            return
        self._fit_mode = True
        self._set_zoom(self._fit_scale())
        self.centerOn(self.after_item)

    def actual_size(self, anchor=None):
        self._fit_mode = False
        self._set_zoom(1.0, anchor)

    def toggle_fit_actual(self, anchor=None):
        if not self.has_image:
            return
        if self._fit_mode or abs(self._zoom - self._fit_scale()) < 1e-3:
            self.actual_size(anchor)
        else:
            self.fit()

    # ── layout ──
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_overlays()
        if self._fit_mode and self.has_image:
            self.fit()

    def _place_overlays(self):
        self.empty_state.setGeometry(self.viewport().geometry())
        self.loading_label.adjustSize()
        self.loading_label.move((self.width() - self.loading_label.width()) // 2, 16)

    # ── drawing ──
    def drawForeground(self, painter, rect):
        if not self.before_item.isVisible():
            return
        painter.save()
        painter.resetTransform()
        painter.setRenderHint(qtg.QPainter.Antialiasing)
        vp = self.viewport().rect()
        x = vp.width() * self.split
        img = self.mapFromScene(self.after_item.sceneBoundingRect()).boundingRect()
        top, bottom = max(vp.top(), img.top()), min(vp.bottom(), img.bottom())
        if bottom > top:
            painter.setPen(qtg.QPen(qtg.QColor(0, 0, 0, 90), 3))
            painter.drawLine(qtc.QPointF(x + 1, top), qtc.QPointF(x + 1, bottom))
            painter.setPen(qtg.QPen(qtg.QColor(255, 255, 255, 235), 1.5))
            painter.drawLine(qtc.QPointF(x, top), qtc.QPointF(x, bottom))
            cy = (top + bottom) / 2
            knob = qtc.QRectF(x - 14, cy - 14, 28, 28)
            painter.setPen(qtg.QPen(qtg.QColor(255, 255, 255, 235), 1.5))
            painter.setBrush(qtg.QColor(20, 20, 22, 210))
            painter.drawEllipse(knob)
            painter.setPen(qtg.QPen(qtg.QColor(255, 255, 255, 235), 1.6, qtc.Qt.SolidLine,
                                    qtc.Qt.RoundCap, qtc.Qt.RoundJoin))
            for d in (-1, 1):
                painter.drawPolyline(qtg.QPolygonF([
                    qtc.QPointF(x + d * 3, cy - 4), qtc.QPointF(x + d * 7, cy), qtc.QPointF(x + d * 3, cy + 4)]))
        # Before / After chips
        font = painter.font()
        font.setPointSizeF(max(8.0, font.pointSizeF() * 0.85))
        font.setWeight(qtg.QFont.DemiBold)
        painter.setFont(font)
        fm = qtg.QFontMetrics(font)
        for text, left in (("Before", True), ("After", False)):
            w, h = fm.horizontalAdvance(text) + 16, fm.height() + 8
            cx = x - 12 - w if left else x + 12
            if (left and cx < 8) or (not left and cx + w > vp.width() - 8):
                continue
            chip = qtc.QRectF(cx, max(vp.top(), img.top()) + 12, w, h)
            painter.setPen(qtc.Qt.NoPen)
            painter.setBrush(qtg.QColor(20, 20, 22, 170))
            painter.drawRoundedRect(chip, h / 2, h / 2)
            painter.setPen(qtg.QColor(255, 255, 255, 230))
            painter.drawText(chip, qtc.Qt.AlignCenter, text)
        painter.restore()

    # ── input: mouse ──
    def _near_split(self, pos):
        return self.before_item.isVisible() and abs(pos.x() - self.viewport().width() * self.split) <= self.SPLIT_GRAB

    def mousePressEvent(self, event):
        if event.button() == qtc.Qt.LeftButton and self._near_split(event.position()):
            self._dragging_split = True
            self.viewport().setCursor(qtc.Qt.SplitHCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging_split:
            w = max(1, self.viewport().width())
            self.split = min(1.0, max(0.0, event.position().x() / w))
            self.viewport().update()
            event.accept()
            return
        if event.buttons() == qtc.Qt.NoButton:
            self.viewport().setCursor(qtc.Qt.SplitHCursor if self._near_split(event.position())
                                      else qtc.Qt.OpenHandCursor if self.has_image else qtc.Qt.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging_split:
            self._dragging_split = False
            event.accept()
            return
        super().mouseReleaseEvent(event)
        if self.has_image:
            self.viewport().setCursor(qtc.Qt.OpenHandCursor)

    def mouseDoubleClickEvent(self, event):
        if self.has_image and event.button() == qtc.Qt.LeftButton and not self._near_split(event.position()):
            self.toggle_fit_actual(event.position())
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if self.before_item.isVisible() and key in (qtc.Qt.Key_Left, qtc.Qt.Key_Right) \
                and event.modifiers() & qtc.Qt.ShiftModifier:
            self.split = min(1.0, max(0.0, self.split + (0.05 if key == qtc.Qt.Key_Right else -0.05)))
            self.viewport().update()
            return
        steps = {qtc.Qt.Key_Left: -1, qtc.Qt.Key_Up: -1, qtc.Qt.Key_Right: 1, qtc.Qt.Key_Down: 1}
        if key in steps and not event.modifiers() and (self._fit_mode or not self.has_image):
            self.step_requested.emit(steps[key])
            return
        super().keyPressEvent(event)

    # ── input: wheel, trackpad ──
    def wheelEvent(self, event):
        if not self.has_image:
            event.ignore()
            return
        mods = event.modifiers()
        pixel = event.pixelDelta()
        angle = event.angleDelta()
        # Trackpads send pixel deltas and/or scroll phases; mouse wheels send neither
        trackpad = (not pixel.isNull()) or event.phase() != qtc.Qt.NoScrollPhase
        zoom_mod = bool(mods & (qtc.Qt.ControlModifier | qtc.Qt.MetaModifier))
        if zoom_mod or not trackpad:
            dy = angle.y() if angle.y() else angle.x()
            if not dy and not pixel.isNull():
                dy = pixel.y() * 2
            if dy:
                # 120 units = one notch = one zoom step; smooth for hi-res wheels/pinch
                self.zoom_by(self.ZOOM_STEP ** (dy / 120.0), event.position())
        else:
            d = pixel if not pixel.isNull() else angle / 4
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
        event.accept()

    # ── input: gestures ──
    def event(self, ev):
        if ev.type() == qtc.QEvent.NativeGesture and self.has_image:
            kind = ev.gestureType()
            pos = self.viewport().mapFromGlobal(ev.globalPosition().toPoint())
            if kind == qtc.Qt.ZoomNativeGesture:
                self.zoom_by(1.0 + ev.value(), qtc.QPointF(pos))
                return True
            if kind == qtc.Qt.SmartZoomNativeGesture:
                self.toggle_fit_actual(qtc.QPointF(pos))
                return True
        return super().event(ev)

    def viewportEvent(self, ev):
        if ev.type() == qtc.QEvent.Gesture:
            pinch = ev.gesture(qtc.Qt.PinchGesture)
            if pinch is not None and self.has_image:
                if pinch.state() == qtc.Qt.GestureStarted:
                    self._last_pinch = 1.0
                total = pinch.totalScaleFactor()
                step = total / self._last_pinch if self._last_pinch else 1.0
                self._last_pinch = total
                center = self.viewport().mapFromGlobal(pinch.centerPoint().toPoint())
                self.zoom_by(step, qtc.QPointF(center))
                ev.accept()
                return True
        return super().viewportEvent(ev)

    # ── drag & drop ──
    @staticmethod
    def _urls(event):
        md = event.mimeData()
        return [u.toLocalFile() for u in md.urls() if u.isLocalFile()] if md.hasUrls() else []

    def dragEnterEvent(self, event):
        if self._urls(event):
            event.acceptProposedAction()
            self.empty_state.set_dragging(True)
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._urls(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.empty_state.set_dragging(False)
        event.accept()

    def dropEvent(self, event):
        self.empty_state.set_dragging(False)
        paths = self._urls(event)
        if paths:
            event.acceptProposedAction()
            self.files_dropped.emit(paths)
        else:
            event.ignore()

    def contextMenuEvent(self, event):
        event.accept()

    def export_view(self, path):
        """Save what the viewer shows (e.g. the comparison) as an image."""
        return self.viewport().grab().save(path)
