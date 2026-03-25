"""
Before/After comparison slider widget.
Composites two images with a draggable vertical divider.
"""
import numpy as np
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw
import PySide6.QtGui as qtg


class ComparisonSlider(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.before_pixmap = None
        self.after_pixmap = None
        self.slider_pos = 0.5  # 0.0 to 1.0
        self._dragging = False
        self.setMouseTracking(True)
        self.setMinimumSize(200, 200)

    def set_before_image(self, image):
        """Set the 'before' image (numpy RGB array or None)."""
        if image is None:
            self.before_pixmap = None
        else:
            h, w, ch = image.shape
            qimg = qtg.QImage(image.data, w, h, w * ch, qtg.QImage.Format_RGB888)
            self.before_pixmap = qtg.QPixmap.fromImage(qimg)
        self.update()

    def set_after_image(self, image):
        """Set the 'after' image (numpy RGB array or None)."""
        if image is None:
            self.after_pixmap = None
        else:
            h, w, ch = image.shape
            qimg = qtg.QImage(image.data, w, h, w * ch, qtg.QImage.Format_RGB888)
            self.after_pixmap = qtg.QPixmap.fromImage(qimg)
        self.update()

    def paintEvent(self, event):
        painter = qtg.QPainter(self)
        painter.setRenderHint(qtg.QPainter.Antialiasing)
        rect = self.rect()

        # Background
        painter.fillRect(rect, qtg.QColor("#232323"))

        if self.before_pixmap is None and self.after_pixmap is None:
            painter.setPen(qtg.QColor("#666666"))
            painter.setFont(qtg.QFont("", 14))
            painter.drawText(rect, qtc.Qt.AlignCenter,
                           "Stack images to see comparison")
            return

        # Scale images to fit the widget while maintaining aspect ratio
        target_rect = self._get_image_rect()
        split_x = int(target_rect.x() + target_rect.width() * self.slider_pos)

        # Draw before (left side)
        if self.before_pixmap:
            painter.setClipRect(qtc.QRect(
                target_rect.x(), target_rect.y(),
                int(target_rect.width() * self.slider_pos), target_rect.height()
            ))
            painter.drawPixmap(target_rect, self.before_pixmap)

        # Draw after (right side)
        if self.after_pixmap:
            painter.setClipRect(qtc.QRect(
                split_x, target_rect.y(),
                target_rect.width() - int(target_rect.width() * self.slider_pos),
                target_rect.height()
            ))
            painter.drawPixmap(target_rect, self.after_pixmap)

        painter.setClipping(False)

        # Draw divider line — thin white with subtle shadow
        painter.setPen(qtg.QPen(qtg.QColor(0, 0, 0, 80), 3))
        painter.drawLine(split_x + 1, target_rect.y(), split_x + 1, target_rect.y() + target_rect.height())
        painter.setPen(qtg.QPen(qtg.QColor(255, 255, 255, 220), 1.5))
        painter.drawLine(split_x, target_rect.y(), split_x, target_rect.y() + target_rect.height())

        # Draw handle — small pill shape
        handle_y = target_rect.y() + target_rect.height() // 2
        pill = qtc.QRectF(split_x - 10, handle_y - 16, 20, 32)
        painter.setPen(qtg.QPen(qtg.QColor(255, 255, 255, 200), 1))
        painter.setBrush(qtg.QColor(40, 40, 40, 220))
        painter.drawRoundedRect(pill, 10, 10)

        # Arrow indicators
        painter.setPen(qtg.QPen(qtg.QColor(255, 255, 255, 200), 1.5))
        # Left arrow
        painter.drawLine(split_x - 4, handle_y - 3, split_x - 1, handle_y)
        painter.drawLine(split_x - 4, handle_y + 3, split_x - 1, handle_y)
        # Right arrow
        painter.drawLine(split_x + 4, handle_y - 3, split_x + 1, handle_y)
        painter.drawLine(split_x + 4, handle_y + 3, split_x + 1, handle_y)

        # Labels
        painter.setPen(qtg.QColor(255, 255, 255, 180))
        font = qtg.QFont("", 11, qtg.QFont.Bold)
        painter.setFont(font)

        label_y = target_rect.y() + 24
        if self.slider_pos > 0.15:
            painter.drawText(target_rect.x() + 12, label_y, "Before")
        if self.slider_pos < 0.85:
            painter.drawText(
                target_rect.x() + target_rect.width() - 55, label_y, "After"
            )

    def _get_image_rect(self):
        """Get the rectangle to draw images in, maintaining aspect ratio."""
        pixmap = self.after_pixmap or self.before_pixmap
        if not pixmap:
            return self.rect()

        pw, ph = pixmap.width(), pixmap.height()
        ww, wh = self.width(), self.height()

        scale = min(ww / pw, wh / ph)
        sw = int(pw * scale)
        sh = int(ph * scale)
        x = (ww - sw) // 2
        y = (wh - sh) // 2

        return qtc.QRect(x, y, sw, sh)

    def mousePressEvent(self, event):
        if event.button() == qtc.Qt.LeftButton:
            self._dragging = True
            self._update_slider(event.pos())

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._update_slider(event.pos())

        # Change cursor near the divider
        target = self._get_image_rect()
        split_x = target.x() + target.width() * self.slider_pos
        if abs(event.pos().x() - split_x) < 15:
            self.setCursor(qtc.Qt.SplitHCursor)
        else:
            self.setCursor(qtc.Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        self._dragging = False

    def _update_slider(self, pos):
        target = self._get_image_rect()
        if target.width() > 0:
            self.slider_pos = max(0.0, min(1.0,
                (pos.x() - target.x()) / target.width()
            ))
            self.update()


class ComparisonWidget(qtw.QWidget):
    """Tab widget containing the comparison slider with export button."""
    def __init__(self):
        super().__init__()
        self.slider = ComparisonSlider()

        layout = qtw.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider)

    def set_before_image(self, image):
        self.slider.set_before_image(image)

    def set_after_image(self, image):
        self.slider.set_after_image(image)

    def export_comparison(self, path):
        """Export the current comparison view to a file."""
        pixmap = self.slider.grab()
        pixmap.save(path)
