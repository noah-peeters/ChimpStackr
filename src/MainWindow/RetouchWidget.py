"""
Retouching tool: paint regions from aligned source frames onto the stacked result.
Works like Zerene Stacker's retouching brush — select a source frame, paint
over areas with artifacts, and it replaces those pixels from the source.
"""
import cv2
import numpy as np
import PySide6.QtCore as qtc
import PySide6.QtGui as qtg
import PySide6.QtWidgets as qtw


class RetouchOverlay(qtw.QGraphicsPathItem):
    """Transparent overlay that tracks brush strokes."""

    def __init__(self, image_size):
        super().__init__()
        self.image_w, self.image_h = image_size
        # Mask tracks where we've painted (0=untouched, 255=painted)
        self.mask = np.zeros((self.image_h, self.image_w), dtype=np.uint8)
        self.setOpacity(0.4)
        self.setPen(qtg.QPen(qtc.Qt.NoPen))
        self.setBrush(qtg.QBrush(qtg.QColor(255, 100, 100, 100)))

    def add_stroke(self, x, y, radius):
        """Add a circular brush stroke to the mask."""
        cv2.circle(self.mask, (int(x), int(y)), int(radius), 255, -1)
        self._update_path()

    def clear_mask(self):
        self.mask[:] = 0
        self._update_path()

    def _update_path(self):
        """Convert mask to QPainterPath for display."""
        # Find contours of painted regions
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        path = qtg.QPainterPath()
        for contour in contours:
            if len(contour) > 2:
                pts = [qtc.QPointF(p[0][0], p[0][1]) for p in contour]
                poly = qtg.QPolygonF(pts)
                path.addPolygon(poly)
        self.setPath(path)


class RetouchEngine:
    """
    Core retouching logic: applies source frame pixels to the result
    where the brush mask indicates.
    """

    def __init__(self):
        self.result_image = None       # float32 BGR, the editable output
        self.original_result = None    # float32 BGR, untouched backup
        self.source_images = []        # list of (path, float32 BGR) aligned frames
        self.current_source_idx = 0
        self._undo_stack = []
        self._redo_stack = []

    def set_result(self, image):
        """Set the stacked result as the base for retouching."""
        self.result_image = image.copy()
        self.original_result = image.copy()
        self._undo_stack.clear()
        self._redo_stack.clear()

    def add_source(self, path, image):
        """Add an aligned source frame."""
        self.source_images.append((path, image))

    def apply_brush(self, mask, source_idx=None, feather_radius=3):
        """
        Apply retouching: where mask > 0, replace result pixels with
        pixels from the selected source frame.

        mask: (H, W) uint8, 255 where painted
        source_idx: index into source_images
        feather_radius: blur radius for soft edges
        """
        if self.result_image is None or not self.source_images:
            return

        if source_idx is None:
            source_idx = self.current_source_idx
        if source_idx >= len(self.source_images):
            return

        # Save undo state
        self._undo_stack.append(self.result_image.copy())
        self._redo_stack.clear()

        _, source = self.source_images[source_idx]

        # Feather the mask edges for smooth blending
        if feather_radius > 0:
            k = feather_radius * 2 + 1
            mask_f = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)
            mask_f = mask_f / (mask_f.max() + 1e-12)
        else:
            mask_f = mask.astype(np.float32) / 255.0

        # Expand mask for color channels
        mask_3ch = mask_f[:, :, np.newaxis]

        # Ensure shapes match
        h, w = self.result_image.shape[:2]
        src = source[:h, :w]
        if src.shape != self.result_image.shape:
            src = cv2.resize(src, (w, h))

        # Blend: result = source * mask + result * (1 - mask)
        self.result_image = (
            src.astype(np.float32) * mask_3ch +
            self.result_image * (1.0 - mask_3ch)
        ).astype(np.float32)

    def undo(self):
        if self._undo_stack:
            self._redo_stack.append(self.result_image.copy())
            self.result_image = self._undo_stack.pop()
            return True
        return False

    def redo(self):
        if self._redo_stack:
            self._undo_stack.append(self.result_image.copy())
            self.result_image = self._redo_stack.pop()
            return True
        return False

    def reset(self):
        """Reset result to original (before any retouching)."""
        if self.original_result is not None:
            self._undo_stack.append(self.result_image.copy())
            self._redo_stack.clear()
            self.result_image = self.original_result.copy()


class RetouchPanel(qtw.QWidget):
    """
    UI panel for retouching controls.
    Shows source frame selector, brush size, and action buttons.
    """
    retouch_applied = qtc.Signal()  # Emitted when retouching changes the result

    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = RetouchEngine()
        self._setup_ui()

    def _setup_ui(self):
        layout = qtw.QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # Header
        header = qtw.QLabel("Retouch")
        header.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(header)

        # Source frame selector
        src_label = qtw.QLabel("Source frame:")
        src_label.setStyleSheet("font-size: 12px; color: #999;")
        layout.addWidget(src_label)

        self.source_combo = qtw.QComboBox()
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        layout.addWidget(self.source_combo)

        # Brush size
        brush_layout = qtw.QHBoxLayout()
        brush_label = qtw.QLabel("Brush size:")
        brush_label.setStyleSheet("font-size: 12px; color: #999;")
        brush_layout.addWidget(brush_label)

        self.brush_slider = qtw.QSlider(qtc.Qt.Horizontal)
        self.brush_slider.setRange(5, 200)
        self.brush_slider.setValue(30)
        brush_layout.addWidget(self.brush_slider)

        self.brush_value = qtw.QLabel("30")
        self.brush_slider.valueChanged.connect(lambda v: self.brush_value.setText(str(v)))
        brush_layout.addWidget(self.brush_value)
        layout.addLayout(brush_layout)

        # Feather
        feather_layout = qtw.QHBoxLayout()
        feather_label = qtw.QLabel("Feather:")
        feather_label.setStyleSheet("font-size: 12px; color: #999;")
        feather_layout.addWidget(feather_label)

        self.feather_slider = qtw.QSlider(qtc.Qt.Horizontal)
        self.feather_slider.setRange(0, 50)
        self.feather_slider.setValue(5)
        feather_layout.addWidget(self.feather_slider)

        self.feather_value = qtw.QLabel("5")
        self.feather_slider.valueChanged.connect(lambda v: self.feather_value.setText(str(v)))
        feather_layout.addWidget(self.feather_value)
        layout.addLayout(feather_layout)

        # Buttons
        btn_layout = qtw.QHBoxLayout()

        undo_btn = qtw.QPushButton("Undo")
        undo_btn.clicked.connect(self._on_undo)
        btn_layout.addWidget(undo_btn)

        redo_btn = qtw.QPushButton("Redo")
        redo_btn.clicked.connect(self._on_redo)
        btn_layout.addWidget(redo_btn)

        reset_btn = qtw.QPushButton("Reset")
        reset_btn.clicked.connect(self._on_reset)
        btn_layout.addWidget(reset_btn)

        layout.addLayout(btn_layout)

        apply_btn = qtw.QPushButton("Apply to result")
        apply_btn.setStyleSheet("font-weight: bold;")
        apply_btn.clicked.connect(self._on_apply)
        layout.addWidget(apply_btn)

        layout.addStretch()

    @property
    def brush_radius(self):
        return self.brush_slider.value()

    @property
    def feather_radius(self):
        return self.feather_slider.value()

    def set_sources(self, source_list):
        """source_list: list of (path, aligned_image) tuples."""
        self.source_combo.clear()
        self.engine.source_images.clear()
        import os
        for path, img in source_list:
            self.engine.add_source(path, img)
            self.source_combo.addItem(os.path.basename(path))

    def set_result(self, image):
        self.engine.set_result(image)

    def _on_source_changed(self, idx):
        self.engine.current_source_idx = max(0, idx)

    def _on_undo(self):
        if self.engine.undo():
            self.retouch_applied.emit()

    def _on_redo(self):
        if self.engine.redo():
            self.retouch_applied.emit()

    def _on_reset(self):
        self.engine.reset()
        self.retouch_applied.emit()

    def _on_apply(self):
        """Signal that the retouched result should be used as the new output."""
        self.retouch_applied.emit()
