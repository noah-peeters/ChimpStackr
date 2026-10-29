"""
Export: one dialog for format, bit depth and quality (remembered between
exports), then the native save panel, then the file is written.
"""
import os

import cv2
import numpy as np
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw

import src.settings as settings

FORMATS = [
    # key, label, extension, bit depths
    ("JPG", "JPEG", "jpg", (8,)),
    ("PNG", "PNG", "png", (8, 16)),
    ("TIFF", "TIFF", "tif", (8, 16)),
    ("EXR", "OpenEXR", "exr", (32,)),
]
_BY_KEY = {f[0]: f for f in FORMATS}


def _convert_to_uint8(imageArray):
    """float32 image (0-255) to uint8."""
    return np.around(np.clip(imageArray, 0, 255)).astype(np.uint8)


def _convert_to_uint16(imageArray):
    """float32 image (0-255) to uint16 (0-65535), keeping the extra
    precision of the float pipeline (256x more tonal steps than 8-bit)."""
    return np.around(np.clip(imageArray, 0, 255) * 257.0).astype(np.uint16)


def encode(image, fmt, bit_depth=8, quality=95, compression=4):
    """(array to write, cv2.imwrite params) for an export."""
    if fmt == "EXR":
        return np.clip(image, 0, 255).astype(np.float32) / 255.0, []
    out = _convert_to_uint16(image) if bit_depth == 16 else _convert_to_uint8(image)
    if fmt == "JPG":
        return out, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    if fmt == "PNG":
        return out, [cv2.IMWRITE_PNG_COMPRESSION, int(compression)]
    return out, []


def write_image(path, image, fmt, bit_depth=8, quality=95, compression=4):
    """Write the export; raises on failure."""
    data, params = encode(image, fmt, bit_depth, quality, compression)
    ok, buf = cv2.imencode("." + _BY_KEY[fmt][2], data, params)
    if not ok:
        raise IOError(f"Could not encode the image as {fmt}")
    # imencode + write handles non-ASCII paths on Windows, which imwrite doesn't
    with open(path, "wb") as f:
        f.write(buf.tobytes())


class ExportDialog(qtw.QDialog):
    def __init__(self, image_shape, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Image")
        self.setMinimumWidth(400)
        qs = settings.globalVars["QSettings"]

        layout = qtw.QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(14)

        h, w = image_shape[:2]
        info = qtw.QLabel(f"{w} × {h} px · {w * h / 1e6:.1f} MP")
        info.setObjectName("Secondary")
        layout.addWidget(info)

        form = qtw.QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        form.setLabelAlignment(qtc.Qt.AlignRight | qtc.Qt.AlignVCenter)

        self.format_combo = qtw.QComboBox()
        for key, label, ext, _d in FORMATS:
            self.format_combo.addItem(f"{label} (.{ext})", key)
        saved = str(qs.value("export/format") or "JPG")
        self.format_combo.setCurrentIndex(max(0, self.format_combo.findData(saved)))
        form.addRow("Format", self.format_combo)

        self.depth_combo = qtw.QComboBox()
        form.addRow("Bit depth", self.depth_combo)
        self.depth_label = form.labelForField(self.depth_combo)

        self.quality = qtw.QSlider(qtc.Qt.Horizontal)
        self.quality.setRange(50, 100)
        self.quality.setValue(int(qs.value("export/jpeg_quality") or 95))
        self.quality_value = qtw.QLabel()
        self.quality_value.setMinimumWidth(32)
        self.quality.valueChanged.connect(lambda v: self.quality_value.setText(str(v)))
        self.quality_value.setText(str(self.quality.value()))
        qrow = qtw.QWidget()
        ql = qtw.QHBoxLayout(qrow)
        ql.setContentsMargins(0, 0, 0, 0)
        ql.addWidget(self.quality)
        ql.addWidget(self.quality_value)
        form.addRow("Quality", qrow)
        self.quality_row = qrow
        self.quality_label = form.labelForField(qrow)

        self.compression = qtw.QComboBox()
        for level, title in ((1, "Fast (larger file)"), (4, "Balanced"), (9, "Smallest (slow)")):
            self.compression.addItem(title, level)
        self.compression.setCurrentIndex(max(0, self.compression.findData(int(qs.value("export/png_compression") or 4))))
        form.addRow("Compression", self.compression)
        self.compression_label = form.labelForField(self.compression)

        layout.addLayout(form)
        self.note = qtw.QLabel()
        self.note.setObjectName("Hint")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)

        buttons = qtw.QDialogButtonBox()
        self.ok = buttons.addButton("Export…", qtw.QDialogButtonBox.AcceptRole)
        self.ok.setObjectName("PrimaryButton")
        self.ok.setDefault(True)
        buttons.addButton(qtw.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.format_combo.currentIndexChanged.connect(self._update)
        self._update()

    def _update(self):
        qs = settings.globalVars["QSettings"]
        key = self.format_combo.currentData()
        depths = _BY_KEY[key][3]
        self.depth_combo.blockSignals(True)
        self.depth_combo.clear()
        for d in depths:
            self.depth_combo.addItem({8: "8-bit", 16: "16-bit", 32: "32-bit float"}[d], d)
        saved = int(qs.value(f"export/{key.lower()}_depth") or (16 if 16 in depths else depths[0]))
        self.depth_combo.setCurrentIndex(max(0, self.depth_combo.findData(saved)))
        self.depth_combo.blockSignals(False)
        multi = len(depths) > 1
        self.depth_combo.setEnabled(multi)
        for widget, on in ((self.quality_row, key == "JPG"), (self.quality_label, key == "JPG"),
                           (self.compression, key == "PNG"), (self.compression_label, key == "PNG")):
            widget.setVisible(on)
        self.note.setText({
            "JPG": "Best for sharing. 8-bit, lossy.",
            "PNG": "Lossless. 16-bit keeps the full precision of the stack.",
            "TIFF": "Lossless, for editing in Lightroom, Photoshop or Capture One.",
            "EXR": "32-bit float, linear 0–1 range, for compositing.",
        }[key])
        self.adjustSize()

    def options(self):
        key = self.format_combo.currentData()
        return {
            "fmt": key,
            "ext": _BY_KEY[key][2],
            "label": _BY_KEY[key][1],
            "bit_depth": self.depth_combo.currentData(),
            "quality": self.quality.value(),
            "compression": self.compression.currentData(),
        }

    def accept(self):
        qs = settings.globalVars["QSettings"]
        o = self.options()
        qs.setValue("export/format", o["fmt"])
        qs.setValue(f"export/{o['fmt'].lower()}_depth", o["bit_depth"])
        qs.setValue("export/jpeg_quality", o["quality"])
        qs.setValue("export/png_compression", o["compression"])
        super().accept()


def ask_export_path(parent, options, directory, base_name):
    """Native save panel; returns the chosen path with the right extension."""
    ext = options["ext"]
    start = os.path.join(directory, f"{base_name}.{ext}")
    path, _ = qtw.QFileDialog.getSaveFileName(
        parent, "Export Image", start, f"{options['label']} (*.{ext})")
    if not path:
        return None
    allowed = {f".{ext}"} | ({".jpeg"} if ext == "jpg" else {".tiff"} if ext == "tif" else set())
    if os.path.splitext(path)[1].lower() not in allowed:
        path += f".{ext}"
    return os.path.abspath(path)
