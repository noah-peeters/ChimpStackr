"""
Inspector (stacking settings, right panel) and the app Settings window.

The inspector shows what matters for a stack: method, alignment, output.
Parameters only appear under "Advanced" when the chosen method uses them.
App-wide preferences (appearance, GPU) live in the Settings window
(Cmd+, / Ctrl+,). Everything is stored in QSettings under the same keys as
before, so existing user settings carry over.
"""
import threading

import PySide6.QtCore as qtc
import PySide6.QtGui as qtg
import PySide6.QtWidgets as qtw

import src.settings as settings
import src.algorithms.stacking_algorithms.ocl as OCL
import src.MainWindow.theme as theme
from src.MainWindow.icons import icon

HAS_GPU = OCL.available()

METHODS = [
    ("laplacian", "Pyramid", "Finest detail: hairs, bristles, edges"),
    ("hybrid", "Hybrid", "Pyramid detail without halos, cleaner backgrounds"),
    ("weighted_average", "Weighted Average", "Smooth contrast blend, faithful color"),
    ("depth_map", "Depth Map", "Sharpest source per pixel, original color"),
    ("exposure_fusion", "HDR (Exposure)", "Merge different exposures, not focus"),
]

# (key, label, description); "off" stacks without aligning
ALIGNMENTS = [
    ("standard", "Standard", "Shift, rotation and scale. Best for most stacks"),
    ("breathing", "Focus breathing", "Frame-to-frame chain for lenses that zoom while focusing"),
    ("shift", "Shift only", "Fast; for stacks on a rail without rotation"),
    ("euclidean", "Shift + rotation", "No scale correction"),
    ("affine", "Full affine", "Also corrects shear; for difficult stacks"),
    ("off", "Off", "Frames are already aligned"),
]


def qs():
    return settings.globalVars["QSettings"]


def updater_display(version):
    import src.updater as updater
    return updater.display_version(version)


def get_int(key, default):
    v = qs().value(key)
    try:
        return int(float(v)) if v is not None and v != "" else default
    except (TypeError, ValueError):
        return default


def get_float(key, default):
    v = qs().value(key)
    try:
        return float(v) if v is not None and v != "" else default
    except (TypeError, ValueError):
        return default


def get_str(key, default):
    v = qs().value(key)
    return str(v) if v not in (None, "") else default


def alignment_choice():
    """Current alignment choice key (see ALIGNMENTS), from the stored keys."""
    if not get_int("algorithm/align_enabled", 1):
        return "off"
    if not get_int("algorithm/align_rst", 1):
        return "shift"
    mode = get_str("algorithm/alignment_mode", "similarity")
    return {"similarity": "standard", "breathing": "breathing", "euclidean": "euclidean",
            "affine": "affine"}.get(mode, "standard")


def store_alignment_choice(key):
    s = qs()
    s.setValue("algorithm/align_enabled", 0 if key == "off" else 1)
    if key == "off":
        return
    s.setValue("algorithm/align_rst", 0 if key == "shift" else 1)
    if key != "shift":
        s.setValue("algorithm/alignment_mode", {"standard": "similarity"}.get(key, key))


# ── building blocks ──
class OptionCard(qtw.QAbstractButton):
    """Selectable row: title and one-line description, radio behaviour."""

    def __init__(self, title, description):
        super().__init__()
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.title, self.description = title, description
        self.setCursor(qtc.Qt.PointingHandCursor)
        self.setFocusPolicy(qtc.Qt.StrongFocus)
        self.setAccessibleName(title)
        self.setAccessibleDescription(description)
        self.setToolTip(description)
        self.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Fixed)
        self._hover = False

    def sizeHint(self):
        fm = self.fontMetrics()
        return qtc.QSize(200, fm.height() * 2 + 14)

    def enterEvent(self, e):
        self._hover = True
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        self.update()

    def paintEvent(self, event):
        t = theme.tokens()
        p = qtg.QPainter(self)
        p.setRenderHint(qtg.QPainter.Antialiasing)
        r = qtc.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        checked = self.isChecked()
        if checked:
            fill = qtg.QColor(t.accent)
            fill.setAlpha(40 if t.dark else 28)
            p.setBrush(fill)
            p.setPen(qtg.QPen(qtg.QColor(t.accent), 1.2))
        else:
            p.setBrush(qtg.QColor(t.surface_hover if self._hover else t.surface))
            p.setPen(qtg.QPen(qtg.QColor(t.border), 1))
        if self.hasFocus():
            p.setPen(qtg.QPen(qtg.QColor(t.focus), 2))
        p.drawRoundedRect(r, theme.RADIUS_SM, theme.RADIUS_SM)

        # radio dot
        cy = r.center().y()
        dot = qtc.QRectF(r.left() + 10, cy - 7, 14, 14)
        p.setPen(qtg.QPen(qtg.QColor(t.accent if checked else t.text_tertiary), 1.4))
        p.setBrush(qtc.Qt.NoBrush)
        p.drawEllipse(dot)
        if checked:
            p.setPen(qtc.Qt.NoPen)
            p.setBrush(qtg.QColor(t.accent))
            p.drawEllipse(dot.adjusted(3.5, 3.5, -3.5, -3.5))

        text_x = dot.right() + 10
        fm = self.fontMetrics()
        title_font = qtg.QFont(self.font())
        title_font.setWeight(qtg.QFont.DemiBold)
        p.setFont(title_font)
        p.setPen(qtg.QColor(t.text if self.isEnabled() else t.text_tertiary))
        line_h = fm.height()
        top = cy - line_h - 1
        p.drawText(qtc.QRectF(text_x, top, r.right() - text_x - 8, line_h),
                   qtc.Qt.AlignLeft | qtc.Qt.AlignVCenter, self.title)
        small = qtg.QFont(self.font())
        small.setPointSizeF(self.font().pointSizeF() * 0.9)
        p.setFont(small)
        p.setPen(qtg.QColor(t.text_secondary if self.isEnabled() else t.text_tertiary))
        desc = qtg.QFontMetrics(small).elidedText(self.description, qtc.Qt.ElideRight,
                                                  int(r.right() - text_x - 8))
        p.drawText(qtc.QRectF(text_x, cy + 1, r.right() - text_x - 8, line_h),
                   qtc.Qt.AlignLeft | qtc.Qt.AlignVCenter, desc)

    def keyPressEvent(self, event):
        if event.key() in (qtc.Qt.Key_Down, qtc.Qt.Key_Up):
            self.focusNextPrevChild(event.key() == qtc.Qt.Key_Down)
            return
        super().keyPressEvent(event)


class OptionGroup(qtw.QWidget):
    changed = qtc.Signal(str)

    def __init__(self, options, current):
        super().__init__()
        layout = qtw.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.cards = {}
        group = qtw.QButtonGroup(self)
        for key, title, desc in options:
            card = OptionCard(title, desc)
            card.setChecked(key == current)
            card.toggled.connect(lambda on, k=key: on and self.changed.emit(k))
            group.addButton(card)
            layout.addWidget(card)
            self.cards[key] = card

    def set_current(self, key):
        if key in self.cards:
            self.cards[key].setChecked(True)


class Section(qtw.QWidget):
    def __init__(self, title):
        super().__init__()
        self.layout_ = qtw.QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(6)
        label = qtw.QLabel(title.upper())
        label.setObjectName("SectionTitle")
        self.layout_.addWidget(label)

    def add(self, widget):
        self.layout_.addWidget(widget)
        return widget


class FormCard(qtw.QFrame):
    """Rounded card with label/control rows."""

    def __init__(self):
        super().__init__()
        self.setObjectName("Card")
        self.grid = qtw.QFormLayout(self)
        self.grid.setContentsMargins(12, 8, 12, 8)
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(8)
        self.grid.setLabelAlignment(qtc.Qt.AlignLeft | qtc.Qt.AlignVCenter)
        self.grid.setFieldGrowthPolicy(qtw.QFormLayout.ExpandingFieldsGrow)
        self.grid.setRowWrapPolicy(qtw.QFormLayout.DontWrapRows)
        self.rows = {}

    def add_row(self, key, label, control, tooltip=None):
        if isinstance(control, qtw.QAbstractSpinBox):
            control.setFixedWidth(112)
            field = qtw.QWidget()
            fl = qtw.QHBoxLayout(field)
            fl.setContentsMargins(0, 0, 0, 0)
            fl.addStretch()
            fl.addWidget(control)
        else:
            field = control
            if isinstance(control, qtw.QComboBox):
                control.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Fixed)
        lab = qtw.QLabel(label)
        if tooltip:
            lab.setToolTip(tooltip)
            control.setToolTip(tooltip)
        lab.setBuddy(control)
        control.setAccessibleName(label)
        self.grid.addRow(lab, field)
        self.rows[key] = (lab, field)
        return control

    def set_row_visible(self, key, visible):
        lab, field = self.rows[key]
        lab.setVisible(visible)
        field.setVisible(visible)


class Switch(qtw.QAbstractButton):
    """iOS/macOS-style toggle switch."""

    def __init__(self, checked=False):
        super().__init__()
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(qtc.Qt.PointingHandCursor)
        self.setFocusPolicy(qtc.Qt.StrongFocus)
        self.setSizePolicy(qtw.QSizePolicy.Fixed, qtw.QSizePolicy.Fixed)

    def sizeHint(self):
        return qtc.QSize(36, 20)

    def paintEvent(self, event):
        t = theme.tokens()
        p = qtg.QPainter(self)
        p.setRenderHint(qtg.QPainter.Antialiasing)
        r = qtc.QRectF(0, 0, 36, 20).translated(0, (self.height() - 20) / 2)
        on = self.isChecked()
        track = qtg.QColor(t.accent if on else t.control_pressed)
        if not self.isEnabled():
            track.setAlpha(90)
        p.setPen(qtg.QPen(qtg.QColor(t.focus), 2) if self.hasFocus() else qtc.Qt.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(r, 10, 10)
        knob = qtc.QRectF(r.left() + (18 if on else 2), r.top() + 2, 16, 16)
        p.setPen(qtc.Qt.NoPen)
        p.setBrush(qtg.QColor("#ffffff") if self.isEnabled() else qtg.QColor(t.text_tertiary))
        p.drawEllipse(knob)


def switch_row(switch):
    w = qtw.QWidget()
    lay = qtw.QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addStretch()
    lay.addWidget(switch)
    w.setAccessibleName(switch.accessibleName())
    return w


def spin(lo, hi, value, step=1, decimals=None, special=None, suffix=""):
    if decimals is None:
        w = qtw.QSpinBox()
    else:
        w = qtw.QDoubleSpinBox()
        w.setDecimals(decimals)
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setValue(value)
    if special:
        w.setSpecialValueText(special)
    if suffix:
        w.setSuffix(suffix)
    w.setKeyboardTracking(False)
    w.setAlignment(qtc.Qt.AlignRight)
    return w


class Inspector(qtw.QWidget):
    """Right panel with the stacking settings."""
    setting_updated = qtc.Signal(tuple)

    default_settings = {
        "algorithm": {
            "stacking_method": "laplacian",
            "kernel_size": 6,
            "pyramid_levels": 8,
            "scale_factor": 10,
            "alignment_ref": "first",
            "align_rst": 1,
            "align_enabled": 1,
            "auto_crop": 1,
            "contrast_threshold": 0.0,
            "feather_radius": 2,
            "depthmap_smoothing": 5,
            "breathing_max_scale_step": 5,
            "breathing_high_precision": 0,
        },
        "computing": {"use_gpu": 1, "selected_gpu_id": 0},
    }

    def __init__(self):
        super().__init__()
        self.setObjectName("Inspector")
        self.setAttribute(qtc.Qt.WA_StyledBackground, True)
        self.setMinimumWidth(280)

        s = qs()
        # GPU stacking is on by default since v0.4. Earlier versions stored
        # their default (off), so switch it on once for existing settings.
        if not s.contains("computing/gpu_default_on"):
            s.setValue("computing/use_gpu", 1)
            s.setValue("computing/gpu_default_on", 1)
        for group, values in self.default_settings.items():
            for name, value in values.items():
                if not s.contains(f"{group}/{name}"):
                    s.setValue(f"{group}/{name}", value)

        scroll = qtw.QScrollArea()
        scroll.setObjectName("InspectorScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(qtw.QFrame.NoFrame)
        qtw.QScroller.grabGesture(scroll.viewport(), qtw.QScroller.TouchGesture)

        content = qtw.QWidget()
        layout = qtw.QVBoxLayout(content)
        layout.setContentsMargins(14, 12, 14, 16)
        layout.setSpacing(16)

        # Method
        method_section = Section("Method")
        self.method_group = OptionGroup(METHODS, get_str("algorithm/stacking_method", "laplacian"))
        self.method_group.changed.connect(self._on_method)
        method_section.add(self.method_group)
        layout.addWidget(method_section)

        # Alignment
        align_section = Section("Alignment")
        self.align_combo = qtw.QComboBox()
        for key, title, desc in ALIGNMENTS:
            self.align_combo.addItem(title, key)
            self.align_combo.setItemData(self.align_combo.count() - 1, desc, qtc.Qt.ToolTipRole)
        self.align_combo.setAccessibleName("Alignment")
        self.align_hint = qtw.QLabel()
        self.align_hint.setObjectName("Hint")
        self.align_hint.setWordWrap(True)
        idx = self.align_combo.findData(alignment_choice())
        self.align_combo.setCurrentIndex(max(0, idx))
        self.align_combo.currentIndexChanged.connect(self._on_alignment)
        align_section.add(self.align_combo)
        align_section.add(self.align_hint)
        layout.addWidget(align_section)

        # Output
        out_section = Section("Output")
        out_card = FormCard()
        self.autocrop_switch = Switch(bool(get_int("algorithm/auto_crop", 1)))
        self.autocrop_switch.setAccessibleName("Auto-crop edges")
        self.autocrop_switch.toggled.connect(lambda v: self.change_setting("algorithm/auto_crop", int(v)))
        out_card.add_row("autocrop", "Auto-crop edges", switch_row(self.autocrop_switch),
                         "Crop the borders that alignment leaves empty")
        out_section.add(out_card)
        layout.addWidget(out_section)

        # Advanced
        self.adv_toggle = qtw.QToolButton()
        self.adv_toggle.setObjectName("Disclosure")
        self.adv_toggle.setText("ADVANCED")
        self.adv_toggle.setCheckable(True)
        self.adv_toggle.setToolButtonStyle(qtc.Qt.ToolButtonTextBesideIcon)
        self.adv_toggle.setIconSize(qtc.QSize(12, 12))
        self.adv_toggle.setCursor(qtc.Qt.PointingHandCursor)
        self.adv_toggle.toggled.connect(self._on_advanced_toggled)
        layout.addWidget(self.adv_toggle)

        self.advanced = qtw.QWidget()
        adv = qtw.QVBoxLayout(self.advanced)
        adv.setContentsMargins(0, 0, 0, 0)
        adv.setSpacing(12)

        self.params = FormCard()
        p = self.params
        self.kernel_spin = p.add_row("kernel", "Kernel size", spin(2, 20, get_int("algorithm/kernel_size", 6)),
                                     "Focus comparison window. Larger = smoother transitions, slower")
        self.pyramid_spin = p.add_row("levels", "Pyramid levels", spin(2, 16, get_int("algorithm/pyramid_levels", 8)),
                                      "Pyramid depth. More = finer detail, slower")
        self.contrast_thresh_spin = p.add_row(
            "contrast", "Contrast threshold",
            spin(0.0, 50.0, get_float("algorithm/contrast_threshold", 0.0), 0.5, 1, "Off"),
            "Minimum contrast to switch source frame. Higher ignores flat, noisy areas")
        self.feather_spin = p.add_row("feather", "Feather radius",
                                      spin(0, 20, get_int("algorithm/feather_radius", 2), special="Off"),
                                      "Soft transitions between source frames. Off = hard edges")
        self.smoothing_spin = p.add_row("smoothing", "Depth smoothing",
                                        spin(0, 30, get_int("algorithm/depthmap_smoothing", 5), special="Off"),
                                        "Smooths the depth map. Higher = fewer artifacts, slightly softer")
        self.slab_spin = p.add_row("slab", "Slab size",
                                   spin(0, 500, get_int("algorithm/slab_size", 0), special="Off", suffix=" frames"),
                                   "Stack deep stacks in chunks of this many frames, then stack the chunks")
        self.kernel_spin.valueChanged.connect(lambda v: self.change_setting("algorithm/kernel_size", v))
        self.pyramid_spin.valueChanged.connect(lambda v: self.change_setting("algorithm/pyramid_levels", v))
        self.contrast_thresh_spin.valueChanged.connect(lambda v: self.change_setting("algorithm/contrast_threshold", v))
        self.feather_spin.valueChanged.connect(lambda v: self.change_setting("algorithm/feather_radius", v))
        self.smoothing_spin.valueChanged.connect(lambda v: self.change_setting("algorithm/depthmap_smoothing", v))
        self.slab_spin.valueChanged.connect(lambda v: self.change_setting("algorithm/slab_size", v))

        params_section = Section("Method parameters")
        params_section.add(self.params)
        reset = qtw.QPushButton("Reset to Recommended")
        reset.setToolTip("Pick parameters for the loaded frames' size and count")
        reset.clicked.connect(self._auto_detect_params)
        params_section.add(reset)
        adv.addWidget(params_section)

        self.align_adv = FormCard()
        a = self.align_adv
        self.ref_combo = qtw.QComboBox()
        for key, title in (("first", "First frame"), ("middle", "Middle frame"), ("previous", "Previous frame (chain)")):
            self.ref_combo.addItem(title, key)
        self.ref_combo.setCurrentIndex(max(0, self.ref_combo.findData(get_str("algorithm/alignment_ref", "first"))))
        self.ref_combo.currentIndexChanged.connect(
            lambda _i: (self.change_setting("algorithm/alignment_ref", self.ref_combo.currentData()),
                        self._update_visibility()))
        a.add_row("ref", "Reference", self.ref_combo, "Frame the others are aligned to")
        self.scale_spin = a.add_row("precision", "Shift precision",
                                    spin(1, 50, get_int("algorithm/scale_factor", 10)),
                                    "Sub-pixel upscale factor for shift-only alignment")
        self.scale_spin.valueChanged.connect(lambda v: self.change_setting("algorithm/scale_factor", v))
        self.breathing_spin = a.add_row(
            "breath", "Max scale step",
            spin(1, 50, int(round(get_float("algorithm/breathing_max_scale_step", 5.0))), suffix=" %"),
            "Largest plausible magnification change between neighbouring frames")
        self.breathing_spin.valueChanged.connect(
            lambda v: self.change_setting("algorithm/breathing_max_scale_step", v))
        self.precision_switch = Switch(bool(get_int("algorithm/breathing_high_precision", 0)))
        self.precision_switch.setAccessibleName("Maximum precision")
        self.precision_switch.toggled.connect(self._on_precision_toggled)
        a.add_row("hq", "Maximum precision", switch_row(self.precision_switch),
                  "Full-resolution float refinement per frame pair, on the CPU. Several times slower")
        self.breathing_warning = qtw.QLabel(
            "Maximum precision runs on the CPU and makes stacking several times slower.")
        self.breathing_warning.setObjectName("Warning")
        self.breathing_warning.setWordWrap(True)
        align_adv_section = Section("Alignment details")
        align_adv_section.add(self.align_adv)
        align_adv_section.add(self.breathing_warning)
        adv.addWidget(align_adv_section)

        layout.addWidget(self.advanced)
        layout.addStretch()
        scroll.setWidget(content)

        outer = qtw.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self.adv_toggle.setChecked(bool(get_int("ui/inspector_advanced", 0)))
        self._on_advanced_toggled(self.adv_toggle.isChecked())
        self._update_visibility()
        tm = theme.manager()
        if tm is not None:
            tm.changed.connect(lambda: self._on_advanced_toggled(self.adv_toggle.isChecked()))

    # ── handlers ──
    def change_setting(self, key, value):
        qs().setValue(key, value)
        self.setting_updated.emit((key, value))

    def _on_method(self, key):
        self.change_setting("algorithm/stacking_method", key)
        self._update_visibility()

    def _on_alignment(self, _index):
        key = self.align_combo.currentData()
        store_alignment_choice(key)
        self.setting_updated.emit(("algorithm/alignment", key))
        self._update_visibility()

    def _on_precision_toggled(self, on):
        self.change_setting("algorithm/breathing_high_precision", int(on))
        self._update_visibility()

    def _on_advanced_toggled(self, on):
        self.advanced.setVisible(on)
        self.adv_toggle.setIcon(icon("chevron_down" if on else "chevron_right", "secondary"))
        qs().setValue("ui/inspector_advanced", int(on))

    def _update_visibility(self):
        method = get_str("algorithm/stacking_method", "laplacian")
        pyramid = method in ("laplacian", "hybrid")
        p = self.params
        p.set_row_visible("kernel", method in ("laplacian", "hybrid", "weighted_average"))
        p.set_row_visible("levels", pyramid)
        p.set_row_visible("contrast", method == "laplacian")
        p.set_row_visible("feather", method == "laplacian")
        p.set_row_visible("smoothing", method == "depth_map")
        p.set_row_visible("slab", method != "exposure_fusion")

        choice = self.align_combo.currentData()
        desc = dict((k, d) for k, _t, d in ALIGNMENTS).get(choice, "")
        self.align_hint.setText(desc)
        sequential = choice == "breathing" or (choice != "off" and self.ref_combo.currentData() == "previous")
        a = self.align_adv
        a.set_row_visible("ref", choice not in ("off", "breathing"))
        a.set_row_visible("precision", choice == "shift")
        a.set_row_visible("breath", sequential)
        a.set_row_visible("hq", sequential)
        self.breathing_warning.setVisible(sequential and self.precision_switch.isChecked())
        self.align_adv.parentWidget().setVisible(choice != "off")

    def alignment_enabled(self):
        return self.align_combo.currentData() != "off"

    def _auto_detect_params(self):
        """Pick parameters for the loaded frames' size and count."""
        from src.config import auto_detect_params
        paths = settings.globalVars.get("LoadedImagePaths", [])
        if not paths:
            return
        import cv2
        sample = cv2.imread(paths[0], cv2.IMREAD_REDUCED_GRAYSCALE_2)
        if sample is None:
            return
        shape = (sample.shape[0] * 2, sample.shape[1] * 2)
        params = auto_detect_params(shape, len(paths))
        self.kernel_spin.setValue(params["fusion_kernel_size"])
        self.pyramid_spin.setValue(params["pyramid_num_levels"])
        self.scale_spin.setValue(params["alignment_scale_factor"])
        self.feather_spin.setValue(params.get("feather_radius", 2))
        self.contrast_thresh_spin.setValue(params.get("contrast_threshold", 0.0))

    def get_algorithm_config(self):
        from src.config import AlgorithmConfig
        return AlgorithmConfig(
            stacking_method=get_str("algorithm/stacking_method", "laplacian"),
            fusion_kernel_size=get_int("algorithm/kernel_size", 6),
            pyramid_num_levels=get_int("algorithm/pyramid_levels", 8),
            alignment_scale_factor=get_int("algorithm/scale_factor", 10),
            use_gpu=bool(get_int("computing/use_gpu", 1)) and HAS_GPU,
            selected_gpu_id=get_int("computing/selected_gpu_id", 0),
            alignment_reference=get_str("algorithm/alignment_ref", "first"),
            align_rotation_scale=bool(get_int("algorithm/align_rst", 1)),
            alignment_mode=(get_str("algorithm/alignment_mode", "similarity")
                            if get_int("algorithm/align_rst", 1) else "translation"),
            contrast_threshold=get_float("algorithm/contrast_threshold", 0.0),
            feather_radius=get_int("algorithm/feather_radius", 2),
            depthmap_smoothing=get_int("algorithm/depthmap_smoothing", 5),
            slab_size=get_int("algorithm/slab_size", 0),
            breathing_max_scale_step=get_float("algorithm/breathing_max_scale_step", 5.0) / 100.0,
            breathing_high_precision=bool(get_int("algorithm/breathing_high_precision", 0)),
        )


# ── Settings window ──
class GpuStatus(qtc.QObject):
    """Builds the GPU kernels in the background at startup (the first stack
    doesn't wait for the compile) and remembers whether the device works."""
    changed = qtc.Signal()

    def __init__(self):
        super().__init__()
        self.error = None if HAS_GPU else "No OpenCL GPU found"
        self.ready = False
        if HAS_GPU:
            threading.Thread(target=self._check, args=(get_int("computing/selected_gpu_id", 0),),
                             daemon=True).start()

    def _check(self, index):
        try:
            OCL.get(index)
            self.error, self.ready = None, True
        except Exception as exc:
            self.error = str(exc)
        self.changed.emit()

    def recheck(self):
        if HAS_GPU:
            self.ready = False
            threading.Thread(target=self._check, args=(get_int("computing/selected_gpu_id", 0),),
                             daemon=True).start()

    def usable(self):
        return HAS_GPU and self.error is None

    def summary(self):
        if not get_int("computing/use_gpu", 1) or not self.usable():
            return "CPU"
        names = OCL.device_names()
        i = min(get_int("computing/selected_gpu_id", 0), len(names) - 1)
        return f"GPU · {names[i]}" if names else "CPU"


class SettingsDialog(qtw.QDialog):
    """App preferences: appearance and processing."""

    def __init__(self, gpu_status, updates=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(False)
        self.gpu_status = gpu_status
        self.updates = updates
        self.setMinimumWidth(460)

        layout = qtw.QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(16)

        appearance = Section("Appearance")
        ac = FormCard()
        self.theme_combo = qtw.QComboBox()
        for key, title in (("system", "Match system"), ("dark", "Dark"), ("light", "Light")):
            self.theme_combo.addItem(title, key)
        self.theme_combo.setCurrentIndex(max(0, self.theme_combo.findData(get_str("ui/theme", "system"))))
        self.theme_combo.currentIndexChanged.connect(self._on_theme)
        ac.add_row("theme", "Theme", self.theme_combo)
        appearance.add(ac)
        layout.addWidget(appearance)

        processing = Section("Processing")
        pc = FormCard()
        self.gpu_switch = Switch(bool(get_int("computing/use_gpu", 1)) and HAS_GPU)
        self.gpu_switch.setAccessibleName("Use GPU")
        self.gpu_switch.toggled.connect(lambda v: (qs().setValue("computing/use_gpu", int(v)),
                                                   self.gpu_status.changed.emit()))
        pc.add_row("gpu", "Use GPU", switch_row(self.gpu_switch),
                   "Stacking and alignment run on the GPU (OpenCL, any vendor), with an automatic "
                   "CPU fallback")
        self.gpu_combo = qtw.QComboBox()
        for name in OCL.device_names():
            self.gpu_combo.addItem(name)
        if not HAS_GPU:
            self.gpu_combo.addItem("No OpenCL GPU found")
        self.gpu_combo.setCurrentIndex(min(get_int("computing/selected_gpu_id", 0),
                                           max(0, self.gpu_combo.count() - 1)))
        self.gpu_combo.currentIndexChanged.connect(self._on_device)
        pc.add_row("device", "Device", self.gpu_combo)
        processing.add(pc)
        self.gpu_note = qtw.QLabel()
        self.gpu_note.setObjectName("Hint")
        self.gpu_note.setWordWrap(True)
        self.gpu_note.setTextInteractionFlags(qtc.Qt.TextSelectableByMouse)
        processing.add(self.gpu_note)
        layout.addWidget(processing)

        import src.MainWindow.UpdateDialog as UpdateDialog
        import src.updater as updater
        upd = Section("Updates")
        uc = FormCard()
        self.auto_update_switch = Switch(UpdateDialog.auto_check_enabled())
        self.auto_update_switch.setAccessibleName("Automatically check for updates")
        self.auto_update_switch.toggled.connect(lambda v: qs().setValue("updates/auto_check", int(v)))
        uc.add_row("auto", "Automatically check for updates", switch_row(self.auto_update_switch),
                   "Once a day, ChimpStackr asks GitHub whether a newer release exists")
        self.prerelease_switch = Switch(UpdateDialog.include_prereleases())
        self.prerelease_switch.setAccessibleName("Include pre-releases")
        self.prerelease_switch.toggled.connect(lambda v: qs().setValue("updates/prerelease", int(v)))
        uc.add_row("pre", "Include pre-releases", switch_row(self.prerelease_switch),
                   "Also offer beta versions")
        upd.add(uc)
        row = qtw.QHBoxLayout()
        self.update_note = qtw.QLabel()
        self.update_note.setObjectName("Hint")
        self.update_note.setWordWrap(True)
        check = qtw.QPushButton("Check Now")
        check.setEnabled(updates is not None)
        check.clicked.connect(lambda: (updates.check(manual=True), self._refresh_update_note()))
        row.addWidget(self.update_note, 1)
        row.addWidget(check)
        upd.layout_.addLayout(row)
        layout.addWidget(upd)
        self._version = updater.current_version()
        self._refresh_update_note()

        layout.addStretch()
        buttons = qtw.QDialogButtonBox(qtw.QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

        gpu_status.changed.connect(self._refresh_gpu)
        self._refresh_gpu()

    def _refresh_update_note(self):
        import time
        last = get_float("updates/last_check", 0)
        when = time.strftime("%d %b %Y, %H:%M", time.localtime(last)) if last else "never"
        self.update_note.setText(f"You have version {updater_display(self._version)}. Last checked: {when}.")

    def showEvent(self, event):
        self._refresh_update_note()
        super().showEvent(event)

    def _on_theme(self, _i):
        mode = self.theme_combo.currentData()
        qs().setValue("ui/theme", mode)
        tm = theme.manager()
        if tm is not None:
            tm.set_mode(mode)

    def _on_device(self, index):
        if HAS_GPU:
            qs().setValue("computing/selected_gpu_id", index)
            self.gpu_status.recheck()

    def _refresh_gpu(self):
        usable = self.gpu_status.usable()
        self.gpu_switch.setEnabled(usable)
        self.gpu_combo.setEnabled(HAS_GPU and len(OCL.device_names()) > 1)
        if not HAS_GPU:
            self.gpu_note.setText("No OpenCL GPU was found; stacking runs on the CPU.")
        elif self.gpu_status.error:
            self.gpu_note.setText(f"This GPU's driver can't run ChimpStackr's GPU code, so stacking "
                                  f"runs on the CPU.\n{self.gpu_status.error}")
        elif not self.gpu_status.ready:
            self.gpu_note.setText("Preparing the GPU…")
        else:
            self.gpu_note.setText("Falls back to the CPU automatically if anything fails on the GPU.")


# Old names
SettingsPanel = Inspector
SettingsWidget = Inspector
