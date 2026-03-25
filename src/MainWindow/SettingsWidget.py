"""
Settings panel — docked on the right side of the main window.
Modern macOS-style grouped settings with sections.
"""
import PySide6.QtWidgets as qtw
import PySide6.QtCore as qtc
import PySide6.QtGui as qtg

try:
    import numba.cuda as cuda
    HAS_CUDA = cuda.is_available()
except ImportError:
    HAS_CUDA = False

import src.settings as settings


class SettingRow(qtw.QWidget):
    """A single label + control row."""
    def __init__(self, label_text, control, tooltip=None):
        super().__init__()
        layout = qtw.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        label = qtw.QLabel(label_text)
        label.setStyleSheet("color: #ececec; font-size: 12px; background: transparent;")
        layout.addWidget(label)
        layout.addStretch()

        control.setFixedWidth(120)
        layout.addWidget(control)

        if tooltip:
            self.setToolTip(tooltip)


class SettingSection(qtw.QWidget):
    """A titled group of settings (macOS-style grouped section)."""
    def __init__(self, title):
        super().__init__()
        self._layout = qtw.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 12)
        self._layout.setSpacing(0)

        # Section title
        title_label = qtw.QLabel(title)
        title_label.setStyleSheet(
            "color: #999999; font-size: 11px; font-weight: 600; "
            "text-transform: uppercase; padding: 8px 0 4px 0; background: transparent;"
        )
        self._layout.addWidget(title_label)

        # Container for rows
        self.container = qtw.QWidget()
        self.container.setStyleSheet(
            "QWidget { background: #2a2a2a; border-radius: 8px; }"
        )
        self.container_layout = qtw.QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(12, 6, 12, 6)
        self.container_layout.setSpacing(0)
        self._row_count = 0
        self._layout.addWidget(self.container)

    def add_row(self, label_text, control, tooltip=None):
        if self._row_count > 0:
            sep = qtw.QFrame()
            sep.setFrameShape(qtw.QFrame.HLine)
            sep.setStyleSheet("background: #3d3d3d; max-height: 1px;")
            self.container_layout.addWidget(sep)

        row = SettingRow(label_text, control, tooltip)
        row.setStyleSheet("background: transparent;")
        self.container_layout.addWidget(row)
        self._row_count += 1


class SettingsPanel(qtw.QWidget):
    """Right-side settings panel (replaces the old floating window)."""
    setting_updated = qtc.Signal(tuple)

    default_settings = {
        "user_interface": {"theme": 2},
        "computing": {"use_gpu": 0, "selected_gpu_id": 0},
        "algorithm": {
            "stacking_method": "laplacian",
            "kernel_size": 6,
            "pyramid_levels": 8,
            "scale_factor": 10,
            "alignment_ref": "first",
            "align_rst": 1,
            "auto_crop": 1,
            "contrast_threshold": 0.0,
            "feather_radius": 2,
            "depthmap_smoothing": 5,
        },
    }

    def __init__(self):
        super().__init__()
        self.setFixedWidth(300)
        self.setStyleSheet("background: #1e1e1e;")

        # Initialize default QSettings values
        for key, dict_val in self.default_settings.items():
            for name, value in dict_val.items():
                path = f"{key}/{name}"
                if settings.globalVars["QSettings"].contains(path):
                    value = settings.globalVars["QSettings"].value(path)
                self.change_setting(path, value)

        scroll = qtw.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: #1e1e1e; }")

        content = qtw.QWidget()
        content.setStyleSheet("background: #1e1e1e;")
        layout = qtw.QVBoxLayout(content)
        layout.setContentsMargins(16, 8, 16, 16)
        layout.setSpacing(4)

        # Header
        header = qtw.QLabel("Settings")
        header.setStyleSheet(
            "font-size: 16px; font-weight: 700; color: #ececec; "
            "padding: 4px 0 8px 0; background: transparent;"
        )
        layout.addWidget(header)

        # ── Stacking Method Section (segmented control) ──
        method_label = qtw.QLabel("STACKING METHOD")
        method_label.setStyleSheet(
            "color: #999999; font-size: 11px; font-weight: 600; "
            "padding: 8px 0 4px 0; background: transparent;"
        )
        layout.addWidget(method_label)

        self._method_buttons = {}
        methods = [
            ("laplacian", "Pyramid", "Best for fine detail\n(hairs, bristles, edges)"),
            ("weighted_average", "Weighted", "Smooth contrast-based blend\n(good color fidelity)"),
            ("depth_map", "Depth Map", "Per-pixel sharpest source\n(best original color)"),
            ("exposure_fusion", "HDR", "Exposure/HDR fusion\n(varying lighting, not focus)"),
        ]
        method_container = qtw.QWidget()
        method_container.setStyleSheet("background: #2a2a2a; border-radius: 8px;")
        method_layout = qtw.QHBoxLayout(method_container)
        method_layout.setContentsMargins(3, 3, 3, 3)
        method_layout.setSpacing(2)

        saved_method = settings.globalVars["QSettings"].value("algorithm/stacking_method") or "laplacian"
        for key, label, tooltip in methods:
            btn = qtw.QPushButton(label)
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            btn.setStyleSheet("""
                QPushButton {
                    background: transparent; color: #999; border: none;
                    border-radius: 6px; padding: 6px 4px; font-size: 11px; font-weight: 500;
                }
                QPushButton:hover { color: #ccc; }
                QPushButton:checked {
                    background: #505050; color: #fff; font-weight: 600;
                }
            """)
            btn.setChecked(key == saved_method)
            btn.clicked.connect(lambda checked, k=key: self._select_method(k))
            method_layout.addWidget(btn)
            self._method_buttons[key] = btn

        layout.addWidget(method_container)

        # ── Advanced Parameters (collapsible section) ──
        advanced_container = qtw.QWidget()
        advanced_container.setStyleSheet("background: #2a2a2a; border-radius: 8px;")
        advanced_outer = qtw.QVBoxLayout(advanced_container)
        advanced_outer.setContentsMargins(0, 0, 0, 0)
        advanced_outer.setSpacing(0)

        self._advanced_toggle = qtw.QPushButton("  \u25b6  Advanced")
        self._advanced_toggle.setCheckable(True)
        self._advanced_toggle.setChecked(False)
        self._advanced_toggle.setStyleSheet(
            "QPushButton { background: #2a2a2a; color: #999; font-size: 11px; "
            "font-weight: 600; text-align: left; padding: 8px 12px; border: none; "
            "border-radius: 8px; }"
            "QPushButton:hover { color: #ccc; background: #333; }"
            "QPushButton:checked { border-bottom-left-radius: 0; border-bottom-right-radius: 0; }"
        )
        advanced_outer.addWidget(self._advanced_toggle)

        self.advanced_widget = qtw.QWidget()
        self.advanced_widget.setVisible(False)
        self.advanced_widget.setStyleSheet("background: #2a2a2a;")

        def _toggle_advanced(checked):
            self.advanced_widget.setVisible(checked)
            arrow = "\u25bc" if checked else "\u25b6"
            self._advanced_toggle.setText(f"  {arrow}  Advanced")

        self._advanced_toggle.toggled.connect(_toggle_advanced)

        advanced_layout = qtw.QVBoxLayout(self.advanced_widget)
        advanced_layout.setContentsMargins(12, 0, 12, 8)
        advanced_layout.setSpacing(4)

        # Auto-detect button
        auto_btn = qtw.QPushButton("Reset to recommended")
        auto_btn.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 4px 8px; background: #383838; "
            "border-radius: 4px; color: #ccc; border: none; }"
            "QPushButton:hover { background: #444; color: #fff; }"
        )
        auto_btn.clicked.connect(self._auto_detect_params)
        advanced_layout.addWidget(auto_btn)

        algo_section = SettingSection("PARAMETERS")

        self.kernel_spin = qtw.QSpinBox()
        self.kernel_spin.setRange(2, 20)
        self.kernel_spin.setValue(int(settings.globalVars["QSettings"].value("algorithm/kernel_size") or 6))
        self.kernel_spin.valueChanged.connect(lambda v: self.change_setting("algorithm/kernel_size", v))
        algo_section.add_row("Kernel size", self.kernel_spin,
            "Focus comparison kernel. Larger = smoother transitions, slower")

        self.pyramid_spin = qtw.QSpinBox()
        self.pyramid_spin.setRange(2, 16)
        self.pyramid_spin.setValue(int(settings.globalVars["QSettings"].value("algorithm/pyramid_levels") or 8))
        self.pyramid_spin.valueChanged.connect(lambda v: self.change_setting("algorithm/pyramid_levels", v))
        algo_section.add_row("Pyramid levels", self.pyramid_spin,
            "Laplacian pyramid depth. More = finer detail, slower.\n"
            "Only used by Pyramid method.")

        self.contrast_thresh_spin = qtw.QDoubleSpinBox()
        self.contrast_thresh_spin.setRange(0.0, 50.0)
        self.contrast_thresh_spin.setSingleStep(0.5)
        self.contrast_thresh_spin.setDecimals(1)
        self.contrast_thresh_spin.setValue(float(settings.globalVars["QSettings"].value("algorithm/contrast_threshold") or 0.0))
        self.contrast_thresh_spin.valueChanged.connect(lambda v: self.change_setting("algorithm/contrast_threshold", v))
        algo_section.add_row("Contrast threshold", self.contrast_thresh_spin,
            "Minimum contrast to switch sources.\n"
            "0 = off. Higher = ignore flat areas (reduces noise).\n"
            "Only used by Pyramid method.")

        self.feather_spin = qtw.QSpinBox()
        self.feather_spin.setRange(0, 20)
        self.feather_spin.setValue(int(settings.globalVars["QSettings"].value("algorithm/feather_radius") or 2))
        self.feather_spin.valueChanged.connect(lambda v: self.change_setting("algorithm/feather_radius", v))
        algo_section.add_row("Feather radius", self.feather_spin,
            "Blur radius for soft transitions between sources.\n"
            "0 = hard edges. Higher = smoother blending.\n"
            "Only used by Pyramid method.")

        self.smoothing_spin = qtw.QSpinBox()
        self.smoothing_spin.setRange(0, 30)
        self.smoothing_spin.setValue(int(settings.globalVars["QSettings"].value("algorithm/depthmap_smoothing") or 5))
        self.smoothing_spin.valueChanged.connect(lambda v: self.change_setting("algorithm/depthmap_smoothing", v))
        algo_section.add_row("DMap smoothing", self.smoothing_spin,
            "Smoothing of the depth/focus selection map.\n"
            "Higher = fewer artifacts, slightly softer.\n"
            "Lower = sharper but may have halos.\n"
            "Only used by Depth Map method.")

        # ── Alignment (visible) ──
        align_section = SettingSection("ALIGNMENT")

        self.rst_checkbox = qtw.QCheckBox()
        self.rst_checkbox.setChecked(
            bool(int(settings.globalVars["QSettings"].value("algorithm/align_rst") or 1))
        )
        self.rst_checkbox.toggled.connect(lambda v: self.change_setting("algorithm/align_rst", int(v)))
        rst_control = qtw.QWidget()
        rst_control.setStyleSheet("background: transparent;")
        rst_layout = qtw.QHBoxLayout(rst_control)
        rst_layout.setContentsMargins(0, 0, 0, 0)
        rst_layout.addStretch()
        rst_layout.addWidget(self.rst_checkbox)
        align_section.add_row("Rotation + Scale", rst_control,
            "Corrects rotation, scale, and focus breathing.\n"
            "Recommended for most stacks.\n"
            "Disable only if images are already pre-aligned.")

        layout.addWidget(align_section)

        # ── Advanced: alignment params, algorithm params, GPU ──
        self.scale_spin = qtw.QSpinBox()
        self.scale_spin.setRange(1, 50)
        self.scale_spin.setValue(int(settings.globalVars["QSettings"].value("algorithm/scale_factor") or 10))
        self.scale_spin.valueChanged.connect(lambda v: self.change_setting("algorithm/scale_factor", v))

        self.ref_combo = qtw.QComboBox()
        self.ref_combo.addItems(["first", "middle", "previous"])
        saved = settings.globalVars["QSettings"].value("algorithm/alignment_ref") or "first"
        idx = self.ref_combo.findText(saved)
        if idx >= 0:
            self.ref_combo.setCurrentIndex(idx)
        self.ref_combo.currentTextChanged.connect(lambda v: self.change_setting("algorithm/alignment_ref", v))

        align_adv_section = SettingSection("ALIGNMENT")
        align_adv_section.add_row("Precision", self.scale_spin,
            "DFT subpixel upscale factor.\n"
            "Higher = more precise translation, slower.\n"
            "Only used when Rotation+Scale is off.")
        align_adv_section.add_row("Reference", self.ref_combo,
            "first: align all to first image\n"
            "middle: align all to middle image")

        self.autocrop_checkbox = qtw.QCheckBox()
        self.autocrop_checkbox.setChecked(
            bool(int(settings.globalVars["QSettings"].value("algorithm/auto_crop") or 1))
        )
        self.autocrop_checkbox.toggled.connect(lambda v: self.change_setting("algorithm/auto_crop", int(v)))
        autocrop_control = qtw.QWidget()
        autocrop_control.setStyleSheet("background: transparent;")
        ac_layout = qtw.QHBoxLayout(autocrop_control)
        ac_layout.setContentsMargins(0, 0, 0, 0)
        ac_layout.addStretch()
        ac_layout.addWidget(self.autocrop_checkbox)
        align_adv_section.add_row("Auto-crop edges", autocrop_control,
            "Automatically crop black edges\ncaused by alignment shifts.")

        advanced_layout.addWidget(align_adv_section)

        advanced_layout.addWidget(algo_section)

        # ── GPU Section (in advanced) ──
        gpu_section = SettingSection("GPU ACCELERATION")

        self.gpu_checkbox = qtw.QCheckBox()
        self.gpu_checkbox.setEnabled(HAS_CUDA)
        self.gpu_checkbox.setChecked(
            bool(int(settings.globalVars["QSettings"].value("computing/use_gpu") or 0)) and HAS_CUDA
        )
        self.gpu_checkbox.toggled.connect(lambda v: self.change_setting("computing/use_gpu", int(v)))
        gpu_control = qtw.QWidget()
        gpu_control.setStyleSheet("background: transparent;")
        gpu_layout = qtw.QHBoxLayout(gpu_control)
        gpu_layout.setContentsMargins(0, 0, 0, 0)
        gpu_layout.addStretch()
        gpu_layout.addWidget(self.gpu_checkbox)
        gpu_section.add_row(
            "Use CUDA" if HAS_CUDA else "CUDA (not available)",
            gpu_control
        )

        if HAS_CUDA:
            self.gpu_combo = qtw.QComboBox()
            for device in cuda.list_devices():
                cc = device.compute_capability
                name = str(device.name)[2:-1]
                self.gpu_combo.addItem(f"{name} ({cc[0]}.{cc[1]})")
            saved_id = int(settings.globalVars["QSettings"].value("computing/selected_gpu_id") or 0)
            self.gpu_combo.setCurrentIndex(min(saved_id, self.gpu_combo.count() - 1))
            self.gpu_combo.currentIndexChanged.connect(
                lambda v: self.change_setting("computing/selected_gpu_id", v)
            )
            gpu_section.add_row("GPU device", self.gpu_combo)

        advanced_layout.addWidget(gpu_section)

        advanced_outer.addWidget(self.advanced_widget)
        layout.addWidget(advanced_container)

        layout.addStretch()

        scroll.setWidget(content)

        outer = qtw.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Left border separator
        wrapper = qtw.QHBoxLayout()
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.setSpacing(0)
        sep = qtw.QFrame()
        sep.setFrameShape(qtw.QFrame.VLine)
        sep.setStyleSheet("background: #3d3d3d; max-width: 1px;")
        wrapper.addWidget(sep)
        wrapper.addWidget(scroll)

        outer.addLayout(wrapper)

    def _select_method(self, method_key):
        """Handle segmented control method selection."""
        for key, btn in self._method_buttons.items():
            btn.setChecked(key == method_key)
        self.change_setting("algorithm/stacking_method", method_key)

    def _auto_detect_params(self):
        """Auto-detect optimal parameters from currently loaded images."""
        from src.config import auto_detect_params
        paths = settings.globalVars.get("LoadedImagePaths", [])
        if not paths:
            return
        import cv2
        sample = cv2.imread(paths[0])
        if sample is None:
            return
        params = auto_detect_params(sample.shape, len(paths))
        self.kernel_spin.setValue(params["fusion_kernel_size"])
        self.pyramid_spin.setValue(params["pyramid_num_levels"])
        self.scale_spin.setValue(params["alignment_scale_factor"])
        if hasattr(self, 'feather_spin'):
            self.feather_spin.setValue(params.get("feather_radius", 2))
        if hasattr(self, 'contrast_thresh_spin'):
            self.contrast_thresh_spin.setValue(params.get("contrast_threshold", 0.0))

    def change_setting(self, key, value):
        settings.globalVars["QSettings"].setValue(key, value)
        self.setting_updated.emit((key, value))

    def get_algorithm_config(self):
        from src.config import AlgorithmConfig
        qs = settings.globalVars["QSettings"]
        return AlgorithmConfig(
            stacking_method=str(qs.value("algorithm/stacking_method") or "laplacian"),
            fusion_kernel_size=int(qs.value("algorithm/kernel_size") or 6),
            pyramid_num_levels=int(qs.value("algorithm/pyramid_levels") or 8),
            alignment_scale_factor=int(qs.value("algorithm/scale_factor") or 10),
            use_gpu=bool(int(qs.value("computing/use_gpu") or 0)),
            selected_gpu_id=int(qs.value("computing/selected_gpu_id") or 0),
            alignment_reference=str(qs.value("algorithm/alignment_ref") or "first"),
            align_rotation_scale=bool(int(qs.value("algorithm/align_rst") or 0)),
            contrast_threshold=float(qs.value("algorithm/contrast_threshold") or 0.0),
            feather_radius=int(qs.value("algorithm/feather_radius") or 2),
            depthmap_smoothing=int(qs.value("algorithm/depthmap_smoothing") or 5),
        )


# Keep backward compatibility name
SettingsWidget = SettingsPanel
