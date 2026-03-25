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
            "kernel_size": 6,
            "pyramid_levels": 8,
            "scale_factor": 10,
            "alignment_ref": "first",
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

        # ── Algorithm Section ──
        algo_section = SettingSection("ALGORITHM")

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
            "Laplacian pyramid depth. More = finer detail, slower")

        self.scale_spin = qtw.QSpinBox()
        self.scale_spin.setRange(1, 50)
        self.scale_spin.setValue(int(settings.globalVars["QSettings"].value("algorithm/scale_factor") or 10))
        self.scale_spin.valueChanged.connect(lambda v: self.change_setting("algorithm/scale_factor", v))
        algo_section.add_row("Align precision", self.scale_spin,
            "DFT alignment scale factor. Higher = more precise, slower")

        self.ref_combo = qtw.QComboBox()
        self.ref_combo.addItems(["first", "middle", "previous"])
        saved = settings.globalVars["QSettings"].value("algorithm/alignment_ref") or "first"
        idx = self.ref_combo.findText(saved)
        if idx >= 0:
            self.ref_combo.setCurrentIndex(idx)
        self.ref_combo.currentTextChanged.connect(lambda v: self.change_setting("algorithm/alignment_ref", v))
        algo_section.add_row("Align reference", self.ref_combo,
            "first: align all to first image\n"
            "middle: align all to middle image\n"
            "previous: chain-align to previous")

        layout.addWidget(algo_section)

        # ── GPU Section ──
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

        layout.addWidget(gpu_section)

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

    def change_setting(self, key, value):
        settings.globalVars["QSettings"].setValue(key, value)
        self.setting_updated.emit((key, value))

    def get_algorithm_config(self):
        from src.config import AlgorithmConfig
        qs = settings.globalVars["QSettings"]
        return AlgorithmConfig(
            fusion_kernel_size=int(qs.value("algorithm/kernel_size") or 6),
            pyramid_num_levels=int(qs.value("algorithm/pyramid_levels") or 8),
            alignment_scale_factor=int(qs.value("algorithm/scale_factor") or 10),
            use_gpu=bool(int(qs.value("computing/use_gpu") or 0)),
            selected_gpu_id=int(qs.value("computing/selected_gpu_id") or 0),
            alignment_reference=str(qs.value("algorithm/alignment_ref") or "first"),
        )


# Keep backward compatibility name
SettingsWidget = SettingsPanel
