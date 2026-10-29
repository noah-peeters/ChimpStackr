"""
Theme: design tokens, palette and stylesheet for the whole app.

Follows the system light/dark setting (and switches live when it changes),
with a user override in Settings → Appearance. Fusion is the base style on
every platform so the app looks and behaves the same everywhere; the system
font is kept (San Francisco on macOS, Segoe UI on Windows, the desktop font
on Linux).
"""
from dataclasses import dataclass

import PySide6.QtCore as qtc
import PySide6.QtGui as qtg
import PySide6.QtWidgets as qtw


@dataclass(frozen=True)
class Tokens:
    dark: bool
    window: str        # window / panel background
    sidebar: str       # side panels
    canvas: str        # image viewer background
    surface: str       # grouped controls, cards
    surface_hover: str
    control: str       # buttons, inputs
    control_hover: str
    control_pressed: str
    border: str
    separator: str
    text: str
    text_secondary: str
    text_tertiary: str
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_text: str   # text on accent
    selection: str     # list selection background
    danger: str
    warning: str
    success: str
    focus: str


DARK = Tokens(
    dark=True,
    window="#1c1c1e", sidebar="#202022", canvas="#141415",
    surface="#2a2a2d", surface_hover="#323236",
    control="#38383c", control_hover="#424247", control_pressed="#4c4c52",
    border="#3a3a3e", separator="#2f2f33",
    text="#f2f2f4", text_secondary="#a1a1a8", text_tertiary="#6c6c73",
    accent="#0a84ff", accent_hover="#3396ff", accent_pressed="#0070e0", accent_text="#ffffff",
    selection="#0a4fa3",
    danger="#ff453a", warning="#ffb340", success="#32d74b",
    focus="#0a84ff",
)

LIGHT = Tokens(
    dark=False,
    window="#f5f5f7", sidebar="#ececef", canvas="#e3e3e7",
    surface="#ffffff", surface_hover="#f4f4f6",
    control="#ffffff", control_hover="#f0f0f2", control_pressed="#e4e4e8",
    border="#d2d2d7", separator="#dcdce0",
    text="#1d1d1f", text_secondary="#5e5e66", text_tertiary="#9a9aa2",
    accent="#007aff", accent_hover="#1a88ff", accent_pressed="#0062cc", accent_text="#ffffff",
    selection="#cfe3ff",
    danger="#e5352b", warning="#c77700", success="#1f9d3a",
    focus="#007aff",
)

# Spacing / shape scale (px)
RADIUS = 8
RADIUS_SM = 6
RADIUS_LG = 12
CONTROL_HEIGHT = 28


class ThemeManager(qtc.QObject):
    """Owns the current tokens; emits `changed` after a switch so widgets
    that paint themselves (icons, viewer, comparison handle) can refresh."""
    changed = qtc.Signal()

    MODES = ("system", "dark", "light")

    def __init__(self, app, mode="system"):
        super().__init__(app)
        self.app = app
        self.mode = mode if mode in self.MODES else "system"
        self.tokens = DARK
        app.setStyle(qtw.QStyleFactory.create("Fusion"))
        hints = app.styleHints()
        if hasattr(hints, "colorSchemeChanged"):   # Qt 6.5+
            hints.colorSchemeChanged.connect(lambda _scheme: self.apply())
        self.apply()

    def system_is_dark(self):
        hints = self.app.styleHints()
        if hasattr(hints, "colorScheme"):
            scheme = hints.colorScheme()
            if scheme != qtc.Qt.ColorScheme.Unknown:
                return scheme == qtc.Qt.ColorScheme.Dark
        # Older Qt / unknown: judge by the platform palette
        return self.app.palette().color(qtg.QPalette.Window).lightness() < 128

    def set_mode(self, mode):
        self.mode = mode if mode in self.MODES else "system"
        self.apply()

    def apply(self):
        dark = self.system_is_dark() if self.mode == "system" else self.mode == "dark"
        self.tokens = DARK if dark else LIGHT
        self.app.setPalette(build_palette(self.tokens))
        self.app.setStyleSheet(build_stylesheet(self.tokens))
        self.changed.emit()


_manager = None


def install(app, mode="system"):
    global _manager
    _manager = ThemeManager(app, mode)
    return _manager


def manager():
    return _manager


def tokens():
    return _manager.tokens if _manager is not None else DARK


def build_palette(t):
    p = qtg.QPalette()
    c = qtg.QColor
    roles = {
        qtg.QPalette.Window: t.window,
        qtg.QPalette.WindowText: t.text,
        qtg.QPalette.Base: t.surface,
        qtg.QPalette.AlternateBase: t.surface_hover,
        qtg.QPalette.ToolTipBase: t.surface,
        qtg.QPalette.ToolTipText: t.text,
        qtg.QPalette.PlaceholderText: t.text_tertiary,
        qtg.QPalette.Text: t.text,
        qtg.QPalette.Button: t.control,
        qtg.QPalette.ButtonText: t.text,
        qtg.QPalette.BrightText: t.danger,
        qtg.QPalette.Highlight: t.accent,
        qtg.QPalette.HighlightedText: t.accent_text,
        qtg.QPalette.Link: t.accent,
        qtg.QPalette.Light: t.control_hover,
        qtg.QPalette.Midlight: t.control,
        qtg.QPalette.Mid: t.border,
        qtg.QPalette.Dark: t.separator,
        qtg.QPalette.Shadow: "#000000",
    }
    for role, color in roles.items():
        p.setColor(role, c(color))
    if hasattr(qtg.QPalette, "Accent"):   # Qt 6.6+
        p.setColor(qtg.QPalette.Accent, c(t.accent))
    for role in (qtg.QPalette.WindowText, qtg.QPalette.Text, qtg.QPalette.ButtonText):
        p.setColor(qtg.QPalette.Disabled, role, c(t.text_tertiary))
    p.setColor(qtg.QPalette.Disabled, qtg.QPalette.Highlight, c(t.control_pressed))
    return p


_ASSET_DIR = None


def _assets(t):
    """Theme-colored SVGs the stylesheet points to (Qt stylesheets need files)."""
    import os
    import tempfile
    global _ASSET_DIR
    if _ASSET_DIR is None:
        _ASSET_DIR = tempfile.mkdtemp(prefix="chimpstackr-theme-")
    down = '<path d="M6 9l6 6 6-6"/>'
    up = '<path d="M6 15l6-6 6 6"/>'
    shapes = {
        "chevron_down": (down, t.text_secondary, 2.2),
        "chevron_down_disabled": (down, t.text_tertiary, 2.2),
        "chevron_up": (up, t.accent, 2.2),
        "chevron_up_small": (up, t.text, 2.4),
        "chevron_down_small": (down, t.text, 2.4),
        "chevron_up_small_disabled": (up, t.text_tertiary, 2.4),
        "chevron_down_small_disabled": (down, t.text_tertiary, 2.4),
        "check": ('<path d="M6 12.5l4 4 8-9"/>', t.accent_text, 3.0),
        "dot": ('<circle cx="12" cy="12" r="4.5" fill="' + t.accent_text + '" stroke="none"/>',
                t.accent_text, 0),
    }
    paths = {}
    tag = "dark" if t.dark else "light"
    for name, (body, color, width) in shapes.items():
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
               f'stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
               f'stroke-linejoin="round">{body}</svg>')
        path = os.path.join(_ASSET_DIR, f"{name}-{tag}.svg")
        with open(path, "w") as f:
            f.write(svg)
        paths[name] = path.replace("\\", "/")
    return paths


def build_stylesheet(t):
    a = _assets(t)
    return f"""
    QMainWindow, QDialog {{ background: {t.window}; }}
    QToolTip {{
        background: {t.surface}; color: {t.text}; border: 1px solid {t.border};
        border-radius: {RADIUS_SM}px; padding: 5px 8px;
    }}

    /* Toolbar */
    QToolBar#MainToolbar {{
        background: {t.window}; border: none; border-bottom: 1px solid {t.separator};
        padding: 6px 10px; spacing: 6px;
    }}
    QToolBar#MainToolbar QToolButton {{
        background: transparent; border: 1px solid transparent; border-radius: {RADIUS_SM}px;
        padding: 4px 8px; min-height: 20px; color: {t.text};
    }}
    QToolBar#MainToolbar QToolButton:hover {{ background: {t.control_hover}; }}
    QToolBar#MainToolbar QToolButton:pressed,
    QToolBar#MainToolbar QToolButton:checked {{ background: {t.control_pressed}; }}
    QToolBar#MainToolbar QToolButton:disabled {{ color: {t.text_tertiary}; }}
    QToolBar#MainToolbar QToolButton::menu-indicator {{ image: none; width: 0; }}
    QToolBar#MainToolbar QToolButton::menu-button {{
        border: none; border-left: 1px solid {t.separator}; width: 16px;
    }}
    QToolBar::separator {{ background: {t.separator}; width: 1px; margin: 4px 6px; }}

    /* Primary action */
    QToolButton#PrimaryButton, QPushButton#PrimaryButton {{
        background: {t.accent}; color: {t.accent_text}; border: none;
        border-radius: {RADIUS_SM}px; padding: 5px 14px; font-weight: 600;
    }}
    QToolButton#PrimaryButton:hover, QPushButton#PrimaryButton:hover {{ background: {t.accent_hover}; }}
    QToolButton#PrimaryButton:pressed, QPushButton#PrimaryButton:pressed {{ background: {t.accent_pressed}; }}
    QToolButton#PrimaryButton:disabled, QPushButton#PrimaryButton:disabled {{
        background: {t.control}; color: {t.text_tertiary};
    }}
    QToolBar#MainToolbar QToolButton#PrimaryButton {{
        background: {t.accent}; color: {t.accent_text}; border: none; padding: 5px 14px;
    }}
    QToolBar#MainToolbar QToolButton#PrimaryButton:hover {{ background: {t.accent_hover}; }}
    QToolBar#MainToolbar QToolButton#PrimaryButton:pressed {{ background: {t.accent_pressed}; }}
    QToolBar#MainToolbar QToolButton#PrimaryButton:disabled {{
        background: {t.control}; color: {t.text_tertiary};
    }}
    QToolBar#MainToolbar QToolButton#PrimaryButton::menu-button {{
        border-left: 1px solid rgba(255,255,255,0.25); border-top-right-radius: {RADIUS_SM}px;
        border-bottom-right-radius: {RADIUS_SM}px;
    }}

    /* Buttons */
    QPushButton {{
        background: {t.control}; color: {t.text}; border: 1px solid {t.border};
        border-radius: {RADIUS_SM}px; padding: 5px 12px; min-height: 18px;
    }}
    QPushButton:hover {{ background: {t.control_hover}; }}
    QPushButton:pressed {{ background: {t.control_pressed}; }}
    QPushButton:disabled {{ color: {t.text_tertiary}; }}
    QPushButton:focus {{ border-color: {t.focus}; }}
    QPushButton#FlatButton {{ background: transparent; border: none; color: {t.accent}; padding: 4px 6px; }}
    QPushButton#FlatButton:hover {{ text-decoration: underline; }}
    QPushButton#DangerButton {{ color: {t.danger}; }}

    /* Segmented control */
    QWidget#Segmented {{
        background: {t.control}; border: 1px solid {t.border}; border-radius: {RADIUS_SM + 1}px;
    }}
    QWidget#Segmented QPushButton {{
        background: transparent; border: none; border-radius: {RADIUS_SM - 1}px;
        padding: 4px 10px; color: {t.text_secondary}; min-height: 18px;
    }}
    QWidget#Segmented QPushButton:hover {{ color: {t.text}; }}
    QWidget#Segmented QPushButton:checked {{
        background: {t.control_pressed if t.dark else t.window}; color: {t.text}; font-weight: 600;
        border: 1px solid {t.border};
    }}
    QWidget#Segmented QPushButton:focus {{ border: 1px solid {t.focus}; }}

    /* Inputs */
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{
        background: {t.control}; color: {t.text}; border: 1px solid {t.border};
        border-radius: {RADIUS_SM}px; padding: 4px 10px; min-height: 20px;
        selection-background-color: {t.accent}; selection-color: {t.accent_text};
    }}
    QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover, QLineEdit:hover {{
        background: {t.control_hover};
    }}
    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus,
    QComboBox:on {{ border: 1px solid {t.focus}; }}
    QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled, QLineEdit:disabled {{
        color: {t.text_tertiary}; background: {t.control};
    }}

    /* Combo box: chevron, list popup */
    QComboBox {{ padding-right: 30px; combobox-popup: 0; }}
    QComboBox::drop-down {{
        subcontrol-origin: padding; subcontrol-position: center right;
        width: 28px; border: none; background: transparent;
    }}
    QComboBox::down-arrow {{ image: url("{a['chevron_down']}"); width: 12px; height: 12px; }}
    QComboBox::down-arrow:disabled {{ image: url("{a['chevron_down_disabled']}"); }}
    QComboBox::down-arrow:on {{ image: url("{a['chevron_up']}"); }}
    QComboBox QAbstractItemView {{
        background: {t.surface}; color: {t.text}; border: 1px solid {t.border};
        border-radius: {RADIUS}px; padding: 4px; outline: none;
    }}
    QComboBox QAbstractItemView::item {{
        min-height: 26px; padding: 2px 8px; border-radius: {RADIUS_SM - 2}px;
    }}
    QComboBox QAbstractItemView::item:hover,
    QComboBox QAbstractItemView::item:selected {{
        background: {t.accent}; color: {t.accent_text};
    }}

    /* Spin boxes: stacked chevron buttons */
    QSpinBox, QDoubleSpinBox {{ padding-right: 26px; }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        subcontrol-origin: border; subcontrol-position: top right; width: 22px;
        border: none; border-left: 1px solid {t.separator}; border-top-right-radius: {RADIUS_SM}px;
        background: transparent; margin: 1px 1px 0 0;
    }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border; subcontrol-position: bottom right; width: 22px;
        border: none; border-left: 1px solid {t.separator}; border-bottom-right-radius: {RADIUS_SM}px;
        background: transparent; margin: 0 1px 1px 0;
    }}
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {t.control_pressed}; }}
    QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
    QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{ background: {t.accent}; }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url("{a['chevron_up_small']}"); width: 11px; height: 11px; }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url("{a['chevron_down_small']}"); width: 11px; height: 11px; }}
    QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled,
    QSpinBox::up-arrow:off, QDoubleSpinBox::up-arrow:off {{ image: url("{a['chevron_up_small_disabled']}"); }}
    QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled,
    QSpinBox::down-arrow:off, QDoubleSpinBox::down-arrow:off {{ image: url("{a['chevron_down_small_disabled']}"); }}

    /* Check boxes and radio buttons */
    QCheckBox, QRadioButton {{ spacing: 8px; color: {t.text}; }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 16px; height: 16px; border: 1.5px solid {t.text_tertiary}; background: {t.control};
    }}
    QCheckBox::indicator {{ border-radius: 4px; }}
    QRadioButton::indicator {{ border-radius: 9px; }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {t.accent}; }}
    QCheckBox::indicator:checked {{
        background: {t.accent}; border-color: {t.accent}; image: url("{a['check']}");
    }}
    QRadioButton::indicator:checked {{
        background: {t.accent}; border-color: {t.accent}; image: url("{a['dot']}");
    }}
    QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{ border-color: {t.border}; }}

    /* Lists */
    QListView, QListWidget {{
        background: transparent; border: none; outline: none; color: {t.text};
    }}
    QListView::item, QListWidget::item {{
        border-radius: {RADIUS_SM}px; padding: 3px; margin: 1px 0;
    }}
    QListView::item:hover, QListWidget::item:hover {{ background: {t.surface_hover}; }}
    QListView::item:selected, QListWidget::item:selected {{
        background: {t.selection}; color: {t.text if not t.dark else '#ffffff'};
    }}

    /* Scroll bars: thin overlay style */
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle {{ background: {t.control_pressed}; border-radius: 3px; min-height: 24px; min-width: 24px; }}
    QScrollBar::handle:hover {{ background: {t.text_tertiary}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    /* Panels */
    QWidget#Sidebar, QWidget#Inspector {{ background: {t.sidebar}; }}
    QScrollArea#InspectorScroll, QScrollArea#InspectorScroll > QWidget > QWidget {{
        background: {t.sidebar}; border: none;
    }}
    QLabel#PanelTitle {{ font-weight: 600; color: {t.text}; }}
    QLabel#SectionTitle {{
        color: {t.text_secondary}; font-weight: 600; font-size: 11px; letter-spacing: 0.5px;
    }}
    QLabel#Secondary {{ color: {t.text_secondary}; }}
    QLabel#Hint {{ color: {t.text_tertiary}; font-size: 11px; }}
    QLabel#Warning {{ color: {t.warning}; font-size: 11px; }}
    QLabel#Badge {{
        background: {t.control}; color: {t.text_secondary}; border-radius: 8px;
        padding: 1px 7px; font-size: 11px;
    }}
    QFrame#Card {{
        background: {t.surface}; border: 1px solid {t.border}; border-radius: {RADIUS}px;
    }}
    QFrame#Separator {{ background: {t.separator}; max-height: 1px; min-height: 1px; border: none; }}
    QFrame#VSeparator {{ background: {t.separator}; max-width: 1px; min-width: 1px; border: none; }}
    QToolButton#Disclosure {{
        background: transparent; border: none; color: {t.text_secondary}; font-weight: 600;
        font-size: 11px; padding: 4px 0; text-align: left;
    }}
    QToolButton#Disclosure:hover {{ color: {t.text}; }}
    QToolButton#IconButton {{
        background: transparent; border: none; border-radius: {RADIUS_SM}px; padding: 3px;
    }}
    QToolButton#IconButton:hover {{ background: {t.control_hover}; }}
    QToolButton#IconButton:pressed {{ background: {t.control_pressed}; }}

    QSplitter::handle {{ background: {t.separator}; }}
    QSplitter::handle:horizontal {{ width: 1px; }}
    QSplitter::handle:vertical {{ height: 1px; }}

    /* Activity bar (progress) */
    QWidget#ActivityBar {{ background: {t.window}; border-top: 1px solid {t.separator}; }}
    QProgressBar {{
        background: {t.control}; border: none; border-radius: 3px; max-height: 6px; min-height: 6px;
        text-align: center; color: transparent;
    }}
    QProgressBar::chunk {{ background: {t.accent}; border-radius: 3px; }}

    QStatusBar {{ background: {t.window}; color: {t.text_secondary}; border-top: 1px solid {t.separator}; }}
    QStatusBar::item {{ border: none; }}
    QStatusBar QLabel {{ color: {t.text_secondary}; padding: 0 6px; }}

    QMenu {{
        background: {t.surface}; color: {t.text}; border: 1px solid {t.border};
        border-radius: {RADIUS_SM}px; padding: 4px;
    }}
    QMenu::item {{ padding: 5px 22px 5px 12px; border-radius: 4px; }}
    QMenu::item:selected {{ background: {t.accent}; color: {t.accent_text}; }}
    QMenu::item:disabled {{ color: {t.text_tertiary}; }}
    QMenu::separator {{ height: 1px; background: {t.separator}; margin: 4px 6px; }}

    QSlider::groove:horizontal {{ height: 4px; background: {t.control_pressed}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {t.accent}; border-radius: 2px; }}
    QSlider::handle:horizontal {{
        background: #ffffff; border: 1px solid {t.border}; width: 16px; height: 16px;
        margin: -7px 0; border-radius: 8px;
    }}
    """
