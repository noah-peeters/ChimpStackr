"""
macOS-native dark mode stylesheet for ChimpStackr.
Inspired by Finder, Photos.app, and Final Cut Pro dark appearance.
Uses Apple's system gray palette and native spacing conventions.
"""

# macOS Dark Mode system colors
BG_WINDOW = "#1e1e1e"          # NSColor.windowBackgroundColor (dark)
BG_SIDEBAR = "#2a2a2a"         # Sidebar / source list
BG_CONTENT = "#232323"         # Main content area
BG_HEADER = "#2d2d2d"          # Toolbar / header bars
BG_CONTROL = "#3a3a3a"         # Button / control backgrounds
BG_CONTROL_HOVER = "#454545"   # Hover state
BG_CONTROL_ACTIVE = "#505050"  # Pressed state
BG_SELECTED = "#0058d0"        # macOS selection blue
BG_SELECTED_DIM = "rgba(0, 88, 208, 0.25)"
TEXT = "#ececec"               # Primary text
TEXT_SECONDARY = "#999999"     # Secondary / label text
TEXT_TERTIARY = "#666666"      # Disabled / placeholder
SEPARATOR = "#3d3d3d"         # Separator lines
BORDER = "#3a3a3a"            # Control borders
ACCENT = "#4ca0ff"            # macOS blue accent (active controls)
ACCENT_HOVER = "#6ab4ff"
ACCENT_DIM = "rgba(76, 160, 255, 0.15)"
RED = "#ff6961"               # Destructive / cancel
GREEN = "#63d97b"
RADIUS = "6px"                # macOS standard corner radius
RADIUS_SM = "4px"

# Export for other modules
TEXT_MUTED = TEXT_TERTIARY


def get_stylesheet():
    return f"""

    * {{
        font-family: "Helvetica Neue", "Segoe UI", sans-serif;
        font-size: 13px;
    }}

    QMainWindow {{
        background: {BG_WINDOW};
    }}

    QWidget {{
        background: {BG_WINDOW};
        color: {TEXT};
    }}

    /* ── Menu Bar (native-like) ── */
    QMenuBar {{
        background: {BG_HEADER};
        color: {TEXT};
        border-bottom: 1px solid {SEPARATOR};
        padding: 1px 0;
        spacing: 0;
    }}
    QMenuBar::item {{
        background: transparent;
        padding: 4px 10px;
        border-radius: {RADIUS_SM};
        margin: 2px 1px;
    }}
    QMenuBar::item:selected {{
        background: {BG_SELECTED};
        color: white;
    }}

    QMenu {{
        background: {BG_SIDEBAR};
        color: {TEXT};
        border: 1px solid {SEPARATOR};
        border-radius: {RADIUS};
        padding: 4px 0;
    }}
    QMenu::item {{
        padding: 4px 20px 4px 10px;
        border-radius: {RADIUS_SM};
        margin: 1px 4px;
    }}
    QMenu::item:selected {{
        background: {BG_SELECTED};
        color: white;
    }}
    QMenu::separator {{
        height: 1px;
        background: {SEPARATOR};
        margin: 4px 10px;
    }}
    QMenu::icon {{
        padding-left: 6px;
    }}

    /* ── Toolbar (like Finder / Photos toolbar) ── */
    QToolBar {{
        background: {BG_HEADER};
        border-bottom: 1px solid {SEPARATOR};
        padding: 3px 8px;
        spacing: 2px;
    }}
    QToolBar::separator {{
        width: 1px;
        background: {SEPARATOR};
        margin: 3px 6px;
    }}
    QToolButton {{
        background: transparent;
        color: {TEXT_SECONDARY};
        border: none;
        border-radius: {RADIUS_SM};
        padding: 4px 8px;
    }}
    QToolButton:hover {{
        background: {BG_CONTROL};
        color: {TEXT};
    }}
    QToolButton:pressed {{
        background: {BG_CONTROL_ACTIVE};
        color: white;
    }}

    /* ── Source List / Sidebar (like Finder sidebar) ── */
    QListWidget {{
        background: {BG_SIDEBAR};
        color: {TEXT};
        border: none;
        border-radius: 0;
        outline: none;
        padding: 2px;
    }}
    QListWidget::item {{
        padding: 4px 8px;
        border-radius: {RADIUS_SM};
        border: none;
        margin: 1px 2px;
    }}
    QListWidget::item:selected {{
        background: {BG_SELECTED_DIM};
        color: {ACCENT};
    }}
    QListWidget::item:selected:active {{
        background: {BG_SELECTED};
        color: white;
    }}
    QListWidget::item:hover:!selected {{
        background: rgba(255, 255, 255, 0.05);
    }}

    /* ── Content Area / Image Viewer ── */
    QGraphicsView {{
        background: {BG_CONTENT};
        border: none;
    }}

    /* ── Tab Bar (like Safari / Xcode tabs) ── */
    QTabWidget::pane {{
        background: {BG_WINDOW};
        border: none;
        border-top: 1px solid {SEPARATOR};
    }}
    QTabBar {{
        background: {BG_HEADER};
    }}
    QTabBar::tab {{
        background: transparent;
        color: {TEXT_SECONDARY};
        padding: 6px 16px;
        border: none;
        border-bottom: 2px solid transparent;
        margin: 0 1px;
    }}
    QTabBar::tab:selected {{
        color: white;
        background: {BG_SELECTED};
        border-radius: {RADIUS_SM};
        border-bottom: none;
    }}
    QTabBar::tab:hover:!selected {{
        color: {TEXT};
        background: {BG_CONTROL};
        border-radius: {RADIUS_SM};
    }}

    /* ── Splitter (thin macOS-style) ── */
    QSplitter::handle {{
        background: {SEPARATOR};
    }}
    QSplitter::handle:horizontal {{
        width: 1px;
    }}
    QSplitter::handle:vertical {{
        height: 1px;
    }}

    /* ── Progress Bar ── */
    QProgressBar {{
        background: {BG_CONTROL};
        border: none;
        border-radius: 3px;
        text-align: center;
        color: {TEXT};
        height: 6px;
        font-size: 0;
    }}
    QProgressBar::chunk {{
        background: {ACCENT};
        border-radius: 3px;
    }}

    /* ── Status Bar ── */
    QStatusBar {{
        background: {BG_HEADER};
        color: {TEXT_SECONDARY};
        border-top: 1px solid {SEPARATOR};
        font-size: 11px;
    }}
    QStatusBar::item {{
        border: none;
    }}

    /* ── Buttons (macOS push button style) ── */
    QPushButton {{
        background: {BG_CONTROL};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SM};
        padding: 4px 14px;
        min-height: 20px;
    }}
    QPushButton:hover {{
        background: {BG_CONTROL_HOVER};
    }}
    QPushButton:pressed {{
        background: {BG_CONTROL_ACTIVE};
    }}
    QPushButton:disabled {{
        color: {TEXT_TERTIARY};
        background: {BG_SIDEBAR};
    }}
    QPushButton:default {{
        background: {BG_SELECTED};
        color: white;
        border: none;
    }}
    QPushButton:default:hover {{
        background: #2070e0;
    }}

    /* ── Inputs (macOS control style) ── */
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{
        background: {BG_CONTROL};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: {RADIUS_SM};
        padding: 3px 8px;
        min-height: 20px;
        selection-background-color: {BG_SELECTED};
    }}
    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:on, QLineEdit:focus {{
        border-color: {ACCENT};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    QComboBox QAbstractItemView {{
        background: {BG_SIDEBAR};
        color: {TEXT};
        border: 1px solid {SEPARATOR};
        selection-background-color: {BG_SELECTED};
        selection-color: white;
        outline: none;
        border-radius: {RADIUS};
    }}

    /* ── Scrollbars (thin macOS overlay style) ── */
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: rgba(255, 255, 255, 0.15);
        border-radius: 4px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: rgba(255, 255, 255, 0.3);
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: transparent;
        height: 0;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 8px;
        margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: rgba(255, 255, 255, 0.15);
        border-radius: 4px;
        min-width: 30px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: rgba(255, 255, 255, 0.3);
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
        background: transparent;
        width: 0;
    }}

    /* ── Tooltips ── */
    QToolTip {{
        background: {BG_CONTROL};
        color: {TEXT};
        border: 1px solid {SEPARATOR};
        border-radius: {RADIUS_SM};
        padding: 4px 8px;
        font-size: 12px;
    }}

    /* ── Slider (macOS-like) ── */
    QSlider::groove:horizontal {{
        border: none;
        height: 4px;
        background: {BG_CONTROL};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: white;
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
        border: 0.5px solid rgba(0,0,0,0.2);
    }}
    QSlider::sub-page:horizontal {{
        background: {ACCENT};
        border-radius: 2px;
    }}

    /* ── Group Box ── */
    QGroupBox {{
        background: {BG_SIDEBAR};
        border: 1px solid {SEPARATOR};
        border-radius: {RADIUS};
        margin-top: 14px;
        padding: 20px 10px 10px 10px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 2px 8px;
        color: {TEXT_SECONDARY};
        font-size: 12px;
    }}

    /* ── Labels ── */
    QLabel {{
        background: transparent;
        color: {TEXT};
    }}

    /* ── Dialogs ── */
    QMessageBox, QDialog, QFileDialog {{
        background: {BG_WINDOW};
    }}
    QMessageBox QLabel {{
        color: {TEXT};
    }}

    /* ── Header / Table View ── */
    QHeaderView::section {{
        background: {BG_HEADER};
        color: {TEXT_SECONDARY};
        border: none;
        border-right: 1px solid {SEPARATOR};
        padding: 4px 8px;
        font-weight: 600;
        font-size: 11px;
    }}
    """
