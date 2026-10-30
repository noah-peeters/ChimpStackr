"""
Line icons for ChimpStackr (24-unit grid, 1.6 stroke), drawn from inline SVG.

Icons are rendered on demand at the requested size and device pixel ratio,
so they are sharp on Retina/HiDPI screens, and take their color from the
current theme when painted, so they follow light/dark switches.
"""
from PySide6.QtCore import QByteArray, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QIcon, QIconEngine, QPainter, QPixmap, QGuiApplication
from PySide6.QtSvg import QSvgRenderer

import src.MainWindow.theme as theme

_PATHS = {
    "add": '<path d="M12 5v14M5 12h14"/>',
    "open": '<path d="M3.5 7.5v10a1.5 1.5 0 0 0 1.5 1.5h14a1.5 1.5 0 0 0 1.5-1.5v-8A1.5 1.5 0 0 0 19 8h-7l-2-2.5H5A1.5 1.5 0 0 0 3.5 7z"/>',
    "export": '<path d="M12 15V4M8 8l4-4 4 4"/><path d="M5 13v5.5A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5V13"/>',
    "play": '<path d="M8 5.5v13l10.5-6.5z" fill="currentColor"/>',
    "pause": '<path d="M8.5 5.5v13M15.5 5.5v13" stroke-width="2.4"/>',
    "stop": '<rect x="6.5" y="6.5" width="11" height="11" rx="2" fill="currentColor"/>',
    "stack": '<path d="M12 4l8 4-8 4-8-4z"/><path d="M4 12l8 4 8-4"/><path d="M4 16l8 4 8-4"/>',
    "settings": '<path d="M4 7h10M18 7h2M4 17h4M12 17h8"/><circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/>',
    "sidebar": '<rect x="3.5" y="5" width="17" height="14" rx="2"/><path d="M9 5v14"/>',
    "inspector": '<rect x="3.5" y="5" width="17" height="14" rx="2"/><path d="M15 5v14"/>',
    "zoom_in": '<circle cx="11" cy="11" r="6.5"/><path d="M16 16l4 4M8 11h6M11 8v6"/>',
    "zoom_out": '<circle cx="11" cy="11" r="6.5"/><path d="M16 16l4 4M8 11h6"/>',
    "fit": '<path d="M4 9V5.5A1.5 1.5 0 0 1 5.5 4H9M15 4h3.5A1.5 1.5 0 0 1 20 5.5V9M20 15v3.5a1.5 1.5 0 0 1-1.5 1.5H15M9 20H5.5A1.5 1.5 0 0 1 4 18.5V15"/>',
    "actual": '<rect x="3.5" y="5" width="17" height="14" rx="2"/><path d="M8 10l1.5-1v6M14 10l1.5-1v6"/><circle cx="12" cy="11" r=".4" fill="currentColor"/><circle cx="12" cy="14" r=".4" fill="currentColor"/>',
    "compare": '<rect x="3.5" y="5" width="17" height="14" rx="2"/><path d="M12 3v18"/><path d="M3.5 5h8.5v14H5.5a2 2 0 0 1-2-2z" fill="currentColor" fill-opacity=".25" stroke="none"/>',
    "view": '<rect x="3.5" y="5" width="17" height="14" rx="2"/><path d="M3.5 16l5-5 4 4 2.5-2.5 5.5 5.5"/><circle cx="15.5" cy="9" r="1.3"/>',
    "trash": '<path d="M4.5 7h15M9.5 7V5h5v2M6.5 7l1 12.5h9l1-12.5M10 11v5M14 11v5"/>',
    "reveal": '<path d="M13 5h6v6M19 5l-8 8"/><path d="M17 14v4.5a1.5 1.5 0 0 1-1.5 1.5h-10A1.5 1.5 0 0 1 4 18.5v-10A1.5 1.5 0 0 1 5.5 7H10"/>',
    "image": '<rect x="3.5" y="5" width="17" height="14" rx="2"/><path d="M3.5 16l5-5 4 4 2.5-2.5 5.5 5.5"/><circle cx="15.5" cy="9" r="1.3"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    "chevron_right": '<path d="M10 6l6 6-6 6"/>',
    "chevron_down": '<path d="M6 10l6 6 6-6"/>',
    "info": '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5"/><circle cx="12" cy="8" r=".6" fill="currentColor"/>',
    "crop": '<path d="M7 3v14h14M3 7h14v14"/>',
    "drop": '<path d="M12 4v10M8 10l4 4 4-4"/><path d="M4 15v3.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V15"/>',
}

# color roles a caller can ask for (resolved against the theme when painted)
_ROLES = {
    "text": lambda t: t.text,
    "secondary": lambda t: t.text_secondary,
    "accent": lambda t: t.accent,
    "on_accent": lambda t: t.accent_text,
    "danger": lambda t: t.danger,
}


def _svg(name, color):
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" color="{color}" stroke-width="1.6" stroke-linecap="round" '
        f'stroke-linejoin="round">{_PATHS[name]}</svg>'
    ).replace("currentColor", color)


class _Engine(QIconEngine):
    def __init__(self, name, role):
        super().__init__()
        self.name, self.role = name, role

    def clone(self):
        return _Engine(self.name, self.role)

    def _color(self, mode):
        t = theme.tokens()
        if mode == QIcon.Disabled:
            return t.text_tertiary
        return _ROLES.get(self.role, _ROLES["text"])(t)

    def paint(self, painter, rect, mode, state):
        renderer = QSvgRenderer(QByteArray(_svg(self.name, self._color(mode)).encode()))
        renderer.render(painter, QRectF(rect))

    def pixmap(self, size, mode, state):
        return self.scaledPixmap(size, mode, state, QGuiApplication.instance().devicePixelRatio()
                                 if QGuiApplication.instance() else 1.0)

    def scaledPixmap(self, size, mode, state, scale):
        scale = max(1.0, float(scale or 1.0))
        pm = QPixmap(QSize(round(size.width() * scale), round(size.height() * scale)))
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        self.paint(p, QRect(0, 0, pm.width(), pm.height()), mode, state)
        p.end()
        pm.setDevicePixelRatio(scale)
        return pm


def icon(name, role="text"):
    """Theme-aware QIcon for one of the names in _PATHS."""
    return QIcon(_Engine(name, role))


# Back-compat helpers used across the app
def icon_open():
    return icon("open")


def icon_save():
    return icon("export")


def icon_play():
    return icon("play")


def icon_pause():
    return icon("pause")


def icon_stop():
    return icon("stop")


def icon_settings():
    return icon("settings")


def icon_zoom_in():
    return icon("zoom_in")


def icon_zoom_out():
    return icon("zoom_out")


def icon_fit():
    return icon("fit")


def icon_stack():
    return icon("stack")
