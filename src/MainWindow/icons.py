"""
Modern SVG icons for ChimpStackr.
Clean line-art style, rendered as QIcons from inline SVGs.
"""
from PySide6.QtGui import QIcon, QPixmap, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtCore import QByteArray, QSize, Qt


def _svg_to_icon(svg_str, size=20):
    ba = QByteArray(svg_str.encode())
    renderer = QSvgRenderer(ba)
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)


C = "#cccccc"
A = "#4ca0ff"
S = f'stroke="{C}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="none"'


def icon_open():
    return _svg_to_icon(f'''<svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
        <path d="M3 7V15C3 15.55 3.45 16 4 16H16C16.55 16 17 15.55 17 15V9C17 8.45 16.55 8 16 8H10L8 5H4C3.45 5 3 5.45 3 6V7Z" {S}/>
    </svg>''')


def icon_save():
    return _svg_to_icon(f'''<svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
        <path d="M10 4V12M10 12L7 9M10 12L13 9" {S}/>
        <path d="M4 14V15C4 15.55 4.45 16 5 16H15C15.55 16 16 15.55 16 15V14" {S}/>
    </svg>''')


def icon_play():
    return _svg_to_icon(f'''<svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
        <path d="M7 5L15 10L7 15V5Z" stroke="{A}" stroke-width="1.5" stroke-linejoin="round" fill="{A}" fill-opacity="0.2"/>
    </svg>''')


def icon_pause():
    return _svg_to_icon(f'''<svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
        <rect x="6" y="5" width="2.5" height="10" rx="0.5" fill="#ffb84d"/>
        <rect x="11.5" y="5" width="2.5" height="10" rx="0.5" fill="#ffb84d"/>
    </svg>''')


def icon_stop():
    return _svg_to_icon(f'''<svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
        <rect x="5.5" y="5.5" width="9" height="9" rx="1.5" stroke="#ff6961" stroke-width="1.5" fill="#ff6961" fill-opacity="0.2"/>
    </svg>''')


def icon_settings():
    """Gear/cog wheel icon."""
    return _svg_to_icon(f'''<svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
        <path d="M8.5 3H11.5L12 5.1C12.4 5.25 12.8 5.45 13.15 5.7L15.1 5L16.6 7.6L15 9C15.05 9.33 15.05 9.67 15 10L16.6 11.4L15.1 14L13.15 13.3C12.8 13.55 12.4 13.75 12 13.9L11.5 16H8.5L8 13.9C7.6 13.75 7.2 13.55 6.85 13.3L4.9 14L3.4 11.4L5 10C4.95 9.67 4.95 9.33 5 9L3.4 7.6L4.9 5L6.85 5.7C7.2 5.45 7.6 5.25 8 5.1L8.5 3Z" {S}/>
        <circle cx="10" cy="9.5" r="2" {S}/>
    </svg>''')


def icon_zoom_in():
    return _svg_to_icon(f'''<svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
        <circle cx="9" cy="9" r="5" {S}/>
        <path d="M13 13L17 17" {S}/>
        <path d="M9 7V11M7 9H11" {S}/>
    </svg>''')


def icon_zoom_out():
    return _svg_to_icon(f'''<svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
        <circle cx="9" cy="9" r="5" {S}/>
        <path d="M13 13L17 17" {S}/>
        <path d="M7 9H11" {S}/>
    </svg>''')


def icon_fit():
    return _svg_to_icon(f'''<svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
        <path d="M4 7V4H7M13 4H16V7M16 13V16H13M7 16H4V13" {S}/>
    </svg>''')


def icon_stack():
    return _svg_to_icon(f'''<svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
        <path d="M3 10L10 6L17 10L10 14L3 10Z" {S}/>
        <path d="M3 13L10 17L17 13" {S}/>
        <path d="M3 7L10 3L17 7" {S}/>
    </svg>''')
