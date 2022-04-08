"""
Classes that help the RetouchScene.
Invoked internally by RetouchScene.
"""
import PySide6.QtWidgets as qtw
import PySide6.QtCore as qtc
import PySide6.QtGui as qtg

# Helper for easy undo/redo of pixmap states
class UndoRedoPixmapClass:
    max_length = 100  # Max amount of pixmaps to keep in undo/redo memory

    def __init__(self):
        self.donePixmaps = []  # List of pixmaps the user can undo (Ctrl+Z)
        self.undonePixmaps = []  # List of pixmaps the user can redo (Ctrl+Y)

    def addPixmap(self, pixmap):
        # Append at endpos (copy() pixmap!!!)
        self.donePixmaps.append(pixmap.copy())

        if len(self.donePixmaps) > self.max_length:
            self.donePixmaps.pop(0)

    # Return previous pixmap state
    def undo(self):
        if len(self.donePixmaps) > 1:
            self.undonePixmaps.append(self.donePixmaps.pop())
            if len(self.undonePixmaps) > self.max_length:
                self.undonePixmaps.pop(0)
            return self.donePixmaps[-1]

    # Return previously undone pixmap state
    def redo(self):
        if len(self.undonePixmaps) > 0:
            recoveredPixmap = self.undonePixmaps.pop()
            self.donePixmaps.append(recoveredPixmap)
            if len(self.donePixmaps) > self.max_length:
                self.donePixmaps.pop(0)
            return recoveredPixmap


# Widget overlay that handles painting
class PaintingWidget(qtw.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.mask_visible = True
        self.erase_paint = False
        self.request_save_to_undoredo = False

        # Transparent background
        self.setAttribute(qtc.Qt.WA_NoSystemBackground)

        hLayout = qtw.QHBoxLayout()
        self.label = qtw.QLabel()

        self.default_pixmap = qtg.QPixmap(5000, 5000)
        self.default_pixmap.fill(qtc.Qt.transparent)
        self.pixmap_canvas = self.default_pixmap.copy()
        self.update_visible_pixmap()

        hLayout.addWidget(self.label)
        self.setLayout(hLayout)

    def update_visible_pixmap(self):
        if self.mask_visible:
            self.label.setPixmap(self.pixmap_canvas)
        else:
            self.label.setPixmap(self.default_pixmap)

    def paintEvent(self, event: qtg.QPaintEvent) -> None:
        if not self.parent.pixmapPicture or not self.parent.pixmapPicture.pixmap():
            # Don't paint lines if no image has been set
            self.parent.lines_to_paint = []
            return

        # Resize label/pixmap(s) to image
        size = self.parent.pixmapPicture.pixmap().size()
        if (
            self.size() != size
            or self.default_pixmap.size() != size
            or self.pixmap_canvas.size() != size
        ):
            self.setFixedSize(size)
            self.default_pixmap = self.default_pixmap.scaled(size)
            self.pixmap_canvas = self.pixmap_canvas.scaled(size)

        # Paint a new line
        if len(self.parent.lines_to_paint) > 0:
            qPainter = qtg.QPainter(self.pixmap_canvas)
            qPen = qtg.QPen()
            qPen.setCapStyle(qtc.Qt.PenCapStyle.RoundCap)
            qPen.setWidth(self.parent.current_brush_size)
            qPen.setColor(self.parent.brush_color)
            if self.erase_paint:
                qPainter.setCompositionMode(qtg.QPainter.CompositionMode_Clear)

            qPainter.setPen(qPen)

            lineToDraw = self.parent.lines_to_paint.pop()
            qPainter.drawLine(lineToDraw)

            qPainter.end()
            self.update_visible_pixmap()

            # Keep painting lines
            if len(self.parent.lines_to_paint) > 0:
                self.update()

        # Save current pixmap if all lines were drawn
        if self.request_save_to_undoredo and len(self.parent.lines_to_paint) == 0:
            self.parent.UndoRedoClass.addPixmap(self.pixmap_canvas)
            self.request_save_to_undoredo = False
