"""
Scene for regular image display with retouching.
Inherits ImageScene class.
"""
import PySide6.QtWidgets as qtw
import PySide6.QtCore as qtc
import PySide6.QtGui as qtg

import src.MainWindow.MainLayout.ImageViewers.ImageScene as image_scene
import src.MainWindow.MainLayout.ImageViewers.RetouchHelpers as retouch_helpers


class ImageRetouchScene(image_scene.ImageScene):
    min_brush_size = 4
    max_brush_size = 600
    brush_size_increment = 7
    current_brush_size = 300
    brush_color = qtg.QColor(qtc.Qt.red)

    def __init__(self, graphicsViewer):
        super().__init__(graphicsViewer)
        self.paint_begin_pos = None
        self.paint_end_pos = None
        self.lines_to_paint = []
        self.cursor_circle = None

        self.UndoRedoClass = retouch_helpers.UndoRedoPixmapClass()

        self.painting_widget = retouch_helpers.PaintingWidget(self)
        self.addWidget(self.painting_widget)

        # Add empty pixmap to undo/redo class
        self.painting_widget.request_save_to_undoredo = True
        self.painting_widget.update()

        # Setup cursor circle
        self.cursor_circle = qtw.QGraphicsEllipseItem(
            0, 0, self.current_brush_size, self.current_brush_size
        )
        pen = qtg.QPen(qtg.QColor(qtc.Qt.yellow))
        pen.setWidth(5)
        self.cursor_circle.setPen(pen)

        self.update_cursor_circle()
        self.addItem(self.cursor_circle)

    # Update cursor circle size & position
    def update_cursor_circle(self, newpos=None):
        # Get center of rectangle (unchanged)
        if not newpos:
            newpos = self.cursor_circle.rect().center()

        # Resize with new position;
        # position is set from topleft corner of circle
        inc = self.current_brush_size / 2
        self.cursor_circle.setRect(
            newpos.x() - inc,
            newpos.y() - inc,
            self.current_brush_size,
            self.current_brush_size,
        )

    # Stop drawing (keeping track of pixmap state for undo/redo)
    def stop_drawing(self):
        if self.paint_begin_pos or self.paint_end_pos:
            self.painting_widget.request_save_to_undoredo = True
            self.painting_widget.update()
            self.paint_begin_pos = None
            self.paint_end_pos = None

    """
    Overridden signals
    """

    # Resize brush on scroll + shift press
    def wheelEvent(self, event: qtg.QWheelEvent) -> None:
        if not event.modifiers() & qtc.Qt.ShiftModifier:
            return super().wheelEvent(event)  # Allow other scrolling event(s)

        if event.delta() < 0:
            # Enlarge brush
            new_size = self.current_brush_size + self.brush_size_increment
            if new_size <= self.max_brush_size:
                self.current_brush_size = new_size
            else:
                self.current_brush_size = self.max_brush_size
        else:
            # Reduce brush
            new_size = self.current_brush_size - self.brush_size_increment
            if new_size >= self.min_brush_size:
                self.current_brush_size = new_size
            else:
                self.current_brush_size = self.min_brush_size

        self.mouse_tooltip.showText(
            qtg.QCursor.pos(),
            str(self.current_brush_size) + "px brush size",
            msecShowTime=self.graphicsViewer.tooltip_displaytime_ms,
        )

        # Update cursor size (QPen size gets updated on mouseclick)
        self.update_cursor_circle()
        event.accept()

    def keyPressEvent(self, event: qtg.QKeyEvent) -> None:
        if event.key() == qtc.Qt.Key_T:
            # Toggle painted mask visibility
            self.painting_widget.mask_visible = not self.painting_widget.mask_visible
            self.painting_widget.update_visible_pixmap()

        if (
            event.modifiers() & qtc.Qt.ShiftModifier
            and event.modifiers() & qtc.Qt.AltModifier
        ):
            # Enable eraser
            self.painting_widget.erase_paint = True

        if event.modifiers() & qtc.Qt.ControlModifier:
            # Handle undo/redo pixmap state
            pixmap = None
            if event.key() == qtc.Qt.Key_Z:
                pixmap = self.UndoRedoClass.undo()
            elif event.key() == qtc.Qt.Key_Y:
                pixmap = self.UndoRedoClass.redo()

            if pixmap is not None:
                self.painting_widget.pixmap_canvas = pixmap.copy()
                self.painting_widget.update_visible_pixmap()

        return super().keyPressEvent(event)

    def keyReleaseEvent(self, event: qtg.QKeyEvent) -> None:
        if not event.modifiers() & qtc.Qt.ShiftModifier:
            # Stop drawing
            self.stop_drawing()

        if not event.modifiers() & qtc.Qt.AltModifier:
            # Disable eraser
            self.painting_widget.erase_paint = False

        return super().keyReleaseEvent(event)

    def mousePressEvent(self, event: qtw.QGraphicsSceneMouseEvent) -> None:
        if event.modifiers() & qtc.Qt.ShiftModifier:
            # Start drawing (mouse click + shift)
            self.paint_begin_pos = event.scenePos()
            event.accept()
        else:
            # Pan scene using mouseclick drag
            return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: qtw.QGraphicsSceneMouseEvent) -> None:
        if event.modifiers() & qtc.Qt.ShiftModifier:
            # Stop drawing (mouseclick released)
            self.stop_drawing()
        else:
            return super().mouseReleaseEvent(event)

    # Draw line on mouse movement + update cursor circle
    def mouseMoveEvent(self, event: qtw.QGraphicsSceneMouseEvent) -> None:
        # Update cursor circle
        newPos = event.scenePos()
        self.update_cursor_circle(newPos)

        if not event.modifiers() & qtc.Qt.ShiftModifier:
            return super().mouseMoveEvent(event)
        else:
            if not self.paint_begin_pos:
                return  # Not allowed to draw (no mouse click)

            self.paint_end_pos = newPos
            # TODO: Determine reason for 10px X-offset
            self.lines_to_paint.append(
                qtc.QLineF(
                    self.paint_begin_pos.x() - 10,
                    self.paint_begin_pos.y(),
                    self.paint_end_pos.x() - 10,
                    self.paint_end_pos.y(),
                )
                # qtc.QLineF(self.paint_begin_pos, self.paint_end_pos)
            )

            # Let QWidget handle (re)painting
            self.painting_widget.update()

            # Update start position for next redraw
            self.paint_begin_pos = newPos
