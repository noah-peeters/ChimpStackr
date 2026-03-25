"""
Threading utilities for running long tasks without freezing the UI.
"""
import traceback
import sys
import logging
import PySide6.QtCore as qtc

logger = logging.getLogger(__name__)


class WorkerSignals(qtc.QObject):
    """Signals emitted by Worker threads. All signals are thread-safe (Qt handles marshalling)."""
    finished = qtc.Signal()
    error = qtc.Signal(tuple)  # (exc_type, exc_value, traceback_str)
    finished_inter_task = qtc.Signal(list)


class Worker(qtc.QRunnable):
    """Runs a callable on a QThreadPool thread with signal support."""

    def __init__(self, fn, *args):
        super().__init__()
        self.fn = fn
        self.args = args
        self.signals = WorkerSignals()

    @qtc.Slot()
    def run(self):
        try:
            self.fn(*self.args, self.signals)
        except Exception:
            exctype, value = sys.exc_info()[:2]
            tb = traceback.format_exc()
            logger.error("Worker error: %s", tb)
            self.signals.error.emit((exctype, value, tb))
        finally:
            self.signals.finished.emit()
