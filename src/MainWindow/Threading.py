"""
    Classes that enable easy threading.
    Used to prevent UI freezing when running long tasks.
    src: https://www.pythonguis.com/tutorials/multithreading-pyqt-applications-qthreadpool/
"""
import PySide6.QtCore as qtc
import traceback, sys

# Signals available from a running worker thread
class WorkerSignals(qtc.QObject):
    # Fire when process finishes
    finished = qtc.Signal()
    # Error message
    # TODO: Use error signal
    error = qtc.Signal(tuple)
    # TODO: Use stop signal
    # Make thread stop task in progress
    stop = qtc.Signal()
    # General signal to call once part of a task has finished
    finished_inter_task = qtc.Signal(list)


class Worker(qtc.QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    """

    def __init__(self, fn, *args):
        super(Worker, self).__init__()

        self.fn = fn
        self.args = args
        self.signals = WorkerSignals()

    @qtc.Slot()
    def run(self):
        """
        Initialise the runner function with passed args.
        """

        # Retrieve args/kwargs here; and fire processing using them
        try:
            self.fn(*self.args, self.signals)
        except:
            # Emit error message
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))

        # Emit finished signal
        self.signals.finished.emit()
