"""
    Classes that enable easy threading.
    Used to prevent UI freezing when running long tasks.
    src: https://www.pythonguis.com/tutorials/multithreading-pyqt-applications-qthreadpool/
"""
import PySide6.QtCore as qtc
import traceback, sys


class WorkerSignals(qtc.QObject):
    """
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress
    """

    finished = qtc.Signal()
    error = qtc.Signal(tuple)
    result = qtc.Signal(object)
    progress = qtc.Signal(int)


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
            # TODO: Pass signals, so function can update progress
            result = self.fn(*self.args, self.signals)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        
        self.signals.finished.emit()
