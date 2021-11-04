"""
    Script that houses the MainWindow class.
    It is the "root display".
"""
import os
import cv2
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw
import qt_material

# UI dependencies
import MainWindow.QActions as qt_actions_setup
import MainWindow.MainLayout as main_layout

# Algorithm
import algorithm.API as algorithm_API

SUPPORTED_IMAGE_FORMATS = "(*.jpg *.png)"


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

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        #self.kwargs["progress_callback"] = self.signals.progress

    @qtc.Slot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class Window(qtw.QMainWindow, qt_material.QtStyleTools):
    def __init__(self):
        super().__init__()
        self.statusbar_msg_display_time = 2000  # Time in ms
        self.setWindowTitle("Test")
        # Set min. window size based on pixel size
        geometry = self.screen().availableGeometry()
        self.setMinimumSize(int(geometry.width() * 0.6), int(geometry.height() * 0.6))

        # Setup actions
        qt_actions_setup.setup_actions(self)
        # Set center widget
        self.setCentralWidget(main_layout.CenterWidget(self))

        # Stylesheet
        # TODO: Make setting toggle that saves stylesheet
        self.apply_stylesheet(self, "dark_blue.xml")

        # Setup algorithm API
        # TODO: Allow user to change program settings
        self.LaplacianAlgorithm = algorithm_API.LaplacianPyramid(6, 8)

        # Threadpool for multi-threading (prevent UI freezing)
        self.threadpool = qtc.QThreadPool()
        print(
            "Multithreading with maximum %d threads" % self.threadpool.maxThreadCount()
        )

    # Export output image to file on disk
    def export_output_image(self):
        if self.LaplacianAlgorithm.output_image is not None:
            home_dir = os.path.expanduser("~")
            file_path, _ = qtw.QFileDialog.getSaveFileName(
                self, "Export stacked image", home_dir, SUPPORTED_IMAGE_FORMATS
            )
            if file_path:
                file_path = os.path.abspath(file_path)
                self.statusBar().showMessage(
                    "Exporting output image...", self.statusbar_msg_display_time
                )

                cv2.imwrite(file_path, self.LaplacianAlgorithm.output_image)

    # Clear all loaded images
    def clear_all_images(self):
        self.statusBar().showMessage(
            "Clearing all loaded images...", self.statusbar_msg_display_time
        )

    # Load images from a file on disk
    def load_images_from_file(self):
        self.statusBar().showMessage(
            "Loading selected images...", self.statusbar_msg_display_time
        )
        home_dir = os.path.expanduser("~")
        new_image_files, _ = qtw.QFileDialog.getOpenFileNames(
            self,
            "Select images to load.",
            home_dir,
            "Image files " + SUPPORTED_IMAGE_FORMATS,
        )
        self.centralWidget().set_loaded_images(new_image_files)

        self.LaplacianAlgorithm.update_image_paths(new_image_files)

    # Shutdown all currently running processes, cleanup and close window
    def shutdown_application(self):
        self.close()

    # Save project file to disk
    def save_project_to_file(self):
        self.statusBar().showMessage(
            "Saving project file to disk...", self.statusbar_msg_display_time
        )

    def align_and_stack_loaded_images(self):
        self.statusBar().showMessage(
            "Started aligning & stacking images...", self.statusbar_msg_display_time
        )
        self.LaplacianAlgorithm.align_and_stack_images()

    def stack_loaded_images(self):
        self.statusBar().showMessage(
            "Started stacking images...", self.statusbar_msg_display_time
        )
        # self.LaplacianAlgorithm.stack_images()

        def print_output(s):
            print(s)

        def thread_complete():
            print("COMPLETED")

        def progress_fn(n):
            print("%d%% done" % n)

        worker = Worker(self.LaplacianAlgorithm.stack_images)
        worker.signals.result.connect(print_output)
        worker.signals.finished.connect(thread_complete)
        worker.signals.progress.connect(progress_fn)

        # Execute
        self.threadpool.start(worker)

    """
        Overridden signals
    """
    # Display dialog to confirm if user wants to quit
    def closeEvent(self, event):
        reply = qtw.QMessageBox.question(
            self,
            "Exit program?",
            "Are you sure you want to exit the program? You might lose unsaved work!",
            qtw.QMessageBox.Yes,
            qtw.QMessageBox.No,
        )
        if reply == qtw.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
