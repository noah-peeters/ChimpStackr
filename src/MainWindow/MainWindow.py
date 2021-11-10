"""
    Script that houses the MainWindow class.
    It is the "root display".
"""
import os
import cv2
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw
import qt_material

import MainWindow.QActions as qt_actions_setup
import MainWindow.MainLayout.MainLayout as MainLayout
import MainWindow.Threading as QThreading
import MainWindow.ProgressBar as ProgressBar

import algorithm.API as algorithm_API

# TODO: Make UI more expressive after long operation finished. Show success/error messages


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
        self.setCentralWidget(MainLayout.CenterWidget())

        # Permanent progressbar inside statusbar
        self.progress_widget = ProgressBar.ProgressBar()
        self.statusBar().addPermanentWidget(self.progress_widget)
        self.progress_widget.setVisible(False)

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
                self, "Export stacked image", home_dir
            )
            if file_path:
                file_path = os.path.abspath(file_path)
                self.statusBar().showMessage(
                    "Exporting output image...", self.statusbar_msg_display_time
                )
                try:
                    cv2.imwrite(file_path, self.LaplacianAlgorithm.output_image)
                except Exception as e:
                    # Display Error message
                    msg = qtw.QMessageBox(self)
                    msg.setStandardButtons(qtw.QMessageBox.Ok)
                    msg.setIcon(qtw.QMessageBox.Critical)
                    msg.setWindowTitle("Export failed")
                    msg.setText(
                        "Failed to export output image.\nHave you used a supported file extension?\n"
                    )
                    msg.setInformativeText("Error:\n" + str(e))
                    msg.show()
        else:
            self.statusBar().showMessage(
                "No image to export!", self.statusbar_msg_display_time
            )

    # Clear all loaded images
    # TODO: Clear output images
    def clear_all_images(self):
        reply=qtw.QMessageBox.Yes
        if len(self.LaplacianAlgorithm.image_paths) == 0:
            # Ask confirmation (if there are loaded images)
            reply = qtw.QMessageBox.question(
                self,
                "Clear images?",
                "Are you sure you want to clear all loaded images? Output image(s) will be cleared to!",
                qtw.QMessageBox.Yes,
                qtw.QMessageBox.No,
            )
        if reply == qtw.QMessageBox.Yes:
            self.statusBar().showMessage(
                "Clearing all loaded images...", self.statusbar_msg_display_time
            )
            self.centralWidget().set_loaded_images([])
            self.LaplacianAlgorithm.update_image_paths([])

    # Load images from a file on disk
    def load_images_from_file(self):
        self.statusBar().showMessage(
            "Loading selected images...", self.statusbar_msg_display_time
        )
        home_dir = os.path.expanduser("~")
        new_image_files, _ = qtw.QFileDialog.getOpenFileNames(
            self, "Select images to load.", home_dir
        )
        # TODO: Check if valid (and same??) format; discard unsupported formats + show warning
        self.centralWidget().set_loaded_images(new_image_files)
        self.centralWidget().add_processed_image(None)
        self.LaplacianAlgorithm.update_image_paths(new_image_files)

    # Shutdown all currently running processes, cleanup and close window
    def shutdown_application(self):
        self.close()

    # Save project file to disk
    def save_project_to_file(self):
        self.statusBar().showMessage(
            "Saving project file to disk...", self.statusbar_msg_display_time
        )
        # Display success message
        qtw.QMessageBox.information(
            self,
            "Save completed",
            "Successfully saved project to disk.",
            qtw.QMessageBox.Ok,
        )

    # TODO: re-implement (with QThread + timing percentages)
    def align_and_stack_loaded_images(self):
        self.statusBar().showMessage(
            "Started aligning & stacking images...", self.statusbar_msg_display_time
        )
        self.LaplacianAlgorithm.align_and_stack_images()

    def stack_loaded_images(self):
        # TODO: Handle not having an image loaded
        self.statusBar().showMessage(
            "Started stacking images...", self.statusbar_msg_display_time
        )

        def status_update(msg):
            self.progress_widget.progress_label.setText(msg)

        worker = QThreading.Worker(self.LaplacianAlgorithm.stack_images)
        worker.signals.finished.connect(self.finished_stack)
        worker.signals.progress_update.connect(self.update_progressbar_value)
        worker.signals.status_update.connect(status_update)

        # Execute
        self.threadpool.start(worker)

        self.progress_widget.setVisible(True)

    # Update progressbar value to new number
    def update_progressbar_value(self, number):
        self.progress_widget.progressbar.setValue(number)

    # Handle progressbar reset & output image display
    def finished_stack(self):
        self.progress_widget.reset_and_hide()

        self.centralWidget().add_processed_image(self.LaplacianAlgorithm.output_image)

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
            # TODO: Tell possibly running tasks to quit
            event.accept()
        else:
            event.ignore()
