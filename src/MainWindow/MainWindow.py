"""
    Script that houses the MainWindow class.
    It is the "root display".
"""
import os
import cv2
import PySide6.QtWidgets as qtw
import qt_material

# UI dependencies
import MainWindow.QActions as qt_actions_setup
import MainWindow.MainLayout as main_layout

# Algorithm
import algorithm.API as algorithm_API

SUPPORTED_IMAGE_FORMATS = "(*.jpg *.png)"


class Window(qtw.QMainWindow, qt_material.QtStyleTools):
    def __init__(self):
        super().__init__()
        self.statusbar_msg_display_time = 2000  # Time in ms
        self.setWindowTitle("Test")
        # Set min. window size based on pixel size
        geometry = self.screen().availableGeometry()
        self.setMinimumSize(int(geometry.width() * 0.5), int(geometry.height() * 0.5))

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
        self.LaplacianAlgorithm.stack_images()

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
