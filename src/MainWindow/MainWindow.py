"""
    Script that houses the MainWindow class.
    It is the "root display".
"""
import os
import PySide6.QtGui as qtg
import PySide6.QtWidgets as qtw

import MainWindow.QActions as qt_actions_setup

SUPPORTED_IMAGE_FORMATS = "(*.jpg *.png)"


class Window(qtw.QMainWindow):
    def __init__(self):
        qtw.QMainWindow.__init__(self)

        self.statusbar_msg_display_time = 2000  # Time in ms

        self.setWindowTitle("Test")
        # Setup actions
        qt_actions_setup.setup_actions(self)

        # Set min. window size based on pixel size
        geometry = self.screen().availableGeometry()
        self.setMinimumSize(int(geometry.width() * 0.7), int(geometry.height() * 0.7))

    # Display dialog if user is sure they want to quit
    def closeEvent(self, event):
        reply = qtw.QMessageBox.question(
            self,
            "Exit program",
            "Are you sure you want to exit the program? You might lose unsaved work!",
            qtw.QMessageBox.Yes,
            qtw.QMessageBox.No,
        )
        if reply == qtw.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    # Export output image to file on disk
    def export_output_image(self):
        self.statusBar().showMessage(
            "Exporting output image...", self.statusbar_msg_display_time
        )

        # rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(path, rgb)

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
        new_image_files = qtw.QFileDialog.getOpenFileNames(
            self,
            "Select images to load.",
            home_dir,
            "Image files " + SUPPORTED_IMAGE_FORMATS,
        )

    # Shutdown all currently running processes, cleanup and close window
    def shutdown_application(self):
        self.close()
        # TODO: Display dialog to make sure user wants to quit
        # qtw.qApp.aboutToQuit()

    # Save project file to disk
    def save_project_to_file(self):
        self.statusBar().showMessage(
            "Saving project file to disk...", self.statusbar_msg_display_time
        )

    def align_and_stack_loaded_images(self):
        self.statusBar().showMessage(
            "Started aligning & stacking images...", self.statusbar_msg_display_time
        )

    def stack_loaded_images(self):
        self.statusBar().showMessage(
            "Started stacking images...", self.statusbar_msg_display_time
        )