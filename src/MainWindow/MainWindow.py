"""
    Script that houses the MainWindow class.
    It is the "root display".
"""
import os
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw

import src.settings as settings
import src.MainWindow.QActions as qt_actions_setup
import src.MainWindow.MainLayout.MainLayout as MainLayout
import src.MainWindow.Threading as QThreading
import src.MainWindow.ProgressBar as ProgressBar

# import src.MainWindow.StackSuccessDialog as StackFinishedDialog
import src.MainWindow.TimeRemainingHandler as TimeRemainingHandler
import src.MainWindow.ImageSavingDialog as ImageSavingDialog
import src.MainWindow.SettingsWidget as SettingsWidget

import src.algorithm.API as algorithm_API

# TODO: Make UI more expressive after long operation finished. Show success/error messages


class Window(qtw.QMainWindow):
    loaded_image_names = []
    # Reference dir for image loading/export
    current_image_directory = os.path.expanduser("~")

    def __init__(self):
        super().__init__()

        settings.globalVars["MainWindow"] = self
        settings.globalVars["LoadedImagePaths"] = []

        self.statusbar_msg_display_time = 2000  # (ms)

        self.setWindowTitle("ChimpStackr")
        geometry = self.screen().availableGeometry()
        self.setMinimumSize(int(geometry.width() * 0.6), int(geometry.height() * 0.6))

        self.SettingsWidget = SettingsWidget.SettingsWidget()

        # Setup actions
        qt_actions_setup.setup_actions()
        # Set center widget
        self.setCentralWidget(MainLayout.CenterWidget())

        # Permanent progressbar inside statusbar
        self.progress_widget = ProgressBar.ProgressBar()
        self.statusBar().addPermanentWidget(self.progress_widget)
        self.progress_widget.setVisible(False)

        # Setup algorithm API
        # TODO: Allow user to change program settings
        self.LaplacianAlgorithm = algorithm_API.LaplacianPyramid(
            settings.globalVars["RootTempDir"], 6, 8
        )
        self.TimeRemainingHandler = TimeRemainingHandler.TimeRemainingHandler()

        # Threadpool for multi-threading (prevent UI freezing)
        self.threadpool = qtc.QThreadPool()

    # Export output image to file on disk
    def export_output_image(self):
        if self.LaplacianAlgorithm.output_image is not None:
            outputFilePath, usedFilter = qtw.QFileDialog.getSaveFileName(
                self,
                "Export stacked image",
                self.current_image_directory,
                "JPEG (*.jpg *.jpeg);; PNG (*.png);; TIFF (*.tiff *.tif)",
            )
            if outputFilePath:
                outputFilePath = os.path.abspath(outputFilePath)
                self.current_image_directory = os.path.dirname(outputFilePath)

                self.statusBar().showMessage(
                    "Exporting output image...", self.statusbar_msg_display_time
                )

                # Get used image type from filter
                imgType = None
                if usedFilter == "JPEG (*.jpg *.jpeg)":
                    imgType = "JPG"
                elif usedFilter == "PNG (*.png)":
                    imgType = "PNG"
                elif usedFilter == "TIFF (*.tiff *.tif)":
                    imgType = "TIFF"

                ImageSavingDialog.createDialog(
                    self.LaplacianAlgorithm.output_image, imgType, outputFilePath
                )

        else:
            # Display Error message
            msg = qtw.QMessageBox(self)
            msg.setStandardButtons(qtw.QMessageBox.Ok)
            msg.setIcon(qtw.QMessageBox.Critical)
            msg.setWindowTitle("Export failed")
            msg.setText("Failed to export!\nPlease load images first.\n")
            msg.show()

    # Clear all loaded images
    def clear_all_images(self):
        if len(self.LaplacianAlgorithm.image_paths) > 0:
            # Ask confirmation (if there are loaded images)
            reply = qtw.QMessageBox.question(
                self,
                "Clear images?",
                "Are you sure you want to clear all loaded images? Output image(s) will be cleared to!",
                qtw.QMessageBox.Cancel,
                qtw.QMessageBox.Ok,
            )
            if reply == qtw.QMessageBox.Ok:
                self.statusBar().showMessage(
                    "Clearing images...", self.statusbar_msg_display_time
                )
                # Clear loaded and processed images from list
                settings.globalVars["LoadedImagePaths"] = []
                self.centralWidget().set_loaded_images(
                    settings.globalVars["LoadedImagePaths"]
                )
                self.LaplacianAlgorithm.update_image_paths(
                    settings.globalVars["LoadedImagePaths"]
                )
                self.centralWidget().add_processed_image(None)
                return True
            else:
                return False
        else:
            # No images were originally loaded
            return True

    # Update loaded image files
    def set_new_loaded_image_files(self, new_loaded_images):
        if len(new_loaded_images) > 0:
            if self.clear_all_images() == False:
                return

            # TODO Test
            # TODO: Check if same format?
            # Check if valid format; discard unsupported formats + show warning saying what images were discarded
            supportedformats = []
            for ext in settings["SupportedReadFormats"]:
                supportedformats.append("." + ext)
            validPaths = []
            invalidPaths = []
            for path in new_loaded_images:
                for ext in settings["SupportedReadFormats"]:
                    if path.endswith(supportedformats):
                        validPaths.append(path)
                    else:
                        invalidPaths.append(path)

            if len(invalidPaths) > 0:
                # Display Error message
                msg = qtw.QMessageBox(self)
                msg.setStandardButtons(qtw.QMessageBox.Ok)
                msg.setIcon(qtw.QMessageBox.Critical)
                msg.setWindowTitle("Failed to load")
                msg.setText(
                    "Failed to load certain images.\nThey have automatically been excluded.\nPlease ensure they use a supported format.\n"
                )
                text = ""
                for path in invalidPaths:
                    text += "\n" + path + ";"
                msg.setDetailedText(text)
                msg.show()

            self.statusBar().showMessage(
                "Loading images...", self.statusbar_msg_display_time
            )
            self.current_image_directory = os.path.dirname(validPaths[0])
            self.centralWidget().set_loaded_images(validPaths)
            self.LaplacianAlgorithm.update_image_paths(validPaths)
            settings.globalVars["LoadedImagePaths"] = validPaths

    # Shutdown all currently running processes, cleanup and close window
    # TODO: Shutdown currently running processes
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
        if len(settings.globalVars["LoadedImagePaths"]) == 0:
            # Display Error message
            msg = qtw.QMessageBox(self)
            msg.setStandardButtons(qtw.QMessageBox.Ok)
            msg.setIcon(qtw.QMessageBox.Critical)
            msg.setWindowTitle("Stacking failed")
            msg.setText("Failed to stack images.\nPlease load images first.\n")
            msg.show()
            return

        self.statusBar().showMessage(
            "Started aligning & stacking images...", self.statusbar_msg_display_time
        )
        self.LaplacianAlgorithm.align_and_stack_images()

    def stack_loaded_images(self):
        if len(settings.globalVars["LoadedImagePaths"]) == 0:
            # Display Error message
            msg = qtw.QMessageBox(self)
            msg.setStandardButtons(qtw.QMessageBox.Ok)
            msg.setIcon(qtw.QMessageBox.Critical)
            msg.setWindowTitle("Stacking failed")
            msg.setText("Failed to stack images.\nPlease load images first.\n")
            msg.show()
            return

        self.statusBar().showMessage(
            "Started stacking images...", self.statusbar_msg_display_time
        )

        def finished_inter_task(result_list):
            task_key, num_processed, num_to_process_total, time_taken = result_list
            percentage_finished = num_processed / num_to_process_total * 100

            # Compute and set new progressbar value
            new_progressbar_value = (
                self.TimeRemainingHandler.calculate_progressbar_value(
                    task_key, percentage_finished
                )
            )
            self.progress_widget.progressbar.setValue(new_progressbar_value)

            # Set new statusbar text
            self.progress_widget.progress_label.setText(
                self.TimeRemainingHandler.calculate_time_remaining(
                    task_key,
                    1 / num_to_process_total * 100,
                    100 - percentage_finished,
                    time_taken,
                )
            )

        worker = QThreading.Worker(self.LaplacianAlgorithm.stack_images)
        worker.signals.finished.connect(self.finished_stack)
        worker.signals.finished_inter_task.connect(finished_inter_task)

        # Execute
        self.threadpool.start(worker)
        self.progress_widget.setVisible(True)

    # Handle stack finish
    def finished_stack(self):  # , data_dictionary
        # Display stack info dialog
        # TODO: Properly implement
        # StackFinishedDialog.Message()

        # Reset progressbar and add selectable stack result
        self.progress_widget.reset_and_hide()
        self.centralWidget().add_processed_image(self.LaplacianAlgorithm.output_image)

        # Clear TimeRemaining cache
        self.TimeRemainingHandler.clear_cache()

    """
        Overridden signals
    """
    # Display dialog to confirm if user wants to quit
    def closeEvent(self, event):
        # TODO: Only ask confirmation if any unsaved progress
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
