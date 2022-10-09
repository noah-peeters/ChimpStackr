"""
    Script that houses the MainWindow class.
    It is the "root display".
"""
import os, sys
import PySide6.QtCore as qtc
import PySide6.QtWidgets as qtw

import src.settings as settings
import src.MainWindow.QActions as qt_actions_setup
import src.MainWindow.MainLayout as MainLayout
import src.MainWindow.Threading as QThreading
import src.MainWindow.ProgressBar as ProgressBar

# import src.MainWindow.StackSuccessDialog as StackFinishedDialog
import src.MainWindow.TimeRemainingHandler as TimeRemainingHandler
import src.MainWindow.ImageSavingDialog as ImageSavingDialog
import src.MainWindow.SettingsWidget as SettingsWidget

import src.algorithms.API as algorithm_API

if os.name == "nt":
    current_image_directory = os.path.expanduser("~")
else:
    # Probably running in snap --> get real home dir and not the snap's home dir
    import pwd

    current_image_directory = os.path.expanduser(f"~{pwd.getpwuid(os.geteuid())[0]}/")


class Window(qtw.QMainWindow):
    # Reference dir for image loading/export
    current_image_directory = current_image_directory

    def __init__(self):
        super().__init__()

        settings.globalVars["MainWindow"] = self
        settings.globalVars["LoadedImagePaths"] = []

        self.statusbar_msg_display_time = 2000  # (ms)
        self.supportedReadFormats = []
        for ext in settings.globalVars["SupportedImageReadFormats"]:
            self.supportedReadFormats.append("." + str.lower(ext))
        for ext in settings.globalVars["SupportedRAWFormats"]:
            self.supportedReadFormats.append("." + str.lower(ext))

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

        # Setup algorithm API
        # TODO: Allow user to change program settings
        self.LaplacianAlgorithm = algorithm_API.LaplacianPyramid(6, 8)
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
                options=qtw.QFileDialog.DontUseNativeDialog,
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

                # Attach extension as per selected filter, if not there
                if not os.path.splitext(outputFilePath)[1]:
                    outputFilePath = outputFilePath + "." + imgType.lower()

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
            # No images were originally loaded
            return True

    # Remove some images from loaded images list (used on right click action)
    def remove_some_images(self, paths_to_remove):
        if len(paths_to_remove) > 0:
            new_paths = []
            for old_path in settings.globalVars["LoadedImagePaths"]:
                # Only add to new list if not excluded
                if len([match for match in paths_to_remove if old_path in match]) <= 0:
                    new_paths.append(old_path)

            self.set_new_loaded_image_files(new_paths)

    # Update loaded image files
    def set_new_loaded_image_files(self, new_loaded_images):
        if len(new_loaded_images) > 0:
            if self.clear_all_images() == False:
                return

            # TODO: Check if same format?
            # Check if valid format; discard unsupported formats + show warning saying what images were discarded
            validPaths = []
            invalidPaths = []
            for path in new_loaded_images:
                if str.lower(path).endswith(tuple(self.supportedReadFormats)):
                    validPaths.append(path)
                else:
                    invalidPaths.append(path)

            if len(invalidPaths) > 0:
                # Display Error message
                msg = qtw.QMessageBox(self)
                msg.setStandardButtons(qtw.QMessageBox.Ok)
                msg.setIcon(qtw.QMessageBox.Critical)
                msg.setWindowTitle("Failed to load {} files!".format(len(invalidPaths)))
                msg.setText(
                    "Failed to load certain files.\nThey have automatically been excluded.\nPlease ensure a supported format is used.\n"
                )
                text = ""
                for path in invalidPaths:
                    text += path + "\n"
                msg.setDetailedText(text)
                msg.show()

            self.statusBar().showMessage(
                "Loading images...", self.statusbar_msg_display_time
            )
            if len(validPaths) > 0:
                self.current_image_directory = os.path.dirname(validPaths[0])
                self.centralWidget().set_loaded_images(validPaths)
                self.LaplacianAlgorithm.update_image_paths(validPaths)
                settings.globalVars["LoadedImagePaths"] = validPaths

    # Shutdown all currently running processes, cleanup and close window
    # TODO: Shutdown currently running processes
    def shutdown_application(self):
        self.close()
        sys.exit()  # TODO: Check if running processes are stopped

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

        def finished_inter_task(result_list):
            task_key, num_processed, num_to_process_total, time_taken = result_list
            if task_key == "finished_image":
                # Update progressbar slider and "time remaining" text
                percentage_finished = num_processed / num_to_process_total * 100
                self.progress_widget.update_value(
                    percentage_finished,
                    self.TimeRemainingHandler.calculate_time_remaining(
                        1 / num_to_process_total * 100,
                        100 - percentage_finished,
                        time_taken,
                    ),
                )

        worker = QThreading.Worker(self.LaplacianAlgorithm.align_and_stack_images)
        worker.signals.finished.connect(self.finished_stack)
        worker.signals.finished_inter_task.connect(finished_inter_task)

        # Execute
        self.threadpool.start(worker)
        self.progress_widget.setVisible(True)

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
            if task_key == "finished_image":
                # Update progressbar slider and "time remaining" text
                percentage_finished = num_processed / num_to_process_total * 100
                self.progress_widget.update_value(
                    percentage_finished,
                    self.TimeRemainingHandler.calculate_time_remaining(
                        1 / num_to_process_total * 100,
                        100 - percentage_finished,
                        time_taken,
                    ),
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

        # Reset progressbar and add selectable stack result image
        self.progress_widget.update_value()
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
