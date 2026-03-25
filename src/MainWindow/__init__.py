"""
    Script that houses the MainWindow class.
    It is the "root display".
"""
import os
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
from src.config import AlgorithmConfig

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

        # Session state: track whether the stacked output has been exported
        self._output_exported = False

        self.statusbar_msg_display_time = 2000  # (ms)
        self.supportedReadFormats = []
        for ext in settings.globalVars["SupportedImageReadFormats"]:
            self.supportedReadFormats.append("." + str.lower(ext))
        for ext in settings.globalVars["SupportedRAWFormats"]:
            self.supportedReadFormats.append("." + str.lower(ext))

        self.setWindowTitle("chimpstackr")
        geometry = self.screen().availableGeometry()
        self.setMinimumSize(int(geometry.width() * 0.6), int(geometry.height() * 0.6))

        self.SettingsWidget = SettingsWidget.SettingsPanel()
        self.SettingsWidget.setVisible(False)

        # Setup algorithm API (before actions so menu can reference it)
        self.algorithm_config = AlgorithmConfig()
        self.LaplacianAlgorithm = algorithm_API.LaplacianPyramid(config=self.algorithm_config)
        self.TimeRemainingHandler = TimeRemainingHandler.TimeRemainingHandler()
        self.threadpool = qtc.QThreadPool()

        # Setup actions
        qt_actions_setup.setup_actions()

        # Main content + settings panel side by side
        main_content = MainLayout.CenterWidget()
        content_wrapper = qtw.QWidget()
        content_layout = qtw.QHBoxLayout(content_wrapper)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(main_content)
        content_layout.addWidget(self.SettingsWidget)
        self.setCentralWidget(content_wrapper)
        self._main_content = main_content

        # Image info label in status bar
        self.image_info_label = qtw.QLabel("No image loaded")
        self.image_info_label.setStyleSheet("color: #999999; padding: 0 8px; font-size: 11px;")
        self.statusBar().addWidget(self.image_info_label)

        # Permanent progressbar inside statusbar
        self.progress_widget = ProgressBar.ProgressBar()
        self.statusBar().addPermanentWidget(self.progress_widget)

        # (algorithm, timer, threadpool already initialized above)

    @property
    def has_unsaved_work(self):
        """True only if there's an unexported stacked result."""
        return (
            self.LaplacianAlgorithm.output_image is not None
            and not self._output_exported
        )

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

                imgType = None
                if usedFilter == "JPEG (*.jpg *.jpeg)":
                    imgType = "JPG"
                elif usedFilter == "PNG (*.png)":
                    imgType = "PNG"
                elif usedFilter == "TIFF (*.tiff *.tif)":
                    imgType = "TIFF"

                if not os.path.splitext(outputFilePath)[1]:
                    outputFilePath = outputFilePath + "." + imgType.lower()

                ImageSavingDialog.createDialog(
                    self.LaplacianAlgorithm.output_image, imgType, outputFilePath
                )
                self._output_exported = True
        else:
            msg = qtw.QMessageBox(self)
            msg.setStandardButtons(qtw.QMessageBox.Ok)
            msg.setIcon(qtw.QMessageBox.Critical)
            msg.setWindowTitle("Export failed")
            msg.setText("Failed to export!\nPlease load images first.\n")
            msg.show()

    def batch_export_output_image(self):
        """Export output image in multiple formats at once."""
        if self.LaplacianAlgorithm.output_image is None:
            msg = qtw.QMessageBox(self)
            msg.setStandardButtons(qtw.QMessageBox.Ok)
            msg.setIcon(qtw.QMessageBox.Critical)
            msg.setWindowTitle("Export failed")
            msg.setText("Failed to export!\nPlease stack images first.\n")
            msg.show()
            return

        directory = qtw.QFileDialog.getExistingDirectory(
            self,
            "Select output directory for batch export",
            self.current_image_directory,
            options=qtw.QFileDialog.DontUseNativeDialog,
        )
        if directory:
            import numpy as np
            image = np.clip(np.around(self.LaplacianAlgorithm.output_image), 0, 255).astype(np.uint8)
            import cv2
            base = os.path.join(directory, "stacked_output")
            exported = []
            try:
                cv2.imwrite(base + ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                exported.append(base + ".jpg")
                cv2.imwrite(base + ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 4])
                exported.append(base + ".png")
                cv2.imwrite(base + ".tif", image)
                exported.append(base + ".tif")

                msg = qtw.QMessageBox(self)
                msg.setStandardButtons(qtw.QMessageBox.Ok)
                msg.setIcon(qtw.QMessageBox.Information)
                msg.setWindowTitle("Batch export success")
                msg.setText(f"Exported {len(exported)} files to:\n{directory}")
                msg.show()
            except Exception as e:
                msg = qtw.QMessageBox(self)
                msg.setStandardButtons(qtw.QMessageBox.Ok)
                msg.setIcon(qtw.QMessageBox.Critical)
                msg.setWindowTitle("Batch export failed")
                msg.setText(f"Error during batch export:\n{e}")
                msg.show()

    # Clear all loaded images
    def clear_all_images(self):
        if self.is_stacking:
            self.statusBar().showMessage("Cannot clear while stacking", self.statusbar_msg_display_time)
            return False
        if len(self.LaplacianAlgorithm.image_paths) > 0:
            # Ask confirmation (if there are loaded images)
            reply = qtw.QMessageBox.question(
                self,
                "Clear images?",
                "Are you sure you want to clear all loaded images? Output images will be cleared too!",
                qtw.QMessageBox.Cancel,
                qtw.QMessageBox.Ok,
            )
            if reply == qtw.QMessageBox.Ok:
                self.statusBar().showMessage(
                    "Clearing images...", self.statusbar_msg_display_time
                )
                # Clear loaded and processed images from list
                settings.globalVars["LoadedImagePaths"] = []
                self._main_content.set_loaded_images(
                    settings.globalVars["LoadedImagePaths"]
                )
                self.LaplacianAlgorithm.update_image_paths(
                    settings.globalVars["LoadedImagePaths"]
                )
                self._main_content.add_processed_image(None)
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
        if self.is_stacking:
            self.statusBar().showMessage("Cannot load images while stacking", self.statusbar_msg_display_time)
            return
        if len(new_loaded_images) > 0:
            if self.clear_all_images() == False:
                return

            # Filter out hidden/system files silently
            IGNORED_FILES = {
                "thumbs.db", ".ds_store", "desktop.ini", ".thumbs",
                ".picasa.ini", ".bridgesort", ".bridgelabelsandratings",
            }
            filtered = []
            for path in new_loaded_images:
                basename = os.path.basename(path).lower()
                # Skip hidden files (start with .)
                if basename.startswith("."):
                    continue
                # Skip known system files
                if basename in IGNORED_FILES:
                    continue
                filtered.append(path)

            # Check supported formats
            validPaths = []
            skipped = 0
            for path in filtered:
                if str.lower(path).endswith(tuple(self.supportedReadFormats)):
                    validPaths.append(path)
                else:
                    skipped += 1

            # Show skip count in status bar (no popup)
            if skipped > 0:
                self.statusBar().showMessage(
                    f"Skipped {skipped} unsupported file{'s' if skipped > 1 else ''}",
                    self.statusbar_msg_display_time * 2,
                )
            else:
                self.statusBar().showMessage(
                    "Loading images...", self.statusbar_msg_display_time
                )
            if len(validPaths) > 0:
                self.current_image_directory = os.path.dirname(validPaths[0])
                self._main_content.set_loaded_images(validPaths)
                self.LaplacianAlgorithm.update_image_paths(validPaths)
                settings.globalVars["LoadedImagePaths"] = validPaths
                self._output_exported = False

                # Auto-detect optimal parameters from first image
                self.SettingsWidget._auto_detect_params()

    @property
    def is_stacking(self):
        """True if a stacking operation is currently running or paused."""
        return getattr(self, '_stacking_active', False)

    def _sync_algorithm_config(self):
        """Sync algorithm config from settings widget before stacking."""
        new_config = self.SettingsWidget.get_algorithm_config()
        self.LaplacianAlgorithm.config = new_config

    def _start_stacking(self, method_name):
        """Common logic for starting align+stack or stack-only. Returns True if started."""
        if len(settings.globalVars["LoadedImagePaths"]) == 0:
            msg = qtw.QMessageBox(self)
            msg.setStandardButtons(qtw.QMessageBox.Ok)
            msg.setIcon(qtw.QMessageBox.Critical)
            msg.setWindowTitle("Stacking failed")
            msg.setText("Failed to stack images.\nPlease load images first.\n")
            msg.show()
            return False

        self._sync_algorithm_config()
        self._stacking_active = True
        self.SettingsWidget.setEnabled(False)  # Lock settings during stacking

        self.statusBar().showMessage(
            f"Started {method_name}...", self.statusbar_msg_display_time
        )

        def finished_inter_task(result_list):
            try:
                task_key, num_processed, num_to_process_total, time_taken = result_list
                if task_key == "finished_image" and num_to_process_total > 0:
                    percentage_finished = num_processed / num_to_process_total * 100
                    self.progress_widget.update_value(
                        percentage_finished,
                        self.TimeRemainingHandler.calculate_time_remaining(
                            1 / num_to_process_total * 100,
                            100 - percentage_finished,
                            time_taken,
                        ),
                    )
            except (AttributeError, RuntimeError, ZeroDivisionError):
                pass  # Widget may be destroyed during shutdown

        fn = getattr(self.LaplacianAlgorithm, method_name)
        worker = QThreading.Worker(fn)
        worker.signals.finished.connect(self.finished_stack)
        worker.signals.finished_inter_task.connect(finished_inter_task)

        self.threadpool.start(worker)
        self.progress_widget.setVisible(True)
        return True

    def align_and_stack_loaded_images(self):
        return self._start_stacking("align_and_stack_images")

    def stack_loaded_images(self):
        return self._start_stacking("stack_images")

    def cancel_stacking(self):
        """Cancel the currently running stacking operation."""
        self.LaplacianAlgorithm.cancel()
        self._stacking_active = False
        self.SettingsWidget.setEnabled(True)  # Unlock settings
        self.statusBar().showMessage("Cancelling...", self.statusbar_msg_display_time)

    def auto_crop_result(self):
        """Manually trigger auto-crop on the current output."""
        if self.LaplacianAlgorithm.output_image is None:
            self.statusBar().showMessage("No output to crop", self.statusbar_msg_display_time)
            return
        bounds = self.LaplacianAlgorithm.auto_crop_output()
        if bounds:
            top, bottom, left, right = bounds
            self._main_content.add_processed_image(self.LaplacianAlgorithm.output_image)
            self.statusBar().showMessage(
                f"Cropped: {top}px top, {bottom}px bottom, {left}px left, {right}px right",
                self.statusbar_msg_display_time
            )
        else:
            self.statusBar().showMessage("No alignment shifts to crop", self.statusbar_msg_display_time)

    def toggle_settings_panel(self):
        """Toggle the settings sidebar visibility."""
        self.SettingsWidget.setVisible(not self.SettingsWidget.isVisible())

    def export_comparison(self):
        """Export the current comparison view as an image."""
        path, _ = qtw.QFileDialog.getSaveFileName(
            self, "Export comparison", self.current_image_directory,
            "PNG (*.png);; JPEG (*.jpg)",
        )
        if path:
            self._main_content.ComparisonViewer.export_comparison(path)
            self.statusBar().showMessage(f"Comparison exported to {path}", self.statusbar_msg_display_time)

    def finished_stack(self):
        self._stacking_active = False
        self.SettingsWidget.setEnabled(True)  # Unlock settings
        self.progress_widget.update_value()

        run_btn = settings.globalVars.get("RunButton")
        if run_btn:
            run_btn.on_finished()

        # If cancelled or no output produced, discard everything
        if self.LaplacianAlgorithm.output_image is None:
            self.LaplacianAlgorithm.Algorithm.alignment_shifts = []
            self.statusBar().showMessage("Stacking cancelled", self.statusbar_msg_display_time)
            return

        # Auto-crop black edges if enabled in settings
        auto_crop = bool(int(settings.globalVars["QSettings"].value("algorithm/auto_crop") or 1))
        if auto_crop:
            bounds = self.LaplacianAlgorithm.auto_crop_output()
            if bounds:
                top, bottom, left, right = bounds
                self.statusBar().showMessage(
                    f"Auto-cropped: {top}px top, {bottom}px bottom, {left}px left, {right}px right",
                    self.statusbar_msg_display_time
                )

        self._main_content.add_processed_image(self.LaplacianAlgorithm.output_image)
        self._output_exported = False

    def closeEvent(self, event):
        if not self.has_unsaved_work:
            self.LaplacianAlgorithm.cancel()
            event.accept()
            settings.globalVars["MainApplication"].closeAllWindows()
            return

        # Only case: unexported stacked image
        reply = qtw.QMessageBox.warning(
            self,
            "Unexported result",
            "You have a stacked image that hasn't been exported.\n"
            "Close anyway and lose the result?",
            qtw.QMessageBox.Discard | qtw.QMessageBox.Cancel,
            qtw.QMessageBox.Cancel,
        )
        if reply == qtw.QMessageBox.Discard:
            self.LaplacianAlgorithm.cancel()
            event.accept()
            settings.globalVars["MainApplication"].closeAllWindows()
        else:
            event.ignore()
