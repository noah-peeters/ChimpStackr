"""
    Sets up QActions, toolbar, and menus for the main window.
"""
import os
import PySide6.QtGui as qtg
import PySide6.QtWidgets as qtw
import PySide6.QtCore as qtc

import src.settings as settings
from src.MainWindow.icons import (
    icon_open, icon_save, icon_play, icon_pause, icon_stop,
    icon_settings, icon_zoom_in, icon_zoom_out, icon_fit, icon_stack,
)


class AboutAppWidget(qtw.QMessageBox):
    def __init__(self):
        super().__init__()
        self.setStandardButtons(qtw.QMessageBox.Ok)
        self.setIcon(qtw.QMessageBox.Information)
        self.setWindowTitle("About")

        import PySide6, src, platform
        self.setText(
            f"ChimpStackr {src.__version__}\n"
            f"Qt {PySide6.__version__}\n"
            f"Python {platform.python_version()}\n"
            f"OS: {platform.platform()}\n"
        )
        copy_btn = self.addButton("Copy", qtw.QMessageBox.RejectRole)
        copy_btn.clicked.connect(lambda: (
            qtw.QApplication.clipboard().clear(),
            qtw.QApplication.clipboard().setText(self.text()),
        ))


class RunButton(qtw.QToolButton):
    """
    Adaptive run button: shows Play when idle, Pause+Stop when running.
    Click toggles between run/pause. A small dropdown gives Stop.
    """
    def __init__(self, mainWindow):
        super().__init__()
        self.mainWindow = mainWindow
        self._state = "idle"  # idle, running, paused

        self.setPopupMode(qtw.QToolButton.MenuButtonPopup)
        self.setToolButtonStyle(qtc.Qt.ToolButtonIconOnly)

        self.menu = qtw.QMenu(self)

        self.align_stack_action = qtg.QAction(icon_play(), "Align && Stack", self)
        self.align_stack_action.triggered.connect(self._on_align_stack)

        self.stack_only_action = qtg.QAction(icon_stack(), "Stack Only", self)
        self.stack_only_action.triggered.connect(self._on_stack_only)

        self.stop_action = qtg.QAction(icon_stop(), "Stop", self)
        self.stop_action.triggered.connect(self._on_stop)

        self._update_state("idle")
        self.clicked.connect(self._on_click)

    def _update_state(self, state):
        self._state = state
        self.menu.clear()

        if state == "idle":
            self.setIcon(icon_play())
            self.setToolTip("Align & Stack (click) or choose mode")
            self.menu.addAction(self.align_stack_action)
            self.menu.addAction(self.stack_only_action)
        elif state == "running":
            self.setIcon(icon_pause())
            self.setToolTip("Pause processing")
            self.menu.addAction(self.stop_action)
        elif state == "paused":
            self.setIcon(icon_play())
            self.setToolTip("Resume processing")
            self.menu.addAction(self.stop_action)

        self.setMenu(self.menu)

    def _on_click(self):
        if self._state == "idle":
            self._on_align_stack()
        elif self._state == "running":
            self._on_pause()
        elif self._state == "paused":
            self._on_resume()

    def _on_align_stack(self):
        if self.mainWindow.align_and_stack_loaded_images():
            self._update_state("running")

    def _on_stack_only(self):
        if self.mainWindow.stack_loaded_images():
            self._update_state("running")

    def _on_pause(self):
        self.mainWindow.LaplacianAlgorithm.pause()
        self._update_state("paused")
        self.mainWindow.statusBar().showMessage("Paused", 2000)

    def _on_resume(self):
        self.mainWindow.LaplacianAlgorithm.resume()
        self._update_state("running")
        self.mainWindow.statusBar().showMessage("Resumed", 2000)

    def _on_stop(self):
        self.mainWindow.cancel_stacking()
        self._update_state("idle")

    def on_finished(self):
        """Called when stacking completes."""
        self._update_state("idle")


def setup_actions():
    mainWindow = settings.globalVars["MainWindow"]

    def load_images_from_file():
        fileNames, _ = qtw.QFileDialog.getOpenFileNames(
            mainWindow,
            "Select images to load.",
            mainWindow.current_image_directory,
        )
        mainWindow.set_new_loaded_image_files(fileNames)

    menubar = mainWindow.menuBar()

    # ===== FILE MENU =====
    file_menu = menubar.addMenu("&File")

    load_images = qtg.QAction(icon_open(), "&Load images", mainWindow)
    load_images.setShortcut("Ctrl+O")
    load_images.triggered.connect(load_images_from_file)
    file_menu.addAction(load_images)

    file_menu.addSeparator()

    export_image = qtg.QAction(icon_save(), "E&xport image", mainWindow)
    export_image.setShortcut("Ctrl+E")
    export_image.triggered.connect(mainWindow.export_output_image)
    file_menu.addAction(export_image)

    batch_export = qtg.QAction("&Batch export (JPG+PNG+TIFF)", mainWindow)
    batch_export.setShortcut("Ctrl+Shift+E")
    batch_export.triggered.connect(mainWindow.batch_export_output_image)
    file_menu.addAction(batch_export)

    export_compare = qtg.QAction("Export &comparison view", mainWindow)
    export_compare.triggered.connect(mainWindow.export_comparison)
    file_menu.addAction(export_compare)

    file_menu.addSeparator()

    exit_action = qtg.QAction("&Exit", mainWindow)
    exit_action.setShortcut(qtg.QKeySequence("Ctrl+W"))
    exit_action.triggered.connect(mainWindow.close)
    file_menu.addAction(exit_action)

    # ===== PROCESSING MENU =====
    processing_menu = menubar.addMenu("&Processing")

    align_and_stack_action = qtg.QAction(icon_play(), "&Align and stack", mainWindow)
    align_and_stack_action.setShortcut("Ctrl+A")
    align_and_stack_action.triggered.connect(mainWindow.align_and_stack_loaded_images)
    processing_menu.addAction(align_and_stack_action)

    stack_action = qtg.QAction(icon_stack(), "&Stack only", mainWindow)
    stack_action.setShortcut("Ctrl+Alt+C")
    stack_action.triggered.connect(mainWindow.stack_loaded_images)
    processing_menu.addAction(stack_action)

    processing_menu.addSeparator()

    pause_action = qtg.QAction(icon_pause(), "&Pause", mainWindow)
    pause_action.triggered.connect(mainWindow.LaplacianAlgorithm.pause)
    processing_menu.addAction(pause_action)

    resume_action = qtg.QAction(icon_play(), "&Resume", mainWindow)
    resume_action.triggered.connect(mainWindow.LaplacianAlgorithm.resume)
    processing_menu.addAction(resume_action)

    cancel_action = qtg.QAction(icon_stop(), "&Stop", mainWindow)
    cancel_action.setShortcut("Escape")
    cancel_action.triggered.connect(mainWindow.cancel_stacking)
    processing_menu.addAction(cancel_action)

    processing_menu.addSeparator()

    autocrop_action = qtg.QAction("Auto-&crop edges", mainWindow)
    autocrop_action.setStatusTip("Crop black edges from alignment shifts")
    autocrop_action.triggered.connect(mainWindow.auto_crop_result)
    processing_menu.addAction(autocrop_action)

    # ===== EDIT MENU =====
    edit = menubar.addMenu("&Edit")

    toggle_settings = qtg.QAction(icon_settings(), "&Settings", mainWindow)
    toggle_settings.setShortcut("Ctrl+,")
    toggle_settings.triggered.connect(mainWindow.toggle_settings_panel)
    edit.addAction(toggle_settings)

    # ===== HELP MENU =====
    help_menu = menubar.addMenu("&Help")
    aboutApp = AboutAppWidget()
    about = qtg.QAction("&About", mainWindow)
    about.triggered.connect(lambda: aboutApp.exec())
    help_menu.addAction(about)

    # ===== TOOLBAR =====
    toolbar = mainWindow.addToolBar("Main")
    toolbar.setMovable(False)
    toolbar.setIconSize(qtc.QSize(18, 18))
    toolbar.setToolButtonStyle(qtc.Qt.ToolButtonIconOnly)

    # --- Left: File actions ---
    toolbar.addAction(load_images)
    toolbar.addAction(export_image)

    toolbar.addSeparator()

    # --- Center: Run button (adaptive play/pause/stop) ---
    run_button = RunButton(mainWindow)
    run_button.setIconSize(qtc.QSize(18, 18))
    settings.globalVars["RunButton"] = run_button
    toolbar.addWidget(run_button)

    # --- Spacer ---
    spacer = qtw.QWidget()
    spacer.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Preferred)
    toolbar.addWidget(spacer)

    # --- Right: View controls ---
    def get_viewer():
        return mainWindow._main_content.ImageViewer

    fit_btn = qtw.QToolButton()
    fit_btn.setIcon(icon_fit())
    fit_btn.setToolTip("Fit to view")
    fit_btn.clicked.connect(lambda: get_viewer().fitInView())
    toolbar.addWidget(fit_btn)

    zoom_out_btn = qtw.QToolButton()
    zoom_out_btn.setIcon(icon_zoom_out())
    zoom_out_btn.setToolTip("Zoom out")
    zoom_out_btn.clicked.connect(lambda: get_viewer().zoom_out())
    toolbar.addWidget(zoom_out_btn)

    zoom_label = qtw.QLabel(" 0% ")
    zoom_label.setStyleSheet("color: #999999; font-size: 11px; background: transparent; min-width: 36px;")
    zoom_label.setAlignment(qtc.Qt.AlignCenter)
    settings.globalVars["ZoomLabel"] = zoom_label
    toolbar.addWidget(zoom_label)

    zoom_in_btn = qtw.QToolButton()
    zoom_in_btn.setIcon(icon_zoom_in())
    zoom_in_btn.setToolTip("Zoom in")
    zoom_in_btn.clicked.connect(lambda: get_viewer().zoom_in())
    toolbar.addWidget(zoom_in_btn)

    toolbar.addSeparator()

    # --- Far right: Settings ---
    settings_btn = qtw.QToolButton()
    settings_btn.setIcon(icon_settings())
    settings_btn.setToolTip("Settings (Cmd+,)")
    settings_btn.clicked.connect(mainWindow.toggle_settings_panel)
    toolbar.addWidget(settings_btn)
