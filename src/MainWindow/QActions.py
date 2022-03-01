"""
    Script that sets up QActions for mainWindow QMainWindow.
"""
import PySide6.QtGui as qtg
import PySide6.QtWidgets as qtw

import src.settings as settings


class AboutAppWidget(qtw.QMessageBox):
    def __init__(self):
        super().__init__()

        self.setStandardButtons(qtw.QMessageBox.Ok)
        self.setIcon(qtw.QMessageBox.Information)
        self.setWindowTitle("About")

        import PySide6
        import src
        import platform

        self.setText(
            "ChimpStackr version: {}\n".format(src.__version__)
            + "Qt version: {}\n".format(PySide6.__version__)
            + "OS: {}\n".format(platform.platform())
            + "Python version: {}\n".format(platform.python_version())
        )
        copyButton = self.addButton("Copy", qtw.QMessageBox.RejectRole)
        copyButton.clicked.connect(self.copy_text)

    # Copy text to clipboard
    def copy_text(self):
        cb = qtw.QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        cb.setText(self.text(), mode=cb.Clipboard)


def setup_actions():
    mainWindow = settings.globalVars["MainWindow"]
    # Load images action; user selects images from a QFileDialog
    def load_images_from_file():
        new_loaded_images, _ = qtw.QFileDialog.getOpenFileNames(
            mainWindow, "Select images to load.", mainWindow.current_image_directory
        )
        mainWindow.set_new_loaded_image_files(new_loaded_images)

    menubar = mainWindow.menuBar()

    """ File menu/toolbar """
    file_menu = menubar.addMenu("&File")

    load_images = qtg.QAction("&Load images", mainWindow)
    load_images.setShortcut("Ctrl+D")
    load_images.setStatusTip("Load images from disk.")
    load_images.triggered.connect(load_images_from_file)
    file_menu.addAction(load_images)

    clear_images = qtg.QAction("&Clear images", mainWindow)
    clear_images.setShortcut("Ctrl+F")
    clear_images.setStatusTip("Clear all loaded images.")
    clear_images.triggered.connect(mainWindow.clear_all_images)
    file_menu.addAction(clear_images)

    export_image = qtg.QAction("E&xport image", mainWindow)
    export_image.setShortcut("Ctrl+E")
    export_image.setStatusTip("Export output image.")
    export_image.triggered.connect(mainWindow.export_output_image)
    file_menu.addAction(export_image)

    save_file = qtg.QAction("&Save project", mainWindow)
    save_file.setShortcut(qtg.QKeySequence("Ctrl+S"))
    save_file.setStatusTip("Save a project file to disk.")
    save_file.triggered.connect(mainWindow.save_project_to_file)

    exit = qtg.QAction("&Exit", mainWindow)
    exit.setShortcut(qtg.QKeySequence("Ctrl+W"))
    exit.setStatusTip("Exit from application. You might lose unsaved work!")
    exit.triggered.connect(mainWindow.shutdown_application)
    file_menu.addAction(exit)

    """ Processing menu/toolbar """
    processing_menu = menubar.addMenu("&Processing")

    # align = qtg.QAction("&Align images", mainWindow)
    # align.setShortcut("Ctrl+A")
    # align.setStatusTip("Align loaded images.")
    # align.triggered.connect(mainWindow.load_images_from_file)
    # processing.addAction(align)

    align_and_stack = qtg.QAction("&Align and stack images", mainWindow)
    align_and_stack.setShortcut("Ctrl+A")
    align_and_stack.setStatusTip("Align and stack loaded images.")
    align_and_stack.triggered.connect(mainWindow.align_and_stack_loaded_images)
    processing_menu.addAction(align_and_stack)

    stack = qtg.QAction("&Stack images", mainWindow)
    stack.setShortcut("Ctrl+Alt+C")
    stack.setStatusTip("Stack loaded images.")
    stack.triggered.connect(mainWindow.stack_loaded_images)
    processing_menu.addAction(stack)

    """ Edit menu """
    edit = menubar.addMenu("&Edit")

    edit_settings = qtg.QAction("&Settings", mainWindow)
    edit_settings.setStatusTip("Edit application settings.")
    edit_settings.triggered.connect(mainWindow.SettingsWidget.show)
    edit.addAction(edit_settings)

    """ Help menu """
    help = menubar.addMenu("&Help")

    aboutApp = AboutAppWidget()
    about = qtg.QAction("&About", mainWindow)
    about.setStatusTip("ChimpStackr")
    about.triggered.connect(lambda: aboutApp.exec())
    help.addAction(about)
