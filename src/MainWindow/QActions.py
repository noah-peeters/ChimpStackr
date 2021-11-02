"""
    Script that sets up QActions for parent QMainWindow.
"""
import PySide6.QtGui as qtg
import PySide6.QtWidgets as qtw

# Setup actions for parent
def setup_actions(parent):
    menubar = parent.menuBar()

    """ File menu/toolbar """
    file_menu = menubar.addMenu("&File")
    file_toolbar = parent.addToolBar("File")

    load_images = qtg.QAction("&Load images", parent)
    load_images.setShortcut("Ctrl+D")
    load_images.setStatusTip("Load images from disk.")
    load_images.triggered.connect(parent.load_images_from_file)
    file_menu.addAction(load_images)
    file_toolbar.addAction(load_images)

    clear_images = qtg.QAction("&Clear images", parent)
    clear_images.setShortcut("Ctrl+F")
    clear_images.setStatusTip("Clear all loaded images.")
    clear_images.triggered.connect(parent.clear_all_images)
    file_menu.addAction(clear_images)
    file_toolbar.addAction(clear_images)

    export_image = qtg.QAction("E&xport image", parent)
    export_image.setShortcut("Ctrl+E")
    export_image.setStatusTip("Export output image.")
    export_image.triggered.connect(parent.export_output_image)
    file_menu.addAction(export_image)
    file_toolbar.addAction(export_image)

    save_file = qtg.QAction("&Save project", parent)
    save_file.setShortcut(qtg.QKeySequence("Ctrl+S"))
    save_file.setStatusTip("Save a project file to disk.")
    save_file.triggered.connect(parent.save_project_to_file)
    file_menu.addAction(save_file)

    exit = qtg.QAction("&Exit", parent)
    exit.setShortcut(qtg.QKeySequence("Ctrl+W"))
    exit.setStatusTip("Exit from application. You might lose unsaved work!")
    exit.triggered.connect(parent.shutdown_application)
    file_menu.addAction(exit)

    """ Processing menu/toolbar """
    processing_menu = menubar.addMenu("&Processing")
    processing_toolbar = parent.addToolBar("Processing")

    # align = qtg.QAction("&Align images", parent)
    # align.setShortcut("Ctrl+A")
    # align.setStatusTip("Align loaded images.")
    # align.triggered.connect(parent.load_images_from_file)
    # processing.addAction(align)

    align_and_stack = qtg.QAction("&Align and stack images", parent)
    align_and_stack.setShortcut("Ctrl+A")
    align_and_stack.setStatusTip("Align and stack loaded images.")
    align_and_stack.triggered.connect(parent.align_and_stack_loaded_images)
    processing_menu.addAction(align_and_stack)
    processing_toolbar.addAction(align_and_stack)

    stack = qtg.QAction("&Stack images", parent)
    stack.setShortcut("Ctrl+Alt+C")
    stack.setStatusTip("Stack loaded images.")
    stack.triggered.connect(parent.stack_loaded_images)
    processing_menu.addAction(stack)
    processing_toolbar.addAction(stack)

    """ Help menu """
    help = menubar.addMenu("&Help")

    qt = qtg.QAction("About &Qt", parent)
    qt.setStatusTip("Information on Qt, the UI framework.")
    qt.triggered.connect(qtw.QApplication.aboutQt)
    help.addAction(qt)

