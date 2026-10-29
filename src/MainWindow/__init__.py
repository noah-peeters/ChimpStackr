"""
Main window: toolbar, frames sidebar, image viewer, inspector and the
activity bar that shows stacking progress.

Actions are enabled only when they can run (no "load images first" popups),
follow each platform's standard shortcuts and menu placement, and the window
layout is restored between sessions.
"""
import os
import time

import PySide6.QtCore as qtc
import PySide6.QtGui as qtg
import PySide6.QtWidgets as qtw

import src
import src.settings as settings
import src.ImageLoadingHandler as ImageLoadingHandler
import src.MainWindow.Threading as QThreading
import src.MainWindow.TimeRemainingHandler as TimeRemainingHandler
import src.MainWindow.ImageSavingDialog as Export
import src.MainWindow.SettingsWidget as SettingsWidget
import src.MainWindow.UpdateDialog as UpdateDialog
import src.MainWindow.theme as theme
from src.MainWindow.icons import icon
from src.MainWindow.MainLayout.ImageViewers import ImageViewer
from src.MainWindow.MainLayout.ImageWidgets import Sidebar, reveal_in_file_manager, file_manager_name
import src.algorithms.API as algorithm_API
from src.config import AlgorithmConfig
from src.utilities import int_string_sorting

IS_MAC = qtc.QSysInfo.productType() == "macos"
IGNORED_FILES = {"thumbs.db", ".ds_store", "desktop.ini", ".thumbs", ".picasa.ini",
                 ".bridgesort", ".bridgelabelsandratings"}
METHOD_TITLES = {k: t for k, t, _d in SettingsWidget.METHODS}
REPO_URL = "https://github.com/noah-peeters/ChimpStackr"


def _home_dir():
    if os.name == "nt":
        return os.path.expanduser("~")
    # Snap: the real home dir, not the snap's
    import pwd
    return os.path.expanduser(f"~{pwd.getpwuid(os.geteuid())[0]}/")


def _format_duration(seconds):
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60:02d} s"
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"


class ImageLoader(qtc.QRunnable):
    """Loads a full-resolution image off the UI thread."""

    class Signals(qtc.QObject):
        ready = qtc.Signal(str, object, object)   # path, image, tag

    def __init__(self, path, crop=None, tag=None):
        super().__init__()
        self.path, self.crop, self.tag = path, crop, tag
        self.signals = self.Signals()

    def run(self):
        img = None
        try:
            img = ImageLoadingHandler.ImageLoadingHandler().read_image_from_path(self.path)
            if img is not None and self.crop:
                top, bottom, left, right = self.crop
                h, w = img.shape[:2]
                img = img[top:h - bottom, left:w - right]
        except Exception:
            img = None
        self.signals.ready.emit(self.path, img, self.tag)


class Result:
    def __init__(self, image, method, alignment, frames, seconds, crop):
        self.image = image
        self.method = method
        self.alignment = alignment
        self.frames = frames
        self.seconds = seconds
        self.crop = crop            # (top, bottom, left, right) or None
        self.exported_to = None
        self.created = time.time()

    @property
    def title(self):
        return METHOD_TITLES.get(self.method, self.method)

    @property
    def subtitle(self):
        h, w = self.image.shape[:2]
        return f"{self.frames} frames · {w}×{h} · {_format_duration(self.seconds)}"


class Segmented(qtw.QWidget):
    """Small segmented control (e.g. View | Compare)."""
    changed = qtc.Signal(int)

    def __init__(self, items):
        super().__init__()
        self.setObjectName("Segmented")
        self.setAttribute(qtc.Qt.WA_StyledBackground, True)
        layout = qtw.QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        self.group = qtw.QButtonGroup(self)
        self.buttons = []
        for i, (name, text, tip) in enumerate(items):
            b = qtw.QPushButton(text)
            b.setIcon(icon(name))
            b.setIconSize(qtc.QSize(15, 15))
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setCursor(qtc.Qt.PointingHandCursor)
            self.group.addButton(b, i)
            layout.addWidget(b)
            self.buttons.append(b)
        self.buttons[0].setChecked(True)
        self.group.idToggled.connect(lambda i, on: on and self.changed.emit(i))

    def set_index(self, i):
        self.buttons[i].setChecked(True)

    def set_enabled_at(self, i, on):
        self.buttons[i].setEnabled(on)


class ActivityBar(qtw.QWidget):
    """Stacking progress: title, bar, time left, pause/resume and stop."""
    pause_clicked = qtc.Signal()
    stop_clicked = qtc.Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("ActivityBar")
        self.setAttribute(qtc.Qt.WA_StyledBackground, True)
        layout = qtw.QHBoxLayout(self)
        layout.setContentsMargins(14, 8, 10, 8)
        layout.setSpacing(12)
        self.title = qtw.QLabel("Stacking…")
        f = self.title.font()
        f.setWeight(qtg.QFont.DemiBold)
        self.title.setFont(f)
        self.bar = qtw.QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setAccessibleName("Stacking progress")
        self.detail = qtw.QLabel("")
        self.detail.setObjectName("Secondary")
        self.detail.setMinimumWidth(150)
        self.pause_btn = qtw.QPushButton("Pause")
        self.pause_btn.setIcon(icon("pause"))
        self.pause_btn.clicked.connect(self.pause_clicked)
        self.stop_btn = qtw.QPushButton("Stop")
        self.stop_btn.setObjectName("DangerButton")
        self.stop_btn.setIcon(icon("stop", "danger"))
        self.stop_btn.clicked.connect(self.stop_clicked)
        layout.addWidget(self.title)
        layout.addWidget(self.bar, 1)
        layout.addWidget(self.detail)
        layout.addWidget(self.pause_btn)
        layout.addWidget(self.stop_btn)
        self._anim = qtc.QPropertyAnimation(self.bar, b"value", self)
        self._anim.setDuration(350)
        self._anim.setEasingCurve(qtc.QEasingCurve.OutCubic)
        self.setVisible(False)

    def start(self, title):
        self.title.setText(title)
        self._anim.stop()
        self.bar.setRange(0, 0)       # indeterminate until the first frame is done
        self.detail.setText("Starting…")
        self.set_paused(False)
        self.stop_btn.setEnabled(True)
        self.setVisible(True)

    def set_progress(self, fraction, detail):
        if self.bar.maximum() == 0:
            self.bar.setRange(0, 1000)
            self.bar.setValue(0)
        self._anim.stop()
        self._anim.setEndValue(int(max(0.0, min(1.0, fraction)) * 1000))
        self._anim.start()
        self.detail.setText(detail)

    def set_paused(self, paused):
        self.pause_btn.setText("Resume" if paused else "Pause")
        self.pause_btn.setIcon(icon("play" if paused else "pause"))

    def stopping(self):
        self.detail.setText("Stopping…")
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)

    def finish(self):
        self.pause_btn.setEnabled(True)
        self.setVisible(False)


class AboutDialog(qtw.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About ChimpStackr")
        layout = qtw.QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(8)
        logo = qtw.QLabel()
        logo.setAlignment(qtc.Qt.AlignCenter)
        app_icon = qtw.QApplication.windowIcon()
        if not app_icon.isNull():
            logo.setPixmap(app_icon.pixmap(96, 96))
        name = qtw.QLabel("ChimpStackr")
        name.setAlignment(qtc.Qt.AlignCenter)
        f = name.font()
        f.setPointSizeF(f.pointSizeF() * 1.5)
        f.setWeight(qtg.QFont.Bold)
        name.setFont(f)
        import platform
        import PySide6
        import src.updater as updater
        version = qtw.QLabel(f"Version {updater.display_version()}")
        version.setAlignment(qtc.Qt.AlignCenter)
        version.setObjectName("Secondary")
        tagline = qtw.QLabel("Focus stacking for macro photography")
        tagline.setAlignment(qtc.Qt.AlignCenter)
        details = (f"ChimpStackr {src.__version__}\nQt {PySide6.__version__} · Python "
                   f"{platform.python_version()}\n{platform.platform()}")
        sys_label = qtw.QLabel(details.split("\n", 1)[1])
        sys_label.setObjectName("Hint")
        sys_label.setAlignment(qtc.Qt.AlignCenter)
        sys_label.setTextInteractionFlags(qtc.Qt.TextSelectableByMouse)
        link = qtw.QLabel(f'<a href="{REPO_URL}">{REPO_URL.replace("https://", "")}</a>')
        link.setAlignment(qtc.Qt.AlignCenter)
        link.setOpenExternalLinks(True)
        buttons = qtw.QDialogButtonBox()
        copy = buttons.addButton("Copy Version Info", qtw.QDialogButtonBox.ActionRole)
        copy.clicked.connect(lambda: qtw.QApplication.clipboard().setText(details))
        buttons.addButton(qtw.QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        for w in (logo, name, version, tagline, link, sys_label):
            layout.addWidget(w)
        layout.addSpacing(8)
        layout.addWidget(buttons)


class Window(qtw.QMainWindow):
    MAX_RECENT = 8

    def __init__(self):
        super().__init__()
        settings.globalVars["MainWindow"] = self
        settings.globalVars["LoadedImagePaths"] = []
        self.qs = settings.globalVars["QSettings"]
        self.current_image_directory = str(self.qs.value("ui/last_dir") or _home_dir())
        self.supportedReadFormats = tuple(
            "." + e.lower() for e in settings.globalVars["SupportedImageReadFormats"]
            + settings.globalVars["SupportedRAWFormats"])

        self.algorithm_config = AlgorithmConfig()
        self.LaplacianAlgorithm = algorithm_API.LaplacianPyramid(config=self.algorithm_config)
        self.TimeRemainingHandler = TimeRemainingHandler.TimeRemainingHandler()
        self.threadpool = qtc.QThreadPool(self)
        self.loader_pool = qtc.QThreadPool(self)
        self.loader_pool.setMaxThreadCount(2)

        self.results = []          # newest first, like the sidebar
        self.current_result = None
        self._stacking_active = False
        self._paused = False
        self._stop_requested = False
        self._pending_path = None
        self._stack_started = 0.0
        self._before_key = None
        self._settings_dialog = None
        self.pending_update_helper = None     # set by the update window: run on quit

        self.setWindowTitle("ChimpStackr")
        self.setMinimumSize(880, 560)
        self.setUnifiedTitleAndToolBarOnMac(True)
        self.setDocumentMode(True)
        self.setAcceptDrops(True)

        self.gpu_status = SettingsWidget.GpuStatus()
        self.updates = UpdateDialog.UpdateChecker(self)

        # Panels
        self.sidebar = Sidebar()
        self.viewer = ImageViewer()
        self.inspector = SettingsWidget.Inspector()
        self.SettingsWidget = self.inspector     # old name
        self.activity = ActivityBar()

        center = qtw.QWidget()
        cl = qtw.QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(self.viewer, 1)
        cl.addWidget(self.activity)

        self.splitter = qtw.QSplitter(qtc.Qt.Horizontal)
        self.splitter.setHandleWidth(1)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(center)
        self.splitter.addWidget(self.inspector)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([240, 900, 310])
        self.setCentralWidget(self.splitter)

        # Status bar
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        self.info_label = qtw.QLabel("")
        self.image_info_label = self.info_label   # old name
        sb.addWidget(self.info_label, 1)
        self.device_label = qtw.QLabel("")
        self.device_label.setObjectName("Secondary")
        sb.addPermanentWidget(self.device_label)

        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._connect()
        self._restore_layout()
        self._update_device_label()
        self._update_actions()

    # ══ actions, menus, toolbar ══
    def _action(self, text, slot=None, shortcut=None, icon_name=None, tip=None, checkable=False,
                role=None):
        a = qtg.QAction(text, self)
        if icon_name:
            a.setIcon(icon(icon_name))
        if shortcut is not None:
            if isinstance(shortcut, (list, tuple)):
                a.setShortcuts([qtg.QKeySequence(s) for s in shortcut])
            else:
                a.setShortcut(qtg.QKeySequence(shortcut))
        if tip:
            a.setToolTip(tip)
            a.setStatusTip(tip)
        a.setCheckable(checkable)
        if role is not None:
            a.setMenuRole(role)
        if slot is not None:
            a.triggered.connect(slot)
        self.addAction(a)     # shortcuts work even when the menu bar is hidden
        return a

    def _build_actions(self):
        K = qtg.QKeySequence
        self.act_new = self._action("New Stack", self.new_stack, K.New, tip="Start over with no frames")
        self.act_add = self._action("Add Images…", self.add_images_dialog, K.Open, "add",
                                    "Add images to the stack")
        self.act_add_folder = self._action("Add Folder…", self.add_folder_dialog, "Ctrl+Shift+O", "open",
                                           "Add every image in a folder")
        self.act_export = self._action("Export…", self.export_result, "Ctrl+E", "export",
                                       "Export the result as JPEG, PNG, TIFF or EXR")
        self.act_export_all = self._action("Export JPEG, PNG and TIFF…", self.batch_export_output_image,
                                           "Ctrl+Shift+E")
        self.act_export_view = self._action("Export Current View…", self.export_view)
        self.act_close = self._action("Close Window", self.close, K.Close)
        self.act_quit = self._action("Quit ChimpStackr" if IS_MAC else "Quit",
                                     qtw.QApplication.closeAllWindows, K.Quit,
                                     role=qtg.QAction.QuitRole)

        self.act_remove = self._action("Remove Selected Frames", self.remove_selected_frames,
                                       tip="Remove the selected frames from the stack")
        self.act_select_all = self._action("Select All Frames", self.select_all_frames, K.SelectAll)
        self.act_settings = self._action("Settings…", self.show_settings, K.Preferences, "settings",
                                         role=qtg.QAction.PreferencesRole)
        if not self.act_settings.shortcuts():
            self.act_settings.setShortcut("Ctrl+,")

        self.act_stack = self._action("Stack", self.stack, ["Ctrl+Return", "Ctrl+Enter"], "play",
                                      "Align and stack the frames")
        self.act_stack_no_align = self._action("Stack Without Aligning", self.stack_loaded_images,
                                               "Ctrl+Shift+Return")
        self.act_pause = self._action("Pause", self.toggle_pause, tip="Pause or resume stacking")
        self.act_stop = self._action("Stop", self.cancel_stacking, ["Ctrl+.", "Escape"], "stop",
                                     "Stop stacking")
        self.act_crop = self._action("Auto-Crop Result", self.auto_crop_result, icon_name="crop",
                                     tip="Crop the empty borders alignment left")

        self.act_toggle_sidebar = self._action("Show Sidebar", self.toggle_sidebar, "Ctrl+Alt+S",
                                               "sidebar", "Show or hide the frames sidebar", True)
        self.act_toggle_inspector = self._action("Show Inspector", self.toggle_inspector, "Ctrl+Alt+I",
                                                 "inspector", "Show or hide the stacking settings", True)
        self.act_compare = self._action("Compare Before / After", self.toggle_compare, ["\\", "Y"],
                                        "compare", "Compare the result with a source frame", True)
        self.act_zoom_in = self._action("Zoom In", self.viewer.zoom_in, [K.ZoomIn, "Ctrl+=", "+", "="], "zoom_in")
        self.act_zoom_out = self._action("Zoom Out", self.viewer.zoom_out, [K.ZoomOut, "-"], "zoom_out")
        self.act_fit = self._action("Zoom to Fit", self.viewer.fit, ["Ctrl+0", "0"], "fit")
        self.act_actual = self._action("Actual Size (100 %)", lambda: self.viewer.actual_size(),
                                       ["Ctrl+1", "1"], "actual")
        self.act_prev = self._action("Previous Frame", lambda: self.step_frame(-1), "Ctrl+Left")
        self.act_next = self._action("Next Frame", lambda: self.step_frame(1), "Ctrl+Right")
        self.act_fullscreen = self._action("Enter Full Screen", self.toggle_fullscreen, K.FullScreen)

        self.act_about = self._action("About ChimpStackr", self.show_about, role=qtg.QAction.AboutRole)
        self.act_updates = self._action("Check for Updates…", lambda: self.updates.check(manual=True),
                                        role=qtg.QAction.ApplicationSpecificRole)
        self.act_help = self._action("ChimpStackr Help", lambda: qtg.QDesktopServices.openUrl(
            qtc.QUrl(REPO_URL + "#readme")), K.HelpContents)
        self.act_issue = self._action("Report an Issue…", lambda: qtg.QDesktopServices.openUrl(
            qtc.QUrl(REPO_URL + "/issues/new")))

        # Single-key shortcuts (+ - 0 1 \\ Y) don't fire while typing in a field:
        # text and number fields take printable keys first

    def _build_menus(self):
        mb = self.menuBar()
        f = mb.addMenu("&File")
        f.addAction(self.act_new)
        f.addAction(self.act_add)
        f.addAction(self.act_add_folder)
        self.recent_menu = f.addMenu("Open Recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        f.addSeparator()
        f.addAction(self.act_export)
        f.addAction(self.act_export_all)
        f.addAction(self.act_export_view)
        f.addSeparator()
        f.addAction(self.act_close)
        f.addAction(self.act_quit)

        e = mb.addMenu("&Edit")
        e.addAction(self.act_select_all)
        e.addAction(self.act_remove)
        if not IS_MAC:
            e.addSeparator()
        e.addAction(self.act_settings)   # macOS moves it to the app menu

        v = mb.addMenu("&View")
        v.addAction(self.act_toggle_sidebar)
        v.addAction(self.act_toggle_inspector)
        v.addSeparator()
        v.addAction(self.act_compare)
        v.addSeparator()
        v.addAction(self.act_zoom_in)
        v.addAction(self.act_zoom_out)
        v.addAction(self.act_fit)
        v.addAction(self.act_actual)
        v.addSeparator()
        v.addAction(self.act_prev)
        v.addAction(self.act_next)
        v.addSeparator()
        v.addAction(self.act_fullscreen)

        s = mb.addMenu("&Stack")
        s.addAction(self.act_stack)
        s.addAction(self.act_stack_no_align)
        s.addSeparator()
        s.addAction(self.act_pause)
        s.addAction(self.act_stop)
        s.addSeparator()
        s.addAction(self.act_crop)

        if IS_MAC:
            w = mb.addMenu("&Window")
            w.addAction(self._action("Minimize", self.showMinimized, "Ctrl+M"))
            w.addAction(self._action("Zoom", lambda: self.showNormal() if self.isMaximized()
                                     else self.showMaximized()))

        h = mb.addMenu("&Help")
        h.addAction(self.act_help)
        h.addAction(self.act_issue)
        h.addSeparator()
        h.addAction(self.act_updates)    # macOS moves it to the app menu
        h.addAction(self.act_about)

    def _build_toolbar(self):
        tb = qtw.QToolBar("Toolbar")
        tb.setObjectName("MainToolbar")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setIconSize(qtc.QSize(18, 18))
        tb.setToolButtonStyle(qtc.Qt.ToolButtonIconOnly)
        tb.setContextMenuPolicy(qtc.Qt.PreventContextMenu)
        self.addToolBar(qtc.Qt.TopToolBarArea, tb)
        self.toolbar = tb

        tb.addAction(self.act_toggle_sidebar)
        tb.addAction(self.act_add)

        # Primary: Stack, with a menu for the variant
        self.stack_btn = qtw.QToolButton()
        self.stack_btn.setObjectName("PrimaryButton")
        self.stack_btn.setDefaultAction(self.act_stack)
        self.stack_btn.setIcon(icon("play", "on_accent"))
        self.stack_btn.setToolButtonStyle(qtc.Qt.ToolButtonTextBesideIcon)
        self.stack_btn.setIconSize(qtc.QSize(14, 14))
        self.stack_btn.setCursor(qtc.Qt.PointingHandCursor)
        stack_menu = qtw.QMenu(self.stack_btn)
        stack_menu.addAction(self.act_stack_no_align)
        self.stack_btn.setMenu(stack_menu)
        self.stack_btn.setPopupMode(qtw.QToolButton.MenuButtonPopup)
        tb.addSeparator()
        tb.addWidget(self.stack_btn)

        self.export_btn = qtw.QToolButton()
        self.export_btn.setDefaultAction(self.act_export)
        self.export_btn.setToolButtonStyle(qtc.Qt.ToolButtonTextBesideIcon)
        tb.addWidget(self.export_btn)

        spacer = qtw.QWidget()
        spacer.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Preferred)
        tb.addWidget(spacer)

        self.mode_switch = Segmented([("view", "View", "Show the selected image"),
                                      ("compare", "Compare", "Before / after with a draggable divider "
                                                             "(\\ or Y)")])
        self.mode_switch.changed.connect(lambda i: self.set_compare(i == 1))
        tb.addWidget(self.mode_switch)
        tb.addSeparator()

        tb.addAction(self.act_zoom_out)
        self.zoom_btn = qtw.QToolButton()
        self.zoom_btn.setText("–")
        self.zoom_btn.setToolTip("Zoom level")
        self.zoom_btn.setMinimumWidth(56)
        self.zoom_btn.setPopupMode(qtw.QToolButton.InstantPopup)
        zm = qtw.QMenu(self.zoom_btn)
        zm.addAction(self.act_fit)
        for pct in (25, 50, 100, 200, 400):
            zm.addAction(f"{pct} %", lambda p=pct: self.viewer.zoom_by(p / 100 / self.viewer.zoom()))
        self.zoom_btn.setMenu(zm)
        tb.addWidget(self.zoom_btn)
        tb.addAction(self.act_zoom_in)
        tb.addAction(self.act_fit)
        tb.addSeparator()
        tb.addAction(self.act_toggle_inspector)

    def _connect(self):
        sb, v = self.sidebar, self.viewer
        sb.add_requested.connect(self.add_images_dialog)
        sb.files_dropped.connect(self.add_paths)
        sb.remove_requested.connect(self.remove_some_images)
        sb.frame_selected.connect(self.show_frame)
        sb.result_selected.connect(self.show_result)
        v.add_requested.connect(self.add_images_dialog)
        v.files_dropped.connect(self.add_paths)
        v.step_requested.connect(self.step_frame)
        v.zoom_changed.connect(lambda z: self.zoom_btn.setText(f"{z * 100:.0f} %"))
        self.activity.pause_clicked.connect(self.toggle_pause)
        self.activity.stop_clicked.connect(self.cancel_stacking)
        self.inspector.setting_updated.connect(lambda _kv: self._update_actions())
        self.gpu_status.changed.connect(self._update_device_label)
        tm = theme.manager()
        if tm is not None:
            tm.changed.connect(self._on_theme_changed)

    def _on_theme_changed(self):
        self.stack_btn.setIcon(icon("play", "on_accent"))
        for w in self.findChildren(qtw.QWidget):
            w.update()

    # ══ state ══
    @property
    def is_stacking(self):
        return self._stacking_active

    @property
    def has_unsaved_work(self):
        return any(r.exported_to is None for r in self.results)

    def frames(self):
        return settings.globalVars["LoadedImagePaths"]

    def _update_actions(self):
        n = len(self.frames())
        busy = self._stacking_active
        has_result = self.current_result is not None
        enough = n >= 2
        for a in (self.act_stack, self.act_stack_no_align):
            a.setEnabled(enough and not busy)
        self.act_stack.setToolTip("Align and stack the frames" if enough
                                  else "Add at least two frames to stack")
        self.act_stack.setText("Stack" if self.inspector.alignment_enabled() else "Stack (No Alignment)")
        self.act_pause.setEnabled(busy and not self._stop_requested)
        self.act_pause.setText("Resume" if self._paused else "Pause")
        self.act_stop.setEnabled(busy and not self._stop_requested)
        for a in (self.act_export, self.act_export_all):
            a.setEnabled(bool(self.results) and not busy)   # exports the shown or latest result
        self.act_export_view.setEnabled(self.viewer.has_image)
        self.act_crop.setEnabled(has_result and not busy and bool(
            self.LaplacianAlgorithm.Algorithm.alignment_shifts))
        for a in (self.act_new, self.act_add, self.act_add_folder):
            a.setEnabled(not busy)
        self.act_remove.setEnabled(not busy and bool(self.sidebar.frames.selectedItems()))
        self.act_select_all.setEnabled(n > 0)
        self.act_prev.setEnabled(n > 1)
        self.act_next.setEnabled(n > 1)
        self.act_compare.setEnabled(has_result)
        self.mode_switch.set_enabled_at(1, has_result)
        self.inspector.setEnabled(not busy)
        self.act_toggle_sidebar.setChecked(self.sidebar.isVisible())
        self.act_toggle_inspector.setChecked(self.inspector.isVisible())
        self.act_fullscreen.setText("Exit Full Screen" if self.isFullScreen() else "Enter Full Screen")
        for a in (self.act_zoom_in, self.act_zoom_out, self.act_fit, self.act_actual):
            a.setEnabled(self.viewer.has_image)

    def _update_device_label(self):
        self.device_label.setText(self.gpu_status.summary())

    def _update_title(self):
        paths = self.frames()
        if paths:
            folder = os.path.basename(os.path.dirname(paths[0])) or paths[0]
            self.setWindowTitle(f"{folder} — ChimpStackr" if not IS_MAC else folder)
            self.setWindowFilePath(os.path.dirname(paths[0]))
        else:
            self.setWindowTitle("ChimpStackr")
            self.setWindowFilePath("")

    def show_message(self, text, ms=4000):
        self.statusBar().showMessage(text, ms)

    # ══ frames ══
    def _image_filter(self):
        exts = " ".join(f"*{e}" for e in sorted(set(self.supportedReadFormats)))
        if not IS_MAC and os.name != "nt":
            exts += " " + exts.upper()
        return f"Images ({exts});;All Files (*)"

    def add_images_dialog(self):
        if self._stacking_active:
            return
        paths, _ = qtw.QFileDialog.getOpenFileNames(
            self, "Add Images", self.current_image_directory, self._image_filter())
        if paths:
            self.add_paths(paths)

    def add_folder_dialog(self):
        if self._stacking_active:
            return
        folder = qtw.QFileDialog.getExistingDirectory(self, "Add Folder", self.current_image_directory)
        if folder:
            self.add_paths([folder])

    def _expand(self, paths):
        out = []
        for path in paths:
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                    out.extend(os.path.join(root, f) for f in files)
            elif os.path.isfile(path):
                out.append(path)
        valid, skipped = [], 0
        for p in out:
            base = os.path.basename(p).lower()
            if base.startswith(".") or base in IGNORED_FILES:
                continue
            if p.lower().endswith(self.supportedReadFormats):
                valid.append(os.path.abspath(p))
            else:
                skipped += 1
        return valid, skipped

    def add_paths(self, paths):
        """Add files and/or folders to the stack (duplicates are ignored)."""
        if self._stacking_active:
            self.show_message("Can't change frames while stacking")
            return
        valid, skipped = self._expand(paths)
        current = list(self.frames())
        known = set(current)
        new = [p for p in valid if p not in known]
        if not new:
            self.show_message(f"No supported images found{f' ({skipped} other files skipped)' if skipped else ''}"
                              if not valid else "Those frames are already in the stack")
            return
        self._set_frames(current + new)
        folder = os.path.dirname(new[0])
        self.current_image_directory = folder
        self.qs.setValue("ui/last_dir", folder)
        self._remember_recent(folder)
        msg = f"Added {len(new)} frame{'s' if len(new) != 1 else ''}"
        if skipped:
            msg += f" · skipped {skipped} unsupported file{'s' if skipped != 1 else ''}"
        self.show_message(msg)
        if not current:
            self.inspector._auto_detect_params()
            self.sidebar.select_frame(0)

    # old entry point (CLI tests and drag-drop helpers used it)
    def set_new_loaded_image_files(self, paths):
        self.add_paths(paths)

    def _set_frames(self, paths):
        paths = sorted(paths, key=int_string_sorting)
        settings.globalVars["LoadedImagePaths"] = paths
        self.LaplacianAlgorithm.update_image_paths(paths)
        self.sidebar.set_frames(paths)
        self._before_key = None
        if not paths and not self.results:
            self.viewer.show_empty_state(True)
            self.info_label.setText("")
        self._update_title()
        self._update_actions()

    def remove_some_images(self, paths_to_remove):
        if self._stacking_active or not paths_to_remove:
            return
        drop = set(paths_to_remove)
        row = self.sidebar.current_frame_row()
        self._set_frames([p for p in self.frames() if p not in drop])
        n = len(paths_to_remove)
        self.show_message(f"Removed {n} frame{'s' if n != 1 else ''}")
        if self.frames():
            self.sidebar.select_frame(min(max(row, 0), len(self.frames()) - 1))

    def remove_selected_frames(self):
        self.remove_some_images(self.sidebar.frames.selected_paths())

    def select_all_frames(self):
        self.sidebar.frames.setFocus()
        self.sidebar.frames.selectAll()

    def new_stack(self):
        if self._stacking_active:
            return
        if self.has_unsaved_work and not self._confirm_discard("Start a new stack?"):
            return
        self.results.clear()
        self.current_result = None
        self.sidebar.clear_results()
        self.set_compare(False)
        self._set_frames([])
        self.viewer.show_empty_state(True)

    # old name
    def clear_all_images(self):
        self.new_stack()
        return not self.frames()

    def step_frame(self, delta):
        n = self.sidebar.frames.count()
        if n:
            row = self.sidebar.current_frame_row()
            self.sidebar.select_frame((row + delta) % n if row >= 0 else 0)

    def show_frame(self, path):
        self.current_result = None
        self.set_compare(False)
        self._pending_path = path
        self.viewer.show_loading(True)
        name = os.path.basename(path)
        self.info_label.setText(name)
        loader = ImageLoader(path)
        loader.signals.ready.connect(self._on_frame_loaded)
        self.loader_pool.start(loader)
        self._update_actions()

    def _on_frame_loaded(self, path, img, _tag=None):
        if path != self._pending_path:
            return
        self.viewer.show_loading(False)
        if img is None:
            self.viewer.set_image(None)
            self.info_label.setText(f"{os.path.basename(path)} · can't be read")
            return
        self.viewer.set_image(img)
        h, w = img.shape[:2]
        try:
            size = os.path.getsize(path)
            size_text = f"{size / 1e6:.1f} MB"
        except OSError:
            size_text = ""
        i = self.frames().index(path) + 1 if path in self.frames() else 0
        parts = [os.path.basename(path), f"{w} × {h}", size_text, f"frame {i} of {len(self.frames())}" if i else ""]
        self.info_label.setText("  ·  ".join(p for p in parts if p))
        self._update_actions()

    # ══ results ══
    def show_result(self, index):
        if not (0 <= index < len(self.results)):
            return
        r = self.results[index]
        self._pending_path = None
        self.current_result = r
        self.viewer.show_loading(False)
        self.viewer.set_image(r.image)
        h, w = r.image.shape[:2]
        self.info_label.setText(f"{r.title} result  ·  {w} × {h}  ·  {r.frames} frames  ·  "
                                f"stacked in {_format_duration(r.seconds)}")
        self._load_before(r)
        self._update_actions()

    def _load_before(self, r):
        """Load the reference frame, cropped like the result, for Compare."""
        paths = self.frames()
        if not paths:
            self.viewer.set_before_image(None)
            return
        ref = paths[len(paths) // 2] if self.inspector.get_algorithm_config().alignment_reference == "middle" \
            else paths[0]
        key = (ref, r.crop)
        if key == self._before_key:
            return
        self._before_key = key
        loader = ImageLoader(ref, r.crop, key)
        loader.signals.ready.connect(self._on_before_loaded)
        self.loader_pool.start(loader)

    def _on_before_loaded(self, _path, img, key):
        if key == self._before_key:
            self.viewer.set_before_image(img)

    def toggle_compare(self):
        self.set_compare(not self.viewer.compare)

    def set_compare(self, on):
        on = bool(on) and self.current_result is not None
        self.viewer.set_compare(on)
        self.act_compare.setChecked(on)
        self.mode_switch.set_index(1 if on else 0)

    def auto_crop_result(self):
        r = self.current_result
        if r is None:
            return
        bounds = self.LaplacianAlgorithm.auto_crop_output()
        if bounds and self.LaplacianAlgorithm.output_image is not None:
            r.image = self.LaplacianAlgorithm.output_image
            r.crop = bounds
            self.show_result(self.results.index(r))
            self.show_message("Cropped the empty borders")
        else:
            self.show_message("Nothing to crop")

    # ══ stacking ══
    def _sync_algorithm_config(self):
        self.LaplacianAlgorithm.config = self.inspector.get_algorithm_config()

    def stack(self):
        if self.inspector.alignment_enabled():
            return self.align_and_stack_loaded_images()
        return self.stack_loaded_images()

    def align_and_stack_loaded_images(self):
        return self._start_stacking("align_and_stack_images", True)

    def stack_loaded_images(self):
        return self._start_stacking("stack_images", False)

    def _start_stacking(self, method_name, aligned):
        if self._stacking_active or len(self.frames()) < 2:
            return False
        self._sync_algorithm_config()
        cfg = self.LaplacianAlgorithm.config
        self._stacking_active = True
        self._paused = False
        self._stop_requested = False
        self._stack_aligned = aligned
        self._stack_started = time.time()
        self.TimeRemainingHandler.reset()
        n = len(self.frames())
        method = METHOD_TITLES.get(cfg.stacking_method, cfg.stacking_method)
        self.activity.start(f"{'Aligning and stacking' if aligned else 'Stacking'} {n} frames · {method}")

        def progress(result_list):
            try:
                key, done, total, seconds = result_list
                if key != "finished_image" or total <= 0:
                    return
                left = self.TimeRemainingHandler.calculate_time_remaining(
                    100 / total, 100 - done / total * 100, seconds)
                left = left.replace("Time left: ", "")
                detail = f"{done} of {total}" + (f" · {left} left" if "~" in left else f" · {left}")
                if self._paused:
                    detail = "Paused"
                self.activity.set_progress(done / total, detail)
            except (AttributeError, RuntimeError, ZeroDivisionError, ValueError):
                pass

        worker = QThreading.Worker(getattr(self.LaplacianAlgorithm, method_name))
        worker.signals.finished_inter_task.connect(progress)
        worker.signals.error.connect(self._on_stack_error)
        worker.signals.finished.connect(self.finished_stack)
        self.threadpool.start(worker)
        self._update_actions()
        return True

    def toggle_pause(self):
        if not self._stacking_active or self._stop_requested:
            return
        self._paused = not self._paused
        if self._paused:
            self.LaplacianAlgorithm.pause()
            self.activity.detail.setText("Paused")
        else:
            self.LaplacianAlgorithm.resume()
        self.activity.set_paused(self._paused)
        self._update_actions()

    def cancel_stacking(self):
        if not self._stacking_active or self._stop_requested:
            return
        self._stop_requested = True
        self._paused = False
        self.LaplacianAlgorithm.cancel()
        self.activity.stopping()
        self._update_actions()

    def _on_stack_error(self, err):
        _type, value, tb = err
        box = qtw.QMessageBox(self)
        box.setIcon(qtw.QMessageBox.Critical)
        box.setWindowTitle("Stacking failed")
        box.setText("Stacking stopped because of an error.")
        box.setInformativeText(str(value))
        box.setDetailedText(tb)
        box.open()

    def finished_stack(self):
        seconds = time.time() - self._stack_started
        stopped = self._stop_requested
        self._stacking_active = False
        self._paused = False
        self._stop_requested = False
        self.activity.finish()
        algo = self.LaplacianAlgorithm
        if algo.output_image is None or stopped:
            algo.Algorithm.alignment_shifts = []
            self.show_message("Stacking cancelled")
            self._update_actions()
            return

        crop = None
        if SettingsWidget.get_int("algorithm/auto_crop", 1):
            crop = algo.auto_crop_output() or None
        cfg = algo.config
        r = Result(algo.output_image, cfg.stacking_method, self._stack_aligned,
                   len(self.frames()), seconds, crop)
        self.results.insert(0, r)
        self.sidebar.add_result(r.title, r.subtitle, r.image)
        self.show_result(0)
        self.show_message(f"Stacked {r.frames} frames in {_format_duration(seconds)}")
        qtw.QApplication.alert(self)      # bounce the Dock icon / flash the taskbar if in the background
        self._update_actions()

    # ══ export ══
    def _result_base_name(self, r):
        paths = self.frames()
        folder = os.path.basename(os.path.dirname(paths[0])) if paths else "stack"
        return f"{folder}_{r.method}"

    def export_result(self):
        r = self.current_result or (self.results[0] if self.results else None)
        if r is None or self._stacking_active:
            return
        dialog = Export.ExportDialog(r.image.shape, self)
        if dialog.exec() != qtw.QDialog.Accepted:
            return
        opts = dialog.options()
        path = Export.ask_export_path(self, opts, self.current_image_directory, self._result_base_name(r))
        if not path:
            return
        qtw.QApplication.setOverrideCursor(qtc.Qt.WaitCursor)
        try:
            Export.write_image(path, r.image, opts["fmt"], opts["bit_depth"], opts["quality"],
                               opts["compression"])
        except Exception as exc:
            qtw.QApplication.restoreOverrideCursor()
            qtw.QMessageBox.critical(self, "Export failed", f"Couldn't write {os.path.basename(path)}.\n\n{exc}")
            return
        qtw.QApplication.restoreOverrideCursor()
        r.exported_to = path
        self.current_image_directory = os.path.dirname(path)
        self._show_exported(path)

    # old name
    def export_output_image(self):
        self.export_result()

    def _show_exported(self, path):
        size = os.path.getsize(path) / 1e6
        self.show_message(f"Exported {os.path.basename(path)} ({size:.1f} MB)", 8000)
        link = qtw.QPushButton(f"Show in {file_manager_name()}")
        link.setObjectName("FlatButton")
        link.setCursor(qtc.Qt.PointingHandCursor)
        link.clicked.connect(lambda: reveal_in_file_manager(path))
        self.statusBar().addWidget(link)
        qtc.QTimer.singleShot(8000, link.deleteLater)

    def batch_export_output_image(self):
        r = self.current_result or (self.results[0] if self.results else None)
        if r is None or self._stacking_active:
            return
        directory = qtw.QFileDialog.getExistingDirectory(self, "Export to Folder", self.current_image_directory)
        if not directory:
            return
        base = os.path.join(directory, self._result_base_name(r))
        qtw.QApplication.setOverrideCursor(qtc.Qt.WaitCursor)
        try:
            Export.write_image(base + ".jpg", r.image, "JPG", 8, 95)
            Export.write_image(base + ".png", r.image, "PNG", 16, compression=4)
            Export.write_image(base + ".tif", r.image, "TIFF", 16)
        except Exception as exc:
            qtw.QApplication.restoreOverrideCursor()
            qtw.QMessageBox.critical(self, "Export failed", str(exc))
            return
        qtw.QApplication.restoreOverrideCursor()
        r.exported_to = base + ".jpg"
        self._show_exported(base + ".jpg")
        self.show_message(f"Exported JPEG, 16-bit PNG and 16-bit TIFF to {directory}", 8000)

    def export_view(self):
        path, _ = qtw.QFileDialog.getSaveFileName(
            self, "Export Current View", os.path.join(self.current_image_directory, "view.png"),
            "PNG (*.png);;JPEG (*.jpg)")
        if path and self.viewer.export_view(path):
            self._show_exported(path)

    # old name
    def export_comparison(self):
        self.export_view()

    # ══ panels, dialogs ══
    def toggle_sidebar(self):
        self.sidebar.setVisible(not self.sidebar.isVisible())
        self._update_actions()

    def toggle_inspector(self):
        self.inspector.setVisible(not self.inspector.isVisible())
        self._update_actions()

    # old name
    def toggle_settings_panel(self):
        self.toggle_inspector()

    def toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()
        self._update_actions()

    def show_settings(self):
        if self._settings_dialog is None:
            self._settings_dialog = SettingsWidget.SettingsDialog(self.gpu_status, self.updates, self)
        self._settings_dialog.show()
        self._settings_dialog.raise_()
        self._settings_dialog.activateWindow()

    def show_about(self):
        AboutDialog(self).exec()

    def _remember_recent(self, folder):
        recent = [f for f in (self.qs.value("ui/recent_folders") or []) if f != folder]
        if isinstance(recent, str):
            recent = [recent]
        self.qs.setValue("ui/recent_folders", ([folder] + list(recent))[:self.MAX_RECENT])

    def _fill_recent_menu(self):
        m = self.recent_menu
        m.clear()
        recent = self.qs.value("ui/recent_folders") or []
        if isinstance(recent, str):
            recent = [recent]
        recent = [f for f in recent if os.path.isdir(f)]
        for folder in recent:
            a = m.addAction(os.path.basename(folder) or folder)
            a.setToolTip(folder)
            a.triggered.connect(lambda _c=False, f=folder: self._open_recent(f))
        if recent:
            m.addSeparator()
            m.addAction("Clear Menu", lambda: self.qs.setValue("ui/recent_folders", []))
        else:
            m.addAction("No Recent Folders").setEnabled(False)

    def _open_recent(self, folder):
        if self._stacking_active:
            return
        if self.frames() and not self._confirm_discard("Open another stack?", only_if_unsaved=False):
            return
        self.results.clear()
        self.current_result = None
        self.sidebar.clear_results()
        self._set_frames([])
        self.add_paths([folder])

    def _confirm_discard(self, question, only_if_unsaved=True):
        unsaved = self.has_unsaved_work
        if only_if_unsaved and not unsaved:
            return True
        box = qtw.QMessageBox(self)
        box.setIcon(qtw.QMessageBox.Warning if unsaved else qtw.QMessageBox.Question)
        box.setText(question)
        box.setInformativeText("Results you haven't exported will be lost." if unsaved
                               else "The current frames will be replaced.")
        discard = box.addButton("Discard" if unsaved else "Replace", qtw.QMessageBox.DestructiveRole)
        box.addButton(qtw.QMessageBox.Cancel)
        box.setDefaultButton(qtw.QMessageBox.Cancel)
        box.exec()
        return box.clickedButton() is discard

    # ══ window ══
    def _restore_layout(self):
        geo = self.qs.value("ui/geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            avail = self.screen().availableGeometry()
            self.resize(min(1440, int(avail.width() * 0.85)), min(900, int(avail.height() * 0.85)))
            self.move(avail.center() - self.rect().center())
        sizes = self.qs.value("ui/splitter")
        if sizes:
            try:
                self.splitter.setSizes([int(s) for s in sizes])
            except (TypeError, ValueError):
                pass
        self.sidebar.setVisible(bool(SettingsWidget.get_int("ui/show_sidebar", 1)))
        self.inspector.setVisible(bool(SettingsWidget.get_int("ui/show_inspector", 1)))

    def _save_layout(self):
        self.qs.setValue("ui/geometry", self.saveGeometry())
        self.qs.setValue("ui/splitter", self.splitter.sizes())
        self.qs.setValue("ui/show_sidebar", int(self.sidebar.isVisible()))
        self.qs.setValue("ui/show_inspector", int(self.inspector.isVisible()))

    def changeEvent(self, event):
        if event.type() == qtc.QEvent.WindowStateChange:
            self._update_actions()
        super().changeEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            event.acceptProposedAction()
            self.add_paths(paths)

    def closeEvent(self, event):
        if self.has_unsaved_work:
            n = sum(r.exported_to is None for r in self.results)
            box = qtw.QMessageBox(self)
            box.setIcon(qtw.QMessageBox.Warning)
            box.setText(f"Quit without exporting {'the result' if n == 1 else f'{n} results'}?")
            box.setInformativeText("Stacked results you haven't exported will be lost.")
            export = box.addButton("Export…", qtw.QMessageBox.AcceptRole) if n == 1 else None
            quit_btn = box.addButton("Quit Anyway" if IS_MAC else "Discard", qtw.QMessageBox.DestructiveRole)
            box.addButton(qtw.QMessageBox.Cancel)
            box.setDefaultButton(qtw.QMessageBox.Cancel)
            box.exec()
            clicked = box.clickedButton()
            if export is not None and clicked is export:
                self.export_result()
                if self.has_unsaved_work:
                    event.ignore()
                    return
            elif clicked is not quit_btn:
                event.ignore()
                return
        self.LaplacianAlgorithm.cancel()
        self._save_layout()
        event.accept()
        if self.pending_update_helper:
            import src.updater as updater
            updater.launch(self.pending_update_helper)
