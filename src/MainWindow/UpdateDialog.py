"""
Update checks and the "new version available" window.

Checks run in the background: at startup at most once a day when
"Automatically check for updates" is on (Settings → Updates), and on demand
from "Check for Updates…". The window shows the release notes and offers
Install Update (download, verify, install and relaunch), Remind Me Later
or Skip This Version.
"""
import tempfile
import time

import PySide6.QtCore as qtc
import PySide6.QtGui as qtg
import PySide6.QtWidgets as qtw

import src.settings as settings
import src.updater as updater

CHECK_INTERVAL = 20 * 3600       # seconds between automatic checks


def qs():
    return settings.globalVars["QSettings"]


def auto_check_enabled():
    v = qs().value("updates/auto_check")
    return True if v in (None, "") else bool(int(v))


def include_prereleases():
    return bool(int(qs().value("updates/prerelease") or 0))


def _version_text(tag):
    return updater.display_version(tag)


class _Task(qtc.QRunnable):
    """Runs fn(report) on a worker thread; report(done, total) for progress."""

    class Signals(qtc.QObject):
        done = qtc.Signal(object)
        failed = qtc.Signal(str)
        progress = qtc.Signal(float, float)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = self.Signals()

    def run(self):
        try:
            result = self.fn(lambda d, t: self.signals.progress.emit(float(d), float(t)))
        except Exception as exc:
            self.signals.failed.emit(str(exc))
            return
        self.signals.done.emit(result)


class UpdateChecker(qtc.QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.pool = qtc.QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.dialog = None
        self._busy = False
        self._manual = False

    def start_automatic(self, delay_ms=4000):
        """Check in the background if it's on and due."""
        if not auto_check_enabled():
            return
        last = float(qs().value("updates/last_check") or 0)
        if time.time() - last < CHECK_INTERVAL:
            return
        qtc.QTimer.singleShot(delay_ms, lambda: self.check(manual=False))

    def check(self, manual=True):
        if self._busy:
            return
        if self.dialog is not None and self.dialog.isVisible():
            self.dialog.raise_()
            self.dialog.activateWindow()
            return
        self._busy, self._manual = True, manual
        if manual:
            self.window.show_message("Checking for updates…", 3000)
        prerelease = include_prereleases()
        task = _Task(lambda _report: updater.check(prerelease))
        task.signals.done.connect(self._on_result)
        task.signals.failed.connect(self._on_failed)
        self.pool.start(task)

    def _on_result(self, release):
        self._busy = False
        qs().setValue("updates/last_check", time.time())
        if release is None:
            if self._manual:
                qtw.QMessageBox.information(
                    self.window, "You're up to date",
                    f"ChimpStackr {_version_text(updater.current_version())} is the newest version.")
            return
        if not self._manual and str(qs().value("updates/skipped") or "") == release.tag:
            return
        self.dialog = UpdateDialog(release, self.window)
        self.dialog.show()
        self.dialog.raise_()

    def _on_failed(self, message):
        self._busy = False
        if self._manual:
            box = qtw.QMessageBox(self.window)
            box.setIcon(qtw.QMessageBox.Warning)
            box.setWindowTitle("Couldn't check for updates")
            box.setText("Couldn't check for updates.")
            box.setInformativeText("Check your internet connection and try again.")
            box.setDetailedText(message)
            box.open()


class UpdateDialog(qtw.QDialog):
    def __init__(self, release, window):
        super().__init__(window)
        self.release = release
        self.window = window
        self.downloaded = None
        self.helper = None
        self._cancel = False
        self.pool = qtc.QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.setWindowTitle("Software Update")
        self.setMinimumSize(520, 420)

        layout = qtw.QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 16)
        layout.setSpacing(12)

        head = qtw.QHBoxLayout()
        head.setSpacing(16)
        logo = qtw.QLabel()
        app_icon = qtw.QApplication.windowIcon()
        if app_icon.isNull() and settings.globalVars.get("AppIconPath"):
            app_icon = qtg.QIcon(settings.globalVars["AppIconPath"])
        if not app_icon.isNull():
            logo.setPixmap(app_icon.pixmap(64, 64))
        logo.setAlignment(qtc.Qt.AlignTop)
        head.addWidget(logo)
        texts = qtw.QVBoxLayout()
        title = qtw.QLabel("A new version of ChimpStackr is available")
        f = title.font()
        f.setPointSizeF(f.pointSizeF() * 1.2)
        f.setWeight(qtg.QFont.DemiBold)
        title.setFont(f)
        title.setWordWrap(True)
        sub = qtw.QLabel(f"ChimpStackr {_version_text(release.tag)} is available — you have "
                         f"{_version_text(updater.current_version())}."
                         + (" This is a pre-release." if release.prerelease else ""))
        sub.setObjectName("Secondary")
        sub.setWordWrap(True)
        texts.addWidget(title)
        texts.addWidget(sub)
        head.addLayout(texts, 1)
        layout.addLayout(head)

        notes_label = qtw.QLabel("RELEASE NOTES")
        notes_label.setObjectName("SectionTitle")
        layout.addWidget(notes_label)
        self.notes = qtw.QTextBrowser()
        self.notes.setOpenExternalLinks(True)
        self.notes.setMarkdown(release.notes or "No release notes.")
        layout.addWidget(self.notes, 1)

        self.progress = qtw.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.status = qtw.QLabel("")
        self.status.setObjectName("Secondary")
        self.status.setWordWrap(True)
        self.progress.setVisible(False)
        self.status.setVisible(False)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)

        self.auto = qtw.QCheckBox("Automatically check for updates")
        self.auto.setChecked(auto_check_enabled())
        self.auto.toggled.connect(lambda on: qs().setValue("updates/auto_check", int(on)))
        layout.addWidget(self.auto)

        row = qtw.QHBoxLayout()
        self.skip_btn = qtw.QPushButton("Skip This Version")
        self.skip_btn.clicked.connect(self._skip)
        self.later_btn = qtw.QPushButton("Remind Me Later")
        self.later_btn.clicked.connect(self.close)
        self.can_install = updater.can_install() and release.asset() is not None
        self.main_btn = qtw.QPushButton("Install Update" if self.can_install else "Download…")
        self.main_btn.setObjectName("PrimaryButton")
        self.main_btn.setDefault(True)
        self.main_btn.clicked.connect(self._main_action)
        if not self.can_install:
            self.main_btn.setToolTip("Opens the release page")
        row.addWidget(self.skip_btn)
        row.addStretch()
        row.addWidget(self.later_btn)
        row.addWidget(self.main_btn)
        layout.addLayout(row)

    # ── actions ──
    def _skip(self):
        qs().setValue("updates/skipped", self.release.tag)
        self.close()

    def _main_action(self):
        if self.helper is not None:
            self._install_and_relaunch()
        elif self.can_install:
            self._download()
        else:
            qtg.QDesktopServices.openUrl(qtc.QUrl(self.release.page))
            self.close()

    def _download(self):
        asset = self.release.asset()
        self._cancel = False
        self.progress.setVisible(True)
        self.status.setVisible(True)
        self.progress.setValue(0)
        self.status.setText("Downloading…")
        self.main_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        self.later_btn.setText("Cancel")
        self.later_btn.clicked.disconnect()
        self.later_btn.clicked.connect(self._cancel_download)
        tmp = tempfile.mkdtemp(prefix="chimpstackr-download-")

        def work(report):
            path = updater.download(asset, tmp, report, lambda: self._cancel)
            return updater.prepare_install(path)

        task = _Task(work)
        task.signals.progress.connect(self._on_progress)
        task.signals.done.connect(self._on_ready)
        task.signals.failed.connect(self._on_failed)
        self.pool.start(task)

    def _cancel_download(self):
        self._cancel = True
        self.status.setText("Cancelling…")

    def _on_progress(self, done, total):
        if total > 0:
            self.progress.setValue(int(done / total * 1000))
            self.status.setText(f"Downloading… {done / 1e6:.0f} of {total / 1e6:.0f} MB")
            if done >= total:
                self.progress.setRange(0, 0)
                self.status.setText("Verifying and preparing the update…")

    def _restore_later_button(self):
        self.later_btn.clicked.disconnect()
        self.later_btn.clicked.connect(self.close)

    def _on_ready(self, helper):
        self.helper = helper
        self.progress.setRange(0, 1000)
        self.progress.setValue(1000)
        self.status.setText("The update is ready. ChimpStackr will quit, update and reopen.")
        self.main_btn.setText("Install and Relaunch")
        self.main_btn.setEnabled(True)
        self.later_btn.setText("Install on Quit")
        self._restore_later_button()
        self.later_btn.clicked.disconnect()
        self.later_btn.clicked.connect(self._install_on_quit)

    def _on_failed(self, message):
        self.progress.setRange(0, 1000)
        self.progress.setVisible(False)
        self.status.setText(message)
        self.main_btn.setEnabled(True)
        self.main_btn.setText("Try Again" if "cancelled" not in message.lower() else "Install Update")
        self.skip_btn.setEnabled(True)
        self.later_btn.setText("Remind Me Later")
        self._restore_later_button()

    def _install_on_quit(self):
        self.window.pending_update_helper = self.helper
        self.window.show_message("The update will be installed when you quit ChimpStackr", 6000)
        self.close()

    def _install_and_relaunch(self):
        self.window.pending_update_helper = self.helper
        self.close()
        self.window.close()      # asks about unexported results; installs if it really closes
