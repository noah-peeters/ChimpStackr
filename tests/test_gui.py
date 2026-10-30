"""GUI tests (run offscreen: QT_QPA_PLATFORM=offscreen)."""
import glob
import os
import sys
import time

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import PySide6.QtCore as qtc
    import PySide6.QtGui as qtg
    import PySide6.QtWidgets as qtw
except ImportError as exc:     # e.g. no libEGL on a headless Linux box
    pytest.skip(f"Qt GUI libraries not available: {exc}", allow_module_level=True)

import src.settings as settings  # noqa: E402

LOW_RES_DIR = os.path.join(os.path.dirname(__file__), "low_res_images")
LOW_RES = sorted(glob.glob(os.path.join(LOW_RES_DIR, "*.jpg")))


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qtc.QSettings.setDefaultFormat(qtc.QSettings.IniFormat)
    qtc.QSettings.setPath(qtc.QSettings.IniFormat, qtc.QSettings.UserScope,
                          str(tmp_path_factory.mktemp("settings")))
    import src.run as run
    application = qtw.QApplication.instance() or run._App(["chimpstackr-test"])
    application.setOrganizationName("chimpstackr-tests")
    application.setApplicationName("chimpstackr-tests")
    settings.init()
    import tempfile
    settings.globalVars["RootTempDir"] = tempfile.TemporaryDirectory(prefix="ChimpStackrTest_")
    settings.globalVars["MainApplication"] = application
    import src.MainWindow.theme as theme
    theme.install(application, "dark")
    return application


@pytest.fixture
def window(app):
    qs = qtc.QSettings()
    qs.clear()
    qs.setValue("computing/use_gpu", 0)      # keep tests on the CPU
    qs.setValue("computing/gpu_default_on", 1)
    settings.globalVars["QSettings"] = qs
    import src.MainWindow as MW
    w = MW.Window()
    w.resize(1200, 800)
    w.show()
    pump(app)
    yield w
    w.results.clear()          # no "unexported result" prompt
    w.close()
    w.deleteLater()
    pump(app)


def pump(app, ms=100):
    end = time.time() + ms / 1000
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)


def wait(app, cond, timeout=60):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        app.processEvents()
        time.sleep(0.01)
    return cond()


class TestWindow:
    def test_actions_follow_state(self, app, window):
        assert not window.act_stack.isEnabled()
        assert not window.act_export.isEnabled()
        assert not window.act_compare.isEnabled()
        window.add_paths(LOW_RES[:1])
        assert not window.act_stack.isEnabled()      # one frame isn't a stack
        window.add_paths([LOW_RES_DIR])
        assert len(window.frames()) == len(LOW_RES)
        assert window.act_stack.isEnabled()
        assert not window.act_export.isEnabled()

    def test_add_ignores_duplicates_and_unsupported(self, app, window, tmp_path):
        (tmp_path / "notes.txt").write_text("x")
        (tmp_path / ".DS_Store").write_text("x")
        img = tmp_path / "a.png"
        cv2.imwrite(str(img), np.zeros((8, 8, 3), np.uint8))
        window.add_paths([str(tmp_path)])
        window.add_paths([str(img)])
        assert window.frames() == [str(img)]

    def test_remove_frames(self, app, window):
        window.add_paths(LOW_RES[:4])
        window.remove_some_images(LOW_RES[1:3])
        assert window.frames() == [LOW_RES[0], LOW_RES[3]]
        assert window.sidebar.frames.count() == 2

    def test_frame_preview_and_thumbnails(self, app, window):
        window.add_paths(LOW_RES[:3])
        assert wait(app, lambda: window.viewer.has_image)
        assert wait(app, lambda: "frame 1 of 3" in window.info_label.info())
        item = window.sidebar.frames.item(0)
        assert wait(app, lambda: not item.icon().pixmap(44, 44).isNull()
                    and item.toolTip().count("\n") == 1)

    def test_stack_compare_export(self, app, window, tmp_path):
        window.add_paths(LOW_RES[:3])
        assert window.stack()
        assert window.is_stacking and not window.act_stack.isEnabled()
        assert wait(app, lambda: not window.is_stacking, 180)
        assert len(window.results) == 1
        assert window.current_result is window.results[0]
        assert window.act_export.isEnabled() and window.act_compare.isEnabled()
        assert window.has_unsaved_work
        assert not window.viewer.compare and not window.act_compare.isChecked()   # stays in View
        window.set_compare(True)
        assert wait(app, lambda: window.viewer.before_item.isVisible())
        import src.MainWindow.ImageSavingDialog as Export
        out = tmp_path / "out.tif"
        Export.write_image(str(out), window.current_result.image, "TIFF", 16)
        assert cv2.imread(str(out), cv2.IMREAD_UNCHANGED).dtype == np.uint16

    def test_cancel(self, app, window):
        window.add_paths([LOW_RES_DIR])
        window.stack()
        window.cancel_stacking()
        assert wait(app, lambda: not window.is_stacking, 120)
        assert window.results == []
        assert window.act_stack.isEnabled()


class TestViewer:
    @pytest.fixture
    def viewer(self, app):
        from src.MainWindow.MainLayout.ImageViewers import ImageViewer
        v = ImageViewer()
        v.resize(800, 600)
        v.show()
        v.set_image(cv2.imread(LOW_RES[0]))
        pump(app)
        v.fit()
        yield v
        v.deleteLater()

    def test_zoom_levels(self, viewer):
        fit = viewer.zoom()
        assert 0.9 < fit < 1.1          # 750x500 into 800x600 minus margins
        viewer.actual_size()
        assert viewer.zoom() == pytest.approx(1.0)
        viewer.zoom_in()
        assert viewer.zoom() == pytest.approx(1.25)
        for _ in range(40):
            viewer.zoom_in()
        assert viewer.zoom() == pytest.approx(viewer.MAX_ZOOM)
        viewer.fit()
        assert viewer.zoom() == pytest.approx(fit)

    def _wheel(self, viewer, pixel, angle, mods=qtc.Qt.NoModifier, phase=qtc.Qt.NoScrollPhase):
        pos = qtc.QPointF(400, 300)
        ev = qtg.QWheelEvent(pos, viewer.mapToGlobal(pos.toPoint()).toPointF(), pixel, angle,
                             qtc.Qt.NoButton, mods, phase, False)
        viewer.wheelEvent(ev)

    def test_mouse_wheel_zooms(self, viewer):
        z = viewer.zoom()
        self._wheel(viewer, qtc.QPoint(0, 0), qtc.QPoint(0, 120))
        assert viewer.zoom() == pytest.approx(z * viewer.ZOOM_STEP)

    def test_trackpad_scroll_pans(self, viewer):
        viewer.zoom_by(4)
        z = viewer.zoom()
        h = viewer.horizontalScrollBar().value()
        self._wheel(viewer, qtc.QPoint(-30, 0), qtc.QPoint(-90, 0), phase=qtc.Qt.ScrollUpdate)
        assert viewer.zoom() == pytest.approx(z)
        assert viewer.horizontalScrollBar().value() == h + 30

    def test_ctrl_scroll_zooms(self, viewer):
        z = viewer.zoom()
        self._wheel(viewer, qtc.QPoint(0, 10), qtc.QPoint(0, 30), qtc.Qt.ControlModifier,
                    qtc.Qt.ScrollUpdate)
        assert viewer.zoom() > z

    def test_zoom_keeps_point_under_cursor(self, viewer):
        viewer.zoom_by(3)
        anchor = qtc.QPoint(250, 180)
        before = viewer.mapToScene(anchor)
        viewer.zoom_by(1.7, qtc.QPointF(anchor))
        after = viewer.mapToScene(anchor)
        assert abs(before.x() - after.x()) < 1.5 and abs(before.y() - after.y()) < 1.5

    def test_double_click_toggles_fit_and_100(self, viewer):
        viewer.toggle_fit_actual()
        assert viewer.zoom() == pytest.approx(1.0)
        viewer.toggle_fit_actual()
        assert viewer._fit_mode


class TestSettings:
    def test_alignment_choice_roundtrip(self, app, window):
        import src.MainWindow.SettingsWidget as S
        for key in ("standard", "breathing", "shift", "euclidean", "affine", "off"):
            S.store_alignment_choice(key)
            assert S.alignment_choice() == key
        S.store_alignment_choice("shift")
        assert window.inspector.get_algorithm_config().alignment_mode == "translation"
        S.store_alignment_choice("breathing")
        assert window.inspector.get_algorithm_config().alignment_mode == "breathing"

    def test_device_setting(self, app, window):
        import src.MainWindow.SettingsWidget as S
        qs = settings.globalVars["QSettings"]
        # the window fixture stored use_gpu=0 before the first start: CPU
        assert S.gpu_mode() == "cpu"
        assert window.inspector.get_algorithm_config().gpu_mode == "cpu"
        # someone upgrading with "Use GPU" on gets Automatic
        qs.remove("computing/gpu_mode")
        qs.setValue("computing/use_gpu", 1)
        S.Inspector()
        assert S.gpu_mode() == "auto"
        # the Settings window switches it
        window.show_settings()
        dlg = window._settings_dialog
        dlg.mode_combo.setCurrentIndex(dlg.mode_combo.findData("gpu"))
        cfg = window.inspector.get_algorithm_config()
        assert cfg.gpu_mode == "gpu" and cfg.use_gpu
        dlg.mode_combo.setCurrentIndex(dlg.mode_combo.findData("cpu"))
        assert window.inspector.get_algorithm_config().gpu_mode == "cpu"
        dlg.close()

    def test_theme_switch(self, app, window):
        import src.MainWindow.theme as theme
        theme.manager().set_mode("light")
        light = app.palette().color(qtg.QPalette.Window).lightness()
        theme.manager().set_mode("dark")
        dark = app.palette().color(qtg.QPalette.Window).lightness()
        assert light > 200 and dark < 60


class TestExport:
    @pytest.mark.parametrize("fmt,depth,ext,dtype", [
        ("JPG", 8, "jpg", np.uint8), ("PNG", 16, "png", np.uint16),
        ("PNG", 8, "png", np.uint8), ("TIFF", 16, "tif", np.uint16), ("EXR", 32, "exr", np.float32)])
    def test_formats(self, tmp_path, fmt, depth, ext, dtype):
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import src.MainWindow.ImageSavingDialog as Export
        img = (np.random.RandomState(0).rand(20, 30, 3) * 255).astype(np.float32)
        path = tmp_path / f"o é.{ext}"          # non-ASCII path
        try:
            Export.write_image(str(path), img, fmt, depth)
        except IOError:
            if fmt == "EXR":
                pytest.skip("OpenCV built without OpenEXR")
            raise
        data = np.fromfile(str(path), np.uint8)
        back = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        assert back.dtype == dtype and back.shape == (20, 30, 3)
