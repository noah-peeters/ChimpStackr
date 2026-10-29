"""In-app updater: version ordering, release parsing, verified downloads,
staging and the swap-and-relaunch helper."""
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.updater as updater  # noqa: E402

V = updater.parse_version


class TestVersions:
    def test_ordering(self):
        assert V("v0.4.1") > V("v0.4.0")
        assert V("v0.10.0") > V("v0.9.9")
        assert V("v0.5.0") > V("v0.5.0-rc1") > V("v0.4.9")
        assert V("v0.4.0-3-g77e4d09") > V("v0.4.0")      # dev build after a tag
        assert V("v0.4.0-3-g77e4d09") < V("v0.4.1")
        assert V("0.4") == V("v0.4.0")
        assert V("dev") is None and V("77e4d09") is None

    def test_display(self):
        assert updater.display_version("v0.4.1") == "0.4.1"
        assert updater.display_version("v0.4.0-3-g77e4d09") == "0.4.0 (dev build 77e4d09)"
        assert updater.display_version("dev") == "dev"

    def test_check_compares_with_current(self, monkeypatch):
        rel = updater._release_from_json(RELEASE_JSON)
        monkeypatch.setattr(updater, "latest_release", lambda pre=False: rel)
        assert updater.check(current="v0.4.0") is rel
        assert updater.check(current="v0.4.1") is None
        assert updater.check(current="v0.4.1-2-gabc1234") is None
        assert updater.check(current="dev") is None


RELEASE_JSON = {
    "tag_name": "v0.4.1", "name": "v0.4.1", "body": "Fixes", "prerelease": False,
    "html_url": "https://github.com/noah-peeters/ChimpStackr/releases/tag/v0.4.1",
    "assets": [
        {"name": "ChimpStackr-macOS.dmg", "size": 3, "digest": "sha256:" + "ab" * 32,
         "browser_download_url": "https://example.invalid/mac.dmg"},
        {"name": "ChimpStackr-Windows.zip", "size": 3, "digest": None,
         "browser_download_url": "https://example.invalid/win.zip"},
    ],
}


class TestRelease:
    def test_parse_assets(self):
        rel = updater._release_from_json(RELEASE_JSON)
        assert rel.version == V("v0.4.1")
        mac = rel.asset("darwin")
        assert mac.name == "ChimpStackr-macOS.dmg" and mac.sha256 == "ab" * 32
        assert rel.asset("win32").sha256 == ""
        assert rel.asset("linux") is None


def _asset_for(path):
    data = pathlib.Path(path).read_bytes()
    return updater.Asset(os.path.basename(path), pathlib.Path(path).as_uri(), len(data),
                         hashlib.sha256(data).hexdigest())


class TestDownload:
    def test_verified_download(self, tmp_path):
        src = tmp_path / "src" / "ChimpStackr-macOS.dmg"
        src.parent.mkdir()
        src.write_bytes(os.urandom(300_000))
        seen = []
        out = updater.download(_asset_for(src), str(tmp_path), lambda d, t: seen.append(d), chunk=65536)
        assert pathlib.Path(out).read_bytes() == src.read_bytes()
        assert seen[-1] == 300_000 and len(seen) > 1

    def test_checksum_mismatch_is_rejected(self, tmp_path):
        src = tmp_path / "src" / "x.dmg"
        src.parent.mkdir()
        src.write_bytes(b"genuine")
        asset = _asset_for(src)
        src.write_bytes(b"tampered")
        with pytest.raises(updater.UpdateError, match="checksum"):
            updater.download(asset, str(tmp_path))
        assert not (tmp_path / "x.dmg").exists()

    def test_no_digest_is_not_installed(self, tmp_path):
        asset = updater.Asset("x", "file:///nonexistent", 1, "")
        with pytest.raises(updater.UpdateError, match="checksum"):
            updater.download(asset, str(tmp_path))

    def test_cancel(self, tmp_path):
        src = tmp_path / "src" / "x.dmg"
        src.parent.mkdir()
        src.write_bytes(os.urandom(200_000))
        with pytest.raises(updater.UpdateError, match="cancelled"):
            updater.download(_asset_for(src), str(tmp_path), cancelled=lambda: True)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX helper")
def test_helper_swaps_and_relaunches(tmp_path):
    target = tmp_path / "ChimpStackr.AppImage"
    target.write_text("#!/bin/sh\necho old\n")
    staged = tmp_path / "ChimpStackr.AppImage.update"
    marker = tmp_path / "relaunched"
    staged.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    staged.chmod(0o755)
    app = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.6)"])
    argv = updater.write_helper("appimage", str(target), str(staged), app.pid)
    updater.launch(argv)
    time.sleep(0.2)
    assert "old" in target.read_text()           # waits for the app to quit
    app.wait()
    deadline = time.time() + 10
    while not marker.exists() and time.time() < deadline:
        time.sleep(0.1)
    assert marker.exists()                        # new version started
    assert "touch" in target.read_text()
    assert not staged.exists() and not (tmp_path / "ChimpStackr.AppImage.old").exists()


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("hdiutil"), reason="macOS DMG")
def test_stage_mac_dmg(tmp_path):
    app = tmp_path / "src" / "ChimpStackr.app" / "Contents" / "MacOS"
    app.mkdir(parents=True)
    (app / "chimpstackr").write_text("new")
    dmg = tmp_path / "ChimpStackr-macOS.dmg"
    subprocess.run(["hdiutil", "create", "-quiet", "-volname", "ChimpStackr", "-srcfolder",
                    str(tmp_path / "src" / "ChimpStackr.app"), "-format", "UDZO", str(dmg)], check=True)
    target = tmp_path / "Applications" / "ChimpStackr.app"
    (target / "Contents" / "MacOS").mkdir(parents=True)
    staged = updater._stage_mac(str(dmg), str(target))
    assert pathlib.Path(staged, "Contents", "MacOS", "chimpstackr").read_text() == "new"
    info = subprocess.run(["hdiutil", "info"], capture_output=True, text=True).stdout
    assert str(dmg) not in info                   # the image was detached again


def test_source_checkout_cannot_self_update(monkeypatch):
    monkeypatch.delenv("APPIMAGE", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert updater.install_target() == (None, None)
    assert not updater.can_install()
