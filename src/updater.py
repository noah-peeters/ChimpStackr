"""
In-app updates from the project's GitHub releases.

check()   asks the GitHub API for the newest release (only api.github.com is
          contacted; nothing about the user is sent).
download() fetches this platform's asset and verifies its SHA-256 against
          the digest GitHub publishes for it; an asset without a digest is
          not installed automatically.
prepare_install() stages the new version next to the installed one and
          writes a small helper that, once the app has quit, swaps it in
          and relaunches:
            macOS    ChimpStackr.app inside the DMG replaces the running .app
            Windows  the zip's contents replace the install folder
            Linux    the AppImage replaces the running AppImage
          Running from source (or any other layout), updates are only
          announced, with a link to the release page.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field

REPO = "noah-peeters/ChimpStackr"
API = f"https://api.github.com/repos/{REPO}/releases"
RELEASES_PAGE = f"https://github.com/{REPO}/releases"
TIMEOUT = 15
ASSETS = {"darwin": "ChimpStackr-macOS.dmg", "win32": "ChimpStackr-Windows.zip",
          "linux": "ChimpStackr-Linux-x86_64.AppImage"}


class UpdateError(Exception):
    pass


@dataclass
class Asset:
    name: str
    url: str
    size: int
    sha256: str = ""          # hex, empty when GitHub has no digest


@dataclass
class Release:
    tag: str
    version: tuple
    title: str
    notes: str
    page: str
    prerelease: bool
    assets: list = field(default_factory=list)

    def asset(self, platform=sys.platform):
        name = ASSETS.get(platform)
        return next((a for a in self.assets if a.name == name), None)


def parse_version(text):
    """Comparable version: (major, minor, patch, pre, commits).

    "v0.4.1" → (0, 4, 1, 1, 0); "v0.5.0-rc1" sorts before "v0.5.0";
    a dev build "v0.4.0-3-g77e4d09" (3 commits after v0.4.0) sorts after
    v0.4.0. Unparseable text gives None.
    """
    m = re.match(r"v?(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:[-.]?((?:a|b|rc|alpha|beta)\.?\d*))?"
                 r"(?:-(\d+)-g[0-9a-f]+)?(?:\.dirty|-dirty)?$", (text or "").strip())
    if not m:
        return None
    major, minor, patch = (int(m.group(i) or 0) for i in (1, 2, 3))
    pre = 0 if m.group(4) else 1
    commits = int(m.group(5) or 0)
    return (major, minor, patch, pre, commits)


def current_version():
    import src
    return src.__version__


def display_version(text=None):
    """Version for people: "0.4.0", or "0.4.0 (dev build 3578d8c)" for a
    build between releases."""
    text = (text or current_version() or "").strip()
    m = re.match(r"v?(.+?)-(\d+)-g([0-9a-f]+)(?:-dirty)?$", text)
    if m:
        return f"{m.group(1)} (dev build {m.group(3)})"
    return text[1:] if text.startswith("v") and text[1:2].isdigit() else text


def _request(url, accept="application/vnd.github+json"):
    req = urllib.request.Request(url, headers={
        "Accept": accept, "User-Agent": f"ChimpStackr/{current_version()}",
        "X-GitHub-Api-Version": "2022-11-28"})
    return urllib.request.urlopen(req, timeout=TIMEOUT)


def _release_from_json(data):
    assets = []
    for a in data.get("assets", []):
        digest = a.get("digest") or ""
        sha = digest.split(":", 1)[1] if digest.startswith("sha256:") else ""
        assets.append(Asset(a["name"], a["browser_download_url"], int(a.get("size") or 0), sha))
    return Release(tag=data["tag_name"], version=parse_version(data["tag_name"]),
                   title=data.get("name") or data["tag_name"], notes=data.get("body") or "",
                   page=data.get("html_url") or RELEASES_PAGE, prerelease=bool(data.get("prerelease")),
                   assets=assets)


def latest_release(include_prerelease=False):
    """Newest published release (GitHub lists newest first)."""
    try:
        with _request(f"{API}?per_page=10") as resp:
            items = json.load(resp)
    except Exception as exc:
        raise UpdateError(f"Couldn't reach GitHub: {exc}") from exc
    for data in items:
        if data.get("draft") or (data.get("prerelease") and not include_prerelease):
            continue
        rel = _release_from_json(data)
        if rel.version is not None:
            return rel
    return None


def check(include_prerelease=False, current=None):
    """The newer release, or None when this version is the newest."""
    rel = latest_release(include_prerelease)
    mine = parse_version(current or current_version())
    if rel is None or mine is None:
        return None
    return rel if rel.version > mine else None


# ── installing ──
def install_target():
    """(kind, path) of what an update would replace, or (None, None) when
    this copy can't update itself (source checkout, unknown layout, or a
    location the user can't write)."""
    if os.environ.get("APPIMAGE") and sys.platform.startswith("linux"):
        path = os.environ["APPIMAGE"]
        return ("appimage", path) if os.access(os.path.dirname(path), os.W_OK) else (None, None)
    if not getattr(sys, "frozen", False):
        return None, None
    exe = os.path.realpath(sys.executable)
    if sys.platform == "darwin":
        m = re.match(r"(.+?\.app)/Contents/MacOS/", exe)
        if m and os.access(os.path.dirname(m.group(1)), os.W_OK):
            return "mac_app", m.group(1)
        return None, None
    if sys.platform == "win32":
        folder = os.path.dirname(exe)
        return ("windows_dir", folder) if os.access(os.path.dirname(folder), os.W_OK) else (None, None)
    return None, None


def can_install():
    return install_target()[0] is not None


def download(asset, dest_dir, progress=None, cancelled=None, chunk=1 << 20):
    """Download and verify an asset; returns the file path. `progress(done,
    total)` is called as data arrives; `cancelled()` returning True stops."""
    if not asset.sha256:
        raise UpdateError("This release has no checksum, so it can't be installed automatically.")
    path = os.path.join(dest_dir, asset.name)
    h = hashlib.sha256()
    done = 0
    try:
        with _request(asset.url, accept="application/octet-stream") as resp, open(path, "wb") as f:
            total = int(resp.headers.get("Content-Length") or asset.size or 0)
            while True:
                if cancelled is not None and cancelled():
                    raise UpdateError("Download cancelled")
                block = resp.read(chunk)
                if not block:
                    break
                f.write(block)
                h.update(block)
                done += len(block)
                if progress is not None:
                    progress(done, total)
    except UpdateError:
        raise
    except Exception as exc:
        raise UpdateError(f"Download failed: {exc}") from exc
    if h.hexdigest() != asset.sha256.lower():
        os.remove(path)
        raise UpdateError("The download doesn't match its published checksum, so it wasn't installed.")
    return path


def _stage_mac(dmg, target):
    mount = tempfile.mkdtemp(prefix="chimpstackr-update-")
    try:
        subprocess.run(["hdiutil", "attach", "-nobrowse", "-readonly", "-noautoopen",
                        "-mountpoint", mount, dmg], check=True, capture_output=True, timeout=120)
        apps = [n for n in os.listdir(mount) if n.endswith(".app")]
        if not apps:
            raise UpdateError("The disk image doesn't contain the app.")
        staged = target + ".update"
        shutil.rmtree(staged, ignore_errors=True)
        subprocess.run(["ditto", os.path.join(mount, apps[0]), staged], check=True,
                       capture_output=True, timeout=600)
    except subprocess.CalledProcessError as exc:
        raise UpdateError(f"Couldn't unpack the update: {exc.stderr.decode(errors='replace').strip()}") from exc
    finally:
        subprocess.run(["hdiutil", "detach", "-quiet", mount], capture_output=True, timeout=60)
        shutil.rmtree(mount, ignore_errors=True)
    return staged


def _stage_windows(zip_path, target):
    import zipfile
    staged = target + ".update"
    shutil.rmtree(staged, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(staged)
    if not os.path.isfile(os.path.join(staged, "chimpstackr.exe")):
        raise UpdateError("The archive doesn't contain chimpstackr.exe.")
    return staged


def _stage_appimage(path, target):
    staged = target + ".update"
    shutil.move(path, staged)
    os.chmod(staged, 0o755)
    return staged


def prepare_install(downloaded, pid=None):
    """Stage the update and write the helper that swaps it in after this
    process (pid) exits. Returns the helper's argv; run it with launch()."""
    kind, target = install_target()
    if kind is None:
        raise UpdateError("This copy of ChimpStackr can't update itself.")
    pid = pid or os.getpid()
    staged = {"mac_app": _stage_mac, "windows_dir": _stage_windows,
              "appimage": _stage_appimage}[kind](downloaded, target)
    return write_helper(kind, target, staged, pid)


def write_helper(kind, target, staged, pid):
    """Helper script: wait for pid to exit, replace target with staged
    (keeping the old copy until the new one is in place), relaunch."""
    folder = tempfile.mkdtemp(prefix="chimpstackr-updater-")
    old = target + ".old"
    if kind == "windows_dir":
        exe = os.path.join(target, "chimpstackr.exe")
        script = os.path.join(folder, "update.ps1")
        q = lambda p: "'" + p.replace("'", "''") + "'"   # noqa: E731
        with open(script, "w", encoding="utf-8") as f:
            f.write(f"""$ErrorActionPreference = 'Stop'
try {{ Wait-Process -Id {int(pid)} -Timeout 120 }} catch {{}}
Start-Sleep -Milliseconds 500
if (Test-Path {q(old)}) {{ Remove-Item -Recurse -Force {q(old)} }}
Rename-Item {q(target)} {q(os.path.basename(old))}
try {{ Rename-Item {q(staged)} {q(os.path.basename(target))} }}
catch {{ Rename-Item {q(old)} {q(os.path.basename(target))}; throw }}
Remove-Item -Recurse -Force {q(old)} -ErrorAction SilentlyContinue
Start-Process {q(exe)}
""")
        return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden",
                "-File", script]

    import shlex
    q = shlex.quote
    if kind == "mac_app":
        relaunch = f"xattr -dr com.apple.quarantine {q(target)} 2>/dev/null; open {q(target)}"
    else:
        relaunch = f"chmod +x {q(target)}; nohup {q(target)} >/dev/null 2>&1 &"
    script = os.path.join(folder, "update.sh")
    with open(script, "w") as f:
        f.write(f"""#!/bin/sh
# ChimpStackr updater: wait for the app to quit, swap in the new version, relaunch
for i in $(seq 1 600); do kill -0 {int(pid)} 2>/dev/null || break; sleep 0.2; done
rm -rf {q(old)}
mv {q(target)} {q(old)} || exit 1
if mv {q(staged)} {q(target)}; then
  rm -rf {q(old)}
else
  mv {q(old)} {q(target)}
fi
{relaunch}
rm -rf {q(folder)}
""")
    os.chmod(script, 0o755)
    return ["/bin/sh", script]


def launch(argv):
    """Start the helper detached, so it outlives this process."""
    if sys.platform == "win32":
        flags = 0x00000008 | 0x00000200      # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(argv, creationflags=flags, close_fds=True)
    else:
        subprocess.Popen(argv, start_new_session=True, close_fds=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
