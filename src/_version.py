"""
Single source of truth for ChimpStackr version.

At build time (PyInstaller), the version is baked into _version_static.py
so that git isn't needed at runtime. During development, we read from git tags.
"""
import subprocess
import os

# Try static version first (set by build process)
try:
    from src._version_static import __version__
except ImportError:
    __version__ = None

if __version__ is None:
    # Development mode: read from git
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--always"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode == 0:
            __version__ = result.stdout.strip()
        else:
            __version__ = "dev"
    except Exception:
        __version__ = "dev"


def _pep440(version):
    """Packaging-safe (PEP 440) form of the version, for pip installs.

    `git describe` gives e.g. "v0.3.6-2-g7bebe23", which pip rejects;
    this maps it to "0.3.6.post2+g7bebe23". The display string above is
    left unchanged.
    """
    import re
    m = re.match(r"v?(\d+(?:\.\d+)*)(?:-(\d+)-(g[0-9a-f]+))?$", version or "")
    if not m:
        return "0.0.0+dev"
    base, commits, sha = m.groups()
    return base + (f".post{commits}+{sha}" if commits else "")


__pep440_version__ = _pep440(__version__)
