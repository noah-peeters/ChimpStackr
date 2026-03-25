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
