# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for ChimpStackr.
Produces two executables from shared libraries:
  - chimpstackr (GUI, no console window)
  - chimpstackr-cli (CLI, with console)

Build with:
  pyinstaller chimpstackr.spec

Output goes to dist/chimpstackr/
"""
import sys
import os
import subprocess
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# ── Bake version from git tag at build time ──
try:
    _ver = subprocess.check_output(
        ["git", "describe", "--tags", "--always"],
        text=True, timeout=5,
    ).strip()
except Exception:
    _ver = "dev"

# Write static version file so runtime doesn't need git
with open(os.path.join("src", "_version_static.py"), "w") as f:
    f.write(f'__version__ = "{_ver}"\n')

# ── Collect problematic dependencies ──
# Numba/llvmlite need special handling — their native libs are often missed
numba_datas, numba_binaries, numba_hiddenimports = collect_all('numba')
pyfftw_datas, pyfftw_binaries, pyfftw_hiddenimports = collect_all('pyfftw')
rawpy_datas, rawpy_binaries, rawpy_hiddenimports = collect_all('rawpy')
imageio_datas, imageio_binaries, imageio_hiddenimports = collect_all('imageio')
scipy_datas, scipy_binaries, scipy_hiddenimports = collect_all('scipy')
pyopencl_datas, pyopencl_binaries, pyopencl_hiddenimports = collect_all('pyopencl')

# ── Data files ──
datas = [
    ('packaging/icons', 'packaging/icons'),
    ('packaging/chimpstackr.desktop', 'packaging'),
]
datas += numba_datas
datas += pyfftw_datas
datas += rawpy_datas
datas += imageio_datas
datas += scipy_datas
datas += pyopencl_datas

# ── Binary files ──
binaries = []
binaries += numba_binaries
binaries += pyfftw_binaries
binaries += rawpy_binaries
binaries += imageio_binaries
binaries += scipy_binaries
binaries += pyopencl_binaries

# ── Hidden imports ──
hiddenimports = [
    'src', 'src.settings', 'src.config', 'src.cli', 'src.run',
    'src.algorithms', 'src.algorithms.API', 'src.algorithms.dft_imreg',
    'src.algorithms.stacking_algorithms', 'src.algorithms.stacking_algorithms.cpu',
    'src.algorithms.stacking_algorithms.ocl',
    'src.MainWindow', 'src.MainWindow.icons', 'src.MainWindow.style',
    'src.ImageLoadingHandler',
    'cv2', 'numpy', 'scipy', 'scipy.ndimage', 'scipy.ndimage.interpolation',
    'imageio', 'imageio.v2',
]
hiddenimports += numba_hiddenimports
hiddenimports += pyfftw_hiddenimports
hiddenimports += rawpy_hiddenimports
hiddenimports += pyopencl_hiddenimports
hiddenimports += imageio_hiddenimports
hiddenimports += scipy_hiddenimports

# ── Excludes (reduce size) ──
excludes = [
    'PyQt5', 'PyQt6', 'tkinter', 'matplotlib', 'IPython',
    'notebook', 'jupyter', 'sphinx', 'docutils',
]

# ── Shared Analysis (both executables use the same collected files) ──
a = Analysis(
    ['src/run.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Also analyze CLI entry point (merges into same analysis)
a_cli = Analysis(
    ['src/cli.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ── Linux: use the user's desktop stack, don't bundle the build host's ──
# PyInstaller picks these up from the build machine (ubuntu-24.04). At runtime the
# user's own GIO/GVfs modules, GTK input methods, pixbuf loaders and Mesa drivers
# get loaded into the app and need the user's matching versions of these libs.
# Bundling older copies shadows them and crashes newer distros (issue #179:
# Ubuntu 26.04 GVfs needs g_variant_builder_init_static from GLib 2.84+).
# Wheel-vendored libs have hashed names (e.g. libxcb-ad31f5a3.so) and are kept.
if sys.platform.startswith('linux'):
    import re
    _system_libs = re.compile(r'^lib(' + '|'.join([
        # GLib / GTK3 stack (loaded by Qt's gtk3 platform theme). libgthread
        # stays bundled: Qt needs it, it's an empty stub, and openSUSE omits it.
        # libbz2 too: Fedora names it libbz2.so.1, Python's _bz2 wants .so.1.0.
        'glib-2.0', 'gio-2.0', 'gobject-2.0', 'gmodule-2.0',
        'gtk-3', 'gdk-3', 'gdk_pixbuf-2.0', 'atk-1.0', 'atk-bridge-2.0', 'atspi',
        'pango-1.0', 'pangocairo-1.0', 'pangoft2-1.0', 'cairo', 'cairo-gobject',
        'harfbuzz', 'fribidi', 'thai', 'datrie', 'graphite2', 'pixman-1', 'epoxy',
        'fontconfig', 'freetype', 'png16', 'brotlicommon', 'brotlidec', 'expat',
        'ffi', 'pcre2-8', 'mount', 'blkid', 'selinux', 'z', 'zstd',
        # X11 / xcb (shared with the system's Mesa libGL)
        'X11', 'X11-xcb', 'Xau', 'Xdmcp', 'Xext', 'Xfixes', 'Xi', 'Xinerama',
        'Xrandr', 'Xrender', 'Xcursor', 'Xcomposite', 'Xdamage', 'xkbcommon',
        'xcb-glx', 'xcb-randr', 'xcb-render', 'xcb-shm', 'xcb-sync', 'xcb-xfixes',
        # D-Bus, systemd, crypto and Kerberos (deps of the system's GVfs modules)
        'dbus-1', 'systemd', 'gcrypt', 'gpg-error', 'bsd', 'md', 'ssl', 'crypto',
        'gssapi_krb5', 'krb5', 'krb5support', 'k5crypto', 'com_err', 'keyutils',
        'cap', 'lz4', 'lzma', 'uuid',
        # Toolchain runtime (Mesa's LLVM needs the system's newer libstdc++)
        'stdc\\+\\+', 'gcc_s',
    ]) + r')\.so(\.|$)')

    def _drop_system_libs(toc):
        return [e for e in toc if not _system_libs.match(os.path.basename(e[0]))]

    a.binaries = _drop_system_libs(a.binaries)
    a_cli.binaries = _drop_system_libs(a_cli.binaries)

# Merge analyses so shared libs are only collected once
MERGE(
    (a, 'chimpstackr', 'chimpstackr'),
    (a_cli, 'chimpstackr-cli', 'chimpstackr-cli'),
)

# ── GUI Executable ──
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

gui_exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='chimpstackr',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX can cause antivirus false positives
    console=False,  # No terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['packaging/icons/icon.ico'] if sys.platform == 'win32'
         else ['packaging/icons/chimpstackr_icon.icns'] if sys.platform == 'darwin'
         else ['packaging/icons/icon_512x512.png'],
)

# ── CLI Executable ──
pyz_cli = PYZ(a_cli.pure, a_cli.zipped_data, cipher=block_cipher)

cli_exe = EXE(
    pyz_cli,
    a_cli.scripts,
    [],
    exclude_binaries=True,
    name='chimpstackr-cli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # Terminal window for CLI
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# ── Collect into single directory ──
coll = COLLECT(
    gui_exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    cli_exe,
    a_cli.binaries,
    a_cli.zipfiles,
    a_cli.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='chimpstackr',
)

# ── macOS .app bundle (GUI only) ──
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='ChimpStackr.app',
        icon='packaging/icons/chimpstackr_icon.icns',
        bundle_identifier='noah.peeters.chimpstackr',
        info_plist={
            'CFBundleName': 'ChimpStackr',
            'CFBundleDisplayName': 'ChimpStackr',
            'CFBundleShortVersionString': _ver.lstrip('v'),
            'CFBundleVersion': _ver.lstrip('v'),
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,  # Support dark mode
            'LSMinimumSystemVersion': '11.0',
            'CFBundleDocumentTypes': [
                {
                    'CFBundleTypeName': 'Image Files',
                    'CFBundleTypeRole': 'Viewer',
                    'LSItemContentTypes': [
                        'public.image',
                        'public.jpeg',
                        'public.png',
                        'public.tiff',
                        'com.adobe.raw-image',
                    ],
                },
            ],
        },
    )
