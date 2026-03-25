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
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# ── Collect problematic dependencies ──
# Numba/llvmlite need special handling — their native libs are often missed
numba_datas, numba_binaries, numba_hiddenimports = collect_all('numba')
pyfftw_datas, pyfftw_binaries, pyfftw_hiddenimports = collect_all('pyfftw')
rawpy_datas, rawpy_binaries, rawpy_hiddenimports = collect_all('rawpy')
imageio_datas, imageio_binaries, imageio_hiddenimports = collect_all('imageio')
scipy_datas, scipy_binaries, scipy_hiddenimports = collect_all('scipy')

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

# ── Binary files ──
binaries = []
binaries += numba_binaries
binaries += pyfftw_binaries
binaries += rawpy_binaries
binaries += imageio_binaries
binaries += scipy_binaries

# ── Hidden imports ──
hiddenimports = [
    'src', 'src.settings', 'src.config', 'src.cli', 'src.run',
    'src.algorithms', 'src.algorithms.API', 'src.algorithms.dft_imreg',
    'src.algorithms.stacking_algorithms', 'src.algorithms.stacking_algorithms.cpu',
    'src.MainWindow', 'src.MainWindow.icons', 'src.MainWindow.style',
    'src.ImageLoadingHandler',
    'cv2', 'numpy', 'scipy', 'scipy.ndimage', 'scipy.ndimage.interpolation',
    'imageio', 'imageio.v2',
]
hiddenimports += numba_hiddenimports
hiddenimports += pyfftw_hiddenimports
hiddenimports += rawpy_hiddenimports
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
            'CFBundleShortVersionString': '0.0.25',
            'CFBundleVersion': '0.0.25',
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
