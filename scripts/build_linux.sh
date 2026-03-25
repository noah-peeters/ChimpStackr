#!/usr/bin/env bash
# Build ChimpStackr for Linux.
# Creates AppImage in dist/
#
# Usage:
#   ./scripts/build_linux.sh
#
# Requires: Python 3.10+, FUSE (for AppImage)
#
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== ChimpStackr Linux Build ==="

# ── Clean previous build ──
rm -rf build/ dist/

# ── Create clean venv ──
echo "Creating clean build environment..."
python3 -m venv .venv-build
source .venv-build/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install pyinstaller -q

# ── Build with PyInstaller ──
echo "Building with PyInstaller..."
pyinstaller chimpstackr.spec --noconfirm

echo "Build complete: dist/chimpstackr/"

# ── Create AppImage ──
echo "Creating AppImage..."

APPDIR="dist/ChimpStackr.AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
mkdir -p "$APPDIR/usr/share/applications"
mkdir -p "$APPDIR/usr/share/icons/hicolor/512x512/apps"

# Copy PyInstaller output
cp -r dist/chimpstackr/* "$APPDIR/usr/bin/"

# Desktop entry
cat > "$APPDIR/usr/share/applications/chimpstackr.desktop" << 'DESKTOP'
[Desktop Entry]
Name=ChimpStackr
GenericName=Focus stacking app
Comment=Easily focus stack images
Exec=chimpstackr
Icon=chimpstackr
Terminal=false
Type=Application
Categories=Graphics;Photography;
StartupNotify=true
DESKTOP

# Copy desktop file to root (AppImage requires it)
cp "$APPDIR/usr/share/applications/chimpstackr.desktop" "$APPDIR/"

# Icons
cp packaging/icons/icon_512x512.png "$APPDIR/usr/share/icons/hicolor/512x512/apps/chimpstackr.png"
cp packaging/icons/icon_256x256.png "$APPDIR/chimpstackr.png"

# AppRun script
cat > "$APPDIR/AppRun" << 'APPRUN'
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
export PATH="$HERE/usr/bin:$PATH"
export LD_LIBRARY_PATH="$HERE/usr/bin:$LD_LIBRARY_PATH"

if [ "$1" = "--cli" ] || [ "$(basename "$0")" = "chimpstackr-cli" ]; then
    shift 2>/dev/null || true
    exec "$HERE/usr/bin/chimpstackr-cli" "$@"
else
    exec "$HERE/usr/bin/chimpstackr" "$@"
fi
APPRUN
chmod +x "$APPDIR/AppRun"

# Download appimagetool if not present
if ! command -v appimagetool &>/dev/null; then
    echo "Downloading appimagetool..."
    ARCH=$(uname -m)
    wget -q "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-${ARCH}.AppImage" \
        -O /tmp/appimagetool
    chmod +x /tmp/appimagetool
    APPIMAGETOOL="/tmp/appimagetool"
else
    APPIMAGETOOL="appimagetool"
fi

# Build AppImage
ARCH=$(uname -m) "$APPIMAGETOOL" "$APPDIR" "dist/ChimpStackr-Linux-$(uname -m).AppImage"

# ── Cleanup ──
deactivate
rm -rf .venv-build "$APPDIR"

echo "=== Done ==="
ls -lh dist/ChimpStackr-Linux-*.AppImage
