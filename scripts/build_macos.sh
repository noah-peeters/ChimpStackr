#!/usr/bin/env bash
# Build ChimpStackr for macOS.
# Creates ChimpStackr.app bundle in dist/
#
# Usage:
#   ./scripts/build_macos.sh
#
# For signed + notarized builds (requires Apple Developer account):
#   CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)" \
#   APPLE_ID="your@email.com" \
#   APPLE_TEAM_ID="TEAMID" \
#   APPLE_APP_PASSWORD="xxxx-xxxx-xxxx-xxxx" \
#   ./scripts/build_macos.sh --sign
#
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== ChimpStackr macOS Build ==="

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

echo "Build complete: dist/ChimpStackr.app"

# ── Code signing (optional) ──
if [[ "${1:-}" == "--sign" ]]; then
    if [[ -z "${CODESIGN_IDENTITY:-}" ]]; then
        echo "Error: CODESIGN_IDENTITY not set" >&2
        exit 1
    fi

    echo "Signing application..."
    APP="dist/ChimpStackr.app"
    ENTITLEMENTS="packaging/entitlements.plist"

    # Sign all .dylib and .so files inside the bundle (inside out)
    find "$APP" -name "*.dylib" -o -name "*.so" | while read lib; do
        codesign --force --options runtime --timestamp \
            --entitlements "$ENTITLEMENTS" \
            --sign "$CODESIGN_IDENTITY" "$lib"
    done

    # Sign the main executables
    codesign --force --options runtime --timestamp \
        --entitlements "$ENTITLEMENTS" \
        --sign "$CODESIGN_IDENTITY" "$APP/Contents/MacOS/chimpstackr"
    codesign --force --options runtime --timestamp \
        --entitlements "$ENTITLEMENTS" \
        --sign "$CODESIGN_IDENTITY" "$APP/Contents/MacOS/chimpstackr-cli"

    # Sign the .app bundle itself
    codesign --force --options runtime --timestamp \
        --entitlements "$ENTITLEMENTS" \
        --sign "$CODESIGN_IDENTITY" "$APP"

    echo "Verifying signature..."
    codesign --verify --deep --strict --verbose=2 "$APP"

    # ── Create DMG ──
    echo "Creating DMG..."
    DMG_NAME="ChimpStackr-macOS.dmg"
    rm -f "dist/$DMG_NAME"
    hdiutil create -volname "ChimpStackr" -srcfolder "dist/ChimpStackr.app" \
        -ov -format UDZO "dist/$DMG_NAME"

    # ── Notarize (optional) ──
    if [[ -n "${APPLE_ID:-}" && -n "${APPLE_TEAM_ID:-}" && -n "${APPLE_APP_PASSWORD:-}" ]]; then
        echo "Submitting for notarization..."
        xcrun notarytool submit "dist/$DMG_NAME" \
            --apple-id "$APPLE_ID" \
            --team-id "$APPLE_TEAM_ID" \
            --password "$APPLE_APP_PASSWORD" \
            --wait

        echo "Stapling notarization ticket..."
        xcrun stapler staple "dist/$DMG_NAME"
        echo "Notarization complete."
    else
        echo "Skipping notarization (APPLE_ID/APPLE_TEAM_ID/APPLE_APP_PASSWORD not set)"
    fi

    echo "Signed DMG: dist/$DMG_NAME"
else
    # Create unsigned DMG
    echo "Creating unsigned DMG..."
    hdiutil create -volname "ChimpStackr" -srcfolder "dist/ChimpStackr.app" \
        -ov -format UDZO "dist/ChimpStackr-macOS.dmg"
    echo "DMG: dist/ChimpStackr-macOS.dmg"
fi

# ── Cleanup ──
deactivate
rm -rf .venv-build

echo "=== Done ==="
ls -lh dist/ChimpStackr-macOS.dmg
