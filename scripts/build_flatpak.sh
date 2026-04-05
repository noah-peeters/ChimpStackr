#!/usr/bin/env bash
# Build ChimpStackr as a Flatpak bundle.
#
# Usage:
#   ./scripts/build_flatpak.sh              # Build and install locally
#   ./scripts/build_flatpak.sh --bundle     # Also create a .flatpak bundle file
#
# Prerequisites:
#   flatpak
#   flatpak-builder
#   org.kde.Platform//6.7 and org.kde.Sdk//6.7 runtimes
#
# Install runtimes:
#   flatpak install flathub org.kde.Platform//6.7 org.kde.Sdk//6.7
#
set -euo pipefail
cd "$(dirname "$0")/.."

APP_ID="io.github.noah_peeters.ChimpStackr"
MANIFEST="packaging/flatpak/${APP_ID}.yml"
BUILD_DIR=".flatpak-build"
REPO_DIR=".flatpak-repo"
BUNDLE_FLAG="${1:-}"

echo "=== ChimpStackr Flatpak Build ==="

# ── Verify prerequisites ──
if ! command -v flatpak-builder &>/dev/null; then
    echo "Error: flatpak-builder not found. Install it with:"
    echo "  sudo apt install flatpak-builder    # Debian/Ubuntu"
    echo "  sudo dnf install flatpak-builder    # Fedora"
    exit 1
fi

# ── Install runtimes if missing ──
if ! flatpak info org.kde.Sdk//6.7 &>/dev/null 2>&1; then
    echo "Installing KDE SDK runtime..."
    flatpak install -y flathub org.kde.Platform//6.7 org.kde.Sdk//6.7
fi

# ── Clean previous build ──
rm -rf "$BUILD_DIR"

# ── Build ──
echo "Building Flatpak..."
flatpak-builder --force-clean --repo="$REPO_DIR" "$BUILD_DIR" "$MANIFEST"

# ── Install locally for testing ──
echo "Installing locally..."
flatpak --user remote-add --if-not-exists chimpstackr-local "$REPO_DIR" --no-gpg-verify
flatpak --user install -y --reinstall chimpstackr-local "$APP_ID"

echo ""
echo "Installed. Run with:"
echo "  flatpak run $APP_ID"
echo "  flatpak run $APP_ID --cli --help    # CLI mode"

# ── Optionally create a redistributable bundle ──
if [ "$BUNDLE_FLAG" = "--bundle" ]; then
    echo ""
    echo "Creating Flatpak bundle..."
    flatpak build-bundle "$REPO_DIR" "dist/ChimpStackr-Linux.flatpak" "$APP_ID"
    echo "Bundle created: dist/ChimpStackr-Linux.flatpak"
    echo ""
    echo "Users can install with:"
    echo "  flatpak install ChimpStackr-Linux.flatpak"
fi

# ── Cleanup build artifacts ──
rm -rf "$BUILD_DIR"

echo "=== Done ==="
