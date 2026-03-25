# Packaging & Distribution

ChimpStackr uses PyInstaller to create native packages for all platforms. Builds are automated via GitHub Actions but can also be done locally.

## Quick Reference

| Platform | Build command | Output |
|---|---|---|
| macOS | `./scripts/build_macos.sh` | `dist/ChimpStackr-macOS.dmg` |
| Windows | `.\scripts\build_windows.ps1` | `dist/ChimpStackr-Windows.zip` |
| Linux | `./scripts/build_linux.sh` | `dist/ChimpStackr-Linux-x86_64.AppImage` |

## Prerequisites

- Python 3.9-3.13
- All dependencies from `requirements.txt`
- PyInstaller (`pip install pyinstaller`)
- Platform-specific:
  - **macOS**: Xcode command line tools (`xcode-select --install`)
  - **Windows**: [Inno Setup 6](https://jrsoftware.org/isdownload.php) (optional, for installer)
  - **Linux**: FUSE (`sudo apt install libfuse2`)

## Local Build

```bash
# Quick build (all platforms)
pip install pyinstaller
pyinstaller chimpstackr.spec --noconfirm

# Output:
#   dist/chimpstackr/          -- directory with both executables
#   dist/chimpstackr/chimpstackr       -- GUI executable
#   dist/chimpstackr/chimpstackr-cli   -- CLI executable
#   dist/ChimpStackr.app/      -- macOS .app bundle (macOS only)
```

### Verify the build

```bash
# Test CLI
dist/chimpstackr/chimpstackr-cli --help
dist/chimpstackr/chimpstackr-cli -i /path/to/images/*.jpg -o /tmp/test.jpg

# Test GUI
open dist/ChimpStackr.app          # macOS
dist/chimpstackr/chimpstackr       # Linux
dist\chimpstackr\chimpstackr.exe   # Windows
```

## CI/CD (GitHub Actions)

The build workflow (`.github/workflows/build.yml`) triggers on:
- **Tag push** (`v*`): builds all platforms and creates a draft GitHub Release
- **Manual dispatch**: from the Actions tab, with optional release creation

### Triggering a release

```bash
# Tag and push
git tag v0.1.0
git push origin v0.1.0
```

This runs:
1. **Test** on all 3 platforms (Python 3.11)
2. **Build macOS** -- PyInstaller, optional code signing, DMG creation
3. **Build Windows** -- PyInstaller, ZIP archive
4. **Build Linux** -- PyInstaller, AppImage creation
5. **Release** -- draft GitHub Release with all 3 artifacts

### GitHub Secrets (optional)

For macOS code signing and notarization, set these repository secrets:

| Secret | Description |
|---|---|
| `MACOS_CODESIGN_IDENTITY` | `Developer ID Application: Your Name (TEAMID)` |
| `APPLE_ID` | Your Apple ID email |
| `APPLE_TEAM_ID` | Apple Developer Team ID |
| `APPLE_APP_PASSWORD` | App-specific password from appleid.apple.com |

Without these, the macOS build still works but produces an unsigned DMG (users will need to right-click > Open on first launch).

## Architecture

### Dual executables

The `.spec` file creates two executables from shared libraries:

- **chimpstackr** (GUI): `console=False` -- no terminal window
- **chimpstackr-cli** (CLI): `console=True` -- headless terminal mode

Both share the same `_internal/` directory with all dependencies, so they're only bundled once.

### macOS .app bundle

On macOS, PyInstaller also creates a `ChimpStackr.app` bundle with:
- Proper `Info.plist` (bundle ID, version, dark mode support, file associations)
- `.icns` icon
- Both executables in `Contents/MacOS/`

### Linux AppImage

The build script wraps the PyInstaller output in an AppImage:
- `AppRun` script detects `--cli` flag or binary name to dispatch to GUI or CLI
- Desktop entry and icons included for desktop integration
- Works on most distros with glibc 2.31+

### Heavy dependencies

These need special handling in the `.spec` file:

| Dependency | Issue | Solution |
|---|---|---|
| numba/llvmlite | Native JIT compiler libs often missed | `collect_all('numba')` |
| pyfftw | FFTW3 C libraries | `collect_all('pyfftw')` |
| rawpy | libraw native lib | `collect_all('rawpy')` |
| imageio | Package metadata needed at runtime | `collect_all('imageio')` |
| scipy | Many submodules with C extensions | `collect_all('scipy')` |

### macOS entitlements

The `packaging/entitlements.plist` grants these hardened runtime exceptions:
- `allow-jit` -- required for Numba JIT compilation
- `allow-unsigned-executable-memory` -- required for NumPy/SciPy C extensions
- `disable-library-validation` -- required for loading bundled `.dylib` files

## Troubleshooting

### "No module named X" at runtime

Add it to `hiddenimports` or use `collect_all('X')` in the `.spec` file, rebuild.

### macOS "app is damaged" / "unidentified developer"

Unsigned builds trigger Gatekeeper. Users can: right-click > Open, or run:
```bash
xattr -cr /Applications/ChimpStackr.app
```

### Linux AppImage won't run

Install FUSE: `sudo apt install libfuse2`

### Windows antivirus flags the exe

This is common with PyInstaller. Options:
- Sign the exe with a code-signing certificate
- Submit to antivirus vendors as false positive
- Don't use UPX compression (already disabled in our spec)

### Build is too large

The ~450MB uncompressed size is normal for NumPy + SciPy + Numba + PySide6. The DMG compresses to ~200MB. To reduce:
- Use `--exclude-module` for unused scipy submodules
- Consider Nuitka for 20-30% smaller binaries (future optimization)
