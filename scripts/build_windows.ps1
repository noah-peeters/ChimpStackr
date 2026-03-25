# Build ChimpStackr for Windows.
# Creates installer in dist/
#
# Usage:
#   .\scripts\build_windows.ps1
#
# Requires: Python 3.10+, Inno Setup (optional, for installer)
#
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "=== ChimpStackr Windows Build ==="

# ── Clean previous build ──
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }

# ── Create clean venv ──
Write-Host "Creating clean build environment..."
python -m venv .venv-build
.\.venv-build\Scripts\Activate.ps1
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install pyinstaller -q

# ── Build with PyInstaller ──
Write-Host "Building with PyInstaller..."
pyinstaller chimpstackr.spec --noconfirm

Write-Host "Build complete: dist\chimpstackr\"

# ── Create Inno Setup installer (if available) ──
$InnoSetup = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if (Test-Path $InnoSetup) {
    Write-Host "Creating installer with Inno Setup..."
    & $InnoSetup "packaging\installer.iss"
    Write-Host "Installer created in dist\"
} else {
    Write-Host "Inno Setup not found - skipping installer creation"
    Write-Host "Install from: https://jrsoftware.org/isdownload.php"

    # Create a simple ZIP as fallback
    Write-Host "Creating ZIP archive..."
    Compress-Archive -Path "dist\chimpstackr\*" -DestinationPath "dist\ChimpStackr-Windows.zip" -Force
    Write-Host "ZIP: dist\ChimpStackr-Windows.zip"
}

# ── Cleanup ──
deactivate
Remove-Item -Recurse -Force ".venv-build"

Write-Host "=== Done ==="
