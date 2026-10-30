# ChimpStackr

![GitHub all releases](https://img.shields.io/github/downloads/noah-peeters/ChimpStackr/total) ![GitHub release (latest by date)](https://img.shields.io/github/downloads/noah-peeters/ChimpStackr/latest/total) ![GitHub](https://img.shields.io/github/license/noah-peeters/ChimpStackr) ![GitHub commits since latest release (by date)](https://img.shields.io/github/commits-since/noah-peeters/ChimpStackr/latest)

<p align="center">
  <img src="packaging/icons/chimpstackr_icon.png" width="200"/>
</p>

Open-source focus stacking application for Windows, macOS, and Linux.

## Features

- **5 stacking algorithms:** Laplacian Pyramid, Hybrid (pyramid detail without halos), Weighted Average, Depth Map, Exposure Fusion (HDR)
- **Slabbing:** Stack very deep stacks in overlapping chunks
- **Automatic alignment:** Translation, Euclidean (rotation), Similarity (rotation + scale), Affine correction, or Breathing (sequential chained alignment for focus breathing compensation)
- **GPU acceleration:** every stacking method and the alignment run on any GPU via OpenCL (NVIDIA, AMD, Intel, Apple): a 50-frame 24 MP stack in under 11 s. By default ChimpStackr picks the GPU or the CPU per stack, whichever it measured faster on your computer (see below)
- **16-bit pipeline:** Full bit-depth preservation from RAW to output
- **Auto-crop:** Removes black edges from alignment shifts
- **Auto-tuning:** Parameters auto-detected from image resolution
- **GUI + CLI:** Full graphical interface and headless command-line tool
- **Cross-platform:** Native builds for Windows, macOS, Linux
- **Pause/resume/cancel:** Control long-running stacks
- **Before/after comparison:** Draggable divider over the result, at any zoom
- **Drag & drop:** Drop image files or folders onto the window (or the Dock icon on macOS)
- **Trackpad, mouse and touch:** Pinch to zoom, two-finger pan, wheel zoom at the cursor, double-click or double-tap for Fit / 100 %
- **Light and dark:** Follows the system appearance, or pick one in Settings
- **Automatic updates:** Optional daily check for new releases; downloads are verified against GitHub's SHA-256 and installed in place, then the app relaunches (macOS app, Windows folder, Linux AppImage)

## Using the app

Drop a folder of frames onto the window (or **Add Images…**), pick a method in the inspector on the right, and press **Stack**. Results appear under *Results* in the sidebar; **Export…** saves JPEG, PNG, TIFF or OpenEXR (16-bit PNG/TIFF keep the full precision of the stack).

| Action | macOS | Windows / Linux |
|---|---|---|
| Add images / folder | ⌘O / ⇧⌘O | Ctrl+O / Ctrl+Shift+O |
| Stack | ⌘↩ | Ctrl+Enter |
| Stop stacking | ⌘. or Esc | Esc |
| Export | ⌘E | Ctrl+E |
| Compare before / after | \\ or Y | \\ or Y |
| Zoom in / out | ⌘+ / ⌘− (or + / −) | Ctrl++ / Ctrl+− (or + / −) |
| Fit / 100 % | ⌘0 / ⌘1 (or 0 / 1) | Ctrl+0 / Ctrl+1 (or 0 / 1) |
| Previous / next frame | ← / → | ← / → |
| Remove selected frames | ⌫ | Delete |
| Show sidebar / inspector | ⌥⌘S / ⌥⌘I | Ctrl+Alt+S / Ctrl+Alt+I |
| Settings | ⌘, | Ctrl+, |

## Download

Pre-built packages are available on the [Releases](https://github.com/noah-peeters/ChimpStackr/releases) page:

| Platform | Download | Notes |
|---|---|---|
| **Windows** | `ChimpStackr-Windows.zip` | Extract and run `chimpstackr.exe` |
| **macOS** | `ChimpStackr-macOS.dmg` | Open DMG, drag to Applications |
| **Linux (AppImage)** | `ChimpStackr-Linux-x86_64.AppImage` | `chmod +x` and run |

## CLI Usage

The CLI allows headless focus stacking without a GUI:

```bash
# Basic stack
chimpstackr-cli --input images/*.jpg --output result.tif

# Align + stack with auto parameters
chimpstackr-cli -i images/*.jpg -o result.tif --align --auto --auto-crop

# Full options
chimpstackr-cli -i images/*.jpg -o result.png \
  --align \
  --method laplacian \
  --rotation-scale \
  --kernel-size 6 \
  --pyramid-levels 8 \
  --auto-crop \
  --quality-report

# Strong focus breathing (e.g. macro lens racked by focus): sequential
# chained alignment with a 5% max scale change per frame pair
chimpstackr-cli -i images/*.jpg -o result.tif \
  --align --alignment-mode breathing --alignment-ref middle --max-scale-step 0.05

# Halo-free detail; very deep stack fused in chunks of 40 frames
chimpstackr-cli -i images/*.jpg -o result.tif --align --method hybrid --slab-size 40
```

**Available methods:** `laplacian` (default), `hybrid`, `weighted_average`, `depth_map`, `exposure_fusion`

### GPU or CPU

The app (Settings → Processing) and the CLI share one choice:

| | App | CLI |
|---|---|---|
| Automatic (default) | *Automatic* | no flag |
| Always the GPU (falls back to the CPU if it fails) | *Always GPU* | `--gpu` |
| Always the CPU | *Always CPU* | `--cpu` |

Automatic uses the CPU when no GPU works or the stack doesn't fit in the GPU's memory, and when a short test on your computer finds the CPU clearly (1.5×) faster for that frame size and count; otherwise the GPU. The test aligns and stacks a few synthetic frames at two sizes on both devices, takes about 15 s, runs once per method (the first time you stack with it), and is repeated after a new GPU, driver or ChimpStackr version. Stacks of small frames (under 1 MP) don't start the test; they use the CPU until a measurement exists. On a fast CPU with a modest GPU (an M4 MacBook Air) this means the GPU for nearly everything except tiny stacks; it matters most on computers with a weak GPU. The CLI prints its choice, e.g. `Device: Auto: GPU · Apple M4 (measured 2.4× faster)`. The measurements are kept in `~/Library/Caches/ChimpStackr` (macOS), `%LOCALAPPDATA%\ChimpStackr\Cache` (Windows) or `~/.cache/chimpstackr` (Linux); Settings → Processing → Measure Again forgets them.

## Stacking Algorithms

| Method | Best for | How it works |
|---|---|---|
| **Pyramid** | Fine detail (hairs, bristles, edges) | Laplacian pyramid decomposition, max-contrast selection per frequency band, local tone-mapping |
| **Hybrid** | Fine detail with clean backgrounds | Max-contrast selection on fine bands, sharpness-weighted blend on coarse bands (no halos) |
| **Weighted** | Smooth subjects, good color | Per-pixel contrast weighting with proper accumulation |
| **Depth Map** | Opaque surfaces, best color fidelity | Multi-scale sharpness with edge-aware bilateral smoothing |
| **HDR** | Varying exposure/lighting | Mertens exposure fusion (not for focus stacking) |

## Build from Source

Requires Python 3.9-3.13.

```bash
git clone https://github.com/noah-peeters/ChimpStackr.git
cd ChimpStackr
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run GUI
python src/run.py

# Run CLI
python -m src.cli --help

# Run tests and lint
pip install -e ".[dev]"
pytest
ruff check .
```

## Packaging

Builds use PyInstaller with platform-specific post-processing. You can only build for your current platform.

```bash
# Install build tools
pip install pyinstaller

# Build (creates dist/chimpstackr/ and dist/ChimpStackr.app on macOS)
pyinstaller chimpstackr.spec --noconfirm

# Or use the platform scripts:
./scripts/build_macos.sh        # macOS → .dmg
./scripts/build_linux.sh        # Linux → .AppImage
.\scripts\build_windows.ps1     # Windows → .zip or installer
```

CI/CD automatically builds all platforms on tagged releases via GitHub Actions.

## Gallery

The following stacks were taken at ~4x magnification on a slightly wobbly rig (~150 images each), stacked with ChimpStackr and post-processed in [darktable](https://www.darktable.org/).

![Bij_TranslationAlignment](https://user-images.githubusercontent.com/17707805/196990942-413ea35c-2abb-4bce-9807-3f3d6b3de3c5.jpg)
![Edited](https://user-images.githubusercontent.com/17707805/196991117-dc4f1c76-cc87-4ef1-92ee-9a7484c7ff07.jpg)
![Bewerkt](https://user-images.githubusercontent.com/17707805/196996295-9fb6c365-ef10-4ef5-b451-1a7269156e53.jpg)

## Sources

- Focus stacking algorithm based on: Wang, W., & Chang, F. (2011). A Multi-focus Image Fusion Method Based on Laplacian Pyramid. *Journal of Computers*, 6(12).
- DFT image alignment adapted from: [imreg_dft](https://github.com/matejak/imreg_dft)
- Mertens exposure fusion: Mertens, T., Kautz, J., & Van Reeth, F. (2007). Exposure Fusion.
- Sum Modified Laplacian focus measure: Nayar, S.K., & Nakagawa, Y. (1994).

## License

GPL-3.0 - see [LICENSE](LICENSE) for details.
