# Contributing to ChimpStackr

Thanks for your interest in contributing! This guide covers how to set up the project, run tests, and submit changes.

## Development Setup

```bash
# Clone and create virtual environment
git clone https://github.com/noah-peeters/ChimpStackr.git
cd ChimpStackr
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
pip install pytest

# Verify everything works
pytest tests/ -v
python src/run.py  # Launch GUI
```

**Python version:** 3.9 - 3.13. PySide6 does not yet support 3.14.

## Project Structure

```
ChimpStackr/
  src/
    run.py                  # GUI entry point
    cli.py                  # CLI entry point
    config.py               # AlgorithmConfig dataclass, auto-detection
    settings.py             # Global state (legacy, being reduced)
    ImageLoadingHandler.py  # Image loading (RAW, 16-bit, float32 pipeline)
    algorithms/
      __init__.py           # Algorithm class: alignment, pyramid fusion
      API.py                # LaplacianPyramid API: high-level stack methods
      dft_imreg.py          # DFT-based image registration
      stacking_algorithms/
        cpu.py              # All CPU algorithms (Laplacian, Weighted Avg, Depth Map, Mertens)
    MainWindow/
      __init__.py           # Main window: stacking orchestration, close handling
      QActions.py           # Menus, toolbar, RunButton
      SettingsWidget.py     # Settings sidebar panel
      ProgressBar.py        # Progress bar with time estimation
      ImageSavingDialog.py  # Export dialogs
      RetouchWidget.py      # Retouching engine and UI panel
      icons.py              # SVG icon functions
      style.py              # Qt stylesheet
      MainLayout/
        __init__.py         # Central widget: image lists, viewer, comparison
        ImageWidgets.py     # Loaded/processed image list widgets
        ImageViewers/
          __init__.py       # Image viewer with zoom/pan
          ImageScene.py     # QGraphicsScene for image display
          ComparisonViewer.py  # Before/after slider comparison
  tests/
    test_ImageLoadingHandler.py
    test_algorithm_API.py   # Algorithm, stacking, alignment, config tests
    test_cli.py             # CLI integration tests
  packaging/
    icons/                  # App icons (.ico, .icns, .png at various sizes)
    entitlements.plist      # macOS code signing entitlements
    installer.iss           # Windows Inno Setup script
    chimpstackr.desktop     # Linux desktop entry
  scripts/
    build_macos.sh          # macOS build script
    build_linux.sh          # Linux AppImage build script
    build_windows.ps1       # Windows build script
  chimpstackr.spec          # PyInstaller spec (dual GUI/CLI)
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_algorithm_API.py -v

# Specific test class
pytest tests/test_algorithm_API.py::TestWeightedAverage -v

# With coverage (install pytest-cov first)
pytest tests/ --cov=src --cov-report=html
```

Tests do not require PySide6/Qt -- they test the algorithm layer directly.

## Architecture

### Algorithm Layer (no Qt dependency)

The algorithm layer is fully decoupled from the GUI:

- `config.py` -- `AlgorithmConfig` dataclass with all parameters
- `algorithms/__init__.py` -- `Algorithm` class handles alignment, pyramid generation, fusion
- `algorithms/API.py` -- `LaplacianPyramid` class provides high-level `stack_images()` / `align_and_stack_images()`
- `algorithms/stacking_algorithms/cpu.py` -- Pure NumPy/OpenCV/Numba implementations

You can use the algorithm layer standalone:

```python
from src.algorithms.API import LaplacianPyramid
from src.config import AlgorithmConfig

config = AlgorithmConfig(stacking_method="laplacian", fusion_kernel_size=6)
lp = LaplacianPyramid(config=config)
lp.update_image_paths(["img1.jpg", "img2.jpg", "img3.jpg"])
lp.align_and_stack_images()
# lp.output_image is a float32 numpy array
```

### GUI Layer

- `MainWindow` orchestrates everything: loads images, starts stacking on thread pool, displays results
- Settings panel is a sidebar (not a modal dialog)
- Image preview loads in background thread
- Progress emitted via Qt signals from worker thread

### 16-bit Pipeline

The entire pipeline operates in float32:
1. `ImageLoadingHandler.read_image_as_float32()` -- loads any format to float32 BGR (0-255 range)
2. Alignment operates on float32
3. Stacking operates on float32
4. Output is float32, converted to uint8 only at export time

## Adding a New Stacking Algorithm

1. Add the core function to `src/algorithms/stacking_algorithms/cpu.py`
2. Add a method key to `STACKING_METHODS` in `src/config.py`
3. Add `_align_and_stack_<method>` and `_stack_<method>` to `src/algorithms/API.py`
4. Add the dispatch in `align_and_stack_images()` and `stack_images()`
5. Add to the segmented control in `src/MainWindow/SettingsWidget.py`
6. Add to the name mapping in `src/MainWindow/MainLayout/__init__.py`
7. Add tests in `tests/test_algorithm_API.py`

## Packaging

See [docs/packaging.md](docs/packaging.md) for detailed build instructions.

## Code Style

- No strict formatter enforced, but keep consistent with existing code
- Type hints welcome but not required
- Tests required for new algorithm features
- Keep the algorithm layer free of Qt imports
