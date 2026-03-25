# Stacking Algorithms

ChimpStackr implements four image fusion methods. This document explains how each works and when to use it.

## Laplacian Pyramid (default)

**Best for:** Fine detail -- hairs, bristles, edges, complex overlapping structures.

Based on Burt & Adelson (1983), similar to Zerene Stacker's PMax.

### How it works

1. Each image is decomposed into a Laplacian pyramid -- a multi-scale representation where each level captures detail at a different spatial frequency (fine texture at level 0, coarse structure at higher levels).

2. At each pyramid level and each pixel, the algorithm compares the two sources and picks the one with higher local contrast (deviation in a kernel-sized neighborhood).

3. The selected pyramid levels are recombined to produce the fused image.

4. A local tone-mapping step (CLAHE on the L channel) compensates for the contrast boost inherent in max-contrast selection.

### Parameters

- **Kernel size** (default: auto-detected): Size of the comparison neighborhood. Larger = smoother transitions but slower. Smaller = sharper but may produce noise.
- **Pyramid levels** (default: auto-detected): Depth of the multi-scale decomposition. More levels capture coarser detail. Auto-detected from image dimensions.
- **Contrast threshold** (default: 1.0): Minimum local contrast to switch between sources. Prevents noise in flat/out-of-focus areas from affecting the result. 0 = disabled.
- **Feather radius** (default: 2): Blurs the binary focus map to create soft transitions between sources instead of hard edges. 0 = hard selection.

### Strengths and weaknesses

- Can take fine detail from one frame and coarse illumination from another at the same pixel location
- Handles overlapping structures (crossing hairs) well
- May accumulate noise (favors high-contrast noise over smooth background)
- Can slightly shift colors due to max-contrast selection

## Weighted Average

**Best for:** Smooth subjects, short stacks, good color fidelity.

Similar to Helicon Focus Method A.

### How it works

1. For each image, compute a per-pixel focus weight using Laplacian energy (higher energy = more in focus).

2. Accumulate weighted pixel sums and weight totals across all images (float64 precision).

3. Divide once at the end to get the properly weighted average.

### Why not pairwise

Earlier versions fused pairs incrementally, which compounded blur because the focus weights on an already-blended result were wrong. The current implementation accumulates all weights before dividing, producing correct results.

### Strengths and weaknesses

- Preserves color well (weighted average, not selection)
- Smooth results with few artifacts
- Cannot represent overlapping structures at different depths (blends them)
- Less sharp than Pyramid for fine detail

## Depth Map

**Best for:** Opaque surfaces with continuous depth, best original color preservation.

Similar to Zerene Stacker's DMap and Helicon Focus Method B.

### How it works

1. For each image, compute a multi-scale sharpness map using the Sum Modified Laplacian (SML) -- the standard focus measure operator used by commercial stackers.

2. Sharpness is evaluated at three window sizes and averaged to reduce blockiness.

3. The sharpness map is smoothed using a bilateral filter -- this preserves sharp depth discontinuities at object edges while smoothing noisy estimates in flat areas.

4. At each pixel, the image with the highest sharpness wins -- no blending.

### Parameters

- **DMap smoothing** (default: 5): Controls the bilateral filter strength. Higher = smoother depth transitions, fewer artifacts. Lower = sharper but more halos at depth boundaries. This is the key quality control -- similar to Zerene's smoothing slider.

### Strengths and weaknesses

- Pixels come directly from original source images -- best color fidelity
- Clean results for smooth opaque surfaces
- Cannot represent overlapping structures (single depth per pixel)
- Can produce halo artifacts at depth discontinuities (controlled by smoothing)

## Exposure Fusion (HDR)

**Best for:** Varying lighting/exposure, NOT for focus stacking.

Uses OpenCV's Mertens exposure fusion algorithm.

### How it works

1. Images are collected in batches of 4.
2. Each batch is fused using Mertens' method -- per-pixel weighting based on contrast, saturation, and exposure quality.
3. Batch results are fused together in a final pass.

### When to use

- Images with varying exposure (bracketed shots)
- Scenes where lighting changed during the stack
- NOT recommended for pure focus stacking (use Pyramid or Depth Map instead)

## Algorithm Comparison

| Aspect | Pyramid | Weighted Avg | Depth Map | HDR |
|---|---|---|---|---|
| Detail preservation | Excellent | Good | Good | Moderate |
| Color fidelity | Good (slight shift) | Very good | Excellent | Good |
| Overlapping structures | Handles well | Blends | Cannot | Blends |
| Artifacts | Noise in flat areas | Few | Halos at edges | Few |
| Speed | Fast | Fast | Fast | Moderate |
| Frame ordering required | No | No | No | No |
| Best stack size | Any | 2-20 | Any | 2-10 |

## References

- Burt, P.J. & Adelson, E.H. (1983). The Laplacian Pyramid as a Compact Image Code.
- Burt, P.J. & Adelson, E.H. (1983). A Multiresolution Spline with Application to Image Mosaics.
- Nayar, S.K. & Nakagawa, Y. (1994). Shape from Focus (Sum Modified Laplacian).
- Mertens, T., Kautz, J. & Van Reeth, F. (2007). Exposure Fusion.
- Wang, W. & Chang, F. (2011). A Multi-focus Image Fusion Method Based on Laplacian Pyramid.
