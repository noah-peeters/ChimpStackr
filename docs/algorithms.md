---
layout: default
title: Stacking Algorithms
description: How each of ChimpStackr's four fusion methods works
---

# Stacking Algorithms

[&larr; Back to home](index.html)

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

## Focus Breathing Compensation (sequential alignment)

**Best for:** Lenses with strong focus breathing, i.e. macro lenses and long lenses/macro zooms where the focal length (and therefore magnification) changes as you rack focus, so frames drift in scale across the stack.

Traditional single-reference alignment (all frames vs. the first or middle frame) struggles when cumulative scale drift is large: frame 100 vs. frame 1 may differ by 10–20% while frame 100 vs. frame 99 differs by less than 1%. This is the same situation Helicon Focus's autoadjustments handle with sequential alignment.

### How it works

Based on the same idea as Helicon Focus: align each frame to its *neighbour* and compose the chain.

1. For each consecutive pair of frames, estimate a small transform (shift, rotation, uniform scale) via ORB feature matching + ECC refinement. Per-pair deltas are tiny, so matching is much more robust than bridging the whole stack at once.

2. Plausibility check ("autoadjustment limits"): magnification never changes by more than a few percent between consecutive frames. Suspected steps beyond **Breathing max scale** (default 5%) are rejected and re-estimated as translation-only; if that also fails, the previous delta is held (constant breathing rate assumption).

3. Compose the per-pair transforms onto the anchor frame (first, or middle — see *Reference* below) and warp every frame into the anchor's coordinate space with one resample each.

4. Frame-to-frame estimation runs in parallel (each pair is independent); only the cheap composition step is sequential. The number of parallel workers follows the CPU core count, capped by currently available RAM.

5. Frames with different dimensions are matched to the first frame's size (center crop/pad) before estimation and warping, as in the standard modes.

### When to use

- Macro shots racked by focus (strong breathing) — the biggest wins
- Stacks where standard alignment leaves ghosting or progressive blur in off-center corners
- Enable via **Alignment mode → Breathing**, or by setting **Reference → previous** with any non-translation mode
- Costs somewhat more than Similarity (see *Performance* below)

### Parameters

- **Breathing max scale %** (default: 5): maximum plausible magnification change per consecutive pair. Higher = more forgiving of noisy estimates, lower = stricter plausibility gating. Note this is *per pair*, not across the whole stack (cumulative breathing of 10–20% is fine).

- **Max precision (slow)** (default: off): see below.

### Resampling

Every frame is warped once, with **Lanczos resampling + replicated borders**:
bilinear softening is avoided, and replicated edges remove the dark border
bleed that constant-black borders would feed into pyramid fusion. On real
stacks this resampling accounts for nearly all of the sharpness gain over the
standard modes.

### Max precision (optional)

Checking **Max precision (slow)** in the GUI (available whenever Breathing or
Reference=previous is active) or passing `--max-precision` on the CLI refines
each frame pair much harder. It can help on difficult, texture-poor subjects,
but on typical stacks the output is indistinguishable from the default while
taking roughly 5× longer and using about twice the memory.

- **Scale is refined, not just measured once.** The coarse ORB feature match
  only localizes scale to ~0.1%. The high-precision path refines it with a
  6-DOF ECC optimization on **float32 grayscale** (no 8-bit quantization),
  projects the result back onto the 4-DOF similarity manifold (dropping
  shear), then re-refines rotation and translation *at that scale*. On
  synthetic test content this recovers scale to ~0.002%.
- **Full-resolution optimization.** The ECC cascade extends to 4096 px
  (standard modes stop at 2048 px), with a tighter convergence epsilon.
- **Objective drift arbitration.** Every pair also computes a conservative
  3-DOF estimate (scale frozen at the ORB value) and the two candidates are
  compared by cross-correlation, with highpass detail error breaking ties.
  Texture-poor frames automatically fall back to the drift-free candidate.

Both precision modes use deterministic per-pair estimation (seeded RANSAC,
single-threaded OpenCV inside the parallel workers), so results are
reproducible run to run.

### Performance

Measured on a real 50-frame, 24 MP macro stack (16-thread desktop CPU,
Laplacian stacking, CPU path):

| Alignment | Total time | Peak RAM |
|---|---|---|
| Similarity | ~60 s | 4.0 GB |
| Breathing (default) | ~90 s | 6.6 GB |
| Breathing + Max precision | ~405 s | 6.1 GB |

Peak RAM scales with the number of parallel estimation workers, which is
chosen from the CPU core count and the RAM available when the stack starts.

### References

- Sequential alignment with per-step limits: conceptually equivalent to Helicon Focus's autoadjustment limits.
- Transform chaining follows the classical image-stitching approach (e.g. Brown & Lowe 2007, "Automatic Panoramic Image Stitching").

## References

- Burt, P.J. & Adelson, E.H. (1983). The Laplacian Pyramid as a Compact Image Code.
- Burt, P.J. & Adelson, E.H. (1983). A Multiresolution Spline with Application to Image Mosaics.
- Nayar, S.K. & Nakagawa, Y. (1994). Shape from Focus (Sum Modified Laplacian).
- Mertens, T., Kautz, J. & Van Reeth, F. (2007). Exposure Fusion.
- Wang, W. & Chang, F. (2011). A Multi-focus Image Fusion Method Based on Laplacian Pyramid.
- Brown, M. & Lowe, D.G. (2007). Automatic Panoramic Image Stitching using Invariant Features (transform chaining).
