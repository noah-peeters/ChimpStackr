---
layout: default
title: Stacking Algorithms
description: How each of ChimpStackr's five fusion methods works
---

# Stacking Algorithms

[&larr; Back to home](index.html)

ChimpStackr implements five image fusion methods. This document explains how each works and when to use it.

## Laplacian Pyramid (default)

**Best for:** Fine detail -- hairs, bristles, edges, complex overlapping structures.

Based on Burt & Adelson (1983).

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

## Hybrid

**Best for:** Fine detail with clean backgrounds -- subjects with hairs or edges against smooth bokeh, where the Pyramid method leaves glows or dark rims around edges.

### How it works

Max-contrast selection at every pyramid level is what causes halos: at coarse levels, an out-of-focus frame often has *more* local contrast next to an edge, because the blurred subject spills into the background. Halos only form where there is no in-focus detail, so Hybrid decides per area:

1. **On the subject** (some frame has real fine detail there): every level keeps max-contrast selection, as in the Pyramid method, for full crispness.
2. **Background and the area around edges**: coarse levels and the base are a blend of all frames, weighted by each frame's fine-detail sharpness (sum-modified Laplacian). The sharpness map is aggregated over the area each level covers, so the frame where a nearby edge is actually in focus supplies the low frequencies there -- out-of-focus spill can't leak in.
3. **Flat areas on the fine levels**: where no frame has detail clearly above the noise level (estimated per level from the first frame), frames are blended instead of picking whichever is noisiest.
4. **Contrast**: the fine detail bands get a mild boost (8%) on the subject only, at full precision. This gives the same crispness as the Pyramid method's tone-mapping step without amplifying background noise. Coarse bands are not boosted, since that would itself create halos around strong edges.

The number of fine levels grows with resolution (3 at ~3000 px on the long side, 4 at ~6000 px), since halos are wider in pixels on larger images. Single pass.

### Parameters

- **Kernel size** and **Pyramid levels** as for the Pyramid method.

### Strengths and weaknesses

- Far fewer halos around edges than the Pyramid method, with the same fine-detail crispness
- Much less noise in smooth backgrounds (frames are averaged there)
- Slightly less deep large-scale shadows than the Pyramid method's tone mapping
- Uses more memory than the Pyramid method (~1 GB more on a 24 MP stack)

## Weighted Average

**Best for:** Smooth subjects, short stacks, good color fidelity.

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

### How it works

1. For each image, compute a multi-scale sharpness map using the Sum Modified Laplacian (SML) -- the standard focus measure operator used by commercial stackers.

2. Sharpness is evaluated at three window sizes and averaged to reduce blockiness.

3. The sharpness map is smoothed using a bilateral filter -- this preserves sharp depth discontinuities at object edges while smoothing noisy estimates in flat areas.

4. At each pixel, the image with the highest sharpness wins -- no blending.

### Parameters

- **DMap smoothing** (default: 5): Controls the bilateral filter strength. Higher = smoother depth transitions, fewer artifacts. Lower = sharper but more halos at depth boundaries. This is the key quality control.

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

## GPU stacking

When a GPU is available, stacking and alignment run on it (turn this off with **Use GPU** under Advanced in the GUI, or `--cpu` on the CLI). They run through OpenCL, which every NVIDIA, AMD, Intel and Apple GPU driver provides -- no CUDA or other toolkit needed. This covers every method (with slabbing and the Pyramid contrast threshold) and the alignment estimation.

- Worker threads decode frames; each frame is uploaded once (8-bit when the source is 8-bit), then warped, decomposed and fused on the GPU, and only the final image is read back. Uploads run on their own queue, so they overlap the previous frame's processing.
- Alignment: the CPU only runs ORB feature matching; the pre-warp, the ECC cascade levels and the ECC iterations themselves run on the GPU. With breathing alignment, each frame is stacked as soon as its transform is known, so estimation and stacking overlap.
- The GPU code mirrors the CPU code step by step, so results are the same apart from float rounding, warp interpolation (OpenCV approximates it with 1/32-pixel steps) and the Pyramid tone map's LAB conversion (OpenCV goes through a lookup table; the GPU uses the exact curves).
- Slabbing streams each slab's result straight into the final pass, so there are no temporary files.
- If no OpenCL GPU is found, or anything fails on the GPU (driver error, out of GPU memory), stacking continues on the CPU automatically. Device memory use is roughly 150-200 bytes per image pixel (about 2.5-4 GB for 24 MP frames); with less free GPU memory the driver may swap, which is slow.

A 50-frame 24 MP stack on an RTX 4070 that other apps were using at the same time (total time including alignment):

| Method | Breathing alignment | Similarity alignment |
|---|---|---|
| Pyramid | 6.7 s | 8.9 s |
| Hybrid | 6.7 s | 9.6 s |
| Weighted Average | 6.7 s | 8.7 s |
| Depth Map | 6.6 s | 9.0 s |
| HDR | 10.3 s | 10.7 s |

The CPU path takes 1-2 minutes for the same stack (HDR peaked at 12 GB of RAM there, 2.8 GB on the GPU path).

## Slabbing

For very deep stacks (hundreds of frames), any method except HDR can stack in overlapping chunks ("slabs"), then stack the slab results with the same method.

- Every frame is aligned once to the global reference, so the slab results line up without a second alignment.
- Neighbouring slabs share frames (default 2) so there are no seams between them.
- Slab results are kept in temporary files, so memory use stays flat however deep the stack is.

Set **Slab size** in the GUI, or `--slab-size N` (and optionally `--slab-overlap`) on the CLI. Off by default.

## Algorithm Comparison

| Aspect | Pyramid | Hybrid | Weighted Avg | Depth Map | HDR |
|---|---|---|---|---|---|
| Detail preservation | Excellent | Excellent | Good | Good | Moderate |
| Color fidelity | Good (slight shift) | Very good | Very good | Excellent | Good |
| Overlapping structures | Handles well | Handles well | Blends | Cannot | Blends |
| Artifacts | Halos, noise in flat areas | Few | Few | Patchy flat areas | Few |
| Speed | Fast | Fast | Fast | Fast | Moderate |
| Frame ordering required | No | No | No | No | No |
| Best stack size | Any | Any | 2-20 | Any | 2-10 |

## Focus Breathing Compensation (sequential alignment)

**Best for:** Lenses with strong focus breathing, i.e. macro lenses and long lenses/macro zooms where the focal length (and therefore magnification) changes as you rack focus, so frames drift in scale across the stack.

Traditional single-reference alignment (all frames vs. the first or middle frame) struggles when cumulative scale drift is large: frame 100 vs. frame 1 may differ by 10–20% while frame 100 vs. frame 99 differs by less than 1%.

### How it works

Align each frame to its *neighbour* and compose the chain.

1. For each consecutive pair of frames, estimate a small transform (shift, rotation, uniform scale) via ORB feature matching + ECC refinement. Per-pair deltas are tiny, so matching is much more robust than bridging the whole stack at once. ECC refines the scale as well (6-DOF ECC projected onto a similarity, then rotation and translation re-refined at that scale): ORB alone only gets the scale to ~0.03% per pair, and in a chain that error accumulates into pixels of drift (up to 10 px on ground-truth test stacks, ~0.1 px with the refinement).

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
Reference=previous is active) or passing `--max-precision` on the CLI runs
the same scale refinement on harder settings, on the CPU. On synthetic
ground-truth stacks both modes land within ~0.1 px of the true transforms,
so it is rarely worth the much longer run time; it may help on difficult,
texture-poor subjects.

- **Float32 data.** The ECC optimization runs on **float32 grayscale** (no
  8-bit quantization), pre-warped with Lanczos instead of bilinear.
- **Full-resolution optimization.** The ECC cascade extends to 4096 px
  (the default stops at 2048 px), with a tighter convergence epsilon.
- **Objective drift arbitration.** Every pair also computes a conservative
  3-DOF estimate (scale frozen at the ORB value) and the two candidates are
  compared by cross-correlation, with highpass detail error breaking ties.
  Texture-poor frames automatically fall back to the drift-free candidate.
  (The default instead keeps ORB's scale only when the refined scale jumps
  by more than 1%.)

Both precision modes use deterministic per-pair estimation (seeded RANSAC,
single-threaded OpenCV inside the parallel workers), so results are
reproducible run to run.

### Performance

Measured on a real 50-frame, 6 MP macro stack (16-thread desktop CPU,
Pyramid stacking):

| Alignment | CPU | GPU |
|---|---|---|
| Similarity | 16 s | 3.4 s |
| Breathing (default) | 33 s | 3.0 s |
| Breathing + Max precision | 67 s (4.9 GB RAM) | (CPU only) |

Peak RAM scales with the number of parallel estimation workers, which is
chosen from the CPU core count and the RAM available when the stack starts.

### References

- Transform chaining follows the classical image-stitching approach (e.g. Brown & Lowe 2007, "Automatic Panoramic Image Stitching").

## References

- Burt, P.J. & Adelson, E.H. (1983). The Laplacian Pyramid as a Compact Image Code.
- Burt, P.J. & Adelson, E.H. (1983). A Multiresolution Spline with Application to Image Mosaics.
- Nayar, S.K. & Nakagawa, Y. (1994). Shape from Focus (Sum Modified Laplacian).
- Mertens, T., Kautz, J. & Van Reeth, F. (2007). Exposure Fusion.
- Wang, W. & Chang, F. (2011). A Multi-focus Image Fusion Method Based on Laplacian Pyramid.
- Brown, M. & Lowe, D.G. (2007). Automatic Panoramic Image Stitching using Invariant Features (transform chaining).
