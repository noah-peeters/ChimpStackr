"""
GPU stacking via OpenCL: works on any GPU whose driver ships OpenCL (NVIDIA,
AMD, Intel, Apple); only needs the small pyopencl package.

Frames are uploaded once (8-bit when the source is 8-bit), then warped,
decomposed and fused entirely on the GPU; only the final image is read
back. Every step mirrors the CPU implementation in cpu.py (same formulas,
OpenCV's BORDER_REFLECT_101 borders), so results match the CPU path to
float rounding, apart from warp interpolation (exact here, fixed-point in
OpenCV).
"""
import math
import os
import threading
import warnings
from functools import lru_cache

import cv2
import numpy as np

import src.algorithms.stacking_algorithms.cpu as CPU

try:
    import pyopencl as cl
except ImportError:  # optional: CPU stacking is used without it
    cl = None

_SRC = r"""
inline int r101(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) { if (i < 0) i = -i; if (i >= n) i = 2 * n - 2 - i; }
    return i;
}
inline float gray3(__global const float *p) { return 0.114f * p[0] + 0.587f * p[1] + 0.299f * p[2]; }

// ── Warp (WARP_INVERSE_MAP: src = M * dst) + convert to float ──
// interp: 0 = bilinear, 1 = Lanczos4.  border: 0 = constant 0, 1 = replicate
#define WARP_KERNEL(NAME, T)                                                              \
__kernel void NAME(__global const T *src, __global float *dst, int sw, int sh,           \
                   int dw, int dh, int ch, float m00, float m01, float m02,               \
                   float m10, float m11, float m12, int interp, int border) {              \
    int x = get_global_id(0), y = get_global_id(1);                                        \
    if (x >= dw || y >= dh) return;                                                        \
    float fx = m00 * x + m01 * y + m02, fy = m10 * x + m11 * y + m12;                      \
    int r = interp ? 4 : 1, x0 = (int)floor(fx), y0 = (int)floor(fy);                     \
    float tx = fx - x0, ty = fy - y0, wx[8], wy[8];                                        \
    if (interp) {                                                                          \
        float sx = 0.f, sy = 0.f;                                                          \
        for (int i = 0; i < 8; i++) {                                                      \
            float dx = tx + 3 - i, dy = ty + 3 - i;                                        \
            wx[i] = fabs(dx) < 1e-6f ? 1.f : 16.f * sin(M_PI_F * dx / 4) * sin(M_PI_F * dx) / (M_PI_F * M_PI_F * dx * dx); \
            wy[i] = fabs(dy) < 1e-6f ? 1.f : 16.f * sin(M_PI_F * dy / 4) * sin(M_PI_F * dy) / (M_PI_F * M_PI_F * dy * dy); \
            sx += wx[i]; sy += wy[i];                                                      \
        }                                                                                  \
        for (int i = 0; i < 8; i++) { wx[i] /= sx; wy[i] /= sy; }                          \
    } else { wx[0] = 1.f - tx; wx[1] = tx; wy[0] = 1.f - ty; wy[1] = ty; }                 \
    int n = 2 * r, off = r - 1;                                                            \
    for (int c = 0; c < ch; c++) {                                                         \
        float acc = 0.f;                                                                   \
        for (int j = 0; j < n; j++) {                                                      \
            int yy = y0 + j - off;                                                         \
            if (border == 0 && (yy < 0 || yy >= sh)) continue;                             \
            yy = clamp(yy, 0, sh - 1);                                                     \
            float row = 0.f;                                                               \
            for (int i = 0; i < n; i++) {                                                  \
                int xx = x0 + i - off;                                                     \
                if (border == 0 && (xx < 0 || xx >= sw)) continue;                         \
                row += wx[i] * (float)src[(yy * sw + clamp(xx, 0, sw - 1)) * ch + c];      \
            }                                                                              \
            acc += wy[j] * row;                                                            \
        }                                                                                  \
        dst[(y * dw + x) * ch + c] = acc;                                                  \
    }                                                                                      \
}
WARP_KERNEL(warp_u8, uchar)
WARP_KERNEL(warp_f32, float)

// ── Pyramids ──
__kernel void pyr_down(__global const float *src, __global float *dst,
                       int sw, int sh, int dw, int dh, int ch) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= dw || y >= dh) return;
    const float k[5] = {1.f, 4.f, 6.f, 4.f, 1.f};
    for (int c = 0; c < ch; c++) {
        float acc = 0.f;
        for (int j = 0; j < 5; j++) {
            int sy = r101(2 * y + j - 2, sh);
            float row = 0.f;
            for (int i = 0; i < 5; i++) row += k[i] * src[(sy * sw + r101(2 * x + i - 2, sw)) * ch + c];
            acc += k[j] * row;
        }
        dst[(y * dw + x) * ch + c] = acc * (1.f / 256.f);
    }
}

// out = base + sign * pyrUp(src)   (sign -1: Laplacian level, +1: reconstruction)
__kernel void pyr_up_combine(__global const float *src, __global const float *base,
                             __global float *out, int sw, int sh, int dw, int dh, int ch,
                             float sign) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= dw || y >= dh) return;
    const float k[5] = {1.f, 4.f, 6.f, 4.f, 1.f};
    for (int c = 0; c < ch; c++) {
        float acc = 0.f;
        for (int j = 0; j < 5; j++) {
            int uy = r101(y + j - 2, 2 * sh);
            if (uy & 1) continue;
            int sy = min(uy >> 1, sh - 1);
            float row = 0.f;
            for (int i = 0; i < 5; i++) {
                int ux = r101(x + i - 2, 2 * sw);
                if (ux & 1) continue;
                row += k[i] * src[(sy * sw + min(ux >> 1, sw - 1)) * ch + c];
            }
            acc += k[j] * row;
        }
        int o = (y * dw + x) * ch + c;
        out[o] = base[o] + sign * acc * (4.f / 256.f);
    }
}

// ── Single-channel helpers ──
__kernel void to_gray(__global const float *src, __global float *dst, int n, int ch) {
    int i = get_global_id(0);
    if (i < n) dst[i] = ch == 3 ? gray3(src + i * 3) : src[i];
}

// 1-D correlation along x (dir 0) or y (dir 1); square: correlate v*v
__kernel void conv1d(__global const float *src, __global float *dst, __constant float *kern,
                     int w, int h, int ksize, int anchor, int dir, int square) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float acc = 0.f;
    for (int t = 0; t < ksize; t++) {
        float v = dir == 0 ? src[y * w + r101(x + t - anchor, w)] : src[r101(y + t - anchor, h) * w + x];
        acc += kern[t] * (square ? v * v : v);
    }
    dst[y * w + x] = acc;
}

// sum-modified Laplacian |[1 -2 1] g| + |[1 -2 1]^T g| of a gray image
__kernel void sml(__global const float *g, __global float *dst, int w, int h) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float c = g[y * w + x];
    float lx = g[y * w + r101(x - 1, w)] - 2.f * c + g[y * w + r101(x + 1, w)];
    float ly = g[r101(y - 1, h) * w + x] - 2.f * c + g[r101(y + 1, h) * w + x];
    dst[y * w + x] = fabs(lx) + fabs(ly);
}

// ── Hybrid ──
__kernel void hybrid_add(__global const float *level, __global const float *wsrc,
                         __global const float *var, __global const float *sharp,
                         __global float *num, __global float *den, __global float *best,
                         __global float *best_var, __global float *max_sharp,
                         int n, int ch, int first) {
    int i = get_global_id(0);
    if (i >= n) return;
    float w = wsrc[i] + 1e-3f; w = w * w; w = w * w;      // sharpness^4
    bool take = first || var[i] > best_var[i];
    for (int c = 0; c < ch; c++) {
        float v = level[i * ch + c];
        num[i * ch + c] = first ? v * w : num[i * ch + c] + v * w;
        if (take) best[i * ch + c] = v;
    }
    den[i] = first ? w : den[i] + w;
    if (take) best_var[i] = var[i];
    max_sharp[i] = first ? sharp[i] : fmax(max_sharp[i], sharp[i]);
}

// alpha from a 0..1 gate: clamp(a / b - 1, 0, 1)
__kernel void gate(__global const float *src, __global float *dst, int n, float denom) {
    int i = get_global_id(0);
    if (i < n) dst[i] = clamp(src[i] / denom - 1.f, 0.f, 1.f);
}

__kernel void hybrid_level(__global const float *num, __global const float *den,
                           __global const float *best, __global const float *alpha,
                           __global const float *boost, __global float *out,
                           int n, int ch, float gain) {
    int i = get_global_id(0);
    if (i >= n) return;
    float a = alpha[i], g = boost ? 1.f + gain * boost[i] : 1.f;
    for (int c = 0; c < ch; c++) {
        int o = i * ch + c;
        out[o] = (a * best[o] + (1.f - a) * (num[o] / den[i])) * g;
    }
}

// ── Pyramid (pairwise max-contrast, optional soft focus map) ──
// partial maxima: each work item scans a strided slice (host takes the max)
__kernel void max_partial(__global const float *a, __global float *out, int n) {
    int gid = get_global_id(0), gs = get_global_size(0);
    float m = -INFINITY;
    for (int i = gid; i < n; i += gs) m = fmax(m, a[i]);
    out[gid] = m;
}

// single work-group: reduce the partial maxima to mx[0]
__kernel void max_final(__global const float *parts, __global float *mx, int n,
                        __local float *tmp) {
    int lid = get_local_id(0), ls = get_local_size(0);
    float m = -INFINITY;
    for (int i = lid; i < n; i += ls) m = fmax(m, parts[i]);
    tmp[lid] = m;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = ls / 2; s > 0; s >>= 1) {
        if (lid < s) tmp[lid] = fmax(tmp[lid], tmp[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) mx[0] = tmp[0];
}

// cv2.resize(map, (dw, dh), INTER_AREA) enlargement of a 0/1 map; exact for
// even sizes and when only one dimension is odd (integer coefficients,
// OpenCV's 1/2048 fixed-point weights)
inline void area_coef(int d, int sn, int dn, int *s0, int *s1, int *w) {
    int sx = (int)(((long)d * sn) / dn);
    long num = (long)(d + 1) * sn - (long)(sx + 1) * dn;
    float fx = num <= 0 ? 0.f : (float)(num % sn) / (float)sn;
    *s0 = min(sx, sn - 1); *s1 = min(sx + 1, sn - 1);
    *w = (int)rint(fx * 2048.f);
}
__kernel void area_up(__global const float *src, __global float *dst, int sw, int sh,
                      int dw, int dh) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= dw || y >= dh) return;
    int x0, x1, y0, y1, wx, wy;
    area_coef(x, sw, dw, &x0, &x1, &wx);
    area_coef(y, sh, dh, &y0, &y1, &wy);
    int top = (int)src[y0 * sw + x0] * (2048 - wx) + (int)src[y0 * sw + x1] * wx;
    int bot = (int)src[y1 * sw + x0] * (2048 - wx) + (int)src[y1 * sw + x1] * wx;
    dst[y * dw + x] = (float)((top * (2048 - wy) + bot * wy + (1 << 21)) >> 22);
}

// out = a * (1 - m / mx) + b * m / mx   (mx: device scalar, no host round trip;
// NULL = no normalisation)
__kernel void blend(__global const float *a, __global const float *b, __global const float *m,
                    __global float *out, int n, int ch, __global const float *mx) {
    int i = get_global_id(0);
    if (i >= n) return;
    float t = m[i];
    if (mx) t = mx[0] > 0.f ? t / mx[0] : 0.f;
    for (int c = 0; c < ch; c++) out[i * ch + c] = a[i * ch + c] * (1.f - t) + b[i * ch + c] * t;
}

// compute_focusmap_thresholded: deviation over the slice [y-k, y+k) x
// [x-k, x+k) (Python slice rules, so a negative start wraps), zero-padded
// to ks x ks; 0 where neither is above thr, else 1 where g2 deviates more
inline void py_slice(int c, int k, int n, int *s, int *e) {
    int a = c - k;
    if (a < 0) a = max(a + n, 0);
    *s = a; *e = max(min(c + k, n), a);
}
inline float py_dev(__global const float *g, int w, int ys, int ye, int xs, int xe, int ks) {
    int ph = ye - ys, pw = xe - xs;
    float area = (ph < ks || pw < ks) ? (float)(ks * ks) : (float)(ph * pw);
    float sum = 0.f;
    for (int y = ys; y < ye; y++) for (int x = xs; x < xe; x++) sum += g[y * w + x];
    float mean = sum / area, dev = (area - ph * pw) * mean * mean;
    for (int y = ys; y < ye; y++) for (int x = xs; x < xe; x++) {
        float d = g[y * w + x] - mean; dev += d * d;
    }
    return dev / area;
}
__kernel void focusmap_thresh(__global const float *g1, __global const float *g2,
                              __global float *fm, int w, int h, int ks, float thr) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int ys, ye, xs, xe, k = ks / 2;
    py_slice(y, k, h, &ys, &ye);
    py_slice(x, k, w, &xs, &xe);
    float d1 = py_dev(g1, w, ys, ye, xs, xe, ks), d2 = py_dev(g2, w, ys, ye, xs, xe, ks);
    fm[y * w + x] = (d1 < thr && d2 < thr) ? 0.f : (d2 > d1 ? 1.f : 0.f);
}

// ── Weighted average ──
// cv2.Laplacian(gray, CV_32F) squared
__kernel void lap_energy(__global const float *g, __global float *dst, int w, int h) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float l = g[y * w + r101(x - 1, w)] + g[y * w + r101(x + 1, w)] + g[r101(y - 1, h) * w + x]
            + g[r101(y + 1, h) * w + x] - 4.f * g[y * w + x];
    dst[y * w + x] = l * l;
}

__kernel void wavg_add(__global const float *img, __global const float *wt, __global float *sum,
                       __global float *tot, int n, int ch, int first) {
    int i = get_global_id(0);
    if (i >= n) return;
    float w = wt[i];
    for (int c = 0; c < ch; c++)
        sum[i * ch + c] = (first ? 0.f : sum[i * ch + c]) + img[i * ch + c] * w;
    tot[i] = (first ? 0.f : tot[i]) + w;
}

__kernel void wavg_result(__global const float *sum, __global const float *tot,
                          __global float *out, int n, int ch) {
    int i = get_global_id(0);
    if (i >= n) return;
    float d = tot[i] + 1e-12f;
    for (int c = 0; c < ch; c++) out[i * ch + c] = sum[i * ch + c] / d;
}

// ── Depth map ──
__kernel void mean3(__global const float *a, __global const float *b, __global const float *c,
                    __global float *out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = (a[i] + b[i] + c[i]) / 3.f;
}

// statistics for cv2.bilateralFilter, st = [min, max, mean, std]; first pass
// (st == NULL) min/max/sum, second pass the squared deviations from st[2]
__kernel void stats_partial(__global const float *a, __global float *parts, int n,
                            __global const float *st) {
    int gid = get_global_id(0), gs = get_global_size(0);
    float lo = INFINITY, hi = -INFINITY, s = 0.f;
    for (int i = gid; i < n; i += gs) {
        float v = a[i];
        if (st) { v -= st[2]; s += v * v; }
        else { lo = fmin(lo, v); hi = fmax(hi, v); s += v; }
    }
    parts[gid] = lo; parts[gs + gid] = hi; parts[2 * gs + gid] = s;
}

__kernel void stats_final(__global const float *parts, __global float *st, int np, int n,
                          int second, __local float *tmp) {
    int lid = get_local_id(0), ls = get_local_size(0);
    float lo = INFINITY, hi = -INFINITY, s = 0.f;
    for (int i = lid; i < np; i += ls) {
        lo = fmin(lo, parts[i]); hi = fmax(hi, parts[np + i]); s += parts[2 * np + i];
    }
    tmp[lid] = lo; tmp[ls + lid] = hi; tmp[2 * ls + lid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = ls / 2; k > 0; k >>= 1) {
        if (lid < k) {
            tmp[lid] = fmin(tmp[lid], tmp[lid + k]);
            tmp[ls + lid] = fmax(tmp[ls + lid], tmp[ls + lid + k]);
            tmp[2 * ls + lid] += tmp[2 * ls + lid + k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        if (second) st[3] = sqrt(tmp[2 * ls] / n);
        else { st[0] = tmp[0]; st[1] = tmp[ls]; st[2] = tmp[2 * ls] / n; }
    }
}

// OpenCV's float bilateral colour table: exp(-d^2 / (2 sigma^2)) in `bins`
// steps over the value range, with sigma = 2 * std + 1e-6 (as cpu.py passes)
__kernel void bilateral_lut(__global const float *st, __global float *lut, int bins) {
    int i = get_global_id(0);
    if (i > bins + 1) return;
    float sc = st[3] * 2.f + 1e-6f, range = st[1] - st[0];
    float v = range > 0.f ? i * range / bins : 0.f;
    lut[i] = exp(-0.5f * v * v / (sc * sc));
}

__kernel void bilateral(__global const float *src, __global float *dst, __global const float *st,
                        __global const float *lut, int w, int h, int radius, float space_coeff,
                        int bins) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float v0 = src[y * w + x], range = st[1] - st[0];
    if (range < FLT_EPSILON) { dst[y * w + x] = v0; return; }
    float scale = bins / range, sum = 0.f, wsum = 0.f;
    for (int j = -radius; j <= radius; j++) {
        int yy = r101(y + j, h);
        for (int i = -radius; i <= radius; i++) {
            if (i * i + j * j > radius * radius) continue;
            float v = src[yy * w + r101(x + i, w)];
            float a = fabs(v - v0) * scale;
            int idx = (int)floor(a);
            a -= idx;
            float wt = exp((i * i + j * j) * space_coeff) * (lut[idx] + a * (lut[idx + 1] - lut[idx]));
            sum += v * wt; wsum += wt;
        }
    }
    dst[y * w + x] = sum / wsum;
}

// keep each pixel from the source with the highest sharpness so far
__kernel void depth_select(__global const float *img, __global const float *sharp,
                           __global float *res, __global float *best, int n, int ch, int first) {
    int i = get_global_id(0);
    if (i >= n) return;
    if (!first && !(sharp[i] > best[i])) return;
    for (int c = 0; c < ch; c++) res[i * ch + c] = img[i * ch + c];
    best[i] = sharp[i];
}

// ── Exposure fusion (cv2.MergeMertens) ──
// the 8-bit input MergeMertens gets (cpu._to_u8 truncates), scaled to 0..1
__kernel void mertens_prep(__global const float *src, __global float *dst, int n) {
    int i = get_global_id(0);
    if (i < n) dst[i] = floor(clamp(src[i], 0.f, 255.f)) * (1.f / 255.f);
}

// MergeMertens converts with COLOR_RGB2GRAY, also for BGR images
__kernel void mertens_gray(__global const float *src, __global float *dst, int n, int ch) {
    int i = get_global_id(0);
    if (i >= n) return;
    __global const float *p = src + i * ch;
    dst[i] = ch == 3 ? 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2] : p[0];
}

__kernel void mertens_weight(__global const float *img, __global const float *g,
                             __global float *wt, __global float *wsum, int w, int h, int ch,
                             float wcon, float wsat, float wexp, int first) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int i = y * w + x;
    float con = fabs(g[y * w + r101(x - 1, w)] + g[y * w + r101(x + 1, w)]
                     + g[r101(y - 1, h) * w + x] + g[r101(y + 1, h) * w + x] - 4.f * g[i]);
    __global const float *p = img + i * ch;
    float mean = 0.f, sat = 0.f, well = 1.f;
    for (int c = 0; c < ch; c++) mean += p[c];
    mean /= ch;
    for (int c = 0; c < ch; c++) {
        float d = p[c] - mean, e = p[c] - 0.5f;
        sat += d * d;
        well *= exp(-e * e / 0.08f);
    }
    float v = pow(con, wcon) * pow(well, wexp);
    if (ch == 3) v *= pow(sqrt(sat), wsat);
    v += 1e-12f;
    wt[i] = v;
    wsum[i] = first ? v : wsum[i] + v;
}

__kernel void divide(__global float *a, __global const float *b, int n) {
    int i = get_global_id(0);
    if (i < n) a[i] /= b[i];
}

// res (+)= level * weight (weight shared by all channels)
__kernel void mul_acc(__global float *res, __global const float *lv, __global const float *wt,
                      int n, int ch, int first) {
    int i = get_global_id(0);
    if (i >= n) return;
    for (int c = 0; c < ch; c++) {
        float v = lv[i * ch + c] * wt[i];
        res[i * ch + c] = first ? v : res[i * ch + c] + v;
    }
}

__kernel void to_255(__global const float *src, __global float *dst, int n) {
    int i = get_global_id(0);
    if (i < n) dst[i] = clamp(src[i] * 255.f, 0.f, 255.f);
}

// ── Alignment estimation ──
// cv2.resize(src, INTER_AREA) shrink by a non-integer factor, bit-exact for
// 8-bit images: OpenCV's per-axis weight tables, in its summation order
__kernel void resize_area(__global const float *src, __global float *dst, int sw, int dw, int dh,
                          __global const int *xofs, __global const int *xsrc,
                          __global const float *xw, __global const int *yofs,
                          __global const int *ysrc, __global const float *yw) {
    #pragma OPENCL FP_CONTRACT OFF
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= dw || y >= dh) return;
    float acc = 0.f;
    for (int j = yofs[y]; j < yofs[y + 1]; j++) {
        __global const float *row = src + ysrc[j] * sw;
        float buf = 0.f;
        for (int k = xofs[x]; k < xofs[x + 1]; k++) buf = buf + row[xsrc[k]] * xw[k];
        acc = j == yofs[y] ? yw[j] * buf : acc + yw[j] * buf;
    }
    dst[y * dw + x] = clamp(rint(acc), 0.f, 255.f);
}

__kernel void u8_to_f32(__global const uchar *a, __global float *b, int n) {
    int i = get_global_id(0);
    if (i < n) b[i] = a[i];
}

// float -> 8-bit values (round half to even, saturate), in place
__kernel void round_u8(__global float *a, int n) {
    int i = get_global_id(0);
    if (i < n) a[i] = clamp(rint(a[i]), 0.f, 255.f);
}

// ── ECC (cv2.findTransformECC, euclidean / affine) ──
// gradients [-0.5 0 0.5] along x and y (cv2.filter2D, reflect-101 borders)
__kernel void ecc_grad(__global const float *a, __global float *gx, __global float *gy,
                       int w, int h) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    gx[y * w + x] = -0.5f * a[y * w + r101(x - 1, w)] + 0.5f * a[y * w + r101(x + 1, w)];
    gy[y * w + x] = -0.5f * a[r101(y - 1, h) * w + x] + 0.5f * a[r101(y + 1, h) * w + x];
}

// bilinear sample, zero outside (warpAffine INTER_LINEAR, BORDER_CONSTANT)
inline float ecc_sample(__global const float *a, int w, int h, float x, float y) {
    int x0 = (int)floor(x), y0 = (int)floor(y);
    float tx = x - x0, ty = y - y0, v = 0.f;
    for (int j = 0; j < 2; j++) {
        int yy = y0 + j;
        if (yy < 0 || yy >= h) continue;
        float row = 0.f;
        if (x0 >= 0 && x0 < w) row += (1.f - tx) * a[yy * w + x0];
        if (x0 + 1 >= 0 && x0 + 1 < w) row += tx * a[yy * w + x0 + 1];
        v += (j ? ty : 1.f - ty) * row;
    }
    return v;
}

// the warp mask (nearest-neighbour warp of an all-ones image)
inline bool ecc_inside(float x, float y, int w, int h) {
    int nx = (int)floor(x + 0.5f), ny = (int)floor(y + 0.5f);
    return nx >= 0 && ny >= 0 && nx < w && ny < h;
}

#define ECC_GROUP 64
#define ECC_VALUES 36

// pass 1: masked sums of the warped image and the template, per work-group
__kernel void ecc_means_partial(__global const float *tmpl, __global const float *img,
                                __global float *parts, int w, int h, __global const float *M) {
    __local float tmp[3 * ECC_GROUP];
    int lid = get_local_id(0), n = w * h;
    float m0 = M[0], m1 = M[1], m2 = M[2], m3 = M[3], m4 = M[4], m5 = M[5];
    float sw = 0.f, st = 0.f, cnt = 0.f;
    for (int p = get_global_id(0); p < n; p += get_global_size(0)) {
        float x = p % w, y = p / w;
        float sx = m0 * x + m1 * y + m2, sy = m3 * x + m4 * y + m5;
        if (!ecc_inside(sx, sy, w, h)) continue;
        sw += ecc_sample(img, w, h, sx, sy);
        st += tmpl[p];
        cnt += 1.f;
    }
    tmp[lid] = sw; tmp[ECC_GROUP + lid] = st; tmp[2 * ECC_GROUP + lid] = cnt;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = ECC_GROUP / 2; s > 0; s >>= 1) {
        if (lid < s)
            for (int k = 0; k < 3; k++) tmp[k * ECC_GROUP + lid] += tmp[k * ECC_GROUP + lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid < 3) parts[get_group_id(0) * 3 + lid] = tmp[lid * ECC_GROUP];
}

// means[0] = warped image mean, means[1] = template mean (single work-group)
__kernel void ecc_means_final(__global const float *parts, __global float *means, int groups) {
    __local float tmp[3 * ECC_GROUP];
    int lid = get_local_id(0);
    float a = 0.f, b = 0.f, c = 0.f;
    for (int g = lid; g < groups; g += ECC_GROUP) {
        a += parts[g * 3]; b += parts[g * 3 + 1]; c += parts[g * 3 + 2];
    }
    tmp[lid] = a; tmp[ECC_GROUP + lid] = b; tmp[2 * ECC_GROUP + lid] = c;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = ECC_GROUP / 2; s > 0; s >>= 1) {
        if (lid < s)
            for (int k = 0; k < 3; k++) tmp[k * ECC_GROUP + lid] += tmp[k * ECC_GROUP + lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        float cnt = fmax(tmp[2 * ECC_GROUP], 1.f);
        means[0] = tmp[0] / cnt;
        means[1] = tmp[ECC_GROUP] / cnt;
    }
}

// pass 2, per work-group: [correlation, |image|^2, |template|^2,
// J'J (upper triangle), J'image, J'template] with zero-mean image/template
__kernel void ecc_sums_partial(__global const float *tmpl, __global const float *img,
                               __global const float *gxs, __global const float *gys,
                               __global const float *means, __global float *parts,
                               int w, int h, __global const float *M, int affine) {
    __local float tmp[ECC_VALUES * ECC_GROUP];
    int lid = get_local_id(0), n = w * h, np = affine ? 6 : 3;
    float m0 = M[0], m1 = M[1], m2 = M[2], m3 = M[3], m4 = M[4], m5 = M[5];
    int nv = 3 + np * (np + 1) / 2 + 2 * np;
    float acc[ECC_VALUES];
    for (int k = 0; k < ECC_VALUES; k++) acc[k] = 0.f;
    float mw = means[0], mt = means[1];
    for (int p = get_global_id(0); p < n; p += get_global_size(0)) {
        float x = p % w, y = p / w;
        float sx = m0 * x + m1 * y + m2, sy = m3 * x + m4 * y + m5;
        bool in = ecc_inside(sx, sy, w, h);
        float iw = ecc_sample(img, w, h, sx, sy);
        float gx = ecc_sample(gxs, w, h, sx, sy), gy = ecc_sample(gys, w, h, sx, sy);
        float tz = in ? tmpl[p] - mt : 0.f;
        if (in) iw -= mw;
        float J[6];
        if (affine) {
            J[0] = gx * x; J[1] = gy * x; J[2] = gx * y; J[3] = gy * y; J[4] = gx; J[5] = gy;
        } else {
            // m0 = cos(theta), m3 = sin(theta)
            float hx = -(x * m3) - (y * m0), hy = (x * m0) - (y * m3);
            J[0] = gx * hx + gy * hy; J[1] = gx; J[2] = gy;
        }
        acc[0] += tz * iw;
        if (in) acc[1] += iw * iw;
        acc[2] += tz * tz;
        int k = 3;
        for (int i = 0; i < np; i++)
            for (int j = i; j < np; j++) acc[k++] += J[i] * J[j];
        for (int i = 0; i < np; i++) { acc[k + i] += J[i] * iw; acc[k + np + i] += J[i] * tz; }
    }
    for (int k = 0; k < nv; k++) tmp[k * ECC_GROUP + lid] = acc[k];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = ECC_GROUP / 2; s > 0; s >>= 1) {
        if (lid < s)
            for (int k = 0; k < nv; k++) tmp[k * ECC_GROUP + lid] += tmp[k * ECC_GROUP + lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int k = lid; k < nv; k += ECC_GROUP) parts[get_group_id(0) * nv + k] = tmp[k * ECC_GROUP];
}

// ── Tone map (cpu.local_tone_map: CLAHE on L of LAB, blended) ──
// sRGB float LAB as cv2.cvtColor (which evaluates these curves via splines)
inline float srgb_to_linear(float v) {
    return v <= 0.04045f ? v / 12.92f : pow((v + 0.055f) / 1.055f, 2.4f);
}
inline float linear_to_srgb(float v) {
    return v <= 0.0031308f ? 12.92f * v : 1.055f * pow(v, 1.f / 2.4f) - 0.055f;
}
inline float lab_f(float t) { return t > 0.008856f ? cbrt(t) : 7.787f * t + 16.f / 116.f; }
inline float lab_finv(float t) { return t > 6.f / 29.f ? t * t * t : (t - 16.f / 116.f) / 7.787f; }

// BGR (0..255, clipped) -> LAB (L 0..100)
__kernel void tone_lab(__global const float *src, __global float *lab, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    __global const float *p = src + i * 3;
    float b = srgb_to_linear(clamp(p[0], 0.f, 255.f) / 255.f);
    float g = srgb_to_linear(clamp(p[1], 0.f, 255.f) / 255.f);
    float r = srgb_to_linear(clamp(p[2], 0.f, 255.f) / 255.f);
    float X = (0.412453f * r + 0.357580f * g + 0.180423f * b) / 0.950456f;
    float Y = 0.212671f * r + 0.715160f * g + 0.072169f * b;
    float Z = (0.019334f * r + 0.119193f * g + 0.950227f * b) / 1.088754f;
    float fx = lab_f(X), fy = lab_f(Y), fz = lab_f(Z);
    lab[i * 3] = Y > 0.008856f ? 116.f * fy - 16.f : 903.3f * Y;
    lab[i * 3 + 1] = 500.f * (fx - fy);
    lab[i * 3 + 2] = 200.f * (fy - fz);
}

// CLAHE tile histograms of round(L * 2.55) over the image padded (reflect-101)
// to a multiple of the grid
__kernel void tone_hist(__global const float *lab, __global int *hist, int w, int h,
                        int tw, int th, int grid) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= tw * grid || y >= th * grid) return;
    float l = clamp(lab[(r101(y, h) * w + r101(x, w)) * 3] * 2.55f, 0.f, 255.f);
    atomic_inc(&hist[((y / th) * grid + x / tw) * 256 + (int)rint(l)]);
}

// L -> blend of L and CLAHE(L) (bilinear between tiles, linear between LUT
// entries, as cpu._clahe_apply), then back to BGR (0..255)
__kernel void tone_apply(__global const float *lab, __global const float *luts,
                         __global float *dst, int w, int h, int tw, int th, int grid,
                         float strength) {
    #pragma OPENCL FP_CONTRACT OFF
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int i = y * w + x;
    float l255 = clamp(lab[i * 3] * 2.55f, 0.f, 255.f);
    float lo_f = floor(l255), fr = l255 - lo_f;
    int lo = (int)lo_f, hi = min(lo + 1, 255);
    float cx = (float)x / tw - 0.5f, cy = (float)y / th - 0.5f;
    int kx = clamp((int)floor(cx) + 1, 0, grid), ky = clamp((int)floor(cy) + 1, 0, grid);
    int tx1 = max(kx - 1, 0), tx2 = min(kx, grid - 1), ty1 = max(ky - 1, 0), ty2 = min(ky, grid - 1);
    float wx = cx - (kx - 1), wy = cy - (ky - 1);
    float v[4];
    int tiles[4] = {ty1 * grid + tx1, ty1 * grid + tx2, ty2 * grid + tx1, ty2 * grid + tx2};
    for (int t = 0; t < 4; t++) {
        float a = luts[tiles[t] * 256 + lo], b = luts[tiles[t] * 256 + hi];
        v[t] = a + (b - a) * fr;
    }
    float top = v[0] * (1.f - wx) + v[1] * wx, bottom = v[2] * (1.f - wx) + v[3] * wx;
    float lc = top * (1.f - wy) + bottom * wy;
    float L = (l255 * (1.f - strength) + lc * strength) / 2.55f;

    float fy, Y;
    if (L <= 7.9996f) { Y = L / 903.3f; fy = 7.787f * Y + 16.f / 116.f; }
    else { fy = (L + 16.f) / 116.f; Y = fy * fy * fy; }
    float X = lab_finv(lab[i * 3 + 1] / 500.f + fy) * 0.950456f;
    float Z = lab_finv(fy - lab[i * 3 + 2] / 200.f) * 1.088754f;
    float r = 3.240479f * X - 1.53715f * Y - 0.498535f * Z;
    float g = -0.969256f * X + 1.875991f * Y + 0.041556f * Z;
    float b = 0.055648f * X - 0.204043f * Y + 1.057311f * Z;
    dst[i * 3] = clamp(linear_to_srgb(clamp(b, 0.f, 1.f)) * 255.f, 0.f, 255.f);
    dst[i * 3 + 1] = clamp(linear_to_srgb(clamp(g, 0.f, 1.f)) * 255.f, 0.f, 255.f);
    dst[i * 3 + 2] = clamp(linear_to_srgb(clamp(r, 0.f, 1.f)) * 255.f, 0.f, 255.f);
}

// ── Local variance (box k x k of the gray), fused ──
// rows pass, for one or two images (gray computed on the fly): (1/k)-weighted
// sums of gray and gray^2 along x -> out planes [s(a), s2(a), s(b), s2(b)]
__kernel void var_rows(__global const float *a, __global const float *b, __global float *out,
                       int w, int h, int ch, int k) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float c = 1.f / k;
    int n = w * h;
    for (int img = 0; img < (b ? 2 : 1); img++) {
        __global const float *src = img ? b : a;
        float s = 0.f, s2 = 0.f;
        for (int t = 0; t < k; t++) {
            int i = y * w + r101(x + t - k / 2, w);
            float v = ch == 3 ? gray3(src + i * 3) : src[i];
            s += c * v;
            s2 += c * (v * v);
        }
        out[2 * img * n + y * w + x] = s;
        out[(2 * img + 1) * n + y * w + x] = s2;
    }
}

// columns pass: var = E[g^2] - E[g]^2. focus = 0: out = var(a);
// focus = 1: out = 1 where max(var(b), 0) > max(var(a), 0), else 0
__kernel void var_cols(__global const float *rows, __global float *out, int w, int h, int k,
                       int focus) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float c = 1.f / k, var[2];
    int n = w * h;
    for (int img = 0; img < (focus ? 2 : 1); img++) {
        float m = 0.f, m2 = 0.f;
        for (int t = 0; t < k; t++) {
            int i = r101(y + t - k / 2, h) * w + x;
            m += c * rows[2 * img * n + i];
            m2 += c * rows[(2 * img + 1) * n + i];
        }
        var[img] = m2 - m * m;
    }
    out[y * w + x] = focus ? (fmax(var[1], 0.f) > fmax(var[0], 0.f) ? 1.f : 0.f) : var[0];
}
"""


# Tests can run the kernels on a CPU OpenCL implementation (e.g. pocl)
_ANY_DEVICE = os.environ.get("CHIMPSTACKR_OPENCL_ANY_DEVICE") == "1"
_devices = None
_gpus = {}


def devices():
    """OpenCL GPU devices (all devices if CHIMPSTACKR_OPENCL_ANY_DEVICE=1)."""
    global _devices
    if _devices is None:
        _devices = []
        if cl is not None:
            kind = cl.device_type.ALL if _ANY_DEVICE else cl.device_type.GPU
            try:
                for platform in cl.get_platforms():
                    try:
                        _devices.extend(platform.get_devices(kind))
                    except cl.Error:
                        pass
            except cl.Error:
                pass
    return _devices


def available():
    return bool(devices())


def device_names():
    return [d.name.strip() for d in devices()]


def get(index=0):
    """Shared GPU context for a device (kernels are compiled once)."""
    devs = devices()
    if not devs:
        raise RuntimeError("No OpenCL GPU available")
    index = min(max(int(index), 0), len(devs) - 1)
    if index not in _gpus:
        _gpus[index] = GPU(devs[index])
    return _gpus[index]


class GPU:
    """OpenCL context, compiled kernels and image helpers.

    Device images are pyopencl arrays shaped like numpy images (HxW or HxWxC).
    """

    RING = 3   # frames in flight (upload buffers)

    def __init__(self, device):
        import pyopencl.array as cla
        import pyopencl.tools as clt
        self._cla = cla
        self.name = device.name.strip()
        self.ctx = cl.Context([device])
        self.q = cl.CommandQueue(self.ctx)
        # Reuse device buffers: drivers commit memory on a buffer's first use,
        # which for fresh frame-sized buffers costs ~1 ms per kernel launch
        self.pool = clt.MemoryPool(clt.ImmediateAllocator(self.q))
        with warnings.catch_warnings():
            # some drivers log notes even for a clean build (e.g. NVIDIA's
            # "overriding noinline attribute" for every kernel)
            warnings.simplefilter("ignore", cl.CompilerWarning)
            prg = self.prg = cl.Program(self.ctx, _SRC).build()
        self.k = {name: getattr(prg, name) for name in (
            "warp_u8", "warp_f32", "pyr_down", "pyr_up_combine", "to_gray", "conv1d", "sml",
            "hybrid_add", "gate", "hybrid_level", "blend", "max_partial",
            "max_final", "area_up", "focusmap_thresh", "lap_energy", "wavg_add", "wavg_result",
            "mean3", "stats_partial", "stats_final", "bilateral_lut", "bilateral", "depth_select",
            "mertens_prep", "mertens_gray", "mertens_weight", "divide", "mul_acc", "to_255",
            "resize_area", "round_u8", "u8_to_f32", "tone_lab", "tone_hist", "tone_apply",
            "var_rows", "var_cols")}
        self._coef = {}
        self._estimation = None
        self.uq = cl.CommandQueue(self.ctx)   # frame uploads (see warp)
        self._ring = [{"buf": None, "host": None, "done": None} for _ in range(self.RING)]
        self._ring_pos = 0

    def estimation(self):
        """This device's EstimationGPU (created on first use)."""
        if self._estimation is None:
            self._estimation = EstimationGPU(self)
        return self._estimation

    # ── buffers ──
    def empty(self, shape):
        return self._cla.empty(self.q, shape, np.float32, allocator=self.pool)

    def upload(self, a):
        return self._cla.to_device(self.q, np.ascontiguousarray(a), allocator=self.pool)

    @staticmethod
    def _ch(a):
        return 1 if a.ndim == 2 else a.shape[2]

    @staticmethod
    def _n(a):
        return a.shape[0] * a.shape[1]

    def _coefs(self, values):
        key = tuple(float(v) for v in values)
        if key not in self._coef:
            self._coef[key] = self._cla.to_device(self.q, np.asarray(values, np.float32))
        return self._coef[key]

    # ── image ops ──
    def warp(self, host_img, M=None, lanczos=False, replicate=False):
        """Upload a frame (uint8 or float32) and warp it into the reference
        frame with forward map M (cv2 WARP_INVERSE_MAP convention).

        Frames are copied on a separate queue into a ring of upload buffers,
        so the copy overlaps the previous frame's processing and the caller
        doesn't wait for the GPU (at most RING frames in flight)."""
        host_img = np.ascontiguousarray(host_img)
        slot = self._ring[self._ring_pos]
        self._ring_pos = (self._ring_pos + 1) % len(self._ring)
        if slot["done"] is not None:
            slot["done"].wait()          # the frame that last used this buffer
        if slot["buf"] is None or slot["buf"].size < host_img.nbytes:
            slot["buf"] = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY, host_img.nbytes)
        slot["host"] = host_img          # alive until the copy has run
        copied = cl.enqueue_copy(self.uq, slot["buf"], host_img, is_blocking=False)
        self.uq.flush()
        h, w = host_img.shape[:2]
        if M is None:
            M = np.float32([[1, 0, 0], [0, 1, 0]])
            lanczos = False
        out = self.empty(host_img.shape)
        kern = self.k["warp_u8"] if host_img.dtype == np.uint8 else self.k["warp_f32"]
        m = np.asarray(M, np.float64).ravel()
        slot["done"] = kern(self.q, (w, h), None, slot["buf"], out.data, np.int32(w), np.int32(h),
                            np.int32(w), np.int32(h), np.int32(self._ch(host_img)),
                            *[np.float32(v) for v in m], np.int32(1 if lanczos else 0),
                            np.int32(1 if replicate else 0), wait_for=[copied])
        return out

    def pyr_down(self, a):
        h, w = a.shape[:2]
        dh, dw = (h + 1) // 2, (w + 1) // 2
        out = self.empty((dh, dw) + a.shape[2:])
        self.k["pyr_down"](self.q, (dw, dh), None, a.data, out.data, np.int32(w), np.int32(h),
                           np.int32(dw), np.int32(dh), np.int32(self._ch(a)))
        return out

    def pyr_up_combine(self, small, base, sign):
        """base + sign * cv2.pyrUp(small, dstsize=base size)."""
        sh, sw = small.shape[:2]
        dh, dw = base.shape[:2]
        out = self.empty(base.shape)
        self.k["pyr_up_combine"](self.q, (dw, dh), None, small.data, base.data, out.data,
                                 np.int32(sw), np.int32(sh), np.int32(dw), np.int32(dh),
                                 np.int32(self._ch(base)), np.float32(sign))
        return out

    def laplacian_pyramid(self, img, levels):
        """Same layout as cpu.generate_laplacian_pyramid: [top, ..., finest]."""
        g = [img]
        for _ in range(levels):
            g.append(self.pyr_down(g[-1]))
        lap = [g[-1]]
        for i in range(levels, 0, -1):
            lap.append(self.pyr_up_combine(g[i], g[i - 1], -1.0))
        return lap

    def reconstruct(self, lap):
        top = lap[0]
        for lv in lap[1:]:
            top = self.pyr_up_combine(top, lv, 1.0)
        return top

    def gray(self, a):
        if a.ndim == 2:
            return a
        out = self.empty(a.shape[:2])
        n = self._n(a)
        self.k["to_gray"](self.q, (n,), None, a.data, out.data, np.int32(n), np.int32(3))
        return out

    def sep(self, a, kx, ky, ax, ay, square=False):
        """Separable correlation (rows then columns) of a single-channel image."""
        h, w = a.shape[:2]
        tmp, out = self.empty((h, w)), self.empty((h, w))
        self.k["conv1d"](self.q, (w, h), None, a.data, tmp.data, self._coefs(kx).data,
                         np.int32(w), np.int32(h), np.int32(len(kx)), np.int32(ax),
                         np.int32(0), np.int32(1 if square else 0))
        self.k["conv1d"](self.q, (w, h), None, tmp.data, out.data, self._coefs(ky).data,
                         np.int32(w), np.int32(h), np.int32(len(ky)), np.int32(ay),
                         np.int32(1), np.int32(0))
        return out

    def box(self, a, k, square=False):
        """cv2.blur / cv2.sqrBoxFilter (normalized, anchor at k // 2)."""
        c = [1.0 / k] * k
        return self.sep(a, c, c, k // 2, k // 2, square)

    def gauss(self, a, sigma, ksize=0):
        """cv2.GaussianBlur(a, (ksize, ksize), sigma) for float32 input."""
        if ksize <= 0:
            ksize = int(round(sigma * 4 * 2 + 1)) | 1
        kk = cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F).ravel()
        return self.sep(a, kk, kk, ksize // 2, ksize // 2)

    def _local_variance(self, a, b, k):
        h, w = a.shape[:2]
        rows, out = self.empty((4 if b is not None else 2, h, w)), self.empty((h, w))
        self.run2d("var_rows", a, a.data, b.data if b is not None else None, rows.data,
                   np.int32(w), np.int32(h), np.int32(self._ch(a)), np.int32(k))
        self.run2d("var_cols", a, rows.data, out.data, np.int32(w), np.int32(h), np.int32(k),
                   np.int32(b is not None))
        return out

    def variance(self, level, k):
        """Local variance of a level's gray (box k x k), as on the CPU."""
        return self._local_variance(level, None, k)

    def focusmap(self, a, b, k):
        """cpu.compute_focusmap_fast: 1 where b has more local variance."""
        return self._local_variance(a, b, k)

    def fine_sharpness(self, img):
        """cpu.fine_sharpness: box-filtered sum-modified Laplacian."""
        g = self.gray(img)
        h, w = g.shape
        s = self.empty((h, w))
        self.k["sml"](self.q, (w, h), None, g.data, s.data, np.int32(w), np.int32(h))
        return self.box(s, CPU.HYBRID_SHARPNESS_KSIZE)

    def max_device(self, a):
        """Max of a as a 1-element device array (no host synchronisation)."""
        parts, mx = self.empty((4096,)), self.empty((1,))
        self.k["max_partial"](self.q, (4096,), None, a.data, parts.data, np.int32(a.size))
        self.k["max_final"](self.q, (256,), (256,), parts.data, mx.data, np.int32(4096),
                            cl.LocalMemory(256 * 4))
        return mx

    def max(self, a):
        return float(self.max_device(a).get()[0])

    def run(self, name, like, *args):
        """Launch a 1-D kernel over like's pixels (or over `like` items)."""
        n = like if isinstance(like, int) else self._n(like)
        self.k[name](self.q, (n,), None, *args)

    def run2d(self, name, like, *args):
        """Launch a 2-D kernel over like's (w, h)."""
        h, w = like.shape[:2]
        self.k[name](self.q, (w, h), None, *args)

    def tone_map(self, img, strength=0.3, grid=8):
        """cpu.local_tone_map of a BGR image (the CLAHE tables are built on
        the host from GPU histograms; the LAB conversions use the exact sRGB
        curves, which cv2 approximates)."""
        h, w = img.shape[:2]
        lab, out = self.empty((h, w, 3)), self.empty((h, w, 3))
        self.run("tone_lab", h * w, img.data, lab.data, np.int32(h * w))
        th, tw = -(-h // grid), -(-w // grid)    # tiles of the padded image
        hist = self._cla.zeros(self.q, grid * grid * 256, np.int32, allocator=self.pool)
        self.k["tone_hist"](self.q, (tw * grid, th * grid), None, lab.data, hist.data,
                            np.int32(w), np.int32(h), np.int32(tw), np.int32(th), np.int32(grid))
        luts = CPU.clahe_luts_from_histograms(hist.get().reshape(grid, grid, 256), tw * th)
        self.run2d("tone_apply", img, lab.data, self.upload(luts).data, out.data, np.int32(w),
                   np.int32(h), np.int32(tw), np.int32(th), np.int32(grid), np.float32(strength))
        return out

    def area_up(self, a, shape):
        """cv2.resize(a, INTER_AREA) enlargement of a 0/1 map to shape (h, w)."""
        sh, sw = a.shape[:2]
        dh, dw = shape
        out = self.empty((dh, dw))
        self.k["area_up"](self.q, (dw, dh), None, a.data, out.data, np.int32(sw), np.int32(sh),
                          np.int32(dw), np.int32(dh))
        return out


class HybridGPU:
    """GPU version of cpu.HybridFuser (same algorithm and constants)."""

    def __init__(self, gpu, num_levels, kernel_size=6, detail_gain=CPU.HYBRID_DETAIL_GAIN):
        self.g = gpu
        self.num_levels = num_levels
        self.kernel_size = max(2, int(kernel_size))
        self.detail_gain = detail_gain
        self.frames = 0
        self._state = []
        self._noise_var = {}
        self._sharp_floor = None
        self._n_fine = None

    def add(self, img):
        g = self.g
        pyr = g.laplacian_pyramid(img, self.num_levels)
        sharp = [g.fine_sharpness(img)]
        for _ in range(self.num_levels):
            sharp.append(g.pyr_down(sharp[-1]))
        sharp.reverse()
        n = len(pyr)
        if self._n_fine is None:
            self._n_fine = CPU.hybrid_fine_level_count(img.shape, self.num_levels)
            self._sharp_floor = float(np.percentile(sharp[-1].get(), 10)) + 1e-3
        first_fine = n - self._n_fine
        first = self.frames == 0
        for i, level in enumerate(pyr):
            wsrc = g.gauss(sharp[i], CPU.HYBRID_SPREAD_SIGMA) if i < first_fine else sharp[i]
            var = g.variance(level, self.kernel_size)
            if first:
                shape2 = level.shape[:2]
                self._state.append([g.empty(level.shape), g.empty(shape2), g.empty(level.shape),
                                    g.empty(shape2), g.empty(shape2)])
                if i >= first_fine:
                    # Robust (MAD) noise estimate from the first frame's coefficients
                    sigma = float(np.median(np.abs(g.gray(level).get()))) / 0.6745
                    self._noise_var[i] = max(sigma * sigma, 1e-6)
            num, den, best, best_var, max_sharp = self._state[i]
            npx = g._n(level)
            g.k["hybrid_add"](g.q, (npx,), None, level.data, wsrc.data, var.data, sharp[i].data,
                              num.data, den.data, best.data, best_var.data, max_sharp.data,
                              np.int32(npx), np.int32(g._ch(level)), np.int32(1 if first else 0))
        self.frames += 1

    def _gate(self, src, denom, blur):
        out = self.g.empty(src.shape)
        npx = self.g._n(src)
        self.g.k["gate"](self.g.q, (npx,), None, src.data, out.data, np.int32(npx),
                         np.float32(denom))
        return self.g.gauss(out, blur)

    def result(self):
        """Fused image (device array), or None if no frames were added."""
        if self.frames == 0:
            return None
        g = self.g
        first_fine = len(self._state) - self._n_fine
        subject_denom = CPU.HYBRID_DETAIL_GATE * self._sharp_floor
        levels = []
        for i, (num, den, best, best_var, max_sharp) in enumerate(self._state):
            boost = None
            if i >= first_fine:
                alpha = self._gate(best_var, CPU.HYBRID_NOISE_FACTOR * self._noise_var[i], 2.0)
                if self.detail_gain > 0:
                    boost = self._gate(max_sharp, subject_denom, 2.0)
            else:
                alpha = self._gate(max_sharp, subject_denom, 1.0)
            out = g.empty(num.shape)
            npx = g._n(num)
            g.k["hybrid_level"](g.q, (npx,), None, num.data, den.data, best.data, alpha.data,
                                boost.data if boost is not None else None, out.data,
                                np.int32(npx), np.int32(g._ch(num)), np.float32(self.detail_gain))
            levels.append(out)
        return g.reconstruct(levels)


class PyramidGPU:
    """GPU version of the CPU pairwise max-contrast fusion (soft or hard map).

    Matches Algorithm.focus_fuse_pyramid_pair.
    """

    def __init__(self, gpu, num_levels, kernel_size=6, feather_radius=2, contrast_threshold=0.0):
        self.g = gpu
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.feather = feather_radius
        self.threshold = contrast_threshold
        self.fused = None

    def _focusmap(self, a, b):
        g = self.g
        if self.threshold <= 0:
            return g.focusmap(a, b, self.kernel_size | 1)
        h, w = a.shape[:2]
        fm = g.empty((h, w))
        g.run2d("focusmap_thresh", a, g.gray(a).data, g.gray(b).data, fm.data, np.int32(w),
                np.int32(h), np.int32(self.kernel_size), np.float32(self.threshold))
        return fm

    def add(self, img):
        g = self.g
        pyr = g.laplacian_pyramid(img, self.num_levels)
        if self.fused is None:
            self.fused = pyr
            return
        last = len(pyr) - 1
        prev_map = None
        new = []
        for i in range(len(pyr)):
            a, b = self.fused[i], pyr[i]
            npx = g._n(a)
            if i < last:
                fm = prev_map = self._focusmap(a, b)
            else:
                # finest level reuses the previous level's map (same resize as CPU)
                h, w = a.shape[:2]
                if h % 2 and w % 2:
                    # both dimensions odd: OpenCV's INTER_AREA isn't reproduced
                    # exactly on the GPU, so resize this one map on the host
                    fm = g.upload(cv2.resize(prev_map.get().astype(np.uint8), (w, h),
                                             interpolation=cv2.INTER_AREA).astype(np.float32))
                else:
                    fm = g.area_up(prev_map, (h, w))
            mx = None
            if self.feather > 0:
                fm = g.gauss(fm, 0, 2 * self.feather + 1)
                mx = g.max_device(fm)
            out = g.empty(a.shape)
            g.k["blend"](g.q, (npx,), None, a.data, b.data, fm.data, out.data, np.int32(npx),
                         np.int32(g._ch(a)), mx.data if mx is not None else None)
            new.append(out)
        self.fused = new

    def result(self):
        return None if self.fused is None else self.g.reconstruct(self.fused)


class WeightedAverageGPU:
    """API._weighted_avg_core: average weighted by local Laplacian energy."""

    def __init__(self, gpu, kernel_size=5):
        self.g = gpu
        self.ksize = kernel_size | 1
        self.sum = self.tot = None

    def add(self, img):
        g = self.g
        gray = g.gray(img)
        h, w = gray.shape
        energy = g.empty((h, w))
        g.run2d("lap_energy", gray, gray.data, energy.data, np.int32(w), np.int32(h))
        weight = g.gauss(energy, 0, self.ksize)
        first = self.sum is None
        if first:
            self.sum, self.tot = g.empty(img.shape), g.empty((h, w))
        g.run("wavg_add", img, img.data, weight.data, self.sum.data, self.tot.data,
              np.int32(g._n(img)), np.int32(g._ch(img)), np.int32(first))

    def result(self):
        if self.sum is None:
            return None
        out = self.g.empty(self.sum.shape)
        self.g.run("wavg_result", out, self.sum.data, self.tot.data, out.data,
                   np.int32(self.g._n(out)), np.int32(self.g._ch(out)))
        return out


class DepthMapGPU:
    """API._depthmap_core: per pixel, the source with the highest multi-scale
    sum-modified-Laplacian sharpness (bilateral-smoothed as on the CPU)."""

    BINS = 1 << 12   # OpenCV's float bilateral table size (one channel)

    def __init__(self, gpu, kernel_size=6, smoothing=5):
        self.g = gpu
        self.scales = (5, kernel_size | 1, kernel_size * 3 + 1)
        self.smoothing = smoothing
        self.res = self.best = None

    def sharpness(self, img):
        """cpu.compute_multires_sharpness."""
        g = self.g
        gray = g.gray(img)
        h, w = gray.shape
        ml = g.empty((h, w))
        g.run2d("sml", gray, gray.data, ml.data, np.int32(w), np.int32(h))
        boxes = [g.box(ml, s | 1) for s in self.scales]
        combined = g.empty((h, w))
        g.run("mean3", combined, *[b.data for b in boxes], combined.data, np.int32(h * w))
        if self.smoothing <= 0:
            return combined
        st = self._stats(combined)
        lut = g.empty((self.BINS + 2,))
        g.k["bilateral_lut"](g.q, (self.BINS + 2,), None, st.data, lut.data, np.int32(self.BINS))
        sigma_space = float(self.smoothing * 2)
        out = g.empty((h, w))
        g.run2d("bilateral", combined, combined.data, out.data, st.data, lut.data, np.int32(w),
                np.int32(h), np.int32(max(self.smoothing, 1)),
                np.float32(-0.5 / (sigma_space * sigma_space)), np.int32(self.BINS))
        return out

    def _stats(self, a):
        """[min, max, mean, std] of a, kept on the device."""
        g, n_parts = self.g, 4096
        st, parts = g.empty((4,)), g.empty((3 * n_parts,))
        for second in (0, 1):
            g.k["stats_partial"](g.q, (n_parts,), None, a.data, parts.data, np.int32(a.size),
                                 st.data if second else None)
            g.k["stats_final"](g.q, (256,), (256,), parts.data, st.data, np.int32(n_parts),
                               np.int32(a.size), np.int32(second), cl.LocalMemory(3 * 256 * 4))
        return st

    def add(self, img):
        g = self.g
        sharp = self.sharpness(img)
        first = self.res is None
        if first:
            self.res, self.best = g.empty(img.shape), g.empty(img.shape[:2])
        g.run("depth_select", img, img.data, sharp.data, self.res.data, self.best.data,
              np.int32(g._n(img)), np.int32(g._ch(img)), np.int32(first))

    def result(self):
        return self.res


class ExposureGPU:
    """API._exposure_core: cv2.MergeMertens over batches of frames, then over
    the batch results (same batching, so the same output as the CPU)."""

    def __init__(self, gpu, batch_size=4, contrast_weight=1.0, saturation_weight=1.0,
                 exposure_weight=0.0):
        self.g = gpu
        self.batch_size = batch_size
        self.weights = (contrast_weight, saturation_weight, exposure_weight)
        self.batch = []
        self.partials = []   # batch results as the next stage reads them (8-bit, host)
        self.last = None     # float result of the latest batch (used if it's the only one)

    def _prep(self, img):
        out = self.g.empty(img.shape)
        self.g.run("mertens_prep", img.size, img.data, out.data, np.int32(img.size))
        return out

    def merge(self, images):
        """cv2.createMergeMertens(...).process(images) * 255, clipped; images
        are callables returning the device frames (0..255)."""
        g = self.g
        wsum, weights = None, []
        for i, load in enumerate(images):
            img = self._prep(load())
            h, w = img.shape[:2]
            gray, wt = g.empty((h, w)), g.empty((h, w))
            if wsum is None:
                wsum = g.empty((h, w))
            g.run("mertens_gray", gray, img.data, gray.data, np.int32(h * w), np.int32(g._ch(img)))
            g.run2d("mertens_weight", gray, img.data, gray.data, wt.data, wsum.data, np.int32(w),
                    np.int32(h), np.int32(g._ch(img)), *[np.float32(v) for v in self.weights],
                    np.int32(i == 0))
            weights.append(wt)
        levels = int(np.log(np.float32(min(h, w))) / np.log(np.float32(2)))
        res = None
        for load, wt in zip(images, weights):
            g.run("divide", wt, wt.data, wsum.data, np.int32(h * w))
            lap = g.laplacian_pyramid(self._prep(load()), levels)
            wp = [wt]
            for _ in range(levels):
                wp.append(g.pyr_down(wp[-1]))
            wp.reverse()
            first = res is None
            if first:
                res = [g.empty(lv.shape) for lv in lap]
            for r, lv, wl in zip(res, lap, wp):
                g.run("mul_acc", lv, r.data, lv.data, wl.data, np.int32(g._n(lv)),
                      np.int32(g._ch(lv)), np.int32(first))
        fused = g.reconstruct(res)
        out = g.empty(fused.shape)
        g.run("to_255", out.size, fused.data, out.data, np.int32(out.size))
        return out

    def _flush(self):
        batch, self.batch = self.batch, []
        self.last = self.merge([lambda b=b: b for b in batch]) if len(batch) > 1 else batch[0]
        self.partials.append(np.clip(self.last.get(), 0, 255).astype(np.uint8))

    def add(self, img):
        self.batch.append(img)
        if len(self.batch) >= self.batch_size:
            self._flush()

    def result(self):
        if self.batch:
            self._flush()
        if not self.partials:
            return None
        if len(self.partials) == 1:
            return self.last
        return self.merge([lambda p=p: self.g.warp(p) for p in self.partials])


@lru_cache(maxsize=32)
def _area_table(ssize, dsize, scale):
    """OpenCV's INTER_AREA weights along one axis (computeResizeAreaTab), as
    (offsets per output index, source indices, weights)."""
    ofs, src, wts = [0], [], []
    for d in range(dsize):
        f1 = d * scale
        f2 = f1 + scale
        cell = min(scale, ssize - f1)
        s1, s2 = math.ceil(f1), min(math.floor(f2), ssize - 1)
        s1 = min(s1, s2)
        if s1 - f1 > 1e-3:
            src.append(s1 - 1)
            wts.append((s1 - f1) / cell)
        for k in range(s1, s2):
            src.append(k)
            wts.append(1.0 / cell)
        if f2 - s2 > 1e-3:
            src.append(s2)
            wts.append(min(min(f2 - s2, 1.0), cell) / cell)
        ofs.append(len(src))
    return (np.asarray(ofs, np.int32), np.asarray(src, np.int32),
            np.asarray(wts, np.float32))


class EstimationGPU:
    """Alignment estimation on the GPU: pre-warps a frame, builds the ECC
    cascade levels (cv2.resize INTER_AREA, bit-exact) and runs the ECC
    iterations (cv2.findTransformECC), so the CPU only decodes and runs ORB.

    Estimation workers call it from several threads at once, alongside GPU
    stacking: each thread gets its own queue and kernel objects (kernel
    arguments aren't thread-safe), so their work overlaps on the GPU.
    """

    KERNELS = ("warp_u8", "round_u8", "resize_area", "u8_to_f32", "conv1d", "ecc_grad",
               "ecc_means_partial", "ecc_means_final", "ecc_sums_partial")
    GROUP = 64          # ECC_GROUP in the kernels
    PIXELS_PER_ITEM = 16

    def __init__(self, gpu):
        import pyopencl.array as cla
        import pyopencl.tools as clt
        self._cla, self._clt = cla, clt
        self.ctx, self.prg = gpu.ctx, gpu.prg
        self._local = threading.local()
        self._lock = threading.Lock()
        self._tables = {}
        self._coef = {}

    # ── per-thread resources ──
    @property
    def _t(self):
        t = self._local
        if not hasattr(t, "q"):
            t.q = cl.CommandQueue(self.ctx)
            t.pool = self._clt.MemoryPool(self._clt.ImmediateAllocator(t.q))
            t.k = {n: cl.Kernel(self.prg, n) for n in self.KERNELS}
            t.frames = (None, None, None, None)   # last uploaded (ref, align) and copies
        return t

    def _empty(self, shape, dtype=np.float32):
        t = self._t
        return self._cla.empty(t.q, shape, dtype, allocator=t.pool)

    def _upload(self, a):
        t = self._t
        return self._cla.to_device(t.q, np.ascontiguousarray(a), allocator=t.pool)

    def _shared(self, cache, key, make):
        """Read-only device data shared by all threads (tables, kernels)."""
        with self._lock:
            if key not in cache:
                cache[key] = self._cla.to_device(self._t.q, make())
                self._t.q.finish()
            return cache[key]

    # ── image ops ──
    def _to_float(self, u8):
        h, w = u8.shape
        out = self._empty((h, w))
        self._t.k["u8_to_f32"](self._t.q, (h * w,), None, self._upload(u8).data, out.data,
                               np.int32(h * w))
        return out

    def _frames(self, ref_u8, align_u8):
        """Device copies of the pair (reused while the same pair is refined)."""
        t = self._t
        ref, align, ref_d, align_d = t.frames
        if ref is not ref_u8:
            ref_d = self._to_float(ref_u8)
        if align is not align_u8:
            align_d = self._upload(align_u8)
        t.frames = (ref_u8, align_u8, ref_d, align_d)
        return ref_d, align_d

    def _shrink(self, img, s):
        """cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)."""
        h, w = img.shape
        dw, dh = int(round(w * s)), int(round(h * s))
        scale = 1.0 / s
        if abs(scale - round(scale)) < 1e-12:
            # integer factors take OpenCV's separate fast path
            small = cv2.resize(img.get().astype(np.uint8), None, fx=s, fy=s,
                               interpolation=cv2.INTER_AREA)
            return self._upload(small.astype(np.float32))
        out = self._empty((dh, dw))
        tabs = [self._shared(self._tables, (n, d, scale, i),
                             lambda n=n, d=d, i=i: _area_table(n, d, scale)[i])
                for n, d in ((w, dw), (h, dh)) for i in range(3)]
        self._t.k["resize_area"](self._t.q, (dw, dh), None, img.data, out.data, np.int32(w),
                                 np.int32(dw), np.int32(dh), *[t.data for t in tabs])
        return out

    def _gauss(self, a, ksize):
        """cv2.GaussianBlur(a, (ksize, ksize), 0) of a float image."""
        h, w = a.shape
        coef = self._shared(self._coef, ksize, lambda: cv2.getGaussianKernel(
            ksize, 0, cv2.CV_32F).ravel().astype(np.float32))
        tmp, out = self._empty((h, w)), self._empty((h, w))
        k = self._t.k["conv1d"]
        for src, dst, axis in ((a, tmp, 0), (tmp, out, 1)):
            k(self._t.q, (w, h), None, src.data, dst.data, coef.data, np.int32(w), np.int32(h),
              np.int32(ksize), np.int32(ksize // 2), np.int32(axis), np.int32(0))
        return out

    def ecc_levels(self, ref_u8, align_u8, M, scales):
        """Per scale, (ref, align pre-warped by M) as cv2 would give them:
        warpAffine(bilinear, replicate) to 8 bits (M None: no pre-warp),
        then resize(INTER_AREA) unless the scale is 1. Device arrays."""
        t = self._t
        h, w = ref_u8.shape
        ref, src = self._frames(ref_u8, align_u8)
        warped = self._empty((h, w))
        if M is None:
            t.k["u8_to_f32"](t.q, (h * w,), None, src.data, warped.data, np.int32(h * w))
        else:
            m = np.asarray(M, np.float64).ravel()
            t.k["warp_u8"](t.q, (w, h), None, src.data, warped.data, np.int32(w), np.int32(h),
                           np.int32(w), np.int32(h), np.int32(1), *[np.float32(v) for v in m],
                           np.int32(0), np.int32(1))
            t.k["round_u8"](t.q, (h * w,), None, warped.data, np.int32(h * w))
        return [(self._shrink(ref, s), self._shrink(warped, s)) if s < 1.0 else (ref, warped)
                for s in scales]

    def find_transform_ecc(self, template, image, warp, motion, criteria, gauss):
        """cv2.findTransformECC(template, image, warp, motion, criteria, None,
        gauss) for device images (MOTION_EUCLIDEAN or MOTION_AFFINE).
        Returns (rho, warp); raises cv2.error where OpenCV does."""
        t = self._t
        h, w = template.shape
        affine = motion == cv2.MOTION_AFFINE
        if not affine and motion != cv2.MOTION_EUCLIDEAN:
            raise ValueError("unsupported motion type")
        n_iter = criteria[1] if criteria[0] & cv2.TERM_CRITERIA_COUNT else 200
        eps = criteria[2] if criteria[0] & cv2.TERM_CRITERIA_EPS else -1
        tmpl, img = self._gauss(template, gauss), self._gauss(image, gauss)
        gx, gy = self._empty((h, w)), self._empty((h, w))
        t.k["ecc_grad"](t.q, (w, h), None, img.data, gx.data, gy.data, np.int32(w), np.int32(h))

        n = h * w
        groups = max(1, -(-n // (self.GROUP * self.PIXELS_PER_ITEM)))
        np_ = 6 if affine else 3
        nv = 3 + np_ * (np_ + 1) // 2 + 2 * np_
        mparts, means = self._empty((groups * 3,)), self._empty((2,))
        parts, mbuf = self._empty((groups * nv,)), self._empty((6,))
        host = np.empty(groups * nv, np.float32)
        gsize, lsize = (groups * self.GROUP,), (self.GROUP,)
        # arguments are set once; each iteration only uploads the warp
        k_means, k_final, k_sums = (t.k[n] for n in (
            "ecc_means_partial", "ecc_means_final", "ecc_sums_partial"))
        k_means.set_args(tmpl.data, img.data, mparts.data, np.int32(w), np.int32(h), mbuf.data)
        k_final.set_args(mparts.data, means.data, np.int32(groups))
        k_sums.set_args(tmpl.data, img.data, gx.data, gy.data, means.data, parts.data,
                        np.int32(w), np.int32(h), mbuf.data, np.int32(affine))
        iu = np.triu_indices(np_)
        warp = np.array(warp, np.float32).reshape(2, 3)
        rho, last_rho = -1.0, -eps
        i = 1
        while i <= n_iter and abs(rho - last_rho) >= eps:
            # (the blocking read below keeps `m` alive until it's uploaded)
            m = warp.ravel().copy()
            cl.enqueue_copy(t.q, mbuf.data, m, is_blocking=False)
            cl.enqueue_nd_range_kernel(t.q, k_means, gsize, lsize)
            cl.enqueue_nd_range_kernel(t.q, k_final, lsize, lsize)
            cl.enqueue_nd_range_kernel(t.q, k_sums, gsize, lsize)
            cl.enqueue_copy(t.q, host, parts.data)
            sums = host.reshape(groups, nv).sum(axis=0, dtype=np.float64)
            corr, img_norm, tmp_norm = sums[0], np.sqrt(sums[1]), np.sqrt(sums[2])
            k = 3 + len(iu[0])
            hess = np.zeros((np_, np_), np.float32)
            hess[iu] = sums[3:k]
            hess = np.triu(hess) + np.triu(hess, 1).T
            img_proj = sums[k:k + np_].astype(np.float32)
            tmp_proj = sums[k + np_:k + 2 * np_].astype(np.float32)
            hess_inv = np.linalg.inv(hess)

            last_rho, rho = rho, corr / (img_norm * tmp_norm)
            if np.isnan(rho):
                raise cv2.error("NaN encountered.")
            img_proj_hess = hess_inv @ img_proj
            lambda_n = img_norm * img_norm - float(img_proj @ img_proj_hess)
            lambda_d = corr - float(tmp_proj @ img_proj_hess)
            if lambda_d <= 0.0:
                raise cv2.error("The algorithm stopped before its convergence. The correlation "
                                "is going to be minimized. Images may be uncorrelated or "
                                "non-overlapped")
            lam = lambda_n / lambda_d
            # J'(lambda * template - image), from the projections (linear)
            error_proj = (lam * tmp_proj.astype(np.float64) - img_proj).astype(np.float32)
            delta = hess_inv @ error_proj
            if affine:
                warp[0, 0] += delta[0]; warp[1, 0] += delta[1]; warp[0, 1] += delta[2]  # noqa: E702
                warp[1, 1] += delta[3]; warp[0, 2] += delta[4]; warp[1, 2] += delta[5]  # noqa: E702
            else:
                theta = float(delta[0]) + math.asin(float(warp[1, 0]))
                warp[0, 2] += delta[1]
                warp[1, 2] += delta[2]
                warp[0, 0] = warp[1, 1] = math.cos(theta)
                warp[1, 0] = math.sin(theta)
                warp[0, 1] = -warp[1, 0]
            i += 1
        return rho, warp
