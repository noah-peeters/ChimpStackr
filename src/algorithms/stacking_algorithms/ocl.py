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
import os

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

// var = sqmean - mean^2, optionally clamped at 0
__kernel void variance(__global const float *mean, __global const float *sqmean,
                       __global float *var, int n, int clamp0) {
    int i = get_global_id(0);
    if (i >= n) return;
    float v = sqmean[i] - mean[i] * mean[i];
    var[i] = clamp0 ? fmax(v, 0.f) : v;
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
__kernel void focusmap(__global const float *v1, __global const float *v2,
                       __global float *fm, int n) {
    int i = get_global_id(0);
    if (i < n) fm[i] = v2[i] > v1[i] ? 1.f : 0.f;
}

// partial maxima: each work item scans a strided slice (host takes the max)
__kernel void max_partial(__global const float *a, __global float *out, int n) {
    int gid = get_global_id(0), gs = get_global_size(0);
    float m = -INFINITY;
    for (int i = gid; i < n; i += gs) m = fmax(m, a[i]);
    out[gid] = m;
}

// out = a * (1 - m / mmax) + b * m / mmax
__kernel void blend(__global const float *a, __global const float *b, __global const float *m,
                    __global float *out, int n, int ch, float inv_max) {
    int i = get_global_id(0);
    if (i >= n) return;
    float t = m[i] * inv_max;
    for (int c = 0; c < ch; c++) out[i * ch + c] = a[i * ch + c] * (1.f - t) + b[i * ch + c] * t;
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

    def __init__(self, device):
        import pyopencl.array as cla
        self._cla = cla
        self.name = device.name.strip()
        self.ctx = cl.Context([device])
        self.q = cl.CommandQueue(self.ctx)
        prg = cl.Program(self.ctx, _SRC).build()
        self.k = {name: getattr(prg, name) for name in (
            "warp_u8", "warp_f32", "pyr_down", "pyr_up_combine", "to_gray", "conv1d", "sml",
            "variance", "hybrid_add", "gate", "hybrid_level", "focusmap", "blend", "max_partial")}
        self._coef = {}

    # ── buffers ──
    def empty(self, shape):
        return self._cla.empty(self.q, shape, np.float32)

    def upload(self, a):
        return self._cla.to_device(self.q, np.ascontiguousarray(a))

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
        frame with forward map M (cv2 WARP_INVERSE_MAP convention)."""
        src = self.upload(host_img)
        h, w = host_img.shape[:2]
        if M is None:
            M = np.float32([[1, 0, 0], [0, 1, 0]])
            lanczos = False
        out = self.empty(host_img.shape)
        kern = self.k["warp_u8"] if host_img.dtype == np.uint8 else self.k["warp_f32"]
        m = np.asarray(M, np.float64).ravel()
        kern(self.q, (w, h), None, src.data, out.data, np.int32(w), np.int32(h), np.int32(w),
             np.int32(h), np.int32(self._ch(host_img)), *[np.float32(v) for v in m],
             np.int32(1 if lanczos else 0), np.int32(1 if replicate else 0))
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

    def variance(self, level, k, clamp0):
        """Local variance of a level's gray (box k x k), as on the CPU."""
        g = self.gray(level)
        mean, sq = self.box(g, k), self.box(g, k, square=True)
        out = self.empty(g.shape)
        n = self._n(g)
        self.k["variance"](self.q, (n,), None, mean.data, sq.data, out.data, np.int32(n),
                           np.int32(1 if clamp0 else 0))
        return out

    def fine_sharpness(self, img):
        """cpu.fine_sharpness: box-filtered sum-modified Laplacian."""
        g = self.gray(img)
        h, w = g.shape
        s = self.empty((h, w))
        self.k["sml"](self.q, (w, h), None, g.data, s.data, np.int32(w), np.int32(h))
        return self.box(s, CPU.HYBRID_SHARPNESS_KSIZE)

    def max(self, a):
        parts = self.empty((4096,))
        self.k["max_partial"](self.q, (4096,), None, a.data, parts.data, np.int32(a.size))
        return float(parts.get().max())


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
            var = g.variance(level, self.kernel_size, clamp0=False)
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
        """Fused image (float32 numpy), or None if no frames were added."""
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
        return g.reconstruct(levels).get()


class PyramidGPU:
    """GPU version of the CPU pairwise max-contrast fusion (soft or hard map).

    Matches Algorithm.focus_fuse_pyramid_pair without a contrast threshold.
    """

    def __init__(self, gpu, num_levels, kernel_size=6, feather_radius=2):
        self.g = gpu
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.feather = feather_radius
        self.fused = None

    def add(self, img):
        g = self.g
        pyr = g.laplacian_pyramid(img, self.num_levels)
        if self.fused is None:
            self.fused = pyr
            return
        k = self.kernel_size | 1
        last = len(pyr) - 1
        prev_map = None
        new = []
        for i in range(len(pyr)):
            a, b = self.fused[i], pyr[i]
            npx = g._n(a)
            if i < last:
                v1, v2 = g.variance(a, k, clamp0=True), g.variance(b, k, clamp0=True)
                fm = g.empty(a.shape[:2])
                g.k["focusmap"](g.q, (npx,), None, v1.data, v2.data, fm.data, np.int32(npx))
                if i == last - 1:
                    prev_map = fm.get().astype(np.uint8)
            else:
                # finest level reuses the previous level's map (same resize as CPU)
                h, w = a.shape[:2]
                fm = g.upload(cv2.resize(prev_map, (w, h), interpolation=cv2.INTER_AREA)
                              .astype(np.float32))
            inv = 1.0
            if self.feather > 0:
                fm = g.gauss(fm, 0, 2 * self.feather + 1)
                mmax = g.max(fm)
                inv = 1.0 / mmax if mmax > 0 else 0.0
            out = g.empty(a.shape)
            g.k["blend"](g.q, (npx,), None, a.data, b.data, fm.data, out.data, np.int32(npx),
                         np.int32(g._ch(a)), np.float32(inv))
            new.append(out)
        self.fused = new

    def result(self):
        return None if self.fused is None else self.g.reconstruct(self.fused).get()
