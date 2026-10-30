"""Build the macOS app icon from chimpstackr_icon.png on Apple's icon grid.

macOS expects the artwork as an 824 px rounded square (superellipse) centred
on a 1024 px canvas with a soft shadow; full-bleed artwork gets masked into
a circle by the Dock. Writes chimpstackr_icon_macos.png and, on macOS,
chimpstackr_icon.icns (via iconutil).

    python packaging/icons/make_macos_icon.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CANVAS, BODY = 1024, 824


def superellipse_mask(size, n=5.0, supersample=4):
    s = size * supersample
    c = (np.arange(s) + 0.5) / s * 2 - 1
    x, y = np.meshgrid(c, c)
    inside = (np.abs(x) ** n + np.abs(y) ** n) <= 1.0
    return cv2.resize(inside.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)


def main():
    src = cv2.imread(os.path.join(HERE, "chimpstackr_icon.png"), cv2.IMREAD_UNCHANGED)
    alpha = src[..., 3]
    ys, xs = np.nonzero(alpha > 8)
    inset = 20   # drop the old artwork's own rounded border and rim
    art = src[ys.min() + inset:ys.max() + 1 - inset,
              xs.min() + inset:xs.max() + 1 - inset].astype(np.float32)
    # fill the old rounded corners with the background colour before re-masking
    h, w = art.shape[:2]
    bg = np.median(art[h // 2 - 20:h // 2 + 20, 4:14, :3].reshape(-1, 3), axis=0)
    a = art[..., 3:4] / 255.0
    rgb = art[..., :3] * a + bg * (1 - a)
    rgb = cv2.resize(rgb, (BODY, BODY), interpolation=cv2.INTER_AREA)

    mask = superellipse_mask(BODY)
    out = np.zeros((CANVAS, CANVAS, 4), np.float32)
    off = (CANVAS - BODY) // 2
    # shadow: y offset 12, blur 28, 30 % black (Apple's template grid)
    shadow = np.zeros((CANVAS, CANVAS), np.float32)
    shadow[off + 12:off + 12 + BODY, off:off + BODY] = mask
    shadow = cv2.GaussianBlur(shadow, (0, 0), 14) * 0.30
    out[..., 3] = shadow
    body_a = np.zeros((CANVAS, CANVAS), np.float32)
    body_a[off:off + BODY, off:off + BODY] = mask
    body_rgb = np.zeros((CANVAS, CANVAS, 3), np.float32)
    body_rgb[off:off + BODY, off:off + BODY] = rgb
    # composite body over shadow (premultiplied)
    out_a = body_a + out[..., 3] * (1 - body_a)
    out_rgb = body_rgb * body_a[..., None] / np.maximum(out_a[..., None], 1e-6)
    result = np.dstack([np.clip(out_rgb, 0, 255), np.clip(out_a * 255, 0, 255)]).astype(np.uint8)
    png = os.path.join(HERE, "chimpstackr_icon_macos.png")
    cv2.imwrite(png, result)
    print("wrote", png)

    if sys.platform == "darwin" and shutil.which("iconutil"):
        with tempfile.TemporaryDirectory() as tmp:
            iconset = os.path.join(tmp, "chimpstackr.iconset")
            os.mkdir(iconset)
            for size in (16, 32, 128, 256, 512):
                for scale in (1, 2):
                    px = size * scale
                    img = cv2.resize(result, (px, px), interpolation=cv2.INTER_AREA)
                    name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
                    cv2.imwrite(os.path.join(iconset, name), img)
            icns = os.path.join(HERE, "chimpstackr_icon.icns")
            subprocess.run(["iconutil", "-c", "icns", iconset, "-o", icns], check=True)
            print("wrote", icns)


if __name__ == "__main__":
    main()
