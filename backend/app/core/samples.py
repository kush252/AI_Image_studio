"""Built-in sample images, so the app is explorable with zero uploads.

All generated procedurally: no binary assets to ship, and each sample is
designed to exercise a particular part of the platform.
"""
from __future__ import annotations

import cv2
import numpy as np

from .imageio import DecodedImage


def _pink(h, w, beta=1.7, seed=0):
    rng = np.random.default_rng(seed)
    wn = rng.normal(0, 1, (h, w))
    F = np.fft.fftshift(np.fft.fft2(wn))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
    r[int(h / 2), int(w / 2)] = 1.0
    F *= 1.0 / (r ** beta)
    out = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
    out -= out.min()
    out /= max(out.max(), 1e-9)
    return out.astype(np.float32)


def _scene(seed=0, h=720, w=960):
    """A synthetic 'photograph' with natural statistics and real structure."""
    rng = np.random.default_rng(seed)
    lum = _pink(h, w, 1.7, seed * 7 + 1)
    img = np.dstack([lum, lum, lum])
    for c in range(3):
        img[:, :, c] += (_pink(h, w, 2.0, seed * 7 + 11 + c) - 0.5) * 0.12
    img = 0.10 + 0.80 * np.clip(img, 0, 1)
    for _ in range(7):
        cx, cy = int(rng.integers(60, w - 60)), int(rng.integers(60, h - 60))
        rr = int(rng.integers(30, 110))
        col = np.array([float(v) for v in rng.uniform(0.15, 0.88, 3)], np.float32)
        layer = np.zeros((h, w), np.float32)
        if rng.random() < 0.5:
            cv2.circle(layer, (cx, cy), rr, 1.0, -1)
        else:
            cv2.rectangle(layer, (cx - rr, cy - rr // 2), (cx + rr, cy + rr // 2),
                          1.0, -1)
        layer = cv2.GaussianBlur(layer, (0, 0), 1.0)
        img = img * (1 - layer[:, :, None]) + col[None, None, :] * layer[:, :, None]
    img += (_pink(h, w, 0.8, seed + 99) - 0.5)[:, :, None] * 0.05
    img = cv2.GaussianBlur(img, (0, 0), 0.4)
    return np.clip(img + rng.normal(0, 1.5 / 255, img.shape), 0, 1).astype(np.float32)


def _chart(h=720, w=960):
    """A resolution and tone chart: edges, gradients, a wedge and a grid."""
    img = np.full((h, w, 3), 0.5, np.float32)
    # horizontal tonal wedge
    img[: h // 4] = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    # stepped grey wedge
    steps = 11
    for i in range(steps):
        x0, x1 = int(w * i / steps), int(w * (i + 1) / steps)
        img[h // 4 : h // 2, x0:x1] = i / (steps - 1)
    # converging line pairs (resolution target)
    for i in range(1, 40):
        x = int(w * 0.02 + i * (w * 0.96) / 40)
        thick = max(1, int(i / 6))
        cv2.line(img, (x, h // 2 + 10), (x, int(h * 0.78)), (0, 0, 0), thick)
    # colour patches
    colours = [(0.9, 0.1, 0.1), (0.1, 0.8, 0.2), (0.15, 0.3, 0.9),
               (0.95, 0.85, 0.1), (0.85, 0.2, 0.75), (0.1, 0.8, 0.85),
               (0.95, 0.95, 0.95), (0.05, 0.05, 0.05)]
    pw = w // len(colours)
    for i, c in enumerate(colours):
        img[int(h * 0.80) :, i * pw : (i + 1) * pw] = c
    return np.clip(img, 0, 1).astype(np.float32)


def _document(h=900, w=680):
    """A photographed page: uneven illumination, slight tilt, text-like rows."""
    img = np.full((h, w, 3), 0.92, np.float32)
    rng = np.random.default_rng(3)
    y = 70
    while y < h - 70:
        line_h = 9
        words = int(rng.integers(5, 11))
        x = 60
        for _ in range(words):
            wlen = int(rng.integers(40, 110))
            if x + wlen > w - 60:
                break
            cv2.rectangle(img, (x, y), (x + wlen, y + line_h), (0.12, 0.12, 0.14), -1)
            x += wlen + int(rng.integers(12, 24))
        y += int(line_h * 2.6)
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    # uneven lighting from one side, plus a soft shadow in a corner
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = 0.62 + 0.38 * (xx / (w - 1))
    r = np.sqrt((xx - w) ** 2 + (yy - h) ** 2) / np.hypot(w, h)
    ramp *= 0.75 + 0.25 * np.clip(r, 0, 1)
    img *= ramp[:, :, None]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -2.4, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    img += rng.normal(0, 2.5 / 255, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


def _lowlight():
    """A badly underexposed, noisy frame -- deliberately unambiguous, so the
    Advisor has something clear to find and fix."""
    img = _scene(4)
    rng = np.random.default_rng(11)
    # Noise enters BEFORE gamma encoding in a real sensor, so simulate it in
    # linear light: that is what makes it relatively far stronger in the
    # shadows, which is the defining difficulty of low-light imagery.
    lin = np.clip(img, 1e-4, 1.0) ** 2.2
    lin = lin * 0.09                       # about 3.5 stops under
    photons = 90.0
    shot = rng.poisson(np.clip(lin, 0, 1) * photons) / photons
    out = np.clip(shot + rng.normal(0, 0.010, lin.shape), 0, 1)
    out = out ** (1 / 2.2)                 # re-encode
    ycc = cv2.cvtColor(out.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    ycc[:, :, 1] += rng.normal(0, 18 / 255, ycc.shape[:2]).astype(np.float32)
    ycc[:, :, 2] += rng.normal(0, 18 / 255, ycc.shape[:2]).astype(np.float32)
    return np.clip(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB), 0, 1).astype(np.float32)


def _hazy():
    img = _scene(2)
    h, w = img.shape[:2]
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    depth = np.clip(0.6 * (1 - yy) + 0.4 *
                    cv2.GaussianBlur(np.random.default_rng(5).random((h, w))
                                     .astype(np.float32), (0, 0), max(h, w) / 10), 0, 1)
    t = np.exp(-2.0 * depth)[:, :, None]
    return np.clip(img * t + 0.88 * (1 - t), 0, 1).astype(np.float32)


def _periodic():
    img = _scene(1)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Amplitude chosen so the two spectral points are unmistakable: this is a
    # demonstration of notch filtering, not a borderline detection test.
    pat = 0.15 * np.sin(2 * np.pi * (0.11 * xx + 0.07 * yy))
    return np.clip(img + pat[:, :, None], 0, 1).astype(np.float32)


def _shapes(h=720, w=960):
    """Binary-friendly geometry: for morphology, contours and segmentation."""
    img = np.full((h, w, 3), 0.95, np.float32)
    rng = np.random.default_rng(8)
    for _ in range(18):
        cx, cy = int(rng.integers(50, w - 50)), int(rng.integers(50, h - 50))
        r = int(rng.integers(18, 55))
        kind = int(rng.integers(0, 3))
        col = (0.08, 0.10, 0.13)
        if kind == 0:
            cv2.circle(img, (cx, cy), r, col, -1)
        elif kind == 1:
            cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), col, -1)
        else:
            pts = np.array([[cx, cy - r], [cx + r, cy + r], [cx - r, cy + r]], np.int32)
            cv2.fillPoly(img, [pts], col)
    # specks and a thin bridge, so opening/closing have something to do
    for _ in range(260):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(img, (x, y), int(rng.integers(1, 3)), (0.1, 0.1, 0.1), -1)
    cv2.line(img, (int(w * 0.2), int(h * 0.5)), (int(w * 0.8), int(h * 0.52)),
             (0.1, 0.1, 0.1), 2)
    return np.clip(cv2.GaussianBlur(img, (0, 0), 0.5), 0, 1).astype(np.float32)


SAMPLES: dict[str, dict] = {
    "scene": {"fn": lambda: _scene(0),
              "meta": {"name": "Natural scene",
                       "blurb": "Clean, well-exposed, 1/f statistics. The "
                                "baseline: the Advisor should find little wrong.",
                       "tags": ["advisor", "baseline"]}},
    "lowlight": {"fn": _lowlight,
                 "meta": {"name": "Low light",
                          "blurb": "Underexposed with heavy luma and chroma "
                                   "noise, worst in the shadows.",
                          "tags": ["advisor", "denoise", "tone"]}},
    "hazy": {"fn": _hazy,
             "meta": {"name": "Hazy landscape",
                      "blurb": "Atmospheric scattering: flat, bright and "
                               "desaturated. Exercises the dark-channel prior.",
                      "tags": ["advisor", "dehaze"]}},
    "document": {"fn": _document,
                 "meta": {"name": "Photographed document",
                          "blurb": "Uneven lighting and a 2.4 degree tilt. For "
                                   "Sauvola, background division and deskew.",
                          "tags": ["advisor", "threshold", "geometry"]}},
    "periodic": {"fn": _periodic,
                 "meta": {"name": "Periodic interference",
                          "blurb": "A planted sinusoid: two bright points in the "
                                   "spectrum. For notch filtering.",
                          "tags": ["frequency", "denoise"]}},
    "chart": {"fn": lambda: _chart(),
              "meta": {"name": "Test chart",
                       "blurb": "Tonal wedge, grey steps, converging lines and "
                                "colour patches. For transfer curves and "
                                "interpolation comparison.",
                       "tags": ["tone", "teaching"]}},
    "shapes": {"fn": lambda: _shapes(),
               "meta": {"name": "Shapes and specks",
                        "blurb": "Clean geometry with small noise specks and a "
                                 "thin bridge. For morphology, contours and "
                                 "watershed.",
                        "tags": ["morphology", "segmentation"]}},
}


def build_sample(sample_id: str) -> DecodedImage:
    from .errors import ImageNotFound

    spec = SAMPLES.get(sample_id)
    if spec is None:
        raise ImageNotFound(
            f"No sample '{sample_id}'. Available: {sorted(SAMPLES)}."
        )
    arr = spec["fn"]()
    return DecodedImage(
        data=np.ascontiguousarray(arr, dtype=np.float32),
        width=arr.shape[1],
        height=arr.shape[0],
        channels=arr.shape[2] if arr.ndim == 3 else 1,
        source_format="synthetic",
        source_bit_depth=32,
        exif={"generated": True, "sample": sample_id},
        had_alpha=False,
        was_grayscale=arr.ndim == 2,
    )
