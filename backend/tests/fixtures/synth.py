"""Photo-like synthetic fixtures with natural-image statistics.

Used to calibrate the Advisor's thresholds and as the ground truth for its
evaluation set (docs/advisor-spec.md Section 9). Two properties matter:

* a 1/f power spectrum, which is what real photographs have, and
* strongly CORRELATED colour channels. Natural images have RGB correlations
  around 0.9; generating each channel independently gives the plate a genuine
  colour cast and differing per-channel highlights, which would force the
  cast threshold to be loosened to accommodate a flaw in the fixture.
"""
from __future__ import annotations

import cv2
import numpy as np


def pink_noise(h: int, w: int, beta: float = 1.6, rng=None) -> np.ndarray:
    """A random field with a 1/f^beta power spectrum."""
    rng = rng or np.random.default_rng(0)
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


def plate(seed: int = 0, h: int = 320, w: int = 320) -> np.ndarray:
    """A clean, sharp, well-exposed, colour-neutral synthetic photograph."""
    rng = np.random.default_rng(seed)

    # One shared luminance field, then small per-channel chroma modulation:
    # this reproduces the high inter-channel correlation of real photographs.
    lum = pink_noise(h, w, 1.7, np.random.default_rng(seed * 7 + 1))
    base = np.dstack([lum, lum, lum])
    for c in range(3):
        chroma = pink_noise(h, w, 2.0, np.random.default_rng(seed * 7 + 11 + c)) - 0.5
        base[:, :, c] = base[:, :, c] + chroma * 0.10
    base = 0.12 + 0.76 * np.clip(base, 0, 1)

    # Objects with softly-focused edges, as real optics produce.
    for _ in range(5):
        cx, cy = int(rng.integers(40, w - 40)), int(rng.integers(40, h - 40))
        rr = int(rng.integers(18, 48))
        col = np.array([float(v) for v in rng.uniform(0.18, 0.82, 3)], np.float32)
        layer = np.zeros((h, w), np.float32)
        cv2.circle(layer, (cx, cy), rr, 1.0, -1)
        layer = cv2.GaussianBlur(layer, (0, 0), 0.8)
        base = base * (1 - layer[:, :, None]) + col[None, None, :] * layer[:, :, None]

    # Fine texture with 1/f character, like foliage or fabric.
    tex = pink_noise(h, w, 0.8, rng) - 0.5
    base = base + tex[:, :, None] * 0.06

    # Flatten any radial brightness trend the random field happened to contain,
    # so the fixture does not embed an accidental vignette. One gain for all
    # channels, so the correction cannot disturb the channel balance.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rr_n = np.sqrt((xx - (w - 1) / 2) ** 2 + (yy - (h - 1) / 2) ** 2) / np.sqrt(
        2 * ((w - 1) / 2) ** 2
    )
    nb = 24
    bi = np.clip((rr_n * nb).astype(np.int32), 0, nb - 1)
    luma_now = base.mean(axis=2)
    ring = np.array([luma_now[bi == i].mean() for i in range(nb)], np.float32)
    base = base * (ring.mean() / np.maximum(ring, 1e-4))[bi][:, :, None]

    # Make the plate neutral to ALL THREE illuminant estimators the Advisor
    # combines, not just gray-world. Equalising channel means satisfies
    # gray-world but leaves the highlight distribution imbalanced, so
    # white-patch and shades-of-gray still reported a cast. Iterating the
    # combined estimate converges all three.
    for _ in range(8):
        flat = np.clip(base, 1e-4, None).reshape(-1, 3)
        gw = flat.mean(axis=0)
        wp = np.percentile(flat, 99.0, axis=0)
        sg = np.power(np.mean(np.power(flat, 6.0), axis=0), 1.0 / 6.0)
        est = np.median(np.stack([gw, wp, sg]), axis=0)
        est = np.maximum(est, 1e-6)
        base = base * (est.mean() / est)[None, None, :]

    base = cv2.GaussianBlur(base, (0, 0), 0.35)   # every real lens is a low-pass
    base = base + rng.normal(0, 1.2 / 255, base.shape)   # a little sensor noise
    return np.clip(base, 0, 1).astype(np.float32)


def plate_with_lines(seed: int = 0, h: int = 320, w: int = 320) -> np.ndarray:
    """A clean plate plus long straight edges.

    Tilt detection needs linear structure by construction -- a Hough transform
    has nothing to vote on otherwise -- so the deskew case is evaluated against
    a plate that contains some, representing a building edge or a document.
    """
    base = plate(seed, h, w)
    for i in range(4):
        y = int(h * (0.18 + 0.2 * i))
        cv2.line(base, (int(0.05 * w), y), (int(0.95 * w), y),
                 (0.93, 0.92, 0.88), 2, cv2.LINE_AA)
    for i in range(3):
        x = int(w * (0.22 + 0.27 * i))
        cv2.line(base, (x, int(0.06 * h)), (x, int(0.94 * h)),
                 (0.12, 0.13, 0.16), 2, cv2.LINE_AA)
    return np.clip(cv2.GaussianBlur(base, (0, 0), 0.4), 0, 1).astype(np.float32)
