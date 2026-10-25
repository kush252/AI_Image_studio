"""Per-operator masks. PLAN.md Phase 9.

A mask is a float32 HxW array in [0,1]. Every builder is resolution-independent
(it takes normalised coordinates) so masks survive the proxy/full-res switch.
"""
from __future__ import annotations

import cv2
import numpy as np

from .colorspace import luma, to_hsv


def _feather(mask: np.ndarray, amount: float) -> np.ndarray:
    """amount is a fraction of the long edge, so feathering is scale-invariant."""
    if amount <= 0:
        return mask
    long_edge = max(mask.shape[:2])
    sigma = max(0.5, amount * long_edge * 0.25)
    k = int(sigma * 4) | 1
    return cv2.GaussianBlur(mask, (k, k), sigma)


def luminosity_mask(
    img: np.ndarray, lo: float, hi: float, feather: float = 0.05, invert: bool = False
) -> np.ndarray:
    """Select a tonal band. 'Sharpen only the midtones' is lo=0.25, hi=0.75."""
    y = luma(img)
    # Smooth trapezoid rather than a hard window, so edges of the band do not band.
    span = max(hi - lo, 1e-4)
    ramp = max(span * 0.35, 1e-4)
    m = np.clip((y - lo) / ramp, 0, 1) * np.clip((hi - y) / ramp, 0, 1)
    m = np.clip(m, 0, 1).astype(np.float32)
    m = _feather(m, feather)
    return (1.0 - m) if invert else m


def color_range_mask(
    img: np.ndarray,
    hue: tuple[float, float],
    sat: tuple[float, float] = (0.0, 1.0),
    val: tuple[float, float] = (0.0, 1.0),
    feather: float = 0.03,
    invert: bool = False,
) -> np.ndarray:
    """Select by HSV window. Hue is in degrees and wraps at 360."""
    hsv = to_hsv(img)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    h0, h1 = hue
    if h0 <= h1:
        hm = (h >= h0) & (h <= h1)
    else:  # wrapped window, e.g. reds spanning 350..10
        hm = (h >= h0) | (h <= h1)
    m = hm & (s >= sat[0]) & (s <= sat[1]) & (v >= val[0]) & (v <= val[1])
    out = _feather(m.astype(np.float32), feather)
    return (1.0 - out) if invert else out


def gradient_mask(
    shape: tuple[int, int],
    kind: str = "linear",
    p0: tuple[float, float] = (0.0, 0.5),
    p1: tuple[float, float] = (1.0, 0.5),
    invert: bool = False,
) -> np.ndarray:
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx /= max(w - 1, 1)
    yy /= max(h - 1, 1)
    x0, y0 = p0
    x1, y1 = p1
    if kind == "radial":
        r = float(np.hypot(x1 - x0, y1 - y0)) or 1e-4
        d = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2) / r
        m = np.clip(1.0 - d, 0, 1)
    else:
        dx, dy = x1 - x0, y1 - y0
        denom = dx * dx + dy * dy
        if denom < 1e-9:
            m = np.ones((h, w), np.float32)
        else:
            m = np.clip(((xx - x0) * dx + (yy - y0) * dy) / denom, 0, 1)
    m = m.astype(np.float32)
    return (1.0 - m) if invert else m


def rect_mask(
    shape: tuple[int, int], rect: tuple[float, float, float, float], feather: float = 0.0
) -> np.ndarray:
    h, w = shape[:2]
    x, y, rw, rh = rect
    m = np.zeros((h, w), np.float32)
    x0, y0 = int(round(x * w)), int(round(y * h))
    x1, y1 = int(round((x + rw) * w)), int(round((y + rh) * h))
    m[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)] = 1.0
    return _feather(m, feather)


def brush_mask(
    shape: tuple[int, int],
    strokes: list[dict],
    invert: bool = False,
) -> np.ndarray:
    """Rasterise brush strokes. Each stroke: {points, size, hardness, flow, erase}."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.float32)
    long_edge = max(h, w)
    for st in strokes or []:
        pts = st.get("points") or []
        if len(pts) < 1:
            continue
        size = float(st.get("size", 0.05)) * long_edge
        radius = max(1, int(round(size / 2)))
        hardness = float(st.get("hardness", 0.5))
        flow = float(st.get("flow", 1.0))
        erase = bool(st.get("erase", False))
        layer = np.zeros((h, w), np.float32)
        ipts = [(int(round(p[0] * w)), int(round(p[1] * h))) for p in pts]
        if len(ipts) == 1:
            cv2.circle(layer, ipts[0], radius, 1.0, -1, cv2.LINE_AA)
        else:
            cv2.polylines(
                layer, [np.array(ipts, np.int32)], False, 1.0, radius * 2, cv2.LINE_AA
            )
        if hardness < 0.99:
            sigma = max(0.5, radius * (1.0 - hardness))
            k = int(sigma * 4) | 1
            layer = cv2.GaussianBlur(layer, (k, k), sigma)
        layer *= flow
        m = np.clip(m - layer, 0, 1) if erase else np.clip(m + layer, 0, 1)
    return (1.0 - m) if invert else m


def from_labels(labels: np.ndarray, keep: list[int] | None = None) -> np.ndarray:
    """Promote a segmentation result to a mask (PLAN.md Phase 6)."""
    if keep is None:
        return (labels > 0).astype(np.float32)
    m = np.isin(labels, keep)
    return m.astype(np.float32)


def resolve_mask(spec: dict | None, img: np.ndarray) -> np.ndarray | None:
    """Build the mask an op's `mask` field describes, or None for no mask."""
    if not spec:
        return None
    kind = spec.get("kind")
    shape = img.shape[:2]
    inv = bool(spec.get("invert", False))
    feather = float(spec.get("feather", 0.03))

    if kind == "luminosity":
        rng = spec.get("range", [0.0, 1.0])
        m = luminosity_mask(img, float(rng[0]), float(rng[1]), feather, inv)
    elif kind == "color_range":
        m = color_range_mask(
            img,
            tuple(spec.get("hue", (0.0, 360.0))),
            tuple(spec.get("sat", (0.0, 1.0))),
            tuple(spec.get("val", (0.0, 1.0))),
            feather,
            inv,
        )
    elif kind == "gradient":
        m = gradient_mask(
            shape,
            spec.get("gradient_kind", "linear"),
            tuple(spec.get("p0", (0.0, 0.5))),
            tuple(spec.get("p1", (1.0, 0.5))),
            inv,
        )
    elif kind == "rect":
        m = rect_mask(shape, tuple(spec.get("rect", (0, 0, 1, 1))), feather)
        if inv:
            m = 1.0 - m
    elif kind == "brush":
        m = brush_mask(shape, spec.get("strokes", []), inv)
    elif kind == "bitmap":
        # A client-supplied mask, run-length encoded at its own resolution.
        data = spec.get("data")
        if data is None:
            return None
        mw, mh = int(spec.get("width", 0)), int(spec.get("height", 0))
        if mw <= 0 or mh <= 0:
            return None
        flat = _rle_decode(data, mw * mh)
        m = flat.reshape(mh, mw).astype(np.float32)
        if (mh, mw) != shape:
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        m = _feather(m, feather)
        if inv:
            m = 1.0 - m
    else:
        return None

    if spec.get("blur"):
        m = _feather(m, float(spec["blur"]))
    gain = float(spec.get("opacity", 1.0))
    return np.clip(m * gain, 0, 1).astype(np.float32)


def _rle_decode(pairs: list, total: int) -> np.ndarray:
    """[value, run, value, run, ...] with values in [0,255]."""
    out = np.zeros(total, np.float32)
    i = 0
    idx = 0
    while i + 1 < len(pairs) and idx < total:
        val = float(pairs[i]) / 255.0
        run = int(pairs[i + 1])
        end = min(idx + run, total)
        out[idx:end] = val
        idx = end
        i += 2
    return out
