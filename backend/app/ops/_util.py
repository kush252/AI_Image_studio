"""Helpers shared across operator families."""
from __future__ import annotations

import cv2
import numpy as np

from ..core.colorspace import as_color, is_color, join_alpha, luma, split_alpha

EPS = 1e-8

BORDER_MODES = {
    "reflect": cv2.BORDER_REFLECT_101,
    "reflect_full": cv2.BORDER_REFLECT,
    "replicate": cv2.BORDER_REPLICATE,
    "wrap": cv2.BORDER_WRAP,
    "constant": cv2.BORDER_CONSTANT,
}

INTERP_MODES = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
    "area": cv2.INTER_AREA,
}


def border_flag(name: str) -> int:
    return BORDER_MODES.get(name, cv2.BORDER_REFLECT_101)


def interp_flag(name: str) -> int:
    return INTERP_MODES.get(name, cv2.INTER_LINEAR)


def clamp01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def apply_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a float LUT with linear interpolation between entries.

    Nearest-neighbour indexing would quantise the float input to the LUT's
    resolution, so an identity curve would not be an identity operation and a
    16-bit source would silently lose precision -- both of which contradict
    working in float32 throughout (PLAN.md D4). Interpolating costs one extra
    multiply and keeps the operation exact for a linear transfer function.
    """
    n = lut.shape[0]
    x = np.clip(img.astype(np.float32, copy=False), 0.0, 1.0) * (n - 1)
    i0 = np.floor(x).astype(np.int32)
    i1 = np.minimum(i0 + 1, n - 1)
    frac = (x - i0).astype(np.float32)
    return (lut[i0] * (1.0 - frac) + lut[i1] * frac).astype(np.float32)


def build_lut(fn, size: int = 256) -> np.ndarray:
    x = np.linspace(0.0, 1.0, size, dtype=np.float32)
    return np.clip(fn(x), 0.0, 1.0).astype(np.float32)


def per_channel(img: np.ndarray, fn) -> np.ndarray:
    """Run fn on each colour channel, leaving alpha alone."""
    rgb, alpha = split_alpha(img)
    if rgb.ndim == 2:
        return join_alpha(fn(rgb), alpha)
    out = np.dstack([fn(rgb[:, :, i]) for i in range(rgb.shape[2])])
    return join_alpha(out.astype(np.float32), alpha)


def on_luma(img: np.ndarray, fn) -> np.ndarray:
    """Apply a grayscale operator to luminance only, preserving chroma ratios.

    Used wherever applying an operator per-channel would shift hue (histogram
    equalisation being the classic example -- see tone.he).
    """
    rgb, alpha = split_alpha(img)
    if rgb.ndim == 2:
        return join_alpha(clamp01(fn(rgb)), alpha)
    y = luma(rgb)
    y2 = fn(y)
    gain = (y2 + EPS) / (y + EPS)
    out = rgb * gain[:, :, None]
    return join_alpha(clamp01(out), alpha)


def on_lab_l(img: np.ndarray, fn) -> np.ndarray:
    """Apply a grayscale operator to CIE L* only. Perceptually the safest place
    to do tonal work on a colour image."""
    from ..core.colorspace import from_lab, to_lab

    rgb, alpha = split_alpha(img)
    if rgb.ndim == 2:
        return join_alpha(clamp01(fn(rgb)), alpha)
    lab = to_lab(rgb)
    l_norm = lab[:, :, 0] / 100.0
    lab[:, :, 0] = np.clip(fn(l_norm), 0, 1) * 100.0
    return join_alpha(from_lab(lab), alpha)


def gray_u8(img: np.ndarray) -> np.ndarray:
    """uint8 grayscale, for the OpenCV functions that insist on it."""
    return (clamp01(luma(img)) * 255.0 + 0.5).astype(np.uint8)


def to_u8(img: np.ndarray) -> np.ndarray:
    return (clamp01(img) * 255.0 + 0.5).astype(np.uint8)


def from_u8(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32) / 255.0).astype(np.float32)


def colorize(gray: np.ndarray, cmap: str = "viridis") -> np.ndarray:
    """Heatmap a single-channel map for overlay display."""
    maps = {
        "viridis": cv2.COLORMAP_VIRIDIS,
        "magma": cv2.COLORMAP_MAGMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
        "hot": cv2.COLORMAP_HOT,
        "bone": cv2.COLORMAP_BONE,
        "plasma": cv2.COLORMAP_PLASMA,
    }
    g = gray.astype(np.float32)
    lo, hi = float(g.min()), float(g.max())
    norm = (g - lo) / max(hi - lo, EPS)
    u8 = (norm * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, maps.get(cmap, cv2.COLORMAP_VIRIDIS))
    return (bgr[:, :, ::-1].astype(np.float32) / 255.0).astype(np.float32)


def normalize01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi - lo < EPS:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def structuring_element(
    shape: str, size: int, angle: float = 0.0, custom: list | None = None
) -> np.ndarray:
    """Shared by every morphology operator (catalogue Section 8)."""
    size = max(1, odd(size))
    if shape == "custom" and custom:
        se = np.array(custom, dtype=np.uint8)
        return (se > 0).astype(np.uint8)
    if shape == "rect":
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    if shape == "cross":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    if shape == "diamond":
        r = size // 2
        yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
        return (np.abs(xx) + np.abs(yy) <= r).astype(np.uint8)
    if shape == "line":
        se = np.zeros((size, size), np.uint8)
        c = size // 2
        rad = np.deg2rad(angle)
        dx, dy = np.cos(rad), -np.sin(rad)
        for t in np.linspace(-c, c, size * 4):
            x, y = int(round(c + dx * t)), int(round(c + dy * t))
            if 0 <= x < size and 0 <= y < size:
                se[y, x] = 1
        return se
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def monotone_spline(points: list[tuple[float, float]], size: int = 256) -> np.ndarray:
    """Monotone cubic (Fritsch-Carlson) interpolation for the curves editor.

    A natural cubic spline overshoots and would let a curve invert locally,
    producing solarisation artifacts the user did not ask for. Monotone
    interpolation cannot overshoot.
    """
    pts = sorted(points)
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)

    # Deduplicate x so the slope computation cannot divide by zero.
    keep = np.concatenate([[True], np.diff(xs) > 1e-9])
    xs, ys = xs[keep], ys[keep]
    if len(xs) < 2:
        return np.linspace(0, 1, size, dtype=np.float32)

    n = len(xs)
    h = np.diff(xs)
    delta = np.diff(ys) / h
    m = np.zeros(n)
    m[0], m[-1] = delta[0], delta[-1]
    if n > 2:
        m[1:-1] = (delta[:-1] + delta[1:]) / 2.0
    # Fritsch-Carlson limiter
    for i in range(n - 1):
        if abs(delta[i]) < 1e-12:
            m[i] = m[i + 1] = 0.0
        else:
            a, b = m[i] / delta[i], m[i + 1] / delta[i]
            s = a * a + b * b
            if s > 9.0:
                t = 3.0 / np.sqrt(s)
                m[i], m[i + 1] = t * a * delta[i], t * b * delta[i]

    x = np.linspace(0.0, 1.0, size)
    idx = np.clip(np.searchsorted(xs, x) - 1, 0, n - 2)
    hh = h[idx]
    t = (x - xs[idx]) / hh
    t2, t3 = t * t, t * t * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    y = h00 * ys[idx] + h10 * hh * m[idx] + h01 * ys[idx + 1] + h11 * hh * m[idx + 1]
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def resolve_ref_image(ctx: dict, ref: str | None, like: np.ndarray) -> np.ndarray:
    """Fetch a second image from the store and size it to match."""
    from ..core.errors import SecondImageRequired
    from ..core.store import store

    if not ref:
        raise SecondImageRequired("This operator needs a second image.")
    other = store.get(ref)
    arr = other.get(ctx.get("target", "proxy"))
    if arr.shape[:2] != like.shape[:2]:
        arr = cv2.resize(arr, (like.shape[1], like.shape[0]), interpolation=cv2.INTER_AREA)
    if is_color(like) and not is_color(arr):
        arr = as_color(arr)
    if not is_color(like) and is_color(arr):
        arr = luma(arr)
    return np.ascontiguousarray(arr, dtype=np.float32)


def scaled_odd(ctx: dict, base: int) -> int:
    return odd(max(1, int(round(base * ctx.get("scale_factor", 1.0)))))
