"""Per-operator blend modes. PLAN.md Phase 9.

Every function takes (base, blend) as float32 in [0,1] and returns float32.
"""
from __future__ import annotations

import numpy as np

from .colorspace import luma


def _normal(b: np.ndarray, s: np.ndarray) -> np.ndarray:
    return s


def _multiply(b: np.ndarray, s: np.ndarray) -> np.ndarray:
    return b * s


def _screen(b: np.ndarray, s: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - b) * (1.0 - s)


def _overlay(b: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.where(b <= 0.5, 2.0 * b * s, 1.0 - 2.0 * (1.0 - b) * (1.0 - s))


def _hard_light(b: np.ndarray, s: np.ndarray) -> np.ndarray:
    return _overlay(s, b)


def _soft_light(b: np.ndarray, s: np.ndarray) -> np.ndarray:
    # W3C / SVG definition
    d = np.where(
        b <= 0.25, ((16.0 * b - 12.0) * b + 4.0) * b, np.sqrt(np.maximum(b, 0.0))
    )
    return np.where(
        s <= 0.5,
        b - (1.0 - 2.0 * s) * b * (1.0 - b),
        b + (2.0 * s - 1.0) * (d - b),
    )


def _difference(b: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.abs(b - s)


def _exclusion(b: np.ndarray, s: np.ndarray) -> np.ndarray:
    return b + s - 2.0 * b * s


def _darken(b: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.minimum(b, s)


def _lighten(b: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.maximum(b, s)


def _color_dodge(b: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.where(s >= 1.0, 1.0, np.minimum(1.0, b / np.maximum(1.0 - s, 1e-6)))


def _color_burn(b: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.where(s <= 0.0, 0.0, 1.0 - np.minimum(1.0, (1.0 - b) / np.maximum(s, 1e-6)))


def _luminosity(b: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Take the blend layer's luminance, keep the base's colour."""
    if b.ndim == 2:
        return s
    lb, ls = luma(b), luma(s)
    return np.clip(b + (ls - lb)[:, :, None], 0.0, 1.0)


def _color(b: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Take the blend layer's colour, keep the base's luminance."""
    if b.ndim == 2:
        return b
    lb, ls = luma(b), luma(s)
    return np.clip(s + (lb - ls)[:, :, None], 0.0, 1.0)


MODES = {
    "normal": _normal,
    "multiply": _multiply,
    "screen": _screen,
    "overlay": _overlay,
    "soft_light": _soft_light,
    "hard_light": _hard_light,
    "difference": _difference,
    "exclusion": _exclusion,
    "darken": _darken,
    "lighten": _lighten,
    "color_dodge": _color_dodge,
    "color_burn": _color_burn,
    "luminosity": _luminosity,
    "color": _color,
}

MODE_LABELS = [
    {"value": k, "label": k.replace("_", " ").title()} for k in MODES
]


def apply_blend(
    base: np.ndarray,
    result: np.ndarray,
    mode: str = "normal",
    opacity: float = 1.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Composite `result` over `base`.

    Phase 9 acceptance requires the effect to be *exactly* zero where the mask
    is zero, so the mask multiply happens on the final alpha, not inside the
    blend function.
    """
    if base.shape != result.shape:
        # Shape-changing operators (crop, resize, warp) cannot be blended.
        return result

    fn = MODES.get(mode, _normal)
    blended = fn(base.astype(np.float32), result.astype(np.float32))
    blended = np.nan_to_num(blended, nan=0.0, posinf=1.0, neginf=0.0)

    alpha = float(np.clip(opacity, 0.0, 1.0))
    if mask is not None:
        m = mask.astype(np.float32)
        if m.ndim == 2 and base.ndim == 3:
            m = m[:, :, None]
        a = m * alpha
        return (base * (1.0 - a) + blended * a).astype(np.float32)

    if alpha >= 1.0:
        return blended.astype(np.float32)
    if alpha <= 0.0:
        return base.astype(np.float32)
    return (base * (1.0 - alpha) + blended * alpha).astype(np.float32)
