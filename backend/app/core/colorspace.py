"""Colour-space conversions and luma. See PLAN.md D4.

Everything here consumes and returns float32 in [0, 1] unless stated otherwise.
"""
from __future__ import annotations

import cv2
import numpy as np

EPS = 1e-8

# ITU-R BT.709 luma weights (advisor-spec.md Section 1)
BT709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
BT601 = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def is_color(img: np.ndarray) -> bool:
    return img.ndim == 3 and img.shape[2] >= 3


def luma(img: np.ndarray, weights: np.ndarray = BT709) -> np.ndarray:
    """BT.709 luma of a float32 image. Returns a 2-D float32 array."""
    if img.ndim == 2:
        return img.astype(np.float32, copy=False)
    rgb = img[:, :, :3].astype(np.float32, copy=False)
    return np.tensordot(rgb, weights, axes=([2], [0])).astype(np.float32)


def to_gray(img: np.ndarray) -> np.ndarray:
    return luma(img)


def as_color(img: np.ndarray) -> np.ndarray:
    """Promote grayscale to 3-channel so colour operators can run uniformly."""
    if img.ndim == 2:
        return np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    return img


def split_alpha(img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Detach an alpha channel so operators never have to think about it."""
    if img.ndim == 3 and img.shape[2] == 4:
        return np.ascontiguousarray(img[:, :, :3]), np.ascontiguousarray(img[:, :, 3])
    return img, None


def join_alpha(img: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if alpha is None:
        return img
    img = as_color(img)
    if img.shape[:2] != alpha.shape[:2]:
        alpha = cv2.resize(alpha, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.dstack([img, alpha])


# --------------------------------------------------------------------------
# sRGB transfer functions (D4: "work in linear light")
# --------------------------------------------------------------------------

def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """sRGB EOTF. Exact piecewise form, not the 2.2 approximation."""
    x = np.clip(img.astype(np.float32, copy=False), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """sRGB OETF (inverse of srgb_to_linear)."""
    x = np.clip(img.astype(np.float32, copy=False), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055).astype(np.float32)


# --------------------------------------------------------------------------
# Wrapped OpenCV conversions (float32-safe)
# --------------------------------------------------------------------------

def to_hsv(img: np.ndarray) -> np.ndarray:
    """H in [0, 360), S and V in [0, 1]."""
    return cv2.cvtColor(as_color(img).astype(np.float32), cv2.COLOR_RGB2HSV)


def from_hsv(hsv: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB), 0, 1)


def to_hls(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(as_color(img).astype(np.float32), cv2.COLOR_RGB2HLS)


def from_hls(hls: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(hls.astype(np.float32), cv2.COLOR_HLS2RGB), 0, 1)


def to_lab(img: np.ndarray) -> np.ndarray:
    """CIE Lab: L in [0, 100], a and b roughly in [-128, 127]."""
    return cv2.cvtColor(as_color(img).astype(np.float32), cv2.COLOR_RGB2Lab)


def from_lab(lab: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2RGB), 0, 1)


def to_ycrcb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(as_color(img).astype(np.float32), cv2.COLOR_RGB2YCrCb)


def from_ycrcb(ycc: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(ycc.astype(np.float32), cv2.COLOR_YCrCb2RGB), 0, 1)


def to_cmyk(img: np.ndarray) -> np.ndarray:
    """Naive (device-independent) CMYK, for the channel-splitter view only."""
    rgb = as_color(img)
    k = 1.0 - np.max(rgb[:, :, :3], axis=2)
    denom = np.maximum(1.0 - k, EPS)
    c = (1.0 - rgb[:, :, 0] - k) / denom
    m = (1.0 - rgb[:, :, 1] - k) / denom
    y = (1.0 - rgb[:, :, 2] - k) / denom
    return np.clip(np.dstack([c, m, y, k]), 0, 1).astype(np.float32)


SPACES = {
    "rgb": (lambda im: as_color(im), ("R", "G", "B")),
    "hsv": (to_hsv, ("H", "S", "V")),
    "hls": (to_hls, ("H", "L", "S")),
    "lab": (to_lab, ("L*", "a*", "b*")),
    "ycrcb": (to_ycrcb, ("Y", "Cr", "Cb")),
    "cmyk": (to_cmyk, ("C", "M", "Y", "K")),
}


# --------------------------------------------------------------------------
# Correlated colour temperature (advisor-spec.md 2.3)
# --------------------------------------------------------------------------

_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)


def rgb_to_cct(rgb: np.ndarray) -> float:
    """McCamy's cubic approximation of CCT from a linear RGB illuminant estimate."""
    v = np.asarray(rgb, dtype=np.float64).reshape(3)
    v = np.maximum(v, 1e-6)
    xyz = _RGB_TO_XYZ @ v
    s = float(xyz.sum())
    if s <= 0:
        return float("nan")
    x, y = float(xyz[0] / s), float(xyz[1] / s)
    denom = 0.1858 - y
    if abs(denom) < 1e-9:
        return float("nan")
    n = (x - 0.3320) / denom
    return float(449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33)


def cct_label(cct: float) -> str:
    if not np.isfinite(cct):
        return "indeterminate"
    if cct < 3200:
        return "very warm"
    if cct < 4500:
        return "warm"
    if cct < 5500:
        return "neutral-warm"
    if cct < 7000:
        return "neutral"
    if cct < 9000:
        return "cool"
    return "very cool"
