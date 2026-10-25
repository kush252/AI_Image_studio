"""The Advisor evaluation set. docs/advisor-spec.md Section 9.

Twenty cases built as synthetic degradations of clean plates, so the ground
truth is known exactly: each entry names the rule that MUST fire. The last case
is an unmodified plate, where nothing above "minor" may fire -- a diagnostic
tool that always finds something wrong is not diagnosing.
"""
from __future__ import annotations

import cv2
import numpy as np

from .fixtures.synth import plate, plate_with_lines


def _motion_kernel(length: float, angle_deg: float) -> np.ndarray:
    n = max(3, int(round(length)) | 1)
    k = np.zeros((n, n), np.float32)
    c = n // 2
    rad = np.deg2rad(angle_deg)
    dx, dy = np.cos(rad), -np.sin(rad)
    for t in np.linspace(-length / 2, length / 2, int(length * 4) + 2):
        x, y = int(round(c + dx * t)), int(round(c + dy * t))
        if 0 <= x < n and 0 <= y < n:
            k[y, x] = 1.0
    return k / max(k.sum(), 1e-8)


def degrade(img: np.ndarray, name: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(1000 + seed)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    if name == "clean":
        return img
    if name == "under":
        return np.clip(img ** 2.4, 0, 1).astype(np.float32)
    if name == "over":
        return np.clip(img ** 0.40, 0, 1).astype(np.float32)
    if name == "flat":
        return np.clip((img - float(img.mean())) * 0.22 + 0.5, 0, 1).astype(np.float32)
    if name == "warm":
        return np.clip(img * np.array([1.28, 1.0, 0.76], np.float32), 0, 1)
    if name == "cool":
        return np.clip(img * np.array([0.78, 1.0, 1.26], np.float32), 0, 1)
    if name == "blur":
        return cv2.GaussianBlur(img, (0, 0), 2.0)
    if name == "motion":
        return cv2.filter2D(img, -1, _motion_kernel(15.0, 30.0),
                            borderType=cv2.BORDER_REFLECT_101)
    if name == "noise10":
        return np.clip(img + rng.normal(0, 10 / 255, img.shape), 0, 1).astype(np.float32)
    if name == "noise25":
        return np.clip(img + rng.normal(0, 25 / 255, img.shape), 0, 1).astype(np.float32)
    if name == "saltpepper":
        o = img.copy()
        n = int(0.02 * h * w)
        ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
        s = rng.random(n) < 0.5
        o[ys[s], xs[s]] = 1.0
        o[ys[~s], xs[~s]] = 0.0
        return o
    if name == "periodic":
        pat = 0.10 * np.sin(2 * np.pi * (0.12 * xx + 0.08 * yy))
        return np.clip(img + pat[:, :, None], 0, 1).astype(np.float32)
    if name == "chroma_noise":
        ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        ycc[:, :, 1] += rng.normal(0, 20 / 255, (h, w)).astype(np.float32)
        ycc[:, :, 2] += rng.normal(0, 20 / 255, (h, w)).astype(np.float32)
        return np.clip(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB), 0, 1).astype(np.float32)
    if name == "haze":
        depth = np.clip(0.55 * (1 - yy / max(h - 1, 1)) + 0.45 *
                        cv2.GaussianBlur(rng.random((h, w)).astype(np.float32),
                                         (0, 0), max(h, w) / 12.0), 0, 1)
        t = np.exp(-2.2 * depth)[:, :, None]
        return np.clip(img * t + 0.88 * (1 - t), 0, 1).astype(np.float32)
    if name == "tilt":
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 1.8, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REFLECT_101)
    if name == "vignette":
        r = np.sqrt((xx - (w - 1) / 2) ** 2 + (yy - (h - 1) / 2) ** 2) / np.sqrt(
            2 * ((w - 1) / 2) ** 2)
        return np.clip(img * (1 - 0.30 * r ** 2)[:, :, None], 0, 1).astype(np.float32)
    if name == "illum":
        ramp = 0.55 + 0.45 * xx / max(w - 1, 1)
        return np.clip(img * ramp[:, :, None], 0, 1).astype(np.float32)
    if name == "jpeg":
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
        ok, enc = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, 20])
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return (dec[:, :, ::-1].astype(np.float32) / 255.0)
    if name == "banding":
        return (np.floor(np.clip(img, 0, 1) * 31 + 0.5) / 31).astype(np.float32)
    if name == "oversharp":
        b = cv2.GaussianBlur(img, (0, 0), 1.2)
        return np.clip(img + 2.5 * (img - b), 0, 1).astype(np.float32)
    raise ValueError(f"unknown degradation {name!r}")


# (degradation, rule that must fire). None = nothing above "minor" may fire.
EVAL_SET: list[tuple[str, str | None]] = [
    ("under", "EXP_UNDER"),
    ("over", "EXP_OVER"),
    ("flat", "CON_FLAT"),
    ("warm", "COL_CAST"),
    ("cool", "COL_CAST"),
    ("blur", "SHP_SOFT"),
    ("motion", "SHP_MOTION"),
    ("noise10", "NSE_GAUSS"),
    ("noise25", "NSE_GAUSS"),
    ("saltpepper", "NSE_IMPULSE"),
    ("periodic", "NSE_PERIODIC"),
    ("chroma_noise", "NSE_CHROMA"),
    ("haze", "UNI_HAZE"),
    ("tilt", "GEO_TILT"),
    ("vignette", "UNI_VIGNETTE"),
    ("illum", "UNI_ILLUM"),
    ("jpeg", "ART_BLOCK"),
    ("banding", "ART_BAND"),
    ("oversharp", "SHP_OVER"),
    ("clean", None),
]


# Cases whose defect can only be detected on content with linear structure.
NEEDS_LINES = {"tilt"}


def build_case(index: int) -> tuple[str, np.ndarray, str | None]:
    name, expect = EVAL_SET[index]
    maker = plate_with_lines if name in NEEDS_LINES else plate
    base = maker(index % 5)
    return name, degrade(base, name, index), expect
