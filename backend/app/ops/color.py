"""Colour operations. Catalogue Section 3."""
from __future__ import annotations

import cv2
import numpy as np

from ..core.colorspace import (
    from_hsv,
    from_lab,
    join_alpha,
    linear_to_srgb,
    luma,
    rgb_to_cct,
    split_alpha,
    srgb_to_linear,
    to_hsv,
    to_lab,
)
from ..core.params import (
    BoolParam,
    ColorParam,
    EnumParam,
    FloatParam,
    ImageRefParam,
    IntParam,
    LutFileParam,
    MatrixParam,
    PointParam,
    RangeParam,
    enum_opts,
)
from ..core.registry import OpResult, register
from ._util import EPS, clamp01, resolve_ref_image

# --------------------------------------------------------------------- white balance


def _illuminant(img: np.ndarray, method: str, percentile: float = 99.0) -> np.ndarray:
    """Estimate the illuminant RGB. advisor-spec.md 2.3."""
    lin = srgb_to_linear(img[:, :, :3])
    flat = lin.reshape(-1, 3)
    if method == "white_patch":
        e = np.percentile(flat, percentile, axis=0)
    elif method == "shades_of_gray":
        p = 6.0
        e = np.power(np.mean(np.power(np.maximum(flat, 0), p), axis=0), 1.0 / p)
    else:  # gray_world
        e = flat.mean(axis=0)
    e = np.maximum(e, 1e-6)
    return (e / e.mean()).astype(np.float32)


def _apply_gains(img: np.ndarray, gains: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Apply channel gains in LINEAR light -- where an illuminant change lives."""
    rgb, alpha = split_alpha(img)
    if rgb.ndim == 2:
        return img
    g = 1.0 + (np.asarray(gains, np.float32) - 1.0) * float(strength)
    lin = srgb_to_linear(rgb) * g[None, None, :]
    return join_alpha(linear_to_srgb(clamp01(lin)), alpha)


def _wb_op(method: str):
    def fn(img, strength=1.0, percentile=99.0, ctx=None):
        if img.ndim == 2:
            return img
        e = _illuminant(img, method, percentile)
        gains = (e.mean() / e).astype(np.float32)
        out = _apply_gains(img, gains, strength)
        return OpResult(
            image=out,
            data={
                "illuminant": [round(float(v), 4) for v in e],
                "gains": [round(float(v), 4) for v in gains],
                "cct_estimate": round(rgb_to_cct(e), 0),
            },
        )

    return fn


register(
    "color.wb_grayworld",
    "White balance (gray world)",
    "color",
    params=[FloatParam("strength", min=0.0, max=1.5, default=1.0, step=0.05)],
    summary="Assumes the average of the scene is achromatic, and scales each "
    "channel so the means agree.",
    formula_latex=r"g_c = \frac{\bar{L}}{\bar{I_c}},\quad I'_c = g_c I_c",
    when_to_use="A quick cast fix on scenes with varied colour content. Fails on "
    "images dominated by one hue, where the average is legitimately not gray.",
    cost="medium",
    references=["Buchsbaum, A spatial processor model for object colour perception, 1980"],
)(_wb_op("gray_world"))

register(
    "color.wb_whitepatch",
    "White balance (white patch)",
    "color",
    params=[
        FloatParam("strength", min=0.0, max=1.5, default=1.0, step=0.05),
        FloatParam("percentile", min=90.0, max=100.0, default=99.0, step=0.1),
    ],
    summary="Max-RGB / Retinex: assumes the brightest pixels are a white "
    "reference and normalises each channel to them.",
    formula_latex=r"g_c = \frac{\max_c I_c \text{ (robust)}}{I_c}",
    when_to_use="Scenes containing a genuine specular highlight or white object. "
    "A percentile rather than a true maximum keeps one hot pixel from ruining it.",
    cost="medium",
    references=["Land & McCann, Lightness and Retinex Theory, 1971"],
)(_wb_op("white_patch"))

register(
    "color.wb_shades_of_gray",
    "White balance (shades of gray)",
    "color",
    params=[FloatParam("strength", min=0.0, max=1.5, default=1.0, step=0.05)],
    summary="Minkowski-norm illuminant estimate (p = 6), interpolating between "
    "gray-world (p=1) and white-patch (p=inf).",
    formula_latex=r"e_c = \left(\int I_c^p \, dx\right)^{1/p},\quad p = 6",
    when_to_use="The most robust of the three estimators, and what the Advisor "
    "combines by median with the other two.",
    tags=["beyond-syllabus"],
    cost="medium",
    references=["Finlayson & Trezzi, Shades of Gray and Colour Constancy, 2004"],
)(_wb_op("shades_of_gray"))


@register(
    "color.wb_picker",
    "White balance (picker)",
    "color",
    params=[
        PointParam("neutral", default=(0.5, 0.5),
                   help="Click a pixel that should be neutral gray."),
        IntParam("sample_radius", min=0, max=32, default=4, scale_aware=True),
        FloatParam("strength", min=0.0, max=1.5, default=1.0, step=0.05),
    ],
    summary="Neutralises a user-picked pixel by scaling each channel so that "
    "sample becomes gray.",
    formula_latex=r"g_c = \frac{\bar{s}}{s_c}",
    when_to_use="The most reliable white balance when the scene contains a known "
    "neutral surface.",
)
def wb_picker(img, neutral=(0.5, 0.5), sample_radius=4, strength=1.0, ctx=None):
    if img.ndim == 2:
        return img
    h, w = img.shape[:2]
    cx, cy = int(neutral[0] * (w - 1)), int(neutral[1] * (h - 1))
    r = int(sample_radius)
    patch = srgb_to_linear(
        img[max(0, cy - r) : cy + r + 1, max(0, cx - r) : cx + r + 1, :3]
    )
    s = np.maximum(patch.reshape(-1, 3).mean(axis=0), 1e-6)
    gains = (s.mean() / s).astype(np.float32)
    return OpResult(
        image=_apply_gains(img, gains, strength),
        data={"sample": [round(float(v), 4) for v in s],
              "gains": [round(float(v), 4) for v in gains]},
    )


@register(
    "color.temp_tint",
    "Temperature / Tint",
    "color",
    params=[
        FloatParam("temp", min=-100.0, max=100.0, default=0.0, step=1.0,
                   help="Negative cools (blue), positive warms (amber)."),
        FloatParam("tint", min=-100.0, max=100.0, default=0.0, step=1.0,
                   help="Negative green, positive magenta."),
    ],
    summary="Manual white-balance control along the blue-amber and green-magenta axes.",
    formula_latex=r"g = (1 + k_t, 1 - k_m, 1 - k_t)",
    when_to_use="Overriding an automatic estimate, or deliberately warming an image.",
)
def temp_tint(img, temp=0.0, tint=0.0, ctx=None):
    if img.ndim == 2:
        return img
    t = temp / 100.0 * 0.35
    m = tint / 100.0 * 0.25
    gains = np.array([1.0 + t, 1.0 - m * 0.5, 1.0 - t], np.float32)
    gains *= 1.0 + m * 0.0
    if m != 0:
        gains[1] -= m * 0.3
    return _apply_gains(img, gains / gains.mean(), 1.0)


@register(
    "color.channel_gains",
    "Channel gains",
    "color",
    params=[
        FloatParam("r", min=0.0, max=3.0, default=1.0, step=0.01),
        FloatParam("g", min=0.0, max=3.0, default=1.0, step=0.01),
        FloatParam("b", min=0.0, max=3.0, default=1.0, step=0.01),
        BoolParam("linear_light", default=True),
    ],
    summary="Direct per-channel multipliers.",
    formula_latex=r"I'_c = g_c\,I_c",
    when_to_use="Manual colour correction, and the operator the Advisor emits "
    "for a measured cast.",
)
def channel_gains(img, r=1.0, g=1.0, b=1.0, linear_light=True, ctx=None):
    if img.ndim == 2:
        return img
    gains = np.array([r, g, b], np.float32)
    if linear_light:
        return _apply_gains(img, gains, 1.0)
    rgb, alpha = split_alpha(img)
    return join_alpha(clamp01(rgb * gains[None, None, :]), alpha)


@register(
    "color.channel_mixer",
    "Channel mixer",
    "color",
    params=[
        MatrixParam(
            "matrix", rows=3, cols=3,
            default=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            min=-2.0, max=2.0,
            presets=[
                {"label": "Identity", "value": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                {"label": "Sepia", "value": [[0.393, 0.769, 0.189],
                                             [0.349, 0.686, 0.168],
                                             [0.272, 0.534, 0.131]]},
                {"label": "B&W (luminosity)", "value": [[0.2126, 0.7152, 0.0722]] * 3},
                {"label": "Swap R/B", "value": [[0, 0, 1], [0, 1, 0], [1, 0, 0]]},
            ],
        )
    ],
    summary="A full 3x3 linear transform of the colour vector.",
    formula_latex=r"\begin{bmatrix}R'\\G'\\B'\end{bmatrix} = M "
    r"\begin{bmatrix}R\\G\\B\end{bmatrix}",
    when_to_use="Any linear colour remap: sepia, custom black-and-white mixes, "
    "channel swaps, cross-processing.",
    requires={"channels": 3},
)
def channel_mixer(img, matrix=None, ctx=None):
    if img.ndim == 2:
        return img
    m = np.array(matrix, np.float32)
    if m.shape != (3, 3):
        return img
    rgb, alpha = split_alpha(img)
    out = np.einsum("ij,hwj->hwi", m, rgb[:, :, :3])
    return join_alpha(clamp01(out), alpha)


@register(
    "color.hsv_shift",
    "Hue / Saturation / Value",
    "color",
    params=[
        FloatParam("hue", min=-180.0, max=180.0, default=0.0, step=1.0, unit="deg"),
        FloatParam("saturation", min=-1.0, max=2.0, default=0.0, step=0.01),
        FloatParam("value", min=-1.0, max=1.0, default=0.0, step=0.01),
    ],
    summary="Shifts hue angle and scales saturation and value in HSV space.",
    formula_latex=r"H' = (H + \Delta) \bmod 360,\ S' = S(1+s),\ V' = V(1+v)",
    when_to_use="Broad colour adjustments. Note that scaling saturation flatly "
    "also saturates already-vivid pixels -- see Vibrance for the better option.",
    requires={"channels": 3},
)
def hsv_shift(img, hue=0.0, saturation=0.0, value=0.0, ctx=None):
    if img.ndim == 2:
        return img
    rgb, alpha = split_alpha(img)
    hsv = to_hsv(rgb)
    hsv[:, :, 0] = np.mod(hsv[:, :, 0] + hue, 360.0)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + saturation), 0, 1)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + value), 0, 1)
    return join_alpha(from_hsv(hsv), alpha)


@register(
    "color.vibrance",
    "Vibrance",
    "color",
    params=[
        FloatParam("amount", min=-1.0, max=1.5, default=0.0, step=0.01),
        BoolParam("protect_skin", default=True,
                  help="Reduces the effect in the orange hue band."),
    ],
    summary="Saturation weighted by (1 - S), so muted colours gain much more "
    "than already-saturated ones.",
    formula_latex=r"S' = S + a\,S\,(1-S)^{\,2}",
    when_to_use="The right way to make colours richer without turning reds into "
    "flat blocks. This is what the Advisor emits instead of flat saturation.",
    tags=["beyond-syllabus"],
    requires={"channels": 3},
)
def vibrance(img, amount=0.0, protect_skin=True, ctx=None):
    if img.ndim == 2:
        return img
    rgb, alpha = split_alpha(img)
    hsv = to_hsv(rgb)
    s = hsv[:, :, 1]
    w = (1.0 - s) ** 2
    if protect_skin:
        h = hsv[:, :, 0]
        # Skin sits roughly in 10..50 degrees; taper the effect there.
        skin = np.exp(-(((h - 30.0) / 28.0) ** 2))
        w = w * (1.0 - 0.6 * skin)
    hsv[:, :, 1] = np.clip(s + amount * s * w + amount * 0.12 * w, 0, 1)
    return join_alpha(from_hsv(hsv), alpha)


@register(
    "color.selective",
    "Selective colour",
    "color",
    params=[
        RangeParam("hue_window", min=0.0, max=360.0, default=(0.0, 60.0), step=1.0),
        FloatParam("hue_shift", min=-180.0, max=180.0, default=0.0, step=1.0),
        FloatParam("saturation", min=-1.0, max=2.0, default=0.0, step=0.01),
        FloatParam("lightness", min=-1.0, max=1.0, default=0.0, step=0.01),
        FloatParam("feather", min=0.0, max=90.0, default=15.0, step=1.0, unit="deg"),
    ],
    summary="Adjusts only the pixels inside a hue window, with a feathered "
    "transition so the edit does not produce visible boundaries.",
    formula_latex=r"w(H) = \text{smooth window}(H; h_0, h_1, f)",
    when_to_use="Targeting one colour family -- deepening a sky without touching "
    "foliage.",
    requires={"channels": 3},
)
def selective_color(img, hue_window=(0.0, 60.0), hue_shift=0.0, saturation=0.0,
                    lightness=0.0, feather=15.0, ctx=None):
    if img.ndim == 2:
        return img
    rgb, alpha = split_alpha(img)
    hsv = to_hsv(rgb)
    h = hsv[:, :, 0]
    h0, h1 = hue_window
    # Angular distance to the window, handling wraparound.
    def ang_dist(a, b):
        d = np.abs(np.mod(a - b + 180.0, 360.0) - 180.0)
        return d

    centre = np.mod((h0 + h1) / 2.0 if h0 <= h1 else (h0 + h1 + 360.0) / 2.0, 360.0)
    half = (abs(h1 - h0) if h0 <= h1 else (360.0 - h0 + h1)) / 2.0
    d = ang_dist(h, centre)
    w = np.clip((half + feather - d) / max(feather, 1e-3), 0, 1).astype(np.float32)

    hsv[:, :, 0] = np.mod(h + hue_shift * w, 360.0)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + saturation * w), 0, 1)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + lightness * w), 0, 1)
    return join_alpha(from_hsv(hsv), alpha)


@register(
    "color.to_gray",
    "Colour to gray",
    "color",
    params=[
        EnumParam(
            "method",
            options=enum_opts(
                ("luminosity", "Luminosity (BT.709)"),
                ("bt601", "Luminosity (BT.601)"),
                ("average", "Average"),
                ("lightness", "Lightness (max+min)/2"),
                ("lab_l", "CIE L*"),
                ("desaturate", "Desaturate (HSV V)"),
                ("r", "Red channel"),
                ("g", "Green channel"),
                ("b", "Blue channel"),
                ("custom", "Custom weights"),
            ),
            default="luminosity",
        ),
        FloatParam("w_r", min=0.0, max=1.0, default=0.299, step=0.001,
                   depends_on={"param": "method", "equals": "custom"}),
        FloatParam("w_g", min=0.0, max=1.0, default=0.587, step=0.001,
                   depends_on={"param": "method", "equals": "custom"}),
        FloatParam("w_b", min=0.0, max=1.0, default=0.114, step=0.001,
                   depends_on={"param": "method", "equals": "custom"}),
    ],
    summary="Nine different ways to discard colour, which do NOT agree: a "
    "saturated red object is mid-gray under BT.709 and nearly white under the "
    "red channel.",
    formula_latex=r"Y = w_R R + w_G G + w_B B",
    when_to_use="Before any grayscale operator. The choice of method materially "
    "changes what survives into the gray image.",
    tags=["teaching"],
)
def to_gray_op(img, method="luminosity", w_r=0.299, w_g=0.587, w_b=0.114, ctx=None):
    if img.ndim == 2:
        return img
    rgb = img[:, :, :3]
    if method == "luminosity":
        g = luma(rgb)
    elif method == "bt601":
        g = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    elif method == "average":
        g = rgb.mean(axis=2)
    elif method == "lightness":
        g = (rgb.max(axis=2) + rgb.min(axis=2)) / 2.0
    elif method == "lab_l":
        g = to_lab(rgb)[:, :, 0] / 100.0
    elif method == "desaturate":
        g = rgb.max(axis=2)
    elif method in ("r", "g", "b"):
        g = rgb[:, :, {"r": 0, "g": 1, "b": 2}[method]]
    else:
        w = np.array([w_r, w_g, w_b], np.float32)
        w = w / max(w.sum(), EPS)
        g = rgb @ w
    return clamp01(g)


@register(
    "color.sepia",
    "Sepia / Duotone",
    "color",
    params=[
        ColorParam("shadow", default=(0.17, 0.11, 0.07)),
        ColorParam("highlight", default=(1.0, 0.94, 0.80)),
        FloatParam("strength", min=0.0, max=1.0, default=1.0, step=0.05),
    ],
    summary="Maps luminance onto a two-colour gradient.",
    formula_latex=r"I' = (1-Y)\,c_{shadow} + Y\,c_{highlight}",
    when_to_use="Toning, split-toning and film emulation.",
)
def sepia(img, shadow=(0.17, 0.11, 0.07), highlight=(1.0, 0.94, 0.80),
          strength=1.0, ctx=None):
    y = clamp01(luma(img))[:, :, None]
    lo = np.array(shadow, np.float32)[None, None, :]
    hi = np.array(highlight, np.float32)[None, None, :]
    toned = lo * (1.0 - y) + hi * y
    rgb, alpha = split_alpha(img)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    return join_alpha(clamp01(rgb * (1 - strength) + toned * strength), alpha)


@register(
    "color.lut",
    "Apply 3D LUT",
    "color",
    params=[
        LutFileParam("lut", help="A .cube file (Adobe/Resolve interchange format)."),
        FloatParam("strength", min=0.0, max=1.0, default=1.0, step=0.05),
    ],
    summary="Applies a 3D colour lookup table with trilinear interpolation.",
    formula_latex=r"I' = \text{trilerp}\big(\text{LUT},\ I\big)",
    when_to_use="Industry-standard colour grading interchange; one LUT can encode "
    "an entire look.",
    tags=["beyond-syllabus"],
    requires={"channels": 3},
)
def apply_3d_lut(img, lut=None, strength=1.0, ctx=None):
    if img.ndim == 2 or not lut:
        return img
    table, size = _parse_cube(lut)
    if table is None:
        return OpResult(image=img, notes=["LUT could not be parsed."])
    rgb, alpha = split_alpha(img)
    x = clamp01(rgb[:, :, :3]) * (size - 1)
    i0 = np.floor(x).astype(np.int32)
    i1 = np.minimum(i0 + 1, size - 1)
    f = (x - i0).astype(np.float32)

    def at(ir, ig, ib):
        return table[ib * size * size + ig * size + ir]

    r0, g0, b0 = i0[:, :, 0], i0[:, :, 1], i0[:, :, 2]
    r1, g1, b1 = i1[:, :, 0], i1[:, :, 1], i1[:, :, 2]
    fr, fg, fb = f[:, :, 0:1], f[:, :, 1:2], f[:, :, 2:3]

    c00 = at(r0, g0, b0) * (1 - fr) + at(r1, g0, b0) * fr
    c01 = at(r0, g0, b1) * (1 - fr) + at(r1, g0, b1) * fr
    c10 = at(r0, g1, b0) * (1 - fr) + at(r1, g1, b0) * fr
    c11 = at(r0, g1, b1) * (1 - fr) + at(r1, g1, b1) * fr
    c0 = c00 * (1 - fg) + c10 * fg
    c1 = c01 * (1 - fg) + c11 * fg
    out = c0 * (1 - fb) + c1 * fb
    return join_alpha(clamp01(rgb * (1 - strength) + out * strength), alpha)


def _parse_cube(payload) -> tuple[np.ndarray | None, int]:
    """Parse a .cube LUT supplied as text."""
    try:
        text = payload if isinstance(payload, str) else payload.get("text", "")
        size = 0
        vals: list[list[float]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("TITLE"):
                continue
            if line.upper().startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
                continue
            if line.upper().startswith(("DOMAIN_", "LUT_1D")):
                continue
            parts = line.split()
            if len(parts) == 3:
                try:
                    vals.append([float(p) for p in parts])
                except ValueError:
                    continue
        if size <= 1 or len(vals) < size**3:
            return None, 0
        return np.array(vals[: size**3], np.float32), size
    except Exception:
        return None, 0


@register(
    "color.transfer",
    "Colour transfer",
    "color",
    params=[
        ImageRefParam("reference", help="Image whose palette to adopt."),
        FloatParam("strength", min=0.0, max=1.0, default=1.0, step=0.05),
        EnumParam("space", options=enum_opts(("lab", "CIE Lab"), ("rgb", "RGB")),
                  default="lab"),
    ],
    summary="Transfers the mean and standard deviation of each channel from a "
    "reference image, in a decorrelated colour space.",
    formula_latex=r"I' = \frac{\sigma_{ref}}{\sigma_{src}}\,(I - \mu_{src}) + \mu_{ref}",
    when_to_use="Adopting another photograph's entire colour mood with no model "
    "and no manual grading. Lab is used because its axes are near-independent, "
    "so the three transfers do not fight each other.",
    tags=["beyond-syllabus"],
    requires={"channels": 3, "second_image": True, "second_image_param": "reference"},
    references=["Reinhard et al., Color Transfer between Images, IEEE CG&A 2001"],
)
def color_transfer(img, reference=None, strength=1.0, space="lab", ctx=None):
    ctx = ctx or {}
    if img.ndim == 2:
        return img
    ref = resolve_ref_image(ctx, reference, img)
    rgb, alpha = split_alpha(img)
    src = to_lab(rgb) if space == "lab" else rgb[:, :, :3].copy()
    tgt = to_lab(ref) if space == "lab" else ref[:, :, :3]

    out = src.copy()
    for c in range(3):
        ms, ss = float(src[:, :, c].mean()), float(src[:, :, c].std())
        mt, st = float(tgt[:, :, c].mean()), float(tgt[:, :, c].std())
        if ss < 1e-6:
            continue
        out[:, :, c] = (src[:, :, c] - ms) * (st / ss) + mt

    out = src * (1 - strength) + out * strength
    result = from_lab(out) if space == "lab" else clamp01(out)
    return join_alpha(result, alpha)


@register(
    "color.gamut_clip",
    "Gamut clip / soft clip",
    "color",
    params=[
        EnumParam("mode", options=enum_opts(("soft", "Soft knee"),
                                            ("desaturate", "Desaturate highlights"),
                                            ("hard", "Hard clip")),
                  default="soft"),
        FloatParam("knee", min=0.5, max=1.0, default=0.85, step=0.01),
    ],
    summary="Handles out-of-range channels gracefully instead of letting them "
    "clip independently, which shifts hue.",
    formula_latex=r"I' = k + (1-k)\tanh\!\left(\frac{I-k}{1-k}\right)",
    when_to_use="After an aggressive exposure or saturation boost. Fixes the "
    "magenta/cyan fringing that appears when only one channel clips.",
    requires={"channels": 3},
)
def gamut_clip(img, mode="soft", knee=0.85, ctx=None):
    rgb, alpha = split_alpha(img)
    if mode == "hard":
        return join_alpha(clamp01(rgb), alpha)
    if mode == "soft":
        x = rgb.astype(np.float32)
        over = x > knee
        out = x.copy()
        out[over] = knee + (1.0 - knee) * np.tanh((x[over] - knee) / max(1.0 - knee, EPS))
        return join_alpha(clamp01(out), alpha)
    # desaturate: pull colours toward luma as they approach clipping
    if rgb.ndim == 2:
        return join_alpha(clamp01(rgb), alpha)
    y = luma(rgb)[:, :, None]
    excess = np.clip((rgb.max(axis=2, keepdims=True) - knee) / max(1 - knee, EPS), 0, 1)
    return join_alpha(clamp01(rgb * (1 - excess) + y * excess), alpha)


@register(
    "color.ca_correct",
    "Chromatic aberration fix",
    "color",
    params=[
        FloatParam("r_scale", min=0.990, max=1.010, default=1.0, step=0.0001),
        FloatParam("b_scale", min=0.990, max=1.010, default=1.0, step=0.0001),
        BoolParam("auto", default=False,
                  help="Estimate the per-channel scale by phase correlation."),
    ],
    summary="Rescales the red and blue channels about the image centre to "
    "re-register them with green, removing lateral chromatic aberration.",
    formula_latex=r"I'_c(x) = I_c\big(s_c (x - x_0) + x_0\big)",
    when_to_use="Purple/green fringing on high-contrast edges toward the frame "
    "corners. The auto mode measures the misregistration rather than guessing.",
    tags=["beyond-syllabus"],
    requires={"channels": 3},
)
def ca_correct(img, r_scale=1.0, b_scale=1.0, auto=False, ctx=None):
    if img.ndim == 2:
        return img
    rgb, alpha = split_alpha(img)
    h, w = rgb.shape[:2]

    if auto:
        r_scale, b_scale = _estimate_ca(rgb)

    def rescale(ch, s):
        if abs(s - 1.0) < 1e-6:
            return ch
        m = np.array([[s, 0, (1 - s) * (w - 1) / 2.0],
                      [0, s, (1 - s) * (h - 1) / 2.0]], np.float32)
        return cv2.warpAffine(ch, m, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REFLECT_101)

    out = np.dstack([
        rescale(rgb[:, :, 0], r_scale),
        rgb[:, :, 1],
        rescale(rgb[:, :, 2], b_scale),
    ])
    return OpResult(
        image=join_alpha(clamp01(out), alpha),
        data={"r_scale": round(float(r_scale), 5), "b_scale": round(float(b_scale), 5)},
    )


def _estimate_ca(rgb: np.ndarray) -> tuple[float, float]:
    """Estimate radial scale per channel by matching edge magnitude profiles.

    Lateral CA scales R and B relative to G about the optical centre, so the
    radial position of an edge differs per channel. We search a small range of
    scales for the one maximising correlation with green's edge map.
    """
    g = cv2.Sobel(rgb[:, :, 1], cv2.CV_32F, 1, 1, ksize=3)
    g = np.abs(g)
    h, w = g.shape
    best = []
    for ci in (0, 2):
        ch = np.abs(cv2.Sobel(rgb[:, :, ci], cv2.CV_32F, 1, 1, ksize=3))
        scores = []
        for s in np.linspace(0.996, 1.004, 17):
            m = np.array([[s, 0, (1 - s) * (w - 1) / 2.0],
                          [0, s, (1 - s) * (h - 1) / 2.0]], np.float32)
            warped = cv2.warpAffine(ch, m, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT_101)
            scores.append((float(np.sum(warped * g)), float(s)))
        best.append(max(scores)[1])
    return best[0], best[1]
