"""Compose / pipeline tools: masks, blending, ROI. Catalogue Section 12."""
from __future__ import annotations

import cv2
import numpy as np

from ..core.blend import MODE_LABELS, apply_blend
from ..core.colorspace import join_alpha, luma, split_alpha
from ..core.masks import (
    brush_mask,
    color_range_mask,
    gradient_mask,
    luminosity_mask,
    rect_mask,
)
from ..core.params import (
    BoolParam,
    EnumParam,
    FloatParam,
    ImageRefParam,
    MaskRefParam,
    PointParam,
    RangeParam,
    RectParam,
    StringParam,
    enum_opts,
)
from ..core.registry import OpResult, register
from ._util import EPS, clamp01, colorize, resolve_ref_image

BLEND_OPTS = [{"value": m["value"], "label": m["label"]} for m in MODE_LABELS]


@register(
    "compose.blend",
    "Blend with image",
    "compose",
    params=[
        ImageRefParam("reference"),
        EnumParam("mode", options=BLEND_OPTS, default="normal"),
        FloatParam("opacity", min=0.0, max=1.0, default=0.5, step=0.01),
    ],
    summary="Composites a second image over the current one using any of the "
    "thirteen blend modes.",
    formula_latex=r"g = (1-\alpha)\,b + \alpha\,B(b, s)",
    when_to_use="Double exposures, texture overlays, and applying a correction "
    "layer at partial strength.",
    requires={"second_image": True, "second_image_param": "reference"},
)
def blend_with(img, reference=None, mode="normal", opacity=0.5, ctx=None):
    ctx = ctx or {}
    other = resolve_ref_image(ctx, reference, img)
    if other.shape != img.shape:
        other = cv2.resize(other, (img.shape[1], img.shape[0]),
                           interpolation=cv2.INTER_AREA)
        if img.ndim == 3 and other.ndim == 2:
            other = np.dstack([other] * img.shape[2])
        elif img.ndim == 2 and other.ndim == 3:
            other = luma(other)
    return clamp01(apply_blend(img, other, mode, opacity))


@register(
    "compose.difference",
    "Difference view",
    "compose",
    params=[
        ImageRefParam("reference", help="Empty compares against the pipeline input."),
        FloatParam("gain", min=1.0, max=50.0, default=8.0, step=0.5),
        BoolParam("signed", default=False,
                  help="Centres on mid-gray so the sign of the change is visible."),
        BoolParam("heatmap", default=False),
    ],
    summary="Amplified absolute (or signed) difference against a reference.",
    formula_latex=r"d = k\,|f - g| \quad\text{or}\quad d = k(f - g) + 0.5",
    when_to_use="Verifying what an operator actually changed. At gain 8 a "
    "difference invisible to the eye becomes obvious, which is the honest way "
    "to check that a 'subtle' filter did anything at all.",
)
def difference(img, reference=None, gain=8.0, signed=False, heatmap=False, ctx=None):
    ctx = ctx or {}
    if reference:
        ref = resolve_ref_image(ctx, reference, img)
    elif ctx.get("original") is not None:
        ref = ctx["original"]
        if ref.shape[:2] != img.shape[:2]:
            ref = cv2.resize(ref, (img.shape[1], img.shape[0]),
                             interpolation=cv2.INTER_AREA)
    else:
        return OpResult(image=img, notes=["No reference available."])

    a = img if img.ndim == ref.ndim else (luma(img) if ref.ndim == 2 else img)
    if a.shape != ref.shape:
        ref = cv2.resize(ref, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        if a.ndim == 3 and ref.ndim == 2:
            ref = np.dstack([ref] * a.shape[2])

    d = (a - ref) * gain
    out = clamp01(d + 0.5) if signed else clamp01(np.abs(d))
    mag = np.abs(a - ref)
    res = OpResult(image=out, data={
        "max_difference": round(float(mag.max()), 6),
        "mean_difference": round(float(mag.mean()), 6),
        "changed_fraction": round(float((mag > 1 / 255).mean()), 5),
    })
    if heatmap:
        m = luma(mag) if mag.ndim == 3 else mag
        res.overlays["heatmap"] = colorize(np.clip(m * gain, 0, 1), "turbo")
    return res


def _mask_builder(op_id, label, build, params, doc, formula, when, requires=None):
    def fn(img, preview="overlay", invert=False, feather=0.02, opacity=1.0,
           ctx=None, **kw):
        m = build(img, kw)
        if m is None:
            return OpResult(image=img, notes=[f"{label}: nothing to build."])
        if invert:
            m = 1.0 - m
        if feather > 0:
            sigma = max(0.5, feather * max(img.shape[:2]) * 0.25)
            m = cv2.GaussianBlur(m, (0, 0), sigma)
        m = np.clip(m * opacity, 0, 1).astype(np.float32)

        data = {
            "coverage": round(float(m.mean()), 5),
            "promotable_mask": True,
            "mask_width": int(m.shape[1]),
            "mask_height": int(m.shape[0]),
        }
        if preview == "mask":
            out = np.dstack([m] * 3) if img.ndim == 3 else m
        elif preview == "applied":
            mm = m[:, :, None] if img.ndim == 3 else m
            out = clamp01(img * mm)
        else:
            base = img if img.ndim == 3 else np.dstack([img] * 3)
            tint = np.zeros_like(base[:, :, :3])
            tint[:, :, 0] = 1.0
            out = clamp01(base[:, :, :3] * (1 - m[:, :, None] * 0.45)
                          + tint * m[:, :, None] * 0.45)
        res = OpResult(image=out, data=data)
        res.overlays["mask"] = np.dstack([m] * 3)
        return res

    register(
        op_id, label, "compose",
        params=list(params) + [
            EnumParam("preview", options=enum_opts(("overlay", "Red overlay"),
                                                   ("mask", "Mask only"),
                                                   ("applied", "Applied to image")),
                      default="overlay"),
            BoolParam("invert", default=False),
            FloatParam("feather", min=0.0, max=0.3, default=0.02, step=0.005),
            FloatParam("opacity", min=0.0, max=1.0, default=1.0, step=0.05),
        ],
        summary=doc, formula_latex=formula, when_to_use=when,
        requires=requires or {}, tags=["mask"],
    )(fn)


_mask_builder(
    "compose.mask_luminosity", "Luminosity mask",
    lambda img, kw: luminosity_mask(img, *kw.get("range", (0.25, 0.75)), 0.0, False),
    [RangeParam("range", min=0.0, max=1.0, default=(0.25, 0.75))],
    "Selects a band of luminance values with a smooth trapezoidal falloff.",
    r"m(x) = \mathrm{ramp}\big(Y(x); lo, hi\big)",
    "Confining an operator to the midtones, shadows or highlights. 'Sharpen "
    "only the midtones' and 'denoise only the shadows' are both this mask.",
)

_mask_builder(
    "compose.mask_color", "Colour-range mask",
    lambda img, kw: color_range_mask(
        img, tuple(kw.get("hue", (90.0, 150.0))), tuple(kw.get("sat", (0.15, 1.0))),
        tuple(kw.get("val", (0.1, 1.0))), 0.0, False),
    [RangeParam("hue", min=0.0, max=360.0, default=(90.0, 150.0), step=1.0),
     RangeParam("sat", min=0.0, max=1.0, default=(0.15, 1.0)),
     RangeParam("val", min=0.0, max=1.0, default=(0.1, 1.0))],
    "Selects pixels inside an HSV window, with hue wraparound handled.",
    r"m = \mathbb{1}[H \in [h_0,h_1]] \wedge \mathbb{1}[S \in [s_0,s_1]] "
    r"\wedge \mathbb{1}[V \in [v_0,v_1]]",
    "Targeting one colour family -- deepening only the sky, or desaturating "
    "only the foliage.",
    requires={"channels": 3},
)

_mask_builder(
    "compose.mask_gradient", "Gradient mask",
    lambda img, kw: gradient_mask(
        img.shape[:2], kw.get("gradient_kind", "linear"),
        tuple(kw.get("p0", (0.0, 0.0))), tuple(kw.get("p1", (1.0, 1.0))), False),
    [EnumParam("gradient_kind", options=enum_opts(("linear", "Linear"),
                                                  ("radial", "Radial")),
               default="linear"),
     PointParam("p0", default=(0.0, 0.0)),
     PointParam("p1", default=(1.0, 1.0))],
    "A linear or radial ramp between two draggable points.",
    r"m(x) = \mathrm{clamp}\!\left(\frac{(x - p_0)\cdot(p_1 - p_0)}"
    r"{\|p_1 - p_0\|^2}\right)",
    "Graduated adjustments: darkening a sky from the top down, or vignetting an "
    "effect toward the corners.",
)

_mask_builder(
    "compose.mask_brush", "Brush mask",
    lambda img, kw: brush_mask(img.shape[:2], kw.get("strokes") or [], False),
    [MaskRefParam("strokes", help="Stroke list painted on the canvas.")],
    "Rasterises hand-painted brush strokes, honouring size, hardness, flow and "
    "erase mode.",
    r"m = \bigcup_i \mathrm{stroke}_i,\ \text{feathered by hardness}",
    "Local, hand-chosen adjustments -- dodging and burning.",
)

_mask_builder(
    "compose.mask_rect", "Rectangle mask",
    lambda img, kw: rect_mask(img.shape[:2], tuple(kw.get("rect", (0.2, 0.2, 0.6, 0.6))), 0.0),
    [RectParam("rect", default=(0.2, 0.2, 0.6, 0.6))],
    "A rectangular selection in normalised coordinates.",
    r"m = \mathbb{1}\big[(x,y) \in R\big]",
    "Restricting an operator to a region of interest.",
)


@register(
    "compose.mask_ops",
    "Mask operations",
    "compose",
    params=[
        MaskRefParam("mask"),
        EnumParam("op", options=enum_opts(("invert", "Invert"), ("blur", "Blur"),
                                          ("grow", "Grow (dilate)"),
                                          ("shrink", "Shrink (erode)"),
                                          ("levels", "Levels"),
                                          ("contrast", "Contrast")),
                  default="blur"),
        FloatParam("amount", min=0.0, max=1.0, default=0.1, step=0.01),
        FloatParam("black_point", min=0.0, max=1.0, default=0.0, step=0.01,
                   depends_on={"param": "op", "equals": "levels"}),
        FloatParam("white_point", min=0.0, max=1.0, default=1.0, step=0.01,
                   depends_on={"param": "op", "equals": "levels"}),
    ],
    summary="Post-processes an existing mask: invert, blur, grow, shrink, or "
    "remap its levels.",
    formula_latex=r"m' = T(m)",
    when_to_use="Refining a mask that is close but not quite right -- growing a "
    "selection by two pixels to cover an edge, or hardening a soft one.",
    tags=["mask"],
)
def mask_ops(img, mask=None, op="blur", amount=0.1, black_point=0.0,
             white_point=1.0, ctx=None):
    from ..core.masks import resolve_mask

    m = resolve_mask(mask, img)
    if m is None:
        return OpResult(image=img, notes=["No mask supplied."])
    long_edge = max(img.shape[:2])
    if op == "invert":
        m = 1.0 - m
    elif op == "blur":
        sigma = max(0.5, amount * long_edge * 0.1)
        m = cv2.GaussianBlur(m, (0, 0), sigma)
    elif op == "grow":
        k = max(1, int(amount * long_edge * 0.05)) | 1
        m = cv2.dilate(m, np.ones((k, k), np.uint8))
    elif op == "shrink":
        k = max(1, int(amount * long_edge * 0.05)) | 1
        m = cv2.erode(m, np.ones((k, k), np.uint8))
    elif op == "levels":
        span = max(white_point - black_point, 1e-4)
        m = np.clip((m - black_point) / span, 0, 1)
    else:
        m = np.clip((m - 0.5) * (1 + amount * 4) + 0.5, 0, 1)
    m = m.astype(np.float32)
    return OpResult(image=np.dstack([m] * 3) if img.ndim == 3 else m,
                    data={"coverage": round(float(m.mean()), 5),
                          "promotable_mask": True},
                    overlays={"mask": np.dstack([m] * 3)})


@register(
    "compose.roi",
    "Limit to ROI",
    "compose",
    params=[
        RectParam("rect", default=(0.25, 0.25, 0.5, 0.5)),
        FloatParam("feather", min=0.0, max=0.3, default=0.02, step=0.005),
        BoolParam("invert", default=False),
        BoolParam("show_boundary", default=True),
    ],
    summary="Produces a region-of-interest mask that later operators can "
    "reference, so an effect applies only inside (or only outside) it.",
    formula_latex=r"m = \mathrm{feather}\big(\mathbb{1}[(x,y) \in R]\big)",
    when_to_use="Local correction without painting: brightening one face, or "
    "sharpening only the centre of the frame.",
    tags=["mask"],
)
def roi(img, rect=(0.25, 0.25, 0.5, 0.5), feather=0.02, invert=False,
        show_boundary=True, ctx=None):
    m = rect_mask(img.shape[:2], tuple(rect), feather)
    if invert:
        m = 1.0 - m
    base = img if img.ndim == 3 else np.dstack([img] * 3)
    vis = base[:, :, :3].copy()
    if show_boundary:
        h, w = img.shape[:2]
        x, y, rw, rh = rect
        cv2.rectangle(vis, (int(x * w), int(y * h)),
                      (int((x + rw) * w), int((y + rh) * h)), (0.1, 1.0, 0.6), 2)
    return OpResult(image=clamp01(vis),
                    data={"coverage": round(float(m.mean()), 5),
                          "promotable_mask": True, "rect": list(rect)},
                    overlays={"mask": np.dstack([m] * 3)})


@register(
    "compose.snapshot",
    "Snapshot",
    "compose",
    params=[StringParam("name", default="Snapshot")],
    summary="Marks a point in the pipeline for A/B comparison. A no-op on the "
    "image itself.",
    formula_latex=r"g = f",
    when_to_use="Pinning an intermediate state so you can flip between it and "
    "the current result while continuing to edit.",
    is_view=True,
)
def snapshot(img, name="Snapshot", ctx=None):
    y = clamp01(luma(img))
    return OpResult(image=None, data={
        "name": name,
        "mean": round(float(y.mean()), 5),
        "std": round(float(y.std()), 5),
    })
