"""Tone / point operations. Catalogue Section 2."""
from __future__ import annotations

import cv2
import numpy as np

from ..core.colorspace import join_alpha, luma, split_alpha
from ..core.params import (
    BoolParam,
    CurveParam,
    EnumParam,
    FloatParam,
    ImageRefParam,
    IntParam,
    OddIntParam,
    RangeParam,
    enum_opts,
)
from ..core.registry import OpResult, register
from ._util import (
    EPS,
    apply_lut,
    build_lut,
    clamp01,
    from_u8,
    monotone_spline,
    on_lab_l,
    on_luma,
    resolve_ref_image,
)
from ._util import per_channel as map_per_channel

SPACE_OPTS = enum_opts(
    ("luma", "Luminance only"),
    ("lab_l", "CIE L* only"),
    ("per_channel", "Each RGB channel"),
    help_map={
        "luma": "Preserves colour ratios. The usual choice.",
        "lab_l": "Perceptually uniform lightness; safest for strong adjustments.",
        "per_channel": "Equalises R, G and B independently -- this SHIFTS HUE. "
        "Useful to demonstrate why luma-only is preferred.",
    },
)


def _dispatch_space(img, space, fn):
    if space == "per_channel":
        return map_per_channel(img, fn)
    if space == "lab_l":
        return on_lab_l(img, fn)
    return on_luma(img, fn)


# ---------------------------------------------------------------- brightness

@register(
    "tone.brightness",
    "Brightness",
    "tone",
    params=[FloatParam("offset", min=-1.0, max=1.0, default=0.0, step=0.005)],
    summary="Adds a constant to every pixel, shifting the whole histogram.",
    formula_latex=r"s = r + b",
    when_to_use="A uniformly dark or light image with no clipping at either end.",
    references=["Gonzalez & Woods, Digital Image Processing, Ch. 3.2"],
)
def brightness(img, offset=0.0, ctx=None):
    return clamp01(img + offset)


@register(
    "tone.contrast_linear",
    "Contrast (linear)",
    "tone",
    params=[
        FloatParam("amount", min=-1.0, max=2.0, default=0.0, step=0.01),
        FloatParam("pivot", min=0.0, max=1.0, default=0.5, step=0.01,
                   help="Intensity held fixed while the rest expands around it."),
    ],
    summary="Scales deviations from a pivot intensity. Linear contrast stretching.",
    formula_latex=r"s = (r - p)\cdot(1 + a) + p",
    when_to_use="Flat images where the histogram occupies a narrow band.",
)
def contrast_linear(img, amount=0.0, pivot=0.5, ctx=None):
    return clamp01((img - pivot) * (1.0 + amount) + pivot)


@register(
    "tone.contrast_sigmoid",
    "Contrast (S-curve)",
    "tone",
    params=[
        FloatParam("strength", min=0.0, max=12.0, default=3.0, step=0.1),
        FloatParam("pivot", min=0.05, max=0.95, default=0.5, step=0.01),
    ],
    summary="Smooth sigmoidal contrast: steepens midtones while rolling off "
    "shadows and highlights asymptotically, so nothing hard-clips.",
    formula_latex=r"s = \frac{1}{1 + e^{-k(r - p)}}\ \text{(normalised to [0,1])}",
    when_to_use="Whenever linear contrast would clip. This is what 'contrast' "
    "means in a professional editor.",
    tags=["beyond-syllabus"],
)
def contrast_sigmoid(img, strength=3.0, pivot=0.5, ctx=None):
    k = max(strength, 1e-3)

    def f(x):
        lo = 1.0 / (1.0 + np.exp(k * pivot))
        hi = 1.0 / (1.0 + np.exp(-k * (1.0 - pivot)))
        s = 1.0 / (1.0 + np.exp(-k * (x - pivot)))
        return (s - lo) / max(hi - lo, EPS)

    return clamp01(apply_lut(img, build_lut(f)))


@register(
    "tone.gamma",
    "Gamma correction",
    "tone",
    params=[
        FloatParam("gamma", min=0.1, max=5.0, default=1.0, step=0.01, log_scale=True),
        FloatParam("gain", min=0.1, max=3.0, default=1.0, step=0.01, advanced=True),
    ],
    summary="Power-law transformation. Gamma below 1 brightens shadows while "
    "leaving white fixed; above 1 darkens.",
    formula_latex=r"s = c\,r^{\gamma}",
    when_to_use="Underexposed images where highlights are already near clipping "
    "-- gamma cannot blow them further, unlike a multiply.",
    references=["Gonzalez & Woods, Ch. 3.2.2"],
)
def gamma_op(img, gamma=1.0, gain=1.0, ctx=None):
    return clamp01(apply_lut(img, build_lut(lambda x: gain * np.power(x, gamma))))


@register(
    "tone.exposure",
    "Exposure (stops)",
    "tone",
    params=[FloatParam("stops", min=-5.0, max=5.0, default=0.0, step=0.05, unit="EV")],
    summary="Multiplies scene-linear radiance. One stop doubles the light.",
    formula_latex=r"s = r \cdot 2^{\,EV}",
    when_to_use="The physically correct brightness control. Operates in linear "
    "light, so it behaves like a camera exposure change.",
    domain="linear",
    tags=["beyond-syllabus"],
)
def exposure(img, stops=0.0, ctx=None):
    return clamp01(img * (2.0**stops))


@register(
    "tone.log",
    "Log transform",
    "tone",
    params=[FloatParam("c", min=0.1, max=5.0, default=1.0, step=0.05, auto=True)],
    summary="Compresses bright values and expands dark ones, revealing shadow "
    "detail over a wide dynamic range.",
    formula_latex=r"s = c\,\frac{\log(1 + r)}{\log 2}",
    when_to_use="Images with a huge dynamic range, such as a Fourier magnitude "
    "spectrum -- which is exactly why the DFT viewer uses it.",
    references=["Gonzalez & Woods, Ch. 3.2.2"],
)
def log_transform(img, c=1.0, ctx=None):
    return clamp01(apply_lut(img, build_lut(lambda x: c * np.log1p(x) / np.log(2.0))))


@register(
    "tone.inverse_log",
    "Inverse log",
    "tone",
    params=[FloatParam("c", min=0.1, max=5.0, default=1.0, step=0.05)],
    summary="Inverse of the log transform: expands bright values.",
    formula_latex=r"s = \frac{(1+1)^{r/c} - 1}{1}",
    when_to_use="Undoing a log transform, or darkening midtones.",
)
def inverse_log(img, c=1.0, ctx=None):
    return clamp01(apply_lut(img, build_lut(lambda x: (np.power(2.0, x / c) - 1.0))))


@register(
    "tone.stretch_piecewise",
    "Piecewise contrast stretch",
    "tone",
    params=[
        FloatParam("r1", min=0.0, max=1.0, default=0.25, step=0.01),
        FloatParam("s1", min=0.0, max=1.0, default=0.1, step=0.01),
        FloatParam("r2", min=0.0, max=1.0, default=0.75, step=0.01),
        FloatParam("s2", min=0.0, max=1.0, default=0.9, step=0.01),
    ],
    summary="Three-segment linear transfer function through two control points.",
    formula_latex=r"s = \begin{cases} \frac{s_1}{r_1}r & r < r_1 \\ "
    r"s_1 + \frac{s_2-s_1}{r_2-r_1}(r-r_1) & r_1 \le r < r_2 \\ "
    r"s_2 + \frac{1-s_2}{1-r_2}(r-r_2) & r \ge r_2 \end{cases}",
    when_to_use="The textbook contrast-stretching exercise; also a quick way to "
    "crush or lift one end of the range only.",
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 3.2.4"],
)
def stretch_piecewise(img, r1=0.25, s1=0.1, r2=0.75, s2=0.9, ctx=None):
    if r2 <= r1:
        r2 = min(1.0, r1 + 1e-3)

    def f(x):
        y = np.empty_like(x)
        a = x < r1
        b = (x >= r1) & (x < r2)
        c = x >= r2
        y[a] = (s1 / max(r1, EPS)) * x[a]
        y[b] = s1 + (s2 - s1) / max(r2 - r1, EPS) * (x[b] - r1)
        y[c] = s2 + (1.0 - s2) / max(1.0 - r2, EPS) * (x[c] - r2)
        return y

    return clamp01(apply_lut(img, build_lut(f)))


def _auto_levels_params(img, **_):
    y = luma(img)
    return {
        "low_pct": round(float(np.percentile(y, 0.5) * 0 + 0.5), 3),
        "high_pct": 99.5,
    }


@register(
    "tone.autolevels",
    "Auto levels (percentile)",
    "tone",
    params=[
        FloatParam("low_pct", min=0.0, max=10.0, default=0.5, step=0.1, unit="%"),
        FloatParam("high_pct", min=90.0, max=100.0, default=99.5, step=0.1, unit="%"),
        BoolParam("per_channel", default=False,
                  help="Stretch each channel separately, which also neutralises "
                       "a colour cast -- but can shift hues."),
    ],
    summary="Maps a low and high percentile of the histogram to black and white.",
    formula_latex=r"s = \frac{r - p_{lo}}{p_{hi} - p_{lo}}",
    when_to_use="An image using only part of the tonal range. Percentiles rather "
    "than min/max so a few stray pixels cannot defeat the stretch.",
    auto_fn=_auto_levels_params,
)
def autolevels(img, low_pct=0.5, high_pct=99.5, per_channel=False, ctx=None):
    def stretch(ch):
        lo = float(np.percentile(ch, low_pct))
        hi = float(np.percentile(ch, high_pct))
        if hi - lo < 1e-5:
            return ch
        return np.clip((ch - lo) / (hi - lo), 0, 1)

    if per_channel:
        return map_per_channel(img, stretch)
    return on_luma(img, stretch)


@register(
    "tone.levels",
    "Levels",
    "tone",
    params=[
        FloatParam("black", min=0.0, max=1.0, default=0.0, step=0.005),
        FloatParam("mid", min=0.05, max=5.0, default=1.0, step=0.01,
                   help="Midtone gamma applied between the black and white points."),
        FloatParam("white", min=0.0, max=1.0, default=1.0, step=0.005),
        FloatParam("out_low", min=0.0, max=1.0, default=0.0, step=0.005, advanced=True),
        FloatParam("out_high", min=0.0, max=1.0, default=1.0, step=0.005, advanced=True),
        EnumParam("channel", options=enum_opts("rgb", "r", "g", "b"), default="rgb"),
    ],
    summary="Black point, white point and midtone gamma, with optional output "
    "range compression. The professional staple.",
    formula_latex=r"s = o_{lo} + (o_{hi}-o_{lo})\left(\frac{r-b}{w-b}\right)^{1/\gamma_m}",
    when_to_use="Precise tonal control; the first tool to reach for after reading "
    "the histogram.",
)
def levels(img, black=0.0, mid=1.0, white=1.0, out_low=0.0, out_high=1.0,
           channel="rgb", ctx=None):
    if white <= black:
        white = min(1.0, black + 1e-3)

    def f(x):
        t = np.clip((x - black) / max(white - black, EPS), 0, 1)
        t = np.power(t, 1.0 / max(mid, EPS))
        return out_low + (out_high - out_low) * t

    lut = build_lut(f)
    if channel == "rgb" or img.ndim == 2:
        return clamp01(apply_lut(img, lut))
    rgb, alpha = split_alpha(img)
    out = rgb.copy()
    i = {"r": 0, "g": 1, "b": 2}[channel]
    out[:, :, i] = apply_lut(rgb[:, :, i], lut)
    return join_alpha(clamp01(out), alpha)


@register(
    "tone.curves",
    "Curves",
    "tone",
    params=[
        CurveParam("curve", default=((0.0, 0.0), (1.0, 1.0))),
        EnumParam("channel", options=enum_opts("rgb", "r", "g", "b", ("luma", "Luma")),
                  default="rgb"),
    ],
    summary="Arbitrary tonal transfer curve through draggable control points, "
    "interpolated with a monotone cubic spline.",
    formula_latex=r"s = f(r),\quad f \text{ monotone cubic through } \{(x_i,y_i)\}",
    when_to_use="Any tonal adjustment a slider cannot express. Monotone "
    "interpolation is used so the curve can never overshoot and invert locally.",
    references=["Fritsch & Carlson, Monotone Piecewise Cubic Interpolation, 1980"],
)
def curves(img, curve=((0.0, 0.0), (1.0, 1.0)), channel="rgb", ctx=None):
    lut = monotone_spline(list(curve), 256)
    if channel == "luma":
        return on_luma(img, lambda y: apply_lut(y, lut))
    if channel == "rgb" or img.ndim == 2:
        return clamp01(apply_lut(img, lut))
    rgb, alpha = split_alpha(img)
    out = rgb.copy()
    i = {"r": 0, "g": 1, "b": 2}[channel]
    out[:, :, i] = apply_lut(rgb[:, :, i], lut)
    return join_alpha(clamp01(out), alpha)


@register(
    "tone.negative",
    "Negative",
    "tone",
    summary="Inverts intensity, producing a photographic negative.",
    formula_latex=r"s = L - 1 - r",
    when_to_use="Revealing detail embedded in dark regions, where the eye "
    "discriminates poorly.",
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 3.2.1"],
)
def negative(img, ctx=None):
    rgb, alpha = split_alpha(img)
    return join_alpha(clamp01(1.0 - rgb), alpha)


def _otsu_level(img) -> dict:
    g = (clamp01(luma(img)) * 255).astype(np.uint8)
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {"level": round(float(t) / 255.0, 4)}


@register(
    "tone.threshold",
    "Threshold",
    "tone",
    params=[
        FloatParam("level", min=0.0, max=1.0, default=0.5, step=0.005, auto=True),
        EnumParam(
            "mode",
            options=enum_opts(
                ("binary", "Binary"),
                ("binary_inv", "Binary inverted"),
                ("trunc", "Truncate"),
                ("tozero", "To zero"),
                ("tozero_inv", "To zero inverted"),
            ),
            default="binary",
        ),
        BoolParam("otsu", default=False, help="Choose the level automatically by "
                  "maximising between-class variance."),
    ],
    summary="Global intensity thresholding in five variants.",
    formula_latex=r"s = \begin{cases} 1 & r > T \\ 0 & \text{otherwise}\end{cases}",
    when_to_use="Separating foreground from background when illumination is even. "
    "Use adaptive thresholding when it is not.",
    syllabus=True,
    auto_fn=_otsu_level,
)
def threshold_op(img, level=0.5, mode="binary", otsu=False, ctx=None):
    g = clamp01(luma(img))
    if otsu:
        u8 = (g * 255).astype(np.uint8)
        t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        level = float(t) / 255.0
    if mode == "binary":
        out = (g > level).astype(np.float32)
    elif mode == "binary_inv":
        out = (g <= level).astype(np.float32)
    elif mode == "trunc":
        out = np.minimum(g, level)
    elif mode == "tozero":
        out = np.where(g > level, g, 0.0)
    else:
        out = np.where(g <= level, g, 0.0)
    return OpResult(image=out.astype(np.float32), data={"level": round(level, 4)})


@register(
    "tone.gray_slice",
    "Gray-level slicing",
    "tone",
    params=[
        RangeParam("window", min=0.0, max=1.0, default=(0.4, 0.6)),
        BoolParam("preserve_background", default=True),
        FloatParam("highlight", min=0.0, max=1.0, default=1.0, step=0.01),
    ],
    summary="Highlights one band of intensities, optionally discarding the rest.",
    formula_latex=r"s = \begin{cases} H & a \le r \le b \\ r \text{ or } 0 & "
    r"\text{otherwise}\end{cases}",
    when_to_use="Isolating a known intensity range, such as a specific tissue "
    "density in a medical scan.",
    syllabus=True,
)
def gray_slice(img, window=(0.4, 0.6), preserve_background=True, highlight=1.0, ctx=None):
    g = clamp01(luma(img))
    lo, hi = window
    sel = (g >= lo) & (g <= hi)
    base = g if preserve_background else np.zeros_like(g)
    return np.where(sel, highlight, base).astype(np.float32)


@register(
    "tone.posterize",
    "Posterize",
    "tone",
    params=[
        IntParam("levels", min=2, max=64, default=6),
        BoolParam("dither", default=False,
                  help="Adds ordered noise before quantising, trading banding "
                       "for fine texture."),
    ],
    summary="Quantises intensity to a small number of levels.",
    formula_latex=r"s = \frac{\lfloor r\,(L-1) + 0.5\rfloor}{L-1}",
    when_to_use="A poster effect, and the clearest demonstration of why banding "
    "appears when bit depth is insufficient.",
)
def posterize(img, levels=6, dither=False, ctx=None):
    rgb, alpha = split_alpha(img)
    x = clamp01(rgb)
    if dither:
        # 4x4 Bayer matrix, tiled -- deterministic, unlike random dithering.
        bayer = np.array(
            [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
            dtype=np.float32,
        ) / 16.0 - 0.5
        h, w = x.shape[:2]
        tile = np.tile(bayer, (h // 4 + 1, w // 4 + 1))[:h, :w]
        if x.ndim == 3:
            tile = tile[:, :, None]
        x = np.clip(x + tile / max(levels - 1, 1), 0, 1)
    q = np.floor(x * (levels - 1) + 0.5) / (levels - 1)
    return join_alpha(clamp01(q), alpha)


@register(
    "tone.solarize",
    "Solarize",
    "tone",
    params=[FloatParam("threshold", min=0.0, max=1.0, default=0.5, step=0.01)],
    summary="Inverts only the pixels above a threshold.",
    formula_latex=r"s = \begin{cases} 1-r & r \ge T \\ r & \text{otherwise}\end{cases}",
    when_to_use="A darkroom effect; also a standard photometric augmentation.",
)
def solarize(img, threshold=0.5, ctx=None):
    rgb, alpha = split_alpha(img)
    return join_alpha(np.where(rgb >= threshold, 1.0 - rgb, rgb).astype(np.float32), alpha)


@register(
    "tone.he",
    "Histogram equalization",
    "tone",
    params=[EnumParam("space", options=SPACE_OPTS, default="luma")],
    summary="Remaps intensities so the histogram is approximately uniform, using "
    "the cumulative distribution as the transfer function.",
    formula_latex=r"s = T(r) = (L-1)\int_0^r p_r(w)\,dw",
    when_to_use="Low-contrast images with a concentrated histogram. Note it is "
    "global and unconditional: it will amplify noise in flat regions.",
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 3.3.1"],
)
def hist_eq(img, space="luma", ctx=None):
    def eq(ch):
        u8 = (clamp01(ch) * 255).astype(np.uint8)
        return from_u8(cv2.equalizeHist(u8))

    return _dispatch_space(img, space, eq)


@register(
    "tone.ahe",
    "Adaptive HE (no clipping)",
    "tone",
    params=[
        IntParam("tiles", min=2, max=32, default=8,
                 help="Number of tiles per axis. A COUNT, not a length: it "
                      "stays fixed across resolutions so each tile covers the "
                      "same fraction of the image."),
        EnumParam("space", options=SPACE_OPTS, default="luma"),
    ],
    summary="Histogram equalisation computed per tile, with no clip limit.",
    formula_latex=r"s(x,y) = T_{\text{tile}(x,y)}(r)",
    when_to_use="Mainly to demonstrate the problem CLAHE solves: without a clip "
    "limit, flat regions have their noise amplified enormously.",
    tags=["teaching"],
)
def ahe(img, tiles=8, space="luma", ctx=None):
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(int(tiles), int(tiles)))

    def eq(ch):
        return from_u8(clahe.apply((clamp01(ch) * 255).astype(np.uint8)))

    return _dispatch_space(img, space, eq)


@register(
    "tone.clahe",
    "CLAHE",
    "tone",
    params=[
        FloatParam("clip", min=0.5, max=8.0, default=2.0, step=0.1),
        IntParam("tiles_x", min=2, max=32, default=8,
                 help="Tiles per axis: a COUNT, not a length, so it is held "
                      "fixed across resolutions and each tile covers the same "
                      "fraction of the image."),
        IntParam("tiles_y", min=2, max=32, default=8),
        EnumParam("space", options=SPACE_OPTS, default="lab_l"),
    ],
    summary="Contrast-limited adaptive histogram equalisation: per-tile "
    "equalisation with the histogram clipped and the excess redistributed, "
    "then bilinearly interpolated between tiles.",
    formula_latex=r"\text{clip } p_r \text{ at } \beta, \text{ redistribute, then } "
    r"s = T_{\text{tile}}(r)",
    when_to_use="The workhorse for local contrast. Preferred over global HE in "
    "almost every case, and the Advisor's primary contrast fix.",
    references=["Zuiderveld, Contrast Limited Adaptive Histogram Equalization, 1994"],
)
def clahe_op(img, clip=2.0, tiles_x=8, tiles_y=8, space="lab_l", ctx=None):
    c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tiles_x), int(tiles_y)))

    def eq(ch):
        return from_u8(c.apply((clamp01(ch) * 255).astype(np.uint8)))

    return _dispatch_space(img, space, eq)


@register(
    "tone.hist_match",
    "Histogram matching",
    "tone",
    params=[
        ImageRefParam("reference", help="Image whose histogram to adopt."),
        EnumParam(
            "parametric",
            options=enum_opts(
                ("none", "Use reference image"),
                ("uniform", "Uniform"),
                ("gaussian", "Gaussian"),
                ("rayleigh", "Rayleigh"),
            ),
            default="none",
        ),
        FloatParam("strength", min=0.0, max=1.0, default=1.0, step=0.05),
        BoolParam("per_channel", default=True),
    ],
    summary="Histogram specification: remaps the image so its cumulative "
    "distribution matches a target's.",
    formula_latex=r"s = G^{-1}\!\left(T(r)\right),\ T = \text{CDF}_{\text{src}},"
    r"\ G = \text{CDF}_{\text{target}}",
    when_to_use="Matching one photograph's tonality to another, or forcing a "
    "known distribution. The basis of colour transfer.",
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 3.3.2"],
)
def hist_match(img, reference=None, parametric="none", strength=1.0,
               per_channel=True, ctx=None):
    ctx = ctx or {}

    def target_cdf(n=256):
        x = np.linspace(0, 1, n)
        if parametric == "uniform":
            pdf = np.ones(n)
        elif parametric == "gaussian":
            pdf = np.exp(-((x - 0.5) ** 2) / (2 * 0.18**2))
        elif parametric == "rayleigh":
            s = 0.3
            pdf = (x / s**2) * np.exp(-(x**2) / (2 * s**2))
        else:
            return None
        c = np.cumsum(pdf)
        return c / c[-1]

    tc = target_cdf()
    ref = None
    if tc is None:
        ref = resolve_ref_image(ctx, reference, img)

    def match_channel(src, ref_ch):
        s_hist, _ = np.histogram(src, bins=256, range=(0, 1))
        s_cdf = np.cumsum(s_hist).astype(np.float64)
        s_cdf /= max(s_cdf[-1], 1)
        if ref_ch is None:
            r_cdf = tc
        else:
            r_hist, _ = np.histogram(ref_ch, bins=256, range=(0, 1))
            r_cdf = np.cumsum(r_hist).astype(np.float64)
            r_cdf /= max(r_cdf[-1], 1)
        # For each source level, find the target level with the nearest CDF.
        lut = np.interp(s_cdf, r_cdf, np.linspace(0, 1, 256))
        matched = apply_lut(src, lut.astype(np.float32))
        return src * (1 - strength) + matched * strength

    if img.ndim == 2 or not per_channel:
        return on_luma(img, lambda y: match_channel(y, luma(ref) if ref is not None else None))

    rgb, alpha = split_alpha(img)
    out = np.dstack([
        match_channel(rgb[:, :, i], ref[:, :, i] if ref is not None else None)
        for i in range(3)
    ])
    return join_alpha(clamp01(out), alpha)


@register(
    "tone.shadows_highlights",
    "Shadows / Highlights",
    "tone",
    params=[
        FloatParam("shadows", min=-1.0, max=1.0, default=0.0, step=0.01,
                   help="Positive lifts shadows."),
        FloatParam("highlights", min=-1.0, max=1.0, default=0.0, step=0.01,
                   help="Negative recovers blown highlights."),
        FloatParam("radius", min=4.0, max=200.0, default=40.0, step=1.0,
                   scale_aware=True,
                   help="Scale of the base layer. Larger = more global."),
        FloatParam("midtone_contrast", min=-1.0, max=1.0, default=0.0, step=0.01),
    ],
    summary="Local tone mapping. Splits the image into a guided-filter base "
    "layer and a detail layer, adjusts shadows and highlights on the base only, "
    "then recombines -- so detail is not flattened.",
    formula_latex=r"B = \text{guided}(Y),\ s = Y + \alpha_s w_s(B) + \alpha_h w_h(B)",
    when_to_use="The correct fix when an image is BOTH shadow-crushed and "
    "highlight-clipped: no single global curve can lift one without worsening "
    "the other.",
    tags=["beyond-syllabus"],
    cost="fast",
)
def shadows_highlights(img, shadows=0.0, highlights=0.0, radius=40.0,
                       midtone_contrast=0.0, ctx=None):
    y = clamp01(luma(img))
    r = max(2, int(round(radius)))
    eps = 1e-3
    try:
        base = cv2.ximgproc.guidedFilter(
            guide=y.astype(np.float32), src=y.astype(np.float32), radius=r, eps=eps
        )
    except Exception:
        base = cv2.bilateralFilter(y.astype(np.float32), 0, 0.1, r)

    # Weight shadows and highlights by smooth masks derived from the base layer,
    # so the adjustment follows large-scale luminance, not per-pixel noise.
    w_sh = np.clip(1.0 - base * 2.0, 0, 1) ** 1.5
    w_hi = np.clip((base - 0.5) * 2.0, 0, 1) ** 1.5

    adj = np.zeros_like(y)
    if shadows != 0:
        adj += shadows * w_sh * (1.0 - y) * 0.9
    if highlights != 0:
        adj += highlights * w_hi * y * 0.9
    if midtone_contrast != 0:
        mid = np.clip(1.0 - np.abs(base - 0.5) * 2.0, 0, 1)
        adj += midtone_contrast * mid * (y - base) * 1.5

    y2 = clamp01(y + adj)
    gain = (y2 + EPS) / (y + EPS)
    rgb, alpha = split_alpha(img)
    if rgb.ndim == 2:
        return join_alpha(y2, alpha)
    return join_alpha(clamp01(rgb * gain[:, :, None]), alpha)


@register(
    "tone.retinex",
    "Retinex (MSR / MSRCR)",
    "tone",
    params=[
        EnumParam("variant", options=enum_opts(("ssr", "Single-scale"),
                                               ("msr", "Multi-scale"),
                                               ("msrcr", "Multi-scale + colour restore")),
                  default="msr"),
        FloatParam("sigma_small", min=5.0, max=60.0, default=15.0, scale_aware=True),
        FloatParam("sigma_mid", min=20.0, max=200.0, default=80.0, scale_aware=True),
        FloatParam("sigma_large", min=50.0, max=400.0, default=200.0, scale_aware=True),
        FloatParam("gain", min=0.1, max=5.0, default=1.0, step=0.05),
        FloatParam("offset", min=-0.5, max=0.5, default=0.0, step=0.01),
    ],
    summary="Retinex: estimates illumination by large-scale blurring and divides "
    "it out in the log domain, leaving reflectance.",
    formula_latex=r"R_i = \sum_k w_k\left[\log I - \log(I * G_{\sigma_k})\right]",
    when_to_use="Strong recovery of low-light and backlit images, with better "
    "colour constancy than a tone curve.",
    tags=["beyond-syllabus"],
    cost="slow",
    references=["Jobson, Rahman & Woodell, A Multiscale Retinex, IEEE TIP 1997"],
)
def retinex(img, variant="msr", sigma_small=15.0, sigma_mid=80.0, sigma_large=200.0,
            gain=1.0, offset=0.0, ctx=None):
    rgb, alpha = split_alpha(img)
    x = np.maximum(rgb.astype(np.float32), 1e-4)
    sigmas = [sigma_small] if variant == "ssr" else [sigma_small, sigma_mid, sigma_large]

    log_x = np.log(x)
    acc = np.zeros_like(x)
    for s in sigmas:
        blur = cv2.GaussianBlur(x, (0, 0), float(max(s, 1.0)))
        acc += log_x - np.log(np.maximum(blur, 1e-4))
    acc /= len(sigmas)

    if variant == "msrcr" and x.ndim == 3:
        # Colour restoration term re-couples the independently processed channels.
        total = np.sum(x, axis=2, keepdims=True)
        crf = 46.0 * (np.log(125.0 * x + 1.0) - np.log(total + 1e-4))
        acc = acc * crf / 46.0

    # Robust rescale: percentile rather than min/max, so outliers do not flatten it.
    lo, hi = np.percentile(acc, 1.0), np.percentile(acc, 99.0)
    out = (acc - lo) / max(hi - lo, EPS)
    out = out * gain + offset
    return join_alpha(clamp01(out), alpha)


@register(
    "tone.bilateral_tonemap",
    "Durand tone map",
    "tone",
    params=[
        FloatParam("base_contrast", min=0.05, max=1.0, default=0.4, step=0.01),
        FloatParam("sigma_space", min=2.0, max=100.0, default=20.0, scale_aware=True),
        FloatParam("sigma_range", min=0.05, max=1.0, default=0.3, step=0.01),
    ],
    summary="Compresses the base layer's contrast while preserving the detail "
    "layer, using an edge-preserving bilateral split in the log domain.",
    formula_latex=r"\log Y = B + D,\quad \log Y' = \gamma B + D",
    when_to_use="High-dynamic-range content that must fit a display without "
    "losing local texture.",
    tags=["beyond-syllabus"],
    cost="slow",
    references=["Durand & Dorsey, Fast Bilateral Filtering for HDR Display, 2002"],
)
def bilateral_tonemap(img, base_contrast=0.4, sigma_space=20.0, sigma_range=0.3, ctx=None):
    y = np.maximum(clamp01(luma(img)), 1e-4)
    log_y = np.log(y)
    lo, hi = float(log_y.min()), float(log_y.max())
    norm = (log_y - lo) / max(hi - lo, EPS)
    base = cv2.bilateralFilter(
        norm.astype(np.float32), 0, float(sigma_range), float(max(sigma_space, 1))
    )
    detail = norm - base
    out_log = base * base_contrast + detail
    out = np.exp(out_log * (hi - lo) + lo)
    out = out / max(float(out.max()), EPS)
    gain = (out + EPS) / (y + EPS)
    rgb, alpha = split_alpha(img)
    if rgb.ndim == 2:
        return join_alpha(clamp01(out), alpha)
    return join_alpha(clamp01(rgb * gain[:, :, None]), alpha)
