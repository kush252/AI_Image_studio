"""Inspect / Understand views. Catalogue Section 1.

These are VIEWS: they report numbers and overlays and pass the image through
unchanged (is_view=True), so they can sit anywhere in the stack without
affecting the render.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.fft import dct
from skimage.feature import graycomatrix, graycoprops

from ..core.colorspace import SPACES, luma, to_lab
from ..core.params import (
    BoolParam,
    EnumParam,
    FloatParam,
    ImageRefParam,
    IntParam,
    LineParam,
    OddIntParam,
    PointParam,
    enum_opts,
)
from ..core.registry import OpResult, register
from ._util import EPS, clamp01, colorize, gray_u8, normalize01


@register(
    "view.histogram",
    "Histogram",
    "inspect",
    params=[
        EnumParam("channels", options=enum_opts(("rgb", "RGB"), ("luma", "Luma"),
                                                ("all", "RGB + Luma"),
                                                ("lab", "CIE Lab")),
                  default="all"),
        IntParam("bins", min=16, max=256, default=256),
        BoolParam("log_y", default=False),
        BoolParam("cumulative", default=False),
    ],
    summary="Intensity distribution, per channel and for luminance.",
    formula_latex=r"h(r_k) = n_k,\qquad p(r_k) = \frac{n_k}{MN}",
    when_to_use="The first thing to read on any image. A histogram pressed "
    "against either end means clipping; a narrow central lump means low "
    "contrast; gaps mean the image was already stretched.",
    is_view=True,
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 3.3"],
)
def histogram(img, channels="all", bins=256, log_y=False, cumulative=False, ctx=None):
    data: dict = {"bins": int(bins)}
    y = clamp01(luma(img))

    def series(a):
        h, _ = np.histogram(a, bins=int(bins), range=(0.0, 1.0))
        h = h.astype(np.float64)
        if cumulative:
            h = np.cumsum(h)
        return (h / max(h.max(), 1)).round(6).tolist()

    if channels in ("luma", "all"):
        data["luma"] = series(y)
    if channels in ("rgb", "all") and img.ndim == 3:
        for i, n in enumerate(("r", "g", "b")):
            data[n] = series(img[:, :, i])
    if channels == "lab" and img.ndim == 3:
        lab = to_lab(img)
        data["L"] = series(lab[:, :, 0] / 100.0)
        data["a"] = series((lab[:, :, 1] + 128) / 255.0)
        data["b"] = series((lab[:, :, 2] + 128) / 255.0)

    data["stats"] = {
        "mean": round(float(y.mean()), 5),
        "median": round(float(np.median(y)), 5),
        "std": round(float(y.std()), 5),
        "min": round(float(y.min()), 5),
        "max": round(float(y.max()), 5),
        "clip_low": round(float((y <= 2 / 255).mean()), 6),
        "clip_high": round(float((y >= 253 / 255).mean()), 6),
        "occupancy": round(float(
            np.count_nonzero(np.histogram(y, bins=256, range=(0, 1))[0]) / 256), 4),
    }
    return OpResult(image=None, data=data)


@register(
    "view.cdf",
    "Cumulative histogram (CDF)",
    "inspect",
    params=[EnumParam("channel", options=enum_opts(("luma", "Luma"), ("r", "R"),
                                                   ("g", "G"), ("b", "B")),
                      default="luma")],
    summary="The cumulative distribution function of intensity.",
    formula_latex=r"T(r) = (L-1)\int_0^r p_r(w)\,dw",
    when_to_use="The CDF *is* the transfer curve histogram equalisation "
    "applies, so plotting it shows in advance exactly what HE will do to this "
    "image.",
    is_view=True,
    syllabus=True,
    tags=["teaching"],
)
def cdf(img, channel="luma", ctx=None):
    if channel == "luma" or img.ndim == 2:
        a = clamp01(luma(img))
    else:
        a = img[:, :, {"r": 0, "g": 1, "b": 2}[channel]]
    h, _ = np.histogram(a, bins=256, range=(0.0, 1.0))
    c = np.cumsum(h).astype(np.float64)
    c /= max(c[-1], 1)
    return OpResult(image=None, data={
        "cdf": c.round(6).tolist(),
        "percentiles": {p: round(float(np.percentile(a, p)), 5)
                        for p in (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)},
    })


@register(
    "view.channels",
    "Channel splitter",
    "inspect",
    params=[
        EnumParam("space", options=enum_opts(("rgb", "RGB"), ("hsv", "HSV"),
                                             ("hls", "HLS"), ("lab", "CIE Lab"),
                                             ("ycrcb", "YCrCb"), ("cmyk", "CMYK")),
                  default="rgb"),
        BoolParam("colorize_channels", default=False),
    ],
    summary="Decomposes the image into its component channels in a chosen "
    "colour space and tiles them for comparison.",
    formula_latex=r"f \to \{f_1, f_2, \dots, f_n\}",
    when_to_use="Choosing which channel to operate on. A red object has high "
    "contrast in the red channel and almost none in Lab's a* near the "
    "neutral axis -- the right channel makes a segmentation trivial.",
    is_view=True,
    tags=["teaching"],
)
def channels_view(img, space="rgb", colorize_channels=False, ctx=None):
    if img.ndim == 2:
        return OpResult(image=None, data={"channels": ["gray"]},
                        overlays={"channels": np.dstack([img] * 3)})
    conv, names = SPACES[space]
    arr = conv(img)
    n = arr.shape[2]
    tiles = []
    stats = {}
    for i in range(n):
        ch = arr[:, :, i]
        lo, hi = float(ch.min()), float(ch.max())
        stats[names[i]] = {"min": round(lo, 4), "max": round(hi, 4),
                           "mean": round(float(ch.mean()), 4),
                           "std": round(float(ch.std()), 4)}
        norm = normalize01(ch)
        tiles.append(colorize(norm, "viridis") if colorize_channels
                     else np.dstack([norm] * 3))
    cols = 2 if n <= 4 else 3
    rows = int(np.ceil(n / cols))
    while len(tiles) < rows * cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[r * cols : (r + 1) * cols])
                      for r in range(rows)])
    return OpResult(image=None,
                    data={"space": space, "channels": list(names[:n]),
                          "channel_stats": stats, "layout": [rows, cols]},
                    overlays={"channels": clamp01(grid)})


@register(
    "view.bitplanes",
    "Bit-plane slicing",
    "inspect",
    params=[
        IntParam("plane", min=0, max=7, default=7,
                 help="7 is the most significant bit."),
        BoolParam("show_all", default=True),
    ],
    summary="Extracts individual bits of the 8-bit intensity representation.",
    formula_latex=r"b_k(x,y) = \left\lfloor \frac{f(x,y)}{2^k} \right\rfloor "
    r"\bmod 2",
    when_to_use="Showing where image information actually lives: the top two or "
    "three planes carry recognisable structure, while the lowest planes are "
    "essentially noise -- which is why 8 bits can often be truncated to 5.",
    is_view=True,
    syllabus=True,
    tags=["teaching"],
    cost="medium",
    references=["Gonzalez & Woods, Ch. 3.2.4"],
)
def bitplanes(img, plane=7, show_all=True, ctx=None):
    u8 = gray_u8(img)
    planes = [((u8 >> k) & 1).astype(np.float32) for k in range(8)]
    energies = {f"bit_{k}": round(float(planes[k].mean()), 5) for k in range(8)}
    res = OpResult(image=None, data={"plane_density": energies,
                                     "selected_plane": int(plane)},
                   overlays={"plane": np.dstack([planes[int(plane)]] * 3)})
    if show_all:
        tiles = [np.dstack([planes[k]] * 3) for k in range(7, -1, -1)]
        grid = np.vstack([np.hstack(tiles[0:4]), np.hstack(tiles[4:8])])
        res.overlays["all_planes"] = grid
    return res


@register(
    "view.profile",
    "Intensity line profile",
    "inspect",
    params=[
        LineParam("line", default=((0.05, 0.5), (0.95, 0.5))),
        EnumParam("channel", options=enum_opts(("luma", "Luma"), ("rgb", "All RGB")),
                  default="rgb"),
        IntParam("samples", min=32, max=2048, default=512),
        IntParam("thickness", min=1, max=21, default=1,
                 help="Averages across this many parallel lines, reducing noise."),
    ],
    summary="Samples intensity along a draggable line.",
    formula_latex=r"p(t) = f\big((1-t)\,\mathbf{a} + t\,\mathbf{b}\big)",
    when_to_use="Measuring an edge's actual width and overshoot. The profile is "
    "how you tell a genuinely sharp edge from a sharpened one -- the latter "
    "shows a characteristic undershoot/overshoot pair.",
    is_view=True,
    tags=["teaching"],
)
def profile(img, line=((0.05, 0.5), (0.95, 0.5)), channel="rgb", samples=512,
            thickness=1, ctx=None):
    h, w = img.shape[:2]
    (x1, y1), (x2, y2) = line
    p1 = np.array([x1 * (w - 1), y1 * (h - 1)])
    p2 = np.array([x2 * (w - 1), y2 * (h - 1)])
    t = np.linspace(0, 1, int(samples))

    # Perpendicular offset for thickness averaging.
    d = p2 - p1
    length = float(np.hypot(*d))
    perp = np.array([-d[1], d[0]]) / max(length, EPS)

    def sample(arr):
        acc = np.zeros(int(samples), np.float32)
        k = int(thickness)
        offs = np.arange(k) - (k - 1) / 2.0
        for o in offs:
            pts = p1[None, :] + t[:, None] * d[None, :] + perp[None, :] * o
            xs = np.clip(pts[:, 0], 0, w - 1).astype(np.float32)
            ys = np.clip(pts[:, 1], 0, h - 1).astype(np.float32)
            acc += cv2.remap(arr, xs.reshape(-1, 1), ys.reshape(-1, 1),
                             cv2.INTER_LINEAR).ravel()
        return (acc / k).round(5).tolist()

    data = {"length_px": round(length, 1), "samples": int(samples)}
    y = clamp01(luma(img))
    data["luma"] = sample(y)
    if channel == "rgb" and img.ndim == 3:
        for i, n in enumerate(("r", "g", "b")):
            data[n] = sample(np.ascontiguousarray(img[:, :, i]))

    # Report the steepest transition, which is the practical edge-width measure.
    prof = np.array(data["luma"])
    grad = np.abs(np.diff(prof))
    if len(grad):
        data["max_slope"] = round(float(grad.max()), 5)
        data["max_slope_at"] = int(np.argmax(grad))
    return OpResult(image=None, data=data)


@register(
    "view.surface3d",
    "3-D intensity surface",
    "inspect",
    params=[
        IntParam("grid", min=16, max=192, default=64,
                 help="Downsampled grid resolution sent to the client."),
        EnumParam("channel", options=enum_opts(("luma", "Luma"), ("r", "R"),
                                               ("g", "G"), ("b", "B")),
                  default="luma"),
    ],
    summary="Returns a downsampled height field so the image can be rendered as "
    "a surface z = f(x, y).",
    formula_latex=r"z = f(x, y)",
    when_to_use="Building the intuition that an image IS a two-dimensional "
    "function. Edges become cliffs and noise becomes visible roughness.",
    is_view=True,
    tags=["teaching"],
)
def surface3d(img, grid=64, channel="luma", ctx=None):
    if channel == "luma" or img.ndim == 2:
        a = clamp01(luma(img))
    else:
        a = img[:, :, {"r": 0, "g": 1, "b": 2}[channel]]
    g = int(grid)
    small = cv2.resize(a, (g, g), interpolation=cv2.INTER_AREA)
    return OpResult(image=None, data={
        "grid": g,
        "z": np.round(small, 4).tolist(),
        "z_min": round(float(small.min()), 4),
        "z_max": round(float(small.max()), 4),
    })


@register(
    "view.waveform",
    "Waveform monitor",
    "inspect",
    params=[
        EnumParam("mode", options=enum_opts(("luma", "Luma"), ("parade", "RGB parade"),
                                            ("overlay", "RGB overlay")),
                  default="luma"),
        IntParam("height", min=64, max=256, default=128),
    ],
    summary="Broadcast-style waveform: for every image column, the distribution "
    "of intensities in that column.",
    formula_latex=r"W(x, v) = \#\{y : f(x,y) \approx v\}",
    when_to_use="Reading exposure spatially rather than globally. A histogram "
    "cannot tell you that only the left side is clipping; a waveform can.",
    is_view=True,
    cost="fast",
)
def waveform(img, mode="luma", height=128, ctx=None):
    h, w = img.shape[:2]
    H = int(height)
    target_w = min(w, 512)
    small = cv2.resize(img, (target_w, min(h, 256)), interpolation=cv2.INTER_AREA)

    def build(a):
        bins = np.clip((a * (H - 1)).astype(np.int32), 0, H - 1)
        out = np.zeros((H, a.shape[1]), np.float32)
        for x in range(a.shape[1]):
            np.add.at(out[:, x], bins[:, x], 1.0)
        out = out / max(float(out.max()), EPS)
        return np.flipud(out)

    if mode == "luma" or small.ndim == 2:
        wf = build(clamp01(luma(small)))
        vis = np.dstack([wf] * 3)
    elif mode == "parade":
        chans = [build(small[:, :, i]) for i in range(3)]
        tinted = [
            np.dstack([chans[0], chans[0] * 0.15, chans[0] * 0.15]),
            np.dstack([chans[1] * 0.15, chans[1], chans[1] * 0.15]),
            np.dstack([chans[2] * 0.15, chans[2] * 0.15, chans[2]]),
        ]
        vis = np.hstack(tinted)
    else:
        chans = [build(small[:, :, i]) for i in range(3)]
        vis = np.dstack(chans)
    return OpResult(image=None, data={"mode": mode, "height": H},
                    overlays={"waveform": clamp01(vis)})


@register(
    "view.vectorscope",
    "Vectorscope",
    "inspect",
    params=[
        IntParam("size", min=128, max=512, default=256),
        BoolParam("show_skin_line", default=True),
        FloatParam("gain", min=0.5, max=8.0, default=2.0, step=0.1),
    ],
    summary="Plots every pixel's chroma on the Cb/Cr plane, so hue is angle and "
    "saturation is radius.",
    formula_latex=r"(C_b - 0.5,\ C_r - 0.5) \text{ in polar form}",
    when_to_use="Diagnosing a colour cast: a neutral image forms a blob centred "
    "on the origin, while a cast displaces the whole blob in the direction of "
    "the cast. Far more direct than reading three histograms.",
    is_view=True,
    requires={"channels": 3},
    cost="fast",
)
def vectorscope(img, size=256, show_skin_line=True, gain=2.0, ctx=None):
    if img.ndim == 2:
        return OpResult(image=None, data={"monochrome": True})
    S = int(size)
    small = cv2.resize(img[:, :, :3], (min(img.shape[1], 320),
                                       min(img.shape[0], 320)),
                       interpolation=cv2.INTER_AREA)
    ycc = cv2.cvtColor(small.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    cr = (ycc[:, :, 1] - 0.5) * gain
    cb = (ycc[:, :, 2] - 0.5) * gain
    xs = np.clip((cb * 0.5 + 0.5) * (S - 1), 0, S - 1).astype(np.int32)
    ys = np.clip((0.5 - cr * 0.5) * (S - 1), 0, S - 1).astype(np.int32)
    acc = np.zeros((S, S), np.float32)
    np.add.at(acc, (ys.ravel(), xs.ravel()), 1.0)
    acc = np.log1p(acc)
    acc /= max(float(acc.max()), EPS)

    vis = colorize(acc, "inferno")
    c = S // 2
    cv2.circle(vis, (c, c), S // 2 - 2, (0.35, 0.35, 0.35), 1)
    cv2.line(vis, (c, 0), (c, S), (0.3, 0.3, 0.3), 1)
    cv2.line(vis, (0, c), (S, c), (0.3, 0.3, 0.3), 1)
    if show_skin_line:
        # Skin tones cluster along roughly 123 degrees on the vectorscope.
        ang = np.deg2rad(123.0)
        cv2.line(vis, (c, c),
                 (int(c + np.cos(ang) * S * 0.45), int(c - np.sin(ang) * S * 0.45)),
                 (0.95, 0.75, 0.6), 1)

    mean_cb, mean_cr = float(cb.mean()), float(cr.mean())
    return OpResult(image=None, data={
        "mean_cb": round(mean_cb, 5), "mean_cr": round(mean_cr, 5),
        "chroma_offset": round(float(np.hypot(mean_cb, mean_cr)), 5),
        "dominant_hue_deg": round(float(np.degrees(np.arctan2(mean_cr, mean_cb))), 1),
        "mean_saturation": round(float(np.hypot(cb, cr).mean()), 5),
    }, overlays={"vectorscope": clamp01(vis)})


@register(
    "view.falsecolor",
    "False-colour exposure",
    "inspect",
    params=[BoolParam("show_legend", default=True)],
    summary="Maps intensity zones to distinct flat colours, so exposure "
    "problems are unmissable.",
    formula_latex=r"c(f) = \text{LUT}_{\text{zone}}(f)",
    when_to_use="Checking exposure at a glance: pure red means clipped "
    "highlights, pure blue means crushed shadows, green marks the midtone "
    "region where you want your subject.",
    is_view=True,
    tags=["teaching"],
)
def falsecolor(img, show_legend=True, ctx=None):
    y = clamp01(luma(img))
    # Film-style false-colour: purple/blue = crushed, green = mid, red = clipped.
    stops = [
        (0.00, (0.35, 0.0, 0.5)),
        (0.02, (0.0, 0.1, 0.9)),
        (0.12, (0.0, 0.6, 0.9)),
        (0.35, (0.15, 0.85, 0.2)),
        (0.60, (0.95, 0.95, 0.25)),
        (0.85, (1.0, 0.55, 0.1)),
        (0.98, (1.0, 0.05, 0.05)),
        (1.01, (1.0, 1.0, 1.0)),
    ]
    out = np.zeros((*y.shape, 3), np.float32)
    for i in range(len(stops) - 1):
        lo, c = stops[i]
        hi, _ = stops[i + 1]
        sel = (y >= lo) & (y < hi)
        out[sel] = c
    return OpResult(image=None, data={
        "crushed_fraction": round(float((y < 0.02).mean()), 5),
        "clipped_fraction": round(float((y >= 0.98).mean()), 5),
        "midtone_fraction": round(float(((y >= 0.35) & (y < 0.60)).mean()), 5),
        "legend": [{"from": s[0], "color": s[1]} for s in stops[:-1]],
    }, overlays={"falsecolor": out})


@register(
    "view.zonemap",
    "Zone map",
    "inspect",
    params=[IntParam("zones", min=5, max=15, default=11)],
    summary="Quantises luminance into Ansel Adams zones and reports how much of "
    "the image falls in each.",
    formula_latex=r"Z(x,y) = \left\lfloor f(x,y)\,(N-1) + 0.5 \right\rfloor",
    when_to_use="Judging tonal distribution the way a photographer does. The "
    "Advisor consumes exactly this histogram for its tonal-balance metric.",
    is_view=True,
)
def zonemap(img, zones=11, ctx=None):
    y = clamp01(luma(img))
    n = int(zones)
    z = np.clip((y * (n - 1) + 0.5).astype(np.int32), 0, n - 1)
    hist = np.bincount(z.ravel(), minlength=n).astype(np.float64)
    hist /= max(hist.sum(), 1)
    vis = colorize(z.astype(np.float32) / max(n - 1, 1), "turbo")
    third = max(n // 3, 1)
    return OpResult(image=None, data={
        "zones": n,
        "zone_histogram": hist.round(5).tolist(),
        "shadow_mass": round(float(hist[:third].sum()), 5),
        "midtone_mass": round(float(hist[third : n - third].sum()), 5),
        "highlight_mass": round(float(hist[n - third :].sum()), 5),
    }, overlays={"zones": vis})


@register(
    "view.clipmask",
    "Clipping overlay",
    "inspect",
    params=[
        FloatParam("shadow_threshold", min=0.0, max=0.2, default=0.008, step=0.002),
        FloatParam("highlight_threshold", min=0.8, max=1.0, default=0.992, step=0.002),
        BoolParam("per_channel", default=True,
                  help="Also flags single-channel clipping, which shifts hue "
                       "without looking blown."),
    ],
    summary="Highlights pixels at the extremes of the range, where information "
    "has been irreversibly lost.",
    formula_latex=r"m_{hi} = \mathbb{1}[f \ge 1 - \epsilon],\quad "
    r"m_{lo} = \mathbb{1}[f \le \epsilon]",
    when_to_use="Before any tonal edit. Clipped detail cannot be recovered by "
    "any operator, so knowing how much is clipped tells you whether a fix is "
    "even possible.",
    is_view=True,
)
def clipmask(img, shadow_threshold=0.008, highlight_threshold=0.992,
             per_channel=True, ctx=None):
    y = clamp01(luma(img))
    lo = y <= shadow_threshold
    hi = y >= highlight_threshold
    base = img if img.ndim == 3 else np.dstack([img] * 3)
    vis = (base[:, :, :3] * 0.35).copy()
    vis[lo] = (0.1, 0.35, 1.0)
    vis[hi] = (1.0, 0.1, 0.1)

    data = {
        "shadow_clipped": round(float(lo.mean()), 6),
        "highlight_clipped": round(float(hi.mean()), 6),
    }
    if per_channel and img.ndim == 3:
        ch_hi = np.any(img[:, :, :3] >= highlight_threshold, axis=2) & ~hi
        vis[ch_hi] = (1.0, 0.6, 0.0)
        data["channel_clipped"] = round(float(ch_hi.mean()), 6)
        for i, n in enumerate(("r", "g", "b")):
            data[f"clip_{n}"] = round(
                float((img[:, :, i] >= highlight_threshold).mean()), 6)
    return OpResult(image=None, data=data, overlays={"clipping": clamp01(vis)})


@register(
    "view.loupe",
    "Pixel inspector",
    "inspect",
    params=[
        PointParam("center", default=(0.5, 0.5)),
        IntParam("grid", min=3, max=17, default=9),
        EnumParam("space", options=enum_opts(("rgb", "RGB 0-255"),
                                             ("float", "Float 0-1"),
                                             ("hsv", "HSV"), ("lab", "CIE Lab")),
                  default="rgb"),
    ],
    summary="Returns the numeric pixel values in a small neighbourhood around a "
    "point.",
    formula_latex=r"\{f(x_0 + i,\ y_0 + j)\}_{i,j = -k}^{k}",
    when_to_use="Verifying exactly what an operator did, and reading real values "
    "rather than inferring them from the display.",
    is_view=True,
)
def loupe(img, center=(0.5, 0.5), grid=9, space="rgb", ctx=None):
    h, w = img.shape[:2]
    k = int(grid) // 2
    cx = int(np.clip(center[0] * (w - 1), k, w - 1 - k)) if w > 2 * k else w // 2
    cy = int(np.clip(center[1] * (h - 1), k, h - 1 - k)) if h > 2 * k else h // 2
    patch = img[max(0, cy - k) : cy + k + 1, max(0, cx - k) : cx + k + 1]

    if space == "hsv" and patch.ndim == 3:
        vals = cv2.cvtColor(patch.astype(np.float32), cv2.COLOR_RGB2HSV)
        fmt = lambda v: [round(float(v[0]), 1), round(float(v[1]), 3),
                         round(float(v[2]), 3)]  # noqa: E731
    elif space == "lab" and patch.ndim == 3:
        vals = to_lab(patch)
        fmt = lambda v: [round(float(x), 1) for x in v]  # noqa: E731
    elif space == "rgb":
        vals = patch * 255.0
        fmt = lambda v: ([int(round(float(x))) for x in v]  # noqa: E731
                         if np.ndim(v) else int(round(float(v))))
    else:
        vals = patch
        fmt = lambda v: ([round(float(x), 4) for x in v]  # noqa: E731
                         if np.ndim(v) else round(float(v), 4))

    if vals.ndim == 2:
        table = [[fmt(vals[r, c]) for c in range(vals.shape[1])]
                 for r in range(vals.shape[0])]
    else:
        table = [[fmt(vals[r, c]) for c in range(vals.shape[1])]
                 for r in range(vals.shape[0])]
    return OpResult(image=None, data={
        "center_px": [cx, cy], "grid": int(grid), "space": space, "values": table,
        "center_value": table[len(table) // 2][len(table[0]) // 2] if table else None,
    }, overlays={"loupe": clamp01(cv2.resize(
        patch if patch.ndim == 3 else np.dstack([patch] * 3),
        (int(grid) * 24, int(grid) * 24), interpolation=cv2.INTER_NEAREST))})


@register(
    "view.dft",
    "DFT magnitude / phase",
    "inspect",
    params=[
        EnumParam("show", options=enum_opts(("magnitude", "Magnitude"),
                                            ("phase", "Phase"), ("both", "Both")),
                  default="magnitude"),
        BoolParam("log_scale", default=True),
        EnumParam("window", options=enum_opts(("none", "None"), ("hann", "Hann"),
                                              ("hamming", "Hamming")),
                  default="hann"),
        EnumParam("colormap", options=enum_opts("turbo", "viridis", "magma", "gray"),
                  default="turbo"),
    ],
    summary="The two-dimensional discrete Fourier transform, fftshifted so DC "
    "sits at the centre, with magnitude log-scaled.",
    formula_latex=r"F(u,v) = \sum_{x=0}^{M-1}\sum_{y=0}^{N-1} f(x,y)\,"
    r"e^{-j2\pi\left(\frac{ux}{M} + \frac{vy}{N}\right)}",
    when_to_use="Reading an image's frequency content: a bright centre means "
    "most energy is low-frequency (smooth), radial streaks indicate strong "
    "oriented edges, and isolated bright dots away from DC are periodic noise. "
    "Log scaling is essential because DC is typically millions of times larger "
    "than the rest.",
    is_view=True,
    syllabus=True,
    cost="fast",
    references=["Gonzalez & Woods, Ch. 4.5-4.6"],
)
def dft_view(img, show="magnitude", log_scale=True, window="hann",
             colormap="turbo", ctx=None):
    g = clamp01(luma(img)).astype(np.float32)
    h, w = g.shape
    if window != "none":
        wy = np.hanning(h) if window == "hann" else np.hamming(h)
        wx = np.hanning(w) if window == "hann" else np.hamming(w)
        g = g * np.outer(wy, wx).astype(np.float32)

    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(F)
    disp_mag = np.log1p(mag) if log_scale else mag
    phase = np.angle(F)

    data = {
        "dc_magnitude": round(float(mag[h // 2, w // 2]), 3),
        "mean_magnitude": round(float(mag.mean()), 5),
        "peak_offcentre": None,
    }
    # Report the strongest non-DC component as a frequency in cycles/pixel.
    m2 = mag.copy()
    r = max(3, int(0.02 * max(h, w)))
    m2[h // 2 - r : h // 2 + r, w // 2 - r : w // 2 + r] = 0
    idx = int(np.argmax(m2))
    py, px = divmod(idx, w)
    data["peak_offcentre"] = {
        "fx": round((px - w / 2) / w, 4), "fy": round((py - h / 2) / h, 4),
        "magnitude": round(float(m2[py, px]), 3),
        "relative_to_dc": round(float(m2[py, px] / max(mag[h // 2, w // 2], EPS)), 6),
    }

    overlays = {}
    cmap = colormap
    if show in ("magnitude", "both"):
        overlays["magnitude"] = (np.dstack([normalize01(disp_mag)] * 3)
                                 if cmap == "gray"
                                 else colorize(normalize01(disp_mag), cmap))
    if show in ("phase", "both"):
        overlays["phase"] = (np.dstack([normalize01(phase)] * 3) if cmap == "gray"
                             else colorize(normalize01(phase), cmap))
    return OpResult(image=None, data=data, overlays=overlays)


@register(
    "view.radial_spectrum",
    "Radial power spectrum",
    "inspect",
    params=[IntParam("bins", min=32, max=256, default=128)],
    summary="Averages the power spectrum over rings of constant radius, then "
    "fits a straight line to the log-log curve.",
    formula_latex=r"P(f) \propto f^{-\alpha};\quad "
    r"\log P = -\alpha \log f + c",
    when_to_use="Separating blur from noise. Natural images have alpha near 2; "
    "a blurred image falls off faster (more negative slope), while a noisy one "
    "flattens at high frequencies. This single curve drives several Advisor "
    "sharpness metrics.",
    is_view=True,
    cost="fast",
    references=["Field, Relations between the statistics of natural images and "
                "the response properties of cortical cells, 1987"],
)
def radial_spectrum(img, bins=128, ctx=None):
    g = clamp01(luma(img)).astype(np.float32)
    g = g - g.mean()
    h, w = g.shape
    F = np.fft.fftshift(np.fft.fft2(g))
    power = (np.abs(F) ** 2).astype(np.float64)

    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    rad = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rmax = min(cx, cy)
    nb = int(bins)
    idx = np.clip((rad / max(rmax, EPS) * (nb - 1)).astype(np.int32), 0, nb - 1)
    valid = rad <= rmax

    sums = np.bincount(idx[valid], weights=power[valid], minlength=nb)
    counts = np.bincount(idx[valid], minlength=nb)
    radial = sums / np.maximum(counts, 1)

    freqs = np.linspace(0, 0.5, nb)
    # Fit over a mid-band, avoiding DC and the noise-dominated top end.
    lo_i = max(1, int(0.04 * nb))
    hi_i = int(0.8 * nb)
    sel = slice(lo_i, hi_i)
    fx = np.log10(np.maximum(freqs[sel], 1e-6))
    fy = np.log10(np.maximum(radial[sel], 1e-20))
    slope, intercept = np.polyfit(fx, fy, 1)

    total = float(radial.sum()) + EPS
    hf = float(radial[int(0.5 * nb) :].sum())
    return OpResult(image=None, data={
        "radial_power": np.round(np.log10(np.maximum(radial, 1e-20)), 4).tolist(),
        "frequencies": np.round(freqs, 5).tolist(),
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "hf_energy_ratio": round(hf / total, 6),
    })


@register(
    "view.dct",
    "DCT spectrum",
    "inspect",
    params=[
        EnumParam("mode", options=enum_opts(("global", "Whole image"),
                                            ("block8", "8x8 blocks averaged")),
                  default="global"),
        BoolParam("log_scale", default=True),
    ],
    summary="The discrete cosine transform, which packs image energy into far "
    "fewer coefficients than the DFT and uses only real arithmetic.",
    formula_latex=r"C(u,v) = \alpha(u)\alpha(v)\sum_x\sum_y f(x,y)"
    r"\cos\frac{(2x+1)u\pi}{2N}\cos\frac{(2y+1)v\pi}{2N}",
    when_to_use="Understanding JPEG. The averaged 8x8 mode shows the "
    "characteristic concentration into the top-left corner that makes "
    "quantisation of the rest viable.",
    is_view=True,
    tags=["beyond-syllabus", "teaching"],
    cost="fast",
)
def dct_view(img, mode="global", log_scale=True, ctx=None):
    g = clamp01(luma(img)).astype(np.float32) - 0.5
    if mode == "block8":
        h, w = g.shape
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        p = np.pad(g, ((0, ph), (0, pw)), mode="edge")
        blocks = p.reshape(p.shape[0] // 8, 8, p.shape[1] // 8, 8).transpose(0, 2, 1, 3)
        acc = np.zeros((8, 8), np.float64)
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                C = dct(dct(blocks[i, j].T, norm="ortho").T, norm="ortho")
                acc += np.abs(C)
        acc /= max(blocks.shape[0] * blocks.shape[1], 1)
        spec = acc
        energy_tl = float(acc[:2, :2].sum() / max(acc.sum(), EPS))
    else:
        C = dct(dct(g.T, norm="ortho").T, norm="ortho")
        spec = np.abs(C)
        n = max(spec.shape) // 8
        energy_tl = float(spec[:n, :n].sum() / max(spec.sum(), EPS))

    disp = np.log1p(spec) if log_scale else spec
    return OpResult(image=None, data={
        "mode": mode,
        "low_frequency_energy_fraction": round(energy_tl, 5),
        "coefficients": (np.round(spec, 5).tolist() if mode == "block8" else None),
    }, overlays={"dct": colorize(normalize01(disp.astype(np.float32)), "turbo")})


@register(
    "view.glcm",
    "GLCM texture statistics",
    "inspect",
    params=[
        IntParam("distance", min=1, max=16, default=1, scale_aware=True),
        IntParam("levels", min=8, max=256, default=32),
        BoolParam("symmetric", default=True),
        BoolParam("normed", default=True),
    ],
    summary="Grey-level co-occurrence matrix: how often each pair of intensities "
    "occurs at a given offset, summarised by Haralick's descriptors at four "
    "orientations.",
    formula_latex=r"P_{d,\theta}(i,j) = \#\{(p,q) : f(p)=i,\ f(q)=j,\ "
    r"q = p + d_\theta\}",
    when_to_use="Quantifying texture. Contrast and homogeneity separate smooth "
    "from rough regions, and comparing across the four angles reveals "
    "directional texture that a histogram cannot see.",
    is_view=True,
    cost="fast",
    references=["Haralick, Shanmugam & Dinstein, Textural Features for Image "
                "Classification, 1973"],
)
def glcm(img, distance=1, levels=32, symmetric=True, normed=True, ctx=None):
    g = gray_u8(img)
    lv = int(levels)
    q = (g.astype(np.uint16) * lv // 256).astype(np.uint8)
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    try:
        m = graycomatrix(q, distances=[int(distance)], angles=angles,
                         levels=lv, symmetric=bool(symmetric), normed=bool(normed))
    except Exception as exc:
        return OpResult(image=None, data={"error": str(exc)})

    props = {}
    for name in ("contrast", "dissimilarity", "homogeneity", "ASM", "energy",
                 "correlation"):
        vals = graycoprops(m, name).ravel()
        props[name] = {
            "per_angle": [round(float(v), 6) for v in vals],
            "mean": round(float(vals.mean()), 6),
            "anisotropy": round(float(vals.max() - vals.min()), 6),
        }
    # Haralick entropy is not in graycoprops; compute it directly.
    p = m[:, :, 0, :].astype(np.float64)
    p = p / np.maximum(p.sum(axis=(0, 1), keepdims=True), EPS)
    ent = -np.sum(np.where(p > 0, p * np.log2(np.maximum(p, 1e-20)), 0), axis=(0, 1))
    props["entropy"] = {"per_angle": [round(float(v), 6) for v in ent],
                        "mean": round(float(ent.mean()), 6)}

    avg = m[:, :, 0, :].mean(axis=2)
    return OpResult(image=None,
                    data={"distance": int(distance), "levels": lv,
                          "angles_deg": [0, 45, 90, 135], "properties": props},
                    overlays={"glcm": colorize(
                        normalize01(np.log1p(avg).astype(np.float32)), "viridis")})


@register(
    "view.entropy_map",
    "Local entropy map",
    "inspect",
    params=[OddIntParam("window", min=3, max=63, default=9, scale_aware=True)],
    summary="Shannon entropy of the intensity histogram in a sliding window.",
    formula_latex=r"H(x,y) = -\sum_k p_k \log_2 p_k \quad\text{over } S_{xy}",
    when_to_use="Locating where the information is. Flat sky has near-zero "
    "entropy; detailed foliage is high. Useful for deciding where to spend "
    "bits, or where detail loss would be noticed.",
    is_view=True,
    cost="medium",
)
def entropy_map(img, window=9, ctx=None):
    from skimage.filters.rank import entropy as sk_entropy
    from skimage.morphology import disk

    g = gray_u8(img)
    r = max(1, int(window) // 2)
    e = sk_entropy(g, disk(r)).astype(np.float32)
    return OpResult(image=None, data={
        "mean_entropy": round(float(e.mean()), 4),
        "max_entropy": round(float(e.max()), 4),
        "low_entropy_fraction": round(float((e < 2.0).mean()), 5),
    }, overlays={"entropy": colorize(normalize01(e), "viridis")})


def spectral_saliency(gray: np.ndarray) -> np.ndarray:
    """Spectral-residual saliency. Pure FFT -- no learned model.

    Natural image log-spectra are near-scale-invariant, so the *residual* after
    removing the locally averaged log-spectrum highlights what is statistically
    unusual, which correlates with where people look.
    """
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    F = np.fft.fft2(small)
    log_amp = np.log(np.abs(F) + 1e-8)
    phase = np.angle(F)
    avg = cv2.blur(log_amp.astype(np.float32), (3, 3))
    residual = log_amp - avg
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 2.5)
    return normalize01(sal)


@register(
    "view.saliency",
    "Saliency map",
    "inspect",
    params=[
        EnumParam("method", options=enum_opts(("spectral_residual", "Spectral residual"),
                                              ("fine_grained", "Fine-grained")),
                  default="spectral_residual"),
        EnumParam("colormap", options=enum_opts("inferno", "viridis", "turbo"),
                  default="inferno"),
    ],
    summary="Estimates which regions are visually unusual, from the residual of "
    "the log-amplitude spectrum.",
    formula_latex=r"R(f) = \log|F| - h_n * \log|F|,\quad "
    r"S = g * \left|\mathcal{F}^{-1}\{e^{R + j\varphi}\}\right|^2",
    when_to_use="ROI weighting. The Advisor uses this to weight sharpness "
    "measurement toward the subject, so a deliberately blurred background is "
    "not reported as a defect. Entirely FFT-based -- no model.",
    is_view=True,
    tags=["beyond-syllabus"],
    references=["Hou & Zhang, Saliency Detection: A Spectral Residual Approach, "
                "CVPR 2007"],
)
def saliency(img, method="spectral_residual", colormap="inferno", ctx=None):
    g = clamp01(luma(img)).astype(np.float32)
    if method == "fine_grained":
        try:
            sal = cv2.saliency.StaticSaliencyFineGrained_create()
            ok, m = sal.computeSaliency((g * 255).astype(np.uint8))
            s = normalize01(m.astype(np.float32)) if ok else spectral_saliency(g)
        except Exception:
            s = spectral_saliency(g)
    else:
        s = spectral_saliency(g)
    s_full = cv2.resize(s, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    s_full = normalize01(s_full)

    # Centre of mass of saliency = where the subject probably is.
    h, w = s_full.shape
    tot = float(s_full.sum()) + EPS
    yy, xx = np.mgrid[0:h, 0:w]
    cx = float((xx * s_full).sum() / tot) / w
    cy = float((yy * s_full).sum() / tot) / h
    return OpResult(image=None, data={
        "salient_fraction": round(float((s_full > 0.5).mean()), 5),
        "centroid": [round(cx, 4), round(cy, 4)],
        "mean": round(float(s_full.mean()), 5),
    }, overlays={"saliency": colorize(s_full, colormap)})


@register(
    "view.darkchannel",
    "Dark channel",
    "inspect",
    params=[OddIntParam("patch", min=3, max=61, default=15, scale_aware=True)],
    summary="The minimum over colour channels and over a local patch. In a "
    "haze-free outdoor image this is close to zero almost everywhere; haze "
    "raises it uniformly.",
    formula_latex=r"J^{dark}(x) = \min_{y \in \Omega(x)}"
    r"\left(\min_{c \in \{r,g,b\}} I^c(y)\right)",
    when_to_use="Haze detection and removal. The prior holds because most "
    "natural patches contain at least one dark pixel in at least one channel; "
    "airlight adds a floor to every channel at once.",
    is_view=True,
    tags=["beyond-syllabus"],
    references=["He, Sun & Tang, Single Image Haze Removal Using Dark Channel "
                "Prior, CVPR 2009"],
)
def darkchannel(img, patch=15, ctx=None):
    base = img if img.ndim == 3 else np.dstack([img] * 3)
    mn = np.min(base[:, :, :3], axis=2)
    k = int(patch) | 1
    dark = cv2.erode(mn, np.ones((k, k), np.uint8))

    flat = dark.ravel()
    n = max(1, int(0.001 * flat.size))
    airlight = float(np.mean(np.sort(flat)[-n:]))
    omega = 0.95
    transmission = 1.0 - omega * dark / max(airlight, 1e-3)
    return OpResult(image=None, data={
        "dark_channel_mean": round(float(dark.mean()), 5),
        "dark_channel_median": round(float(np.median(dark)), 5),
        "airlight_estimate": round(airlight, 5),
        "mean_transmission": round(float(np.clip(transmission, 0, 1).mean()), 5),
    }, overlays={
        "dark_channel": colorize(dark, "bone"),
        "transmission": colorize(np.clip(transmission, 0, 1), "viridis"),
    })


@register(
    "view.noisemap",
    "Noise map",
    "inspect",
    params=[
        OddIntParam("window", min=7, max=63, default=16, scale_aware=True),
        EnumParam("colormap", options=enum_opts("magma", "inferno", "turbo"),
                  default="magma"),
    ],
    summary="Per-tile noise standard deviation, estimated from the median "
    "absolute deviation of a high-pass residual.",
    formula_latex=r"\hat{\sigma} = 1.4826 \cdot \mathrm{MAD}"
    r"\big(f - \mathrm{median}_{3\times3}(f)\big)",
    when_to_use="Seeing WHERE noise lives. It is usually worst in shadows, and "
    "a map shows that immediately, whereas a single global sigma hides it.",
    is_view=True,
    cost="medium",
)
def noisemap(img, window=16, colormap="magma", ctx=None):
    g = clamp01(luma(img)).astype(np.float32)
    med = cv2.medianBlur(g, 3)
    resid = np.abs(g - med)
    k = int(window) | 1
    local_med = cv2.medianBlur(resid, min(k, 5) | 1)
    sigma_map = cv2.blur(np.abs(resid - local_med), (k, k)) * 1.4826

    h, w = g.shape
    tile = max(8, int(window))
    tiles = []
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            patch = g[y : y + tile, x : x + tile]
            gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
            tiles.append((float(np.mean(gx**2 + gy**2)),
                          float(np.median(np.abs(patch - cv2.medianBlur(patch, 3))))))
    sigma_flat = 0.0
    if tiles:
        tiles.sort(key=lambda t: t[0])
        flat = tiles[: max(1, len(tiles) // 20)]
        sigma_flat = float(np.median([t[1] for t in flat])) * 1.4826

    return OpResult(image=None, data={
        "sigma_flat_regions": round(sigma_flat, 6),
        "sigma_flat_8bit": round(sigma_flat * 255, 3),
        "sigma_map_mean": round(float(sigma_map.mean()), 6),
        "sigma_map_max": round(float(sigma_map.max()), 6),
    }, overlays={"noise": colorize(normalize01(sigma_map), colormap)})


@register(
    "view.metrics",
    "Quality metrics",
    "inspect",
    params=[
        ImageRefParam("reference", help="Leave empty to compare with the "
                                        "pipeline input."),
        BoolParam("include_ms_ssim", default=True),
    ],
    summary="Full-reference image-quality metrics against a reference image.",
    formula_latex=r"\mathrm{PSNR} = 10\log_{10}\frac{L^2}{\mathrm{MSE}},\quad "
    r"\mathrm{SSIM} = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}"
    r"{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}",
    when_to_use="Quantifying how much an operation changed the image. PSNR "
    "measures raw error; SSIM measures structural similarity and tracks "
    "perceived quality considerably better.",
    is_view=True,
    syllabus=True,
    cost="medium",
    references=["Wang et al., Image Quality Assessment: From Error Visibility to "
                "Structural Similarity, IEEE TIP 2004"],
)
def metrics(img, reference=None, include_ms_ssim=True, ctx=None):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    ctx = ctx or {}
    ref = None
    if reference:
        from ._util import resolve_ref_image
        ref = resolve_ref_image(ctx, reference, img)
    elif ctx.get("original") is not None:
        ref = ctx["original"]
        if ref.shape[:2] != img.shape[:2]:
            ref = cv2.resize(ref, (img.shape[1], img.shape[0]),
                             interpolation=cv2.INTER_AREA)
    if ref is None:
        return OpResult(image=None, data={"available": False},
                        notes=["No reference available for comparison."])

    a = clamp01(img if img.ndim == 2 else img[:, :, :3])
    b = clamp01(ref if ref.ndim == 2 else ref[:, :, :3])
    if a.ndim != b.ndim:
        a, b = clamp01(luma(a)), clamp01(luma(b))

    mse = float(np.mean((a - b) ** 2))
    data = {
        "available": True,
        "mse": round(mse, 9),
        "rmse": round(float(np.sqrt(mse)), 6),
        "mae": round(float(np.mean(np.abs(a - b))), 6),
        "psnr_db": round(float(peak_signal_noise_ratio(b, a, data_range=1.0))
                         if mse > 0 else 99.0, 3),
    }
    try:
        data["ssim"] = round(float(structural_similarity(
            b, a, data_range=1.0, channel_axis=-1 if a.ndim == 3 else None)), 5)
    except Exception:
        pass

    ya, yb = clamp01(luma(a)), clamp01(luma(b))
    # Normalised cross-correlation
    da, db = ya - ya.mean(), yb - yb.mean()
    data["ncc"] = round(float(np.sum(da * db) /
                              max(np.sqrt(np.sum(da**2) * np.sum(db**2)), EPS)), 5)
    for name, arr in (("entropy_result", ya), ("entropy_reference", yb)):
        hist, _ = np.histogram(arr, bins=256, range=(0, 1))
        p = hist / max(hist.sum(), 1)
        p = p[p > 0]
        data[name] = round(float(-np.sum(p * np.log2(p))), 4)
    data["sharpness_result"] = round(float(cv2.Laplacian(ya, cv2.CV_32F).var()), 7)
    data["sharpness_reference"] = round(float(cv2.Laplacian(yb, cv2.CV_32F).var()), 7)

    if include_ms_ssim:
        # Multi-scale SSIM: SSIM at successive pyramid levels, geometrically
        # combined, which correlates better with perception than single-scale.
        vals = []
        ca, cb = ya, yb
        for _ in range(4):
            if min(ca.shape) < 16:
                break
            try:
                vals.append(float(structural_similarity(cb, ca, data_range=1.0)))
            except Exception:
                break
            ca, cb = cv2.pyrDown(ca), cv2.pyrDown(cb)
        if vals:
            data["ms_ssim"] = round(float(np.prod(vals) ** (1.0 / len(vals))), 5)
            data["ssim_per_scale"] = [round(v, 5) for v in vals]
    return OpResult(image=None, data=data)
