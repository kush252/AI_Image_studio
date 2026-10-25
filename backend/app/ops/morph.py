"""Morphological operations. Catalogue Section 8 (syllabus block).

Every operator here shares the structuring-element designer: shape, size,
angle, anchor and iteration count.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import (
    area_closing,
    area_opening,
    medial_axis,
    reconstruction,
    skeletonize,
    thin,
)

from ..core.colorspace import luma
from ..core.params import (
    BoolParam,
    EnumParam,
    FloatParam,
    IntParam,
    MaskRefParam,
    MatrixParam,
    OddIntParam,
    enum_opts,
)
from ..core.registry import OpResult, register
from ._util import EPS, clamp01, colorize, normalize01, structuring_element

SHAPE_OPTS = enum_opts(
    ("ellipse", "Ellipse / disc"),
    ("rect", "Rectangle"),
    ("cross", "Cross"),
    ("diamond", "Diamond"),
    ("line", "Line at angle"),
    ("custom", "Custom matrix"),
)


def _se_params(default_size: int = 5) -> list:
    """The shared structuring-element designer."""
    return [
        EnumParam("shape", options=SHAPE_OPTS, default="ellipse", group="Structuring element"),
        OddIntParam("size", min=1, max=99, default=default_size, scale_aware=True,
                    group="Structuring element"),
        FloatParam("angle", min=0.0, max=180.0, default=0.0, step=5.0, unit="deg",
                   group="Structuring element",
                   depends_on={"param": "shape", "equals": "line"}),
        IntParam("iterations", min=1, max=20, default=1, group="Structuring element"),
        MatrixParam("custom_se", rows=3, cols=3,
                    default=((0, 1, 0), (1, 1, 1), (0, 1, 0)), min=0, max=1,
                    group="Structuring element", advanced=True,
                    depends_on={"param": "shape", "equals": "custom"}),
    ]


def _morph(op_id, label, cv_op, doc, formula, when, syllabus=True, refs=None,
           default_size=5):
    def fn(img, shape="ellipse", size=5, angle=0.0, iterations=1, custom_se=None,
           ctx=None):
        se = structuring_element(shape, size, angle, custom_se)
        out = cv2.morphologyEx(img, cv_op, se, iterations=int(iterations))
        return OpResult(image=clamp01(out),
                        data={"se_size": int(se.shape[0]),
                              "se_area": int(se.sum())})

    register(op_id, label, "morph", params=_se_params(default_size), summary=doc,
             formula_latex=formula, when_to_use=when, syllabus=syllabus,
             references=refs or ["Gonzalez & Woods, Ch. 9"])(fn)


@register(
    "morph.dilate",
    "Dilation",
    "morph",
    params=_se_params(5),
    summary="Replaces each pixel with the maximum over the structuring element, "
    "growing bright regions.",
    formula_latex=r"A \oplus B = \{z \mid (\hat{B})_z \cap A \neq \emptyset\}",
    when_to_use="Filling small holes, joining broken strokes, and thickening "
    "thin features.",
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 9.2.2"],
)
def dilate(img, shape="ellipse", size=5, angle=0.0, iterations=1, custom_se=None,
           ctx=None):
    se = structuring_element(shape, size, angle, custom_se)
    return clamp01(cv2.dilate(img, se, iterations=int(iterations)))


@register(
    "morph.erode",
    "Erosion",
    "morph",
    params=_se_params(5),
    summary="Replaces each pixel with the minimum over the structuring element, "
    "shrinking bright regions.",
    formula_latex=r"A \ominus B = \{z \mid (B)_z \subseteq A\}",
    when_to_use="Removing small bright specks, separating touching objects, and "
    "thinning features.",
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 9.2.1"],
)
def erode(img, shape="ellipse", size=5, angle=0.0, iterations=1, custom_se=None,
          ctx=None):
    se = structuring_element(shape, size, angle, custom_se)
    return clamp01(cv2.erode(img, se, iterations=int(iterations)))


_morph("morph.open", "Opening", cv2.MORPH_OPEN,
       "Erosion followed by dilation. Removes structures smaller than the "
       "structuring element while keeping the size of everything that survives.",
       r"A \circ B = (A \ominus B) \oplus B",
       "Removing small bright noise without shrinking the objects you keep. "
       "Idempotent: opening twice equals opening once.")
_morph("morph.close", "Closing", cv2.MORPH_CLOSE,
       "Dilation followed by erosion. Fills gaps and holes smaller than the "
       "structuring element.",
       r"A \bullet B = (A \oplus B) \ominus B",
       "Closing small dark gaps, joining nearby components, filling pinholes.")
_morph("morph.gradient", "Morphological gradient", cv2.MORPH_GRADIENT,
       "Difference between dilation and erosion: a thick outline of every edge.",
       r"g = (A \oplus B) - (A \ominus B)",
       "Edge detection with no derivatives at all -- useful on binary images "
       "where gradient operators behave badly.", default_size=3)
_morph("morph.tophat", "Top-hat (white)", cv2.MORPH_TOPHAT,
       "The image minus its opening: isolates bright features smaller than the "
       "structuring element.",
       r"T_w = A - (A \circ B)",
       "Extracting small bright detail from an uneven background. With a large "
       "SE it is the standard illumination-correction trick.", default_size=15)
_morph("morph.blackhat", "Black-hat", cv2.MORPH_BLACKHAT,
       "The closing minus the image: isolates dark features smaller than the "
       "structuring element.",
       r"T_b = (A \bullet B) - A",
       "Extracting dark text or thin dark lines from a bright, unevenly lit "
       "background -- the usual first step on a photographed document.",
       default_size=15)


@register(
    "morph.hitmiss",
    "Hit-or-miss",
    "morph",
    params=[
        MatrixParam("se_fg", rows=3, cols=3,
                    default=((0, 1, 0), (1, 1, 1), (0, 1, 0)), min=0, max=1,
                    help="Pattern that must be present (foreground)."),
        MatrixParam("se_bg", rows=3, cols=3,
                    default=((1, 0, 1), (0, 0, 0), (1, 0, 1)), min=0, max=1,
                    help="Pattern that must be absent (background)."),
        FloatParam("threshold", min=0.0, max=1.0, default=0.5, step=0.01),
    ],
    summary="Finds pixels where one pattern fits the foreground AND a second "
    "fits the background simultaneously.",
    formula_latex=r"A \circledast B = (A \ominus B_1) \cap (A^c \ominus B_2)",
    when_to_use="Exact shape and template detection on binary images: corners, "
    "line ends, isolated points, junctions.",
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 9.4"],
)
def hitmiss(img, se_fg=None, se_bg=None, threshold=0.5, ctx=None):
    g = luma(img) if img.ndim == 3 else img
    b = (g > threshold).astype(np.uint8)
    fg = np.array(se_fg, np.int8)
    bg = np.array(se_bg, np.int8)
    kernel = fg - bg  # +1 must be set, -1 must be clear, 0 is don't-care
    out = cv2.morphologyEx(b, cv2.MORPH_HITMISS, kernel)
    return OpResult(image=out.astype(np.float32),
                    data={"matches": int(out.sum())})


@register(
    "morph.thin",
    "Thinning",
    "morph",
    params=[
        IntParam("iterations", min=1, max=100, default=20),
        FloatParam("threshold", min=0.0, max=1.0, default=0.5, step=0.01),
    ],
    summary="Iteratively erodes the boundary while preserving connectivity and "
    "end points.",
    formula_latex=r"A \otimes B = A - (A \circledast B)",
    when_to_use="Reducing strokes toward unit width without breaking them -- "
    "handwriting, fingerprints, circuit traces.",
    syllabus=True,
    cost="medium",
)
def thin_op(img, iterations=20, threshold=0.5, ctx=None):
    g = luma(img) if img.ndim == 3 else img
    b = g > threshold
    out = thin(b, max_num_iter=int(iterations))
    return out.astype(np.float32)


@register(
    "morph.skeleton",
    "Skeletonize",
    "morph",
    params=[
        EnumParam("method", options=enum_opts(("zhang", "Zhang-Suen"),
                                              ("lee", "Lee (3-D capable)"),
                                              ("medial_axis", "Medial axis")),
                  default="zhang"),
        FloatParam("threshold", min=0.0, max=1.0, default=0.5, step=0.01),
    ],
    summary="Reduces a binary shape to a one-pixel-wide centreline that "
    "preserves its topology.",
    formula_latex=r"S(A) = \bigcup_{k=0}^{K} S_k(A),\quad "
    r"S_k(A) = (A \ominus kB) - (A \ominus kB)\circ B",
    when_to_use="Shape analysis, character recognition, and measuring the length "
    "of elongated structures.",
    syllabus=True,
    cost="medium",
    references=["Zhang & Suen, A fast parallel algorithm for thinning, 1984"],
)
def skeleton(img, method="zhang", threshold=0.5, ctx=None):
    g = luma(img) if img.ndim == 3 else img
    b = g > threshold
    if method == "medial_axis":
        out = medial_axis(b)
    elif method == "lee":
        out = skeletonize(b, method="lee")
    else:
        out = skeletonize(b)
    return OpResult(image=out.astype(np.float32),
                    data={"skeleton_pixels": int(np.sum(out))})


@register(
    "morph.reconstruct",
    "Reconstruction by dilation",
    "morph",
    params=[
        EnumParam("marker", options=enum_opts(("erode", "Eroded image"),
                                              ("border", "Image border"),
                                              ("threshold", "High-intensity seed")),
                  default="erode"),
        OddIntParam("marker_size", min=3, max=99, default=15, scale_aware=True),
        FloatParam("seed_level", min=0.0, max=1.0, default=0.7, step=0.01,
                   depends_on={"param": "marker", "equals": "threshold"}),
    ],
    summary="Geodesic reconstruction: repeatedly dilates a marker image but "
    "clips it to the original at every step, so it regrows exactly the "
    "components the marker touched, with their original shapes intact.",
    formula_latex=r"R_f(g) = \lim_{n\to\infty} \delta_f^{(n)}(g),\quad "
    r"\delta_f(g) = (g \oplus B) \wedge f",
    when_to_use="Far better than opening when shape must be preserved: opening "
    "with a large SE rounds every corner, whereas reconstruction removes whole "
    "objects and leaves the rest pixel-identical.",
    tags=["beyond-syllabus"],
    cost="medium",
    references=["Vincent, Morphological grayscale reconstruction, IEEE TIP 1993"],
)
def morph_reconstruct(img, marker="erode", marker_size=15, seed_level=0.7, ctx=None):
    g = (luma(img) if img.ndim == 3 else img).astype(np.float32)
    if marker == "border":
        seed = np.zeros_like(g)
        seed[0, :] = g[0, :]
        seed[-1, :] = g[-1, :]
        seed[:, 0] = g[:, 0]
        seed[:, -1] = g[:, -1]
    elif marker == "threshold":
        seed = np.where(g >= seed_level, g, 0.0).astype(np.float32)
    else:
        se = structuring_element("ellipse", marker_size)
        seed = cv2.erode(g, se)
    seed = np.minimum(seed, g)
    out = reconstruction(seed, g, method="dilation")
    return clamp01(out.astype(np.float32))


@register(
    "morph.area_open",
    "Area opening / closing",
    "morph",
    params=[
        EnumParam("mode", options=enum_opts(("open", "Area opening (remove bright)"),
                                            ("close", "Area closing (fill dark)")),
                  default="open"),
        IntParam("min_area", min=1, max=20000, default=100, scale_aware=True),
        EnumParam("connectivity", options=enum_opts(("1", "4-connected"),
                                                    ("2", "8-connected")),
                  default="2"),
    ],
    summary="Removes connected components smaller than a given AREA, regardless "
    "of their shape.",
    formula_latex=r"\gamma_\lambda(f) = \bigvee\{\gamma_B(f) : |B| \ge \lambda\}",
    when_to_use="When the thing that distinguishes noise from signal is size "
    "rather than shape -- a long thin scratch and a round blob of the same area "
    "are treated alike, which a shaped SE cannot do.",
    tags=["beyond-syllabus"],
    cost="medium",
)
def area_open_op(img, mode="open", min_area=100, connectivity="2", ctx=None):
    g = (luma(img) if img.ndim == 3 else img).astype(np.float32)
    u16 = (np.clip(g, 0, 1) * 65535).astype(np.uint16)
    conn = int(connectivity)
    if mode == "close":
        out = area_closing(u16, area_threshold=int(min_area), connectivity=conn)
    else:
        out = area_opening(u16, area_threshold=int(min_area), connectivity=conn)
    return (out.astype(np.float32) / 65535.0).astype(np.float32)


@register(
    "morph.fill_holes",
    "Fill holes",
    "morph",
    params=[
        FloatParam("threshold", min=0.0, max=1.0, default=0.5, step=0.01),
        EnumParam("connectivity", options=enum_opts(("1", "4-connected"),
                                                    ("2", "8-connected")),
                  default="1"),
    ],
    summary="Fills every background region not connected to the image border.",
    formula_latex=r"H = \left[R_{A^c}(\text{border})\right]^c",
    when_to_use="Closing interior voids in segmented objects before measuring "
    "their area.",
    syllabus=True,
)
def fill_holes(img, threshold=0.5, connectivity="1", ctx=None):
    g = luma(img) if img.ndim == 3 else img
    b = g > threshold
    out = ndimage.binary_fill_holes(
        b, structure=np.ones((3, 3)) if connectivity == "2" else None
    )
    filled = int(np.sum(out) - np.sum(b))
    return OpResult(image=out.astype(np.float32), data={"pixels_filled": filled})


@register(
    "morph.convex_hull",
    "Convex hull",
    "morph",
    params=[
        FloatParam("threshold", min=0.0, max=1.0, default=0.5, step=0.01),
        BoolParam("per_object", default=True),
        IntParam("min_area", min=1, max=10000, default=30, scale_aware=True),
    ],
    summary="The smallest convex set containing each object.",
    formula_latex=r"C(A) = \bigcap \{K \supseteq A : K \text{ convex}\}",
    when_to_use="Measuring solidity (area / hull area), which separates "
    "star-shaped or indented objects from compact ones.",
    syllabus=True,
)
def convex_hull(img, threshold=0.5, per_object=True, min_area=30, ctx=None):
    g = luma(img) if img.ndim == 3 else img
    b = (g > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(b)
    stats = []
    if per_object:
        for c in contours:
            a = cv2.contourArea(c)
            if a < min_area:
                continue
            hull = cv2.convexHull(c)
            ha = cv2.contourArea(hull)
            cv2.drawContours(out, [hull], -1, 1, -1)
            stats.append({"area": round(a, 1), "hull_area": round(ha, 1),
                          "solidity": round(a / max(ha, EPS), 4)})
    elif contours:
        allpts = np.vstack(contours)
        hull = cv2.convexHull(allpts)
        cv2.drawContours(out, [hull], -1, 1, -1)
    return OpResult(image=out.astype(np.float32),
                    data={"objects": len(stats), "hulls": stats[:200]})


@register(
    "morph.distance",
    "Distance transform",
    "morph",
    params=[
        EnumParam("metric", options=enum_opts(("l2", "Euclidean (L2)"),
                                              ("l1", "City-block (L1)"),
                                              ("chessboard", "Chessboard (L-inf)")),
                  default="l2"),
        FloatParam("threshold", min=0.0, max=1.0, default=0.5, step=0.01),
        BoolParam("normalize", default=True),
        BoolParam("invert", default=False),
        EnumParam("colormap", options=enum_opts("viridis", "turbo", "magma", "jet"),
                  default="turbo"),
    ],
    summary="Distance from each foreground pixel to the nearest background "
    "pixel, under a choice of metric.",
    formula_latex=r"D(p) = \min_{q \in A^c} d(p, q),\quad "
    r"d \in \{L_1, L_2, L_\infty\}",
    when_to_use="The three metrics are the syllabus' distance measures made "
    "visible. Also the standard way to generate watershed markers, since local "
    "maxima of the distance transform sit at object centres.",
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 2.5.3 (distance measures), Ch. 9.5.3"],
)
def distance_transform(img, metric="l2", threshold=0.5, normalize=True,
                       invert=False, colormap="turbo", ctx=None):
    g = luma(img) if img.ndim == 3 else img
    b = (g > threshold).astype(np.uint8)
    if invert:
        b = 1 - b
    dt_type = {"l2": cv2.DIST_L2, "l1": cv2.DIST_L1,
               "chessboard": cv2.DIST_C}[metric]
    # For L2 use the exact transform. OpenCV's default 3x3/5x5 masks are
    # chamfer APPROXIMATIONS of Euclidean distance, off by a few percent, and
    # this operator advertises Euclidean. L1 and chessboard are exact on a
    # 3x3 mask by construction, since both are separable integer metrics.
    mask = cv2.DIST_MASK_PRECISE if metric == "l2" else 3
    d = cv2.distanceTransform(b, dt_type, mask)
    out = normalize01(d) if normalize else np.clip(d / max(float(d.max()), EPS), 0, 1)
    return OpResult(image=out.astype(np.float32),
                    data={"max_distance": round(float(d.max()), 3),
                          "metric": metric},
                    overlays={"distance": colorize(out, colormap)})


@register(
    "morph.granulometry",
    "Granulometry (pattern spectrum)",
    "morph",
    params=[
        EnumParam("shape", options=SHAPE_OPTS, default="ellipse"),
        IntParam("max_size", min=3, max=99, default=31, scale_aware=True),
        IntParam("step", min=2, max=10, default=2),
    ],
    summary="Opens the image with structuring elements of increasing size and "
    "records how much intensity each one removes, giving the size distribution "
    "of the image's structures.",
    formula_latex=r"PS(n) = \frac{\mathrm{area}(f \circ nB) - "
    r"\mathrm{area}(f \circ (n{+}1)B)}{\mathrm{area}(f)}",
    when_to_use="Genuine texture ANALYSIS rather than filtering: the spectrum's "
    "peaks tell you the characteristic grain sizes present, which is how you "
    "measure particles without segmenting them.",
    tags=["beyond-syllabus"],
    cost="medium",
    is_view=True,
    references=["Matheron, Random Sets and Integral Geometry, 1975",
                "Gonzalez & Woods, Ch. 9.6.3"],
)
def granulometry(img, shape="ellipse", max_size=31, step=2, ctx=None):
    g = (luma(img) if img.ndim == 3 else img).astype(np.float32)
    total = float(g.sum()) + EPS
    sizes = list(range(1, int(max_size) + 1, int(step)))
    areas = []
    for s in sizes:
        se = structuring_element(shape, s if s % 2 else s + 1)
        areas.append(float(cv2.morphologyEx(g, cv2.MORPH_OPEN, se).sum()))
    spectrum = [
        round((areas[i] - areas[i + 1]) / total, 6) for i in range(len(areas) - 1)
    ]
    peak = int(np.argmax(spectrum)) if spectrum else 0
    return OpResult(
        image=None,
        data={
            "sizes": sizes[:-1],
            "pattern_spectrum": spectrum,
            "surface_area": [round(a / total, 6) for a in areas],
            "dominant_size": sizes[peak] if spectrum else None,
        },
    )


@register(
    "morph.bg_divide",
    "Background estimate + divide",
    "morph",
    params=[
        EnumParam("method", options=enum_opts(("open", "Morphological opening"),
                                              ("close", "Morphological closing"),
                                              ("median", "Large median"),
                                              ("gaussian", "Large Gaussian")),
                  default="open"),
        OddIntParam("se_size", min=5, max=301, default=61, scale_aware=True),
        EnumParam("combine", options=enum_opts(("divide", "Divide"),
                                               ("subtract", "Subtract")),
                  default="divide"),
        BoolParam("renormalize", default=True),
        BoolParam("show_background", default=False),
    ],
    summary="Estimates the slowly varying background with a large morphological "
    "operator, then divides it out.",
    formula_latex=r"g = \frac{f}{\hat{b}} \cdot \overline{\hat{b}},\quad "
    r"\hat{b} = f \circ B_{large}",
    when_to_use="Flattening uneven illumination, especially on documents. "
    "Division is the physically correct combination because illumination is "
    "multiplicative -- subtraction leaves a residual tint.",
    tags=["beyond-syllabus"],
    cost="medium",
)
def bg_divide(img, method="open", se_size=61, combine="divide", renormalize=True,
              show_background=False, ctx=None):
    g = (luma(img) if img.ndim == 3 else img).astype(np.float32)
    k = int(se_size) | 1
    if method == "median":
        bg = cv2.medianBlur((np.clip(g, 0, 1) * 255).astype(np.uint8),
                            min(k, 255) | 1).astype(np.float32) / 255.0
    elif method == "gaussian":
        bg = cv2.GaussianBlur(g, (0, 0), k / 4.0)
    else:
        se = structuring_element("ellipse", k)
        op = cv2.MORPH_OPEN if method == "open" else cv2.MORPH_CLOSE
        bg = cv2.morphologyEx(g, op, se)

    if combine == "subtract":
        out = g - bg + float(np.mean(bg))
    else:
        out = g / np.maximum(bg, 1e-3)
        if renormalize:
            out = out * float(np.mean(bg))
    out = clamp01(out) if not renormalize else normalize01(out)

    if img.ndim == 3:
        gain = (out + EPS) / (g + EPS)
        res_img = clamp01(img * gain[:, :, None])
    else:
        res_img = clamp01(out)

    res = OpResult(image=res_img,
                   data={"background_mean": round(float(bg.mean()), 4),
                         "background_std": round(float(bg.std()), 4)})
    if show_background:
        res.overlays["background"] = colorize(normalize01(bg), "viridis")
    return res


@register(
    "morph.grayscale",
    "Grayscale morphology",
    "morph",
    params=[
        EnumParam("op", options=enum_opts(("erode", "Erosion (min)"),
                                          ("dilate", "Dilation (max)"),
                                          ("open", "Opening"), ("close", "Closing")),
                  default="open"),
        EnumParam("shape", options=SHAPE_OPTS, default="ellipse"),
        OddIntParam("size", min=3, max=99, default=5, scale_aware=True),
        BoolParam("flat", default=True,
                  help="Flat SE uses min/max; non-flat adds a height offset."),
        FloatParam("height", min=0.0, max=0.5, default=0.1, step=0.01,
                   depends_on={"param": "flat", "equals": False}),
    ],
    summary="Morphology on continuous-tone images, where erosion is a local "
    "minimum and dilation a local maximum.",
    formula_latex=r"(f \ominus b)(x) = \min_{s \in B}\big(f(x+s) - b(s)\big)",
    when_to_use="Showing that morphology is not restricted to binary images; "
    "non-flat structuring elements act like a shaped probe.",
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 9.6"],
)
def grayscale_morph(img, op="open", shape="ellipse", size=5, flat=True,
                    height=0.1, ctx=None):
    se = structuring_element(shape, size)
    if flat:
        ops = {"erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE,
               "open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE}
        return clamp01(cv2.morphologyEx(img, ops[op], se))
    # Non-flat: subtract a height profile over the SE footprint.
    r = se.shape[0] // 2
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1].astype(np.float32)
    prof = height * (1.0 - np.sqrt(xx**2 + yy**2) / max(r, 1))
    prof = np.where(se > 0, prof, -np.inf)

    def apply(ch):
        pad = np.pad(ch, r, mode="reflect")
        wins = np.lib.stride_tricks.sliding_window_view(pad, se.shape)
        if op in ("erode", "open"):
            res = np.min(wins - prof[None, None], axis=(2, 3))
        else:
            res = np.max(wins + prof[None, None], axis=(2, 3))
        return res

    if img.ndim == 2:
        return clamp01(apply(img))
    return clamp01(np.dstack([apply(img[:, :, i]) for i in range(img.shape[2])]))
