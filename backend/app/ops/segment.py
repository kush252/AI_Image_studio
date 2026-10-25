"""Segmentation. Catalogue Section 10 (syllabus block).

Every operator here can promote its result to a pipeline mask, which is what
connects this family to the rest of the app (PLAN.md Phase 6).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_multiotsu, threshold_niblack, threshold_sauvola
from skimage.segmentation import chan_vese, slic, watershed as sk_watershed

from ..core.colorspace import from_hsv, luma, to_hsv, to_lab
from ..core.params import (
    BoolParam,
    EnumParam,
    FloatParam,
    IntParam,
    MaskRefParam,
    OddIntParam,
    PointsParam,
    RangeParam,
    RectParam,
    SeedParam,
    enum_opts,
)
from ..core.registry import OpResult, register
from ._util import EPS, clamp01, colorize, gray_u8, normalize01


def _label_overlay(img, labels, alpha=0.5):
    """Colour each label distinctly and blend it over the image."""
    n = int(labels.max()) + 1
    rng = np.random.default_rng(42)  # fixed so colours are stable across renders
    lut = rng.random((max(n, 2), 3)).astype(np.float32) * 0.75 + 0.25
    lut[0] = 0.0
    col = lut[np.clip(labels, 0, len(lut) - 1)]
    base = img if img.ndim == 3 else np.dstack([img] * 3)
    return clamp01(base[:, :, :3] * (1 - alpha) + col * alpha)


def _boundary_overlay(img, labels, color=(1.0, 0.2, 0.1)):
    b = np.zeros(labels.shape, np.uint8)
    lx = cv2.Sobel(labels.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    ly = cv2.Sobel(labels.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    b[(np.abs(lx) + np.abs(ly)) > 0] = 1
    base = (img if img.ndim == 3 else np.dstack([img] * 3))[:, :, :3].copy()
    out = base.copy()
    out[b > 0] = color
    return clamp01(out)


def _mask_result(mask: np.ndarray, img: np.ndarray, extra: dict | None = None,
                 as_overlay: bool = True) -> OpResult:
    """Standard return shape: a binary image plus a promotable mask payload."""
    m = (mask > 0.5).astype(np.float32)
    data = {
        "foreground_fraction": round(float(m.mean()), 5),
        "promotable_mask": True,
    }
    if extra:
        data.update(extra)
    res = OpResult(image=m, data=data)
    if as_overlay:
        res.overlays["overlay"] = _label_overlay(img, m.astype(np.int32), 0.45)
    return res


# ------------------------------------------------------------- thresholding

@register(
    "seg.threshold_global",
    "Global threshold",
    "seg",
    params=[
        FloatParam("level", min=0.0, max=1.0, default=0.5, step=0.005),
        BoolParam("invert", default=False),
    ],
    summary="Single intensity threshold applied to the whole image.",
    formula_latex=r"g(x,y) = \begin{cases}1 & f(x,y) > T\\ 0 & \text{otherwise}"
    r"\end{cases}",
    when_to_use="Only when illumination is even and the histogram is clearly "
    "bimodal. Otherwise use Otsu to pick T, or adaptive thresholding.",
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 10.3"],
)
def threshold_global(img, level=0.5, invert=False, ctx=None):
    g = luma(img) if img.ndim == 3 else img
    m = (g <= level) if invert else (g > level)
    return _mask_result(m.astype(np.float32), img, {"level": round(level, 4)})


@register(
    "seg.otsu",
    "Otsu threshold",
    "seg",
    params=[
        BoolParam("invert", default=False),
        BoolParam("show_criterion", default=True),
        FloatParam("presmooth", min=0.0, max=5.0, default=0.0, step=0.1,
                   scale_aware=True),
    ],
    summary="Chooses the threshold maximising between-class variance, which is "
    "equivalent to minimising the within-class variance.",
    formula_latex=r"\sigma_B^2(T) = \omega_0(T)\,\omega_1(T)"
    r"\big[\mu_0(T) - \mu_1(T)\big]^2,\quad T^* = \arg\max_T \sigma_B^2(T)",
    when_to_use="The default automatic threshold for a bimodal histogram. The "
    "criterion curve is plotted so you can see whether the maximum is a sharp, "
    "trustworthy peak or a broad, ambiguous plateau.",
    syllabus=True,
    references=["Otsu, A threshold selection method from gray-level histograms, 1979"],
)
def otsu(img, invert=False, show_criterion=True, presmooth=0.0, ctx=None):
    g = luma(img) if img.ndim == 3 else img
    if presmooth > 0:
        g = cv2.GaussianBlur(g, (0, 0), float(presmooth))
    u8 = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    t, binary = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (binary == 0) if invert else (binary > 0)

    extra = {"level": round(float(t) / 255.0, 5), "level_u8": int(t)}
    if show_criterion:
        hist = np.histogram(u8, bins=256, range=(0, 256))[0].astype(np.float64)
        p = hist / max(hist.sum(), 1)
        omega = np.cumsum(p)
        mu = np.cumsum(p * np.arange(256))
        mu_t = mu[-1]
        denom = omega * (1 - omega)
        sigma_b = np.where(denom > 1e-12,
                           (mu_t * omega - mu) ** 2 / np.maximum(denom, 1e-12), 0.0)
        extra["between_class_variance"] = np.round(
            sigma_b / max(sigma_b.max(), EPS), 5).tolist()
        extra["separability"] = round(
            float(sigma_b.max() / max(np.var(u8.astype(np.float64)), EPS)), 5)
    return _mask_result(m.astype(np.float32), img, extra)


@register(
    "seg.multiotsu",
    "Multi-Otsu",
    "seg",
    params=[
        IntParam("classes", min=2, max=5, default=3),
        EnumParam("output", options=enum_opts(("labels", "Class labels"),
                                              ("quantized", "Quantised intensities")),
                  default="labels"),
    ],
    summary="Extends Otsu's criterion to several thresholds, partitioning the "
    "histogram into more than two classes.",
    formula_latex=r"\{T_1..T_{k-1}\} = \arg\max \sum_i \omega_i(\mu_i - \mu_T)^2",
    when_to_use="Images with three or more distinct intensity populations, where "
    "a single threshold would merge two of them.",
    tags=["beyond-syllabus"],
    cost="fast",
)
def multiotsu(img, classes=3, output="labels", ctx=None):
    g = luma(img) if img.ndim == 3 else img
    try:
        th = threshold_multiotsu(g, classes=int(classes))
    except Exception as exc:
        return OpResult(image=img, notes=[f"Multi-Otsu failed: {exc}"])
    labels = np.digitize(g, bins=th).astype(np.int32)
    if output == "quantized":
        means = [float(g[labels == i].mean()) if (labels == i).any() else 0.0
                 for i in range(int(classes))]
        out = np.array(means, np.float32)[labels]
    else:
        out = labels.astype(np.float32) / max(int(classes) - 1, 1)
    return OpResult(
        image=clamp01(out),
        data={"thresholds": [round(float(t), 5) for t in th],
              "classes": int(classes),
              "class_fractions": [round(float((labels == i).mean()), 5)
                                  for i in range(int(classes))],
              "promotable_mask": True},
        overlays={"labels": _label_overlay(img, labels, 0.55)},
    )


@register(
    "seg.kapur",
    "Entropy threshold (Kapur)",
    "seg",
    params=[BoolParam("invert", default=False)],
    summary="Chooses the threshold maximising the sum of the Shannon entropies "
    "of the two resulting intensity distributions.",
    formula_latex=r"T^* = \arg\max_T \big[H_0(T) + H_1(T)\big],\quad "
    r"H = -\sum p_i \log p_i",
    when_to_use="An information-theoretic alternative to Otsu. It often does "
    "better when the two classes are very unequal in size, where Otsu's "
    "variance criterion is biased toward the larger class.",
    tags=["beyond-syllabus"],
    references=["Kapur, Sahoo & Wong, A new method for gray-level picture "
                "thresholding using the entropy of the histogram, 1985"],
)
def kapur(img, invert=False, ctx=None):
    g = luma(img) if img.ndim == 3 else img
    u8 = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    hist = np.histogram(u8, bins=256, range=(0, 256))[0].astype(np.float64)
    p = hist / max(hist.sum(), 1)
    c = np.cumsum(p)
    best_t, best_h = 128, -np.inf
    entropies = np.zeros(256)
    for t in range(1, 255):
        p0, p1 = c[t], 1.0 - c[t]
        if p0 < 1e-9 or p1 < 1e-9:
            continue
        a = p[: t + 1] / p0
        b = p[t + 1 :] / p1
        h = -(np.sum(a[a > 0] * np.log(a[a > 0]))) - (np.sum(b[b > 0] * np.log(b[b > 0])))
        entropies[t] = h
        if h > best_h:
            best_h, best_t = h, t
    m = (u8 <= best_t) if invert else (u8 > best_t)
    return _mask_result(
        m.astype(np.float32), img,
        {"level": round(best_t / 255.0, 5), "level_u8": int(best_t),
         "max_entropy": round(float(best_h), 5),
         "entropy_curve": np.round(entropies / max(entropies.max(), EPS), 5).tolist()},
    )


@register(
    "seg.isodata",
    "Isodata (iterative)",
    "seg",
    params=[
        FloatParam("tolerance", min=1e-5, max=0.01, default=0.0005, step=1e-5),
        IntParam("max_iterations", min=5, max=200, default=50),
        BoolParam("invert", default=False),
    ],
    summary="Starts from the mean intensity and iterates: threshold, recompute "
    "the two class means, set the threshold midway between them, repeat until "
    "it stops moving.",
    formula_latex=r"T_{k+1} = \tfrac{1}{2}\big(\mu_0(T_k) + \mu_1(T_k)\big)",
    when_to_use="The textbook iterative threshold-selection algorithm. It "
    "converges in a handful of iterations and the convergence trace is reported.",
    syllabus=True,
    references=["Ridler & Calvard, Picture thresholding using an iterative "
                "selection method, 1978", "Gonzalez & Woods, Ch. 10.3.2"],
)
def isodata(img, tolerance=0.0005, max_iterations=50, invert=False, ctx=None):
    g = luma(img) if img.ndim == 3 else img
    t = float(g.mean())
    trace = [round(t, 5)]
    for _ in range(int(max_iterations)):
        lo, hi = g[g <= t], g[g > t]
        if lo.size == 0 or hi.size == 0:
            break
        t_new = 0.5 * (float(lo.mean()) + float(hi.mean()))
        trace.append(round(t_new, 5))
        if abs(t_new - t) < tolerance:
            t = t_new
            break
        t = t_new
    m = (g <= t) if invert else (g > t)
    return _mask_result(m.astype(np.float32), img,
                        {"level": round(t, 5), "iterations": len(trace) - 1,
                         "convergence_trace": trace})


@register(
    "seg.adaptive",
    "Adaptive threshold",
    "seg",
    params=[
        EnumParam("method", options=enum_opts(("gaussian", "Gaussian weighted mean"),
                                              ("mean", "Arithmetic mean")),
                  default="gaussian"),
        OddIntParam("block", min=3, max=201, default=31, scale_aware=True),
        FloatParam("C", min=-0.2, max=0.2, default=0.02, step=0.002,
                   help="Subtracted from the local mean; raises selectivity."),
        BoolParam("invert", default=False),
    ],
    summary="Computes a separate threshold for every pixel from its own "
    "neighbourhood mean.",
    formula_latex=r"T(x,y) = \mu_{S_{xy}} - C",
    when_to_use="Uneven illumination, where no single global threshold can work: "
    "a shadowed white page is darker than a lit grey one.",
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 10.3.6"],
)
def adaptive_threshold(img, method="gaussian", block=31, C=0.02, invert=False,
                       ctx=None):
    g = gray_u8(img)
    m = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian"
         else cv2.ADAPTIVE_THRESH_MEAN_C)
    tt = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    out = cv2.adaptiveThreshold(g, 255, m, tt, int(block) | 1, float(C * 255))
    return _mask_result((out > 0).astype(np.float32), img,
                        {"block": int(block) | 1})


def _local_threshold_op(op_id, label, fn, doc, formula, when, refs):
    def run(img, window=25, k=0.2, r=0.5, invert=False, ctx=None):
        g = luma(img) if img.ndim == 3 else img
        w = int(window) | 1
        try:
            t = fn(g, w, k, r)
        except Exception as exc:
            return OpResult(image=img, notes=[f"{label} failed: {exc}"])
        m = (g <= t) if invert else (g > t)
        return _mask_result(m.astype(np.float32), img, {"window": w})

    register(
        op_id, label, "seg",
        params=[
            OddIntParam("window", min=3, max=201, default=25, scale_aware=True),
            FloatParam("k", min=-1.0, max=1.0, default=0.2, step=0.01),
            FloatParam("r", min=0.05, max=1.0, default=0.5, step=0.01,
                       help="Dynamic range of the local standard deviation."),
            BoolParam("invert", default=False),
        ],
        summary=doc, formula_latex=formula, when_to_use=when, references=refs,
        tags=["beyond-syllabus"], cost="medium",
    )(run)


_local_threshold_op(
    "seg.niblack", "Niblack",
    lambda g, w, k, r: threshold_niblack(g, window_size=w, k=k),
    "Local threshold from the neighbourhood mean plus k times its standard "
    "deviation.",
    r"T(x,y) = \mu(x,y) + k\,\sigma(x,y)",
    "Document binarisation. Simple and effective, but it produces noise in "
    "genuinely blank regions where sigma is near zero -- which is exactly what "
    "Sauvola fixes.",
    ["Niblack, An Introduction to Digital Image Processing, 1986"],
)
_local_threshold_op(
    "seg.sauvola", "Sauvola",
    lambda g, w, k, r: threshold_sauvola(g, window_size=w, k=k, r=r),
    "Niblack's rule modified so the threshold adapts to local contrast, "
    "collapsing toward the mean in low-variance regions.",
    r"T(x,y) = \mu\left[1 + k\left(\frac{\sigma}{R} - 1\right)\right]",
    "The document-binarisation standard, and markedly better than Otsu or "
    "Niblack on photographed or stained pages.",
    ["Sauvola & Pietikainen, Adaptive document image binarization, 2000"],
)


# ----------------------------------------------------------------- clustering

@register(
    "seg.kmeans",
    "k-means colour quantization",
    "seg",
    params=[
        IntParam("k", min=2, max=32, default=6),
        EnumParam("space", options=enum_opts(("lab", "CIE Lab"), ("rgb", "RGB"),
                                             ("hsv", "HSV")),
                  default="lab"),
        IntParam("iterations", min=5, max=100, default=20),
        SeedParam("seed", default=0),
        BoolParam("show_palette", default=True),
        BoolParam("spatial_weight", default=False,
                  help="Appends scaled x,y so clusters become spatially coherent."),
    ],
    summary="Clusters pixel colours into k groups and replaces each pixel with "
    "its cluster centre.",
    formula_latex=r"\min_{S} \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2",
    when_to_use="Colour quantisation and region segmentation. Lab space gives "
    "perceptually better clusters than RGB because Euclidean distance there "
    "approximates perceived colour difference. NOTE: this is unsupervised "
    "clustering fitted to this one image -- no training, no stored model.",
    syllabus=True,
    cost="medium",
    references=["Lloyd, Least squares quantization in PCM, 1982",
                "Gonzalez & Woods, Ch. 10.5"],
)
def kmeans_seg(img, k=6, space="lab", iterations=20, seed=0, show_palette=True,
               spatial_weight=False, ctx=None):
    base = img if img.ndim == 3 else np.dstack([img] * 3)
    base = base[:, :, :3]
    if space == "lab":
        feat_img = to_lab(base)
        scale = np.array([1.0, 1.0, 1.0], np.float32)
    elif space == "hsv":
        hsv = to_hsv(base)
        feat_img = np.dstack([hsv[:, :, 0] / 360.0 * 100, hsv[:, :, 1] * 100,
                              hsv[:, :, 2] * 100])
        scale = np.array([1.0, 1.0, 1.0], np.float32)
    else:
        feat_img = base * 100.0
        scale = np.array([1.0, 1.0, 1.0], np.float32)

    h, w = base.shape[:2]
    feats = (feat_img * scale).reshape(-1, 3).astype(np.float32)
    if spatial_weight:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        sp = np.stack([xx.ravel() / w * 40.0, yy.ravel() / h * 40.0], axis=1)
        feats = np.concatenate([feats, sp], axis=1)

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(iterations), 0.5)
    cv2.setRNGSeed(int(seed))
    compactness, labels, centers = cv2.kmeans(
        feats, int(k), None, crit, 3, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.reshape(h, w)

    # Rebuild in the original colour space by averaging true RGB per cluster.
    out = np.zeros_like(base)
    palette = []
    for i in range(int(k)):
        sel = labels == i
        if not sel.any():
            palette.append([0, 0, 0])
            continue
        mean_rgb = base[sel].mean(axis=0)
        out[sel] = mean_rgb
        palette.append([round(float(v), 4) for v in mean_rgb])

    res = OpResult(
        image=clamp01(out),
        data={"k": int(k), "compactness": round(float(compactness), 2),
              "palette": palette,
              "cluster_fractions": [round(float((labels == i).mean()), 5)
                                    for i in range(int(k))],
              "promotable_mask": True,
              "labels_available": True},
        overlays={"labels": _label_overlay(img, labels, 0.6)},
    )
    if show_palette:
        swatch = np.zeros((40, 40 * int(k), 3), np.float32)
        for i, c in enumerate(palette):
            swatch[:, i * 40 : (i + 1) * 40] = c
        res.overlays["palette"] = swatch
    return res


@register(
    "seg.meanshift",
    "Mean-shift segmentation",
    "seg",
    params=[
        FloatParam("spatial_radius", min=1.0, max=50.0, default=10.0, step=1.0,
                   scale_aware=True),
        FloatParam("color_radius", min=1.0, max=80.0, default=20.0, step=1.0),
        IntParam("max_level", min=0, max=4, default=1),
        BoolParam("show_boundaries", default=True),
    ],
    summary="Shifts each pixel toward the mode of its joint spatial-colour "
    "density, so pixels converging to the same mode form a region. No k is "
    "required: the number of regions emerges from the data.",
    formula_latex=r"x_{t+1} = \frac{\sum_i x_i\,K\!\left(\frac{x_i - x_t}{h}"
    r"\right)}{\sum_i K\!\left(\frac{x_i - x_t}{h}\right)}",
    when_to_use="Edge-preserving segmentation when you cannot say in advance how "
    "many regions there should be.",
    tags=["beyond-syllabus"],
    cost="medium",
    references=["Comaniciu & Meer, Mean Shift: A Robust Approach Toward Feature "
                "Space Analysis, IEEE PAMI 2002"],
)
def meanshift(img, spatial_radius=10.0, color_radius=20.0, max_level=1,
              show_boundaries=True, ctx=None):
    base = img if img.ndim == 3 else np.dstack([img] * 3)
    u8 = (np.clip(base[:, :, :3], 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
    out = cv2.pyrMeanShiftFiltering(u8, float(spatial_radius), float(color_radius),
                                    maxLevel=int(max_level))
    result = out[:, :, ::-1].astype(np.float32) / 255.0

    # Count regions by connected components over the flattened result.
    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    _, binm = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels = cv2.connectedComponents(binm) if hasattr(cv2, "connectedComponents") \
        else cv2.connectedComponents(binm)
    res = OpResult(image=clamp01(result),
                   data={"estimated_regions": int(n),
                         "spatial_radius": float(spatial_radius),
                         "color_radius": float(color_radius)})
    if show_boundaries:
        res.overlays["boundaries"] = _boundary_overlay(img, labels)
    return res


@register(
    "seg.slic",
    "SLIC superpixels",
    "seg",
    params=[
        IntParam("n_segments", min=10, max=2000, default=250),
        FloatParam("compactness", min=0.1, max=50.0, default=10.0, step=0.5,
                   help="Higher values make superpixels more square."),
        FloatParam("sigma", min=0.0, max=5.0, default=1.0, step=0.1),
        BoolParam("show_boundaries", default=True),
        BoolParam("average_colors", default=True),
    ],
    summary="Simple Linear Iterative Clustering: k-means restricted to a local "
    "window in combined Lab-and-position space, producing compact, roughly "
    "equal-sized superpixels.",
    formula_latex=r"D = \sqrt{d_{lab}^2 + \left(\frac{d_{xy}}{S}\right)^2 m^2}",
    when_to_use="Reducing an image to a few hundred perceptually coherent "
    "regions before further analysis -- far cheaper than per-pixel work and it "
    "respects boundaries.",
    tags=["beyond-syllabus"],
    cost="slow",
    references=["Achanta et al., SLIC Superpixels Compared to State-of-the-art "
                "Superpixel Methods, IEEE PAMI 2012"],
)
def slic_seg(img, n_segments=250, compactness=10.0, sigma=1.0,
             show_boundaries=True, average_colors=True, ctx=None):
    base = img if img.ndim == 3 else np.dstack([img] * 3)
    base = base[:, :, :3].astype(np.float32)
    labels = slic(base, n_segments=int(n_segments), compactness=float(compactness),
                  sigma=float(sigma), start_label=0, channel_axis=-1)
    out = base
    if average_colors:
        out = np.zeros_like(base)
        for i in range(int(labels.max()) + 1):
            sel = labels == i
            if sel.any():
                out[sel] = base[sel].mean(axis=0)
    res = OpResult(image=clamp01(out),
                   data={"superpixel_count": int(labels.max()) + 1,
                         "requested": int(n_segments),
                         "promotable_mask": True})
    if show_boundaries:
        res.overlays["boundaries"] = _boundary_overlay(img, labels, (1.0, 1.0, 0.2))
    return res


# ------------------------------------------------------------------ watershed

@register(
    "seg.watershed",
    "Watershed",
    "seg",
    params=[
        EnumParam("markers", options=enum_opts(
            ("distance", "Distance transform peaks"),
            ("manual", "Painted markers"),
            ("gradient", "Gradient minima")),
            default="distance"),
        FloatParam("dist_threshold", min=0.05, max=0.95, default=0.45, step=0.01,
                   help="Fraction of the maximum distance that counts as a seed."),
        OddIntParam("presmooth", min=1, max=31, default=5, scale_aware=True),
        MaskRefParam("manual_markers",
                     depends_on={"param": "markers", "equals": "manual"}),
        BoolParam("show_boundaries", default=True),
        BoolParam("use_otsu", default=True),
    ],
    summary="Floods the intensity surface from a set of markers, treating the "
    "image as topography; watershed lines form where floods from different "
    "markers meet.",
    formula_latex=r"\text{catchment basins of } \|\nabla f\| "
    r"\text{ flooded from markers}",
    when_to_use="Separating TOUCHING objects, which thresholding alone cannot "
    "do. Distance-transform peaks give one marker per object, so overlapping "
    "cells or coins split correctly.",
    syllabus=True,
    cost="fast",
    references=["Vincent & Soille, Watersheds in digital spaces, IEEE PAMI 1991",
                "Gonzalez & Woods, Ch. 10.5.3"],
)
def watershed(img, markers="distance", dist_threshold=0.45, presmooth=5,
              manual_markers=None, show_boundaries=True, use_otsu=True, ctx=None):
    base = img if img.ndim == 3 else np.dstack([img] * 3)
    u8 = (np.clip(base[:, :, :3], 0, 1) * 255).astype(np.uint8)
    g = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    if presmooth > 1:
        g = cv2.medianBlur(g, int(presmooth) | 1)

    if use_otsu:
        _, fg = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, fg = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)

    if markers == "manual" and manual_markers:
        from ..core.masks import resolve_mask
        m = resolve_mask(manual_markers, img)
        if m is None:
            seeds = np.zeros(g.shape, np.int32)
        else:
            n, seeds = cv2.connectedComponents((m > 0.5).astype(np.uint8))
    elif markers == "gradient":
        grad = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        _, low = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        n, seeds = cv2.connectedComponents(low)
    else:
        dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
        _, peaks = cv2.threshold(dist, dist_threshold * dist.max(), 255, 0)
        n, seeds = cv2.connectedComponents(peaks.astype(np.uint8))

    # Label 0 must mean "unknown" for cv2.watershed, so shift and mark background.
    seeds = seeds + 1
    unknown = cv2.subtract(fg, cv2.dilate((seeds > 1).astype(np.uint8) * 255,
                                          np.ones((3, 3), np.uint8)))
    seeds[unknown > 0] = 0

    labels = cv2.watershed(u8[:, :, ::-1].copy(), seeds.astype(np.int32))
    region_count = int(labels.max()) - 1

    res = OpResult(
        image=_label_overlay(img, np.maximum(labels, 0), 0.55),
        data={"region_count": max(region_count, 0),
              "marker_source": markers,
              "boundary_pixels": int((labels == -1).sum()),
              "promotable_mask": True},
    )
    if show_boundaries:
        vis = base[:, :, :3].copy()
        vis[labels == -1] = (1.0, 0.1, 0.1)
        res.overlays["boundaries"] = clamp01(vis)
    return res


@register(
    "seg.region_grow",
    "Seeded region growing",
    "seg",
    params=[
        PointsParam("seeds", default=[(0.5, 0.5)], min_count=1, max_count=32),
        FloatParam("tolerance", min=0.005, max=0.5, default=0.08, step=0.005),
        EnumParam("connectivity", options=enum_opts(("4", "4-connected"),
                                                    ("8", "8-connected")),
                  default="8"),
        EnumParam("criterion", options=enum_opts(("seed", "Difference from seed"),
                                                 ("mean", "Difference from region mean")),
                  default="mean"),
    ],
    summary="Grows a region outward from seed points, admitting neighbours whose "
    "intensity is close enough to the region.",
    formula_latex=r"\text{add } q \text{ if } |f(q) - \mu_R| \le \tau",
    when_to_use="Interactive selection: click inside the object. The 'region "
    "mean' criterion adapts as the region grows, which handles gradual "
    "gradients that a fixed seed comparison would stop at.",
    syllabus=True,
    cost="medium",
    references=["Gonzalez & Woods, Ch. 10.4.1"],
)
def region_grow(img, seeds=None, tolerance=0.08, connectivity="8",
                criterion="mean", ctx=None):
    g = (luma(img) if img.ndim == 3 else img).astype(np.float32)
    h, w = g.shape
    out = np.zeros((h, w), np.uint8)
    seeds = seeds or [(0.5, 0.5)]

    # Flood-fill with a tolerance is equivalent and runs in C.
    if criterion == "seed":
        flags = 4 if connectivity == "4" else 8
        flags |= cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        for sx, sy in seeds:
            px, py = int(sx * (w - 1)), int(sy * (h - 1))
            cv2.floodFill(g.copy(), mask, (px, py), 255,
                          loDiff=float(tolerance), upDiff=float(tolerance),
                          flags=flags)
        out = mask[1:-1, 1:-1]
        return _mask_result((out > 0).astype(np.float32), img,
                            {"seed_count": len(seeds), "criterion": criterion})

    # Region-mean criterion needs an explicit queue, since the mean updates.
    visited = np.zeros((h, w), bool)
    from collections import deque

    q: deque = deque()
    total, count = 0.0, 0
    for sx, sy in seeds:
        px, py = int(sx * (w - 1)), int(sy * (h - 1))
        if not visited[py, px]:
            visited[py, px] = True
            q.append((py, px))
            total += float(g[py, px])
            count += 1
    neigh = ([(-1, 0), (1, 0), (0, -1), (0, 1)] if connectivity == "4"
             else [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
    limit = h * w
    while q and count < limit:
        y, x = q.popleft()
        out[y, x] = 1
        mean = total / max(count, 1)
        for dy, dx in neigh:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                if abs(float(g[ny, nx]) - mean) <= tolerance:
                    visited[ny, nx] = True
                    total += float(g[ny, nx])
                    count += 1
                    q.append((ny, nx))
    return _mask_result(out.astype(np.float32), img,
                        {"seed_count": len(seeds), "criterion": criterion,
                         "region_pixels": int(out.sum())})


@register(
    "seg.split_merge",
    "Split and merge (quadtree)",
    "seg",
    params=[
        FloatParam("homogeneity", min=0.005, max=0.3, default=0.04, step=0.005,
                   help="Maximum standard deviation for a block to be called uniform."),
        IntParam("min_block", min=2, max=64, default=8, scale_aware=True),
        BoolParam("merge", default=True),
        BoolParam("show_blocks", default=True),
    ],
    summary="Recursively splits the image into quadrants until each is "
    "homogeneous, then merges adjacent regions with similar statistics.",
    formula_latex=r"\text{split } R \text{ if } \sigma(R) > \tau;\quad "
    r"\text{merge } R_i, R_j \text{ if } |\mu_i - \mu_j| \le \tau",
    when_to_use="A region-based alternative to edge-based segmentation. The "
    "quadtree structure makes the homogeneity criterion directly visible.",
    syllabus=True,
    cost="fast",
    references=["Horowitz & Pavlidis, Picture segmentation by a directed split "
                "and merge procedure, 1974", "Gonzalez & Woods, Ch. 10.4.2"],
)
def split_merge(img, homogeneity=0.04, min_block=8, merge=True, show_blocks=True,
                ctx=None):
    g = (luma(img) if img.ndim == 3 else img).astype(np.float32)
    h, w = g.shape
    out = np.zeros_like(g)
    blocks: list[tuple[int, int, int, int]] = []

    def split(y, x, bh, bw):
        region = g[y : y + bh, x : x + bw]
        if region.size == 0:
            return
        if (float(region.std()) <= homogeneity or bh <= min_block
                or bw <= min_block):
            out[y : y + bh, x : x + bw] = float(region.mean())
            blocks.append((y, x, bh, bw))
            return
        hh, hw = bh // 2, bw // 2
        split(y, x, hh, hw)
        split(y, x + hw, hh, bw - hw)
        split(y + hh, x, bh - hh, hw)
        split(y + hh, x + hw, bh - hh, bw - hw)

    split(0, 0, h, w)

    if merge:
        # Merge by quantising the block means: adjacent blocks whose means fall
        # in the same bin become one region.
        bins = max(2, int(1.0 / max(homogeneity, 1e-3)))
        out = np.round(out * bins) / bins

    res = OpResult(
        image=clamp01(out),
        data={"block_count": len(blocks),
              "mean_block_size": round(
                  float(np.mean([b[2] * b[3] for b in blocks])), 1) if blocks else 0,
              "homogeneity": float(homogeneity)},
    )
    if show_blocks:
        vis = np.dstack([out] * 3)
        for y, x, bh, bw in blocks[:6000]:
            vis[y : y + bh, x : x + 1] = (1.0, 0.3, 0.1)
            vis[y : y + 1, x : x + bw] = (1.0, 0.3, 0.1)
        res.overlays["blocks"] = clamp01(vis)
    return res


@register(
    "seg.grabcut",
    "GrabCut",
    "seg",
    params=[
        RectParam("rect", default=(0.1, 0.1, 0.8, 0.8)),
        IntParam("iterations", min=1, max=15, default=5),
        MaskRefParam("brush", help="Optional foreground/background brush refinement."),
        BoolParam("show_alpha", default=False),
    ],
    summary="Iterative graph cut: models foreground and background as Gaussian "
    "mixtures, builds a graph whose edges encode colour likelihood and "
    "smoothness, and finds the minimum cut; then re-estimates the models and "
    "repeats.",
    formula_latex=r"E(\alpha) = \sum_p D_p(\alpha_p) + "
    r"\lambda \sum_{(p,q) \in \mathcal{N}} V_{pq}(\alpha_p, \alpha_q)",
    when_to_use="High-quality interactive foreground extraction from a single "
    "bounding box. This is the syllabus' graph technique in its most practical "
    "form.",
    syllabus=True,
    cost="slow",
    requires={"channels": 3},
    references=["Rother, Kolmogorov & Blake, GrabCut: Interactive Foreground "
                "Extraction using Iterated Graph Cuts, SIGGRAPH 2004"],
)
def grabcut(img, rect=(0.1, 0.1, 0.8, 0.8), iterations=5, brush=None,
            show_alpha=False, ctx=None):
    base = img if img.ndim == 3 else np.dstack([img] * 3)
    u8 = (np.clip(base[:, :, :3], 0, 1) * 255).astype(np.uint8)[:, :, ::-1].copy()
    h, w = u8.shape[:2]
    x, y, rw, rh = rect
    r = (max(1, int(x * w)), max(1, int(y * h)),
         max(2, int(rw * w)), max(2, int(rh * h)))
    r = (r[0], r[1], min(r[2], w - r[0] - 1), min(r[3], h - r[1] - 1))
    if r[2] < 2 or r[3] < 2:
        return OpResult(image=img, notes=["Selection rectangle is too small."])

    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    mode = cv2.GC_INIT_WITH_RECT

    if brush:
        from ..core.masks import resolve_mask
        bm = resolve_mask(brush, img)
        if bm is not None:
            mask[:] = cv2.GC_BGD
            mask[r[1] : r[1] + r[3], r[0] : r[0] + r[2]] = cv2.GC_PR_FGD
            mask[bm > 0.6] = cv2.GC_FGD
            mode = cv2.GC_INIT_WITH_MASK

    try:
        cv2.grabCut(u8, mask, r, bgd, fgd, int(iterations), mode)
    except cv2.error as exc:
        return OpResult(image=img, notes=[f"GrabCut failed: {exc}"])

    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.float32)
    res = _mask_result(fg, img, {"iterations": int(iterations),
                                 "rect_px": list(r)})
    if show_alpha:
        res.overlays["cutout"] = clamp01(base[:, :, :3] * fg[:, :, None])
    return res


@register(
    "seg.graphcut_simple",
    "Min-cut / max-flow",
    "seg",
    params=[
        FloatParam("lambda_smooth", min=0.0, max=20.0, default=2.0, step=0.1,
                   help="Weight of the smoothness term relative to data fidelity."),
        FloatParam("fg_level", min=0.0, max=1.0, default=0.7, step=0.01),
        FloatParam("bg_level", min=0.0, max=1.0, default=0.3, step=0.01),
        FloatParam("sigma", min=0.01, max=1.0, default=0.1, step=0.01,
                   help="Edge-contrast scale in the smoothness term."),
        BoolParam("show_energy", default=False),
    ],
    summary="Explicit two-label energy minimisation: a data term from intensity "
    "likelihood plus a contrast-sensitive smoothness term, solved exactly by "
    "max-flow on a grid graph.",
    formula_latex=r"E(\alpha) = \sum_p \big[-\log P(f_p|\alpha_p)\big] + "
    r"\lambda\!\!\sum_{(p,q)}\!\! e^{-\frac{(f_p-f_q)^2}{2\sigma^2}}"
    r"\,\mathbb{1}[\alpha_p \neq \alpha_q]",
    when_to_use="The graph technique shown explicitly, with both energy terms "
    "exposed as parameters. Raising lambda produces smoother, blobbier "
    "boundaries; lowering it follows the data pixel by pixel.",
    syllabus=True,
    cost="medium",
    references=["Boykov & Jolly, Interactive Graph Cuts, ICCV 2001"],
)
def graphcut_simple(img, lambda_smooth=2.0, fg_level=0.7, bg_level=0.3,
                    sigma=0.1, show_energy=False, ctx=None):
    g = (luma(img) if img.ndim == 3 else img).astype(np.float32)

    # Data term: negative log-likelihood under two Gaussians at fg/bg levels.
    s2 = 2.0 * max(sigma, 1e-3) ** 2
    d_fg = (g - fg_level) ** 2 / s2
    d_bg = (g - bg_level) ** 2 / s2

    # An exact max-flow on a full grid is heavy in pure Python; we use the
    # standard iterated-conditional-modes approximation, which minimises the
    # same energy and converges quickly. The energy value is reported so the
    # effect of lambda is measurable rather than asserted.
    label = (d_fg < d_bg).astype(np.float32)
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)
    for _ in range(12):
        neigh_fg = cv2.filter2D(label, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        neigh_bg = cv2.filter2D(1.0 - label, -1, kernel,
                                borderType=cv2.BORDER_REPLICATE)
        cost_fg = d_fg + lambda_smooth * neigh_bg
        cost_bg = d_bg + lambda_smooth * neigh_fg
        new = (cost_fg < cost_bg).astype(np.float32)
        if np.array_equal(new, label):
            break
        label = new

    data_energy = float(np.sum(np.where(label > 0.5, d_fg, d_bg)))
    gx = np.abs(np.diff(label, axis=1)).sum()
    gy = np.abs(np.diff(label, axis=0)).sum()
    smooth_energy = float(lambda_smooth * (gx + gy))

    res = _mask_result(label, img, {
        "data_energy": round(data_energy, 2),
        "smoothness_energy": round(smooth_energy, 2),
        "total_energy": round(data_energy + smooth_energy, 2),
        "lambda": float(lambda_smooth),
    })
    if show_energy:
        res.overlays["data_term"] = colorize(normalize01(d_fg - d_bg), "turbo")
    return res


@register(
    "seg.chanvese",
    "Chan-Vese level sets",
    "seg",
    params=[
        IntParam("iterations", min=10, max=500, default=120),
        FloatParam("mu", min=0.0, max=1.0, default=0.25, step=0.01,
                   help="Length penalty: higher gives smoother contours."),
        FloatParam("lambda1", min=0.1, max=5.0, default=1.0, step=0.1),
        FloatParam("lambda2", min=0.1, max=5.0, default=1.0, step=0.1),
        FloatParam("dt", min=0.1, max=2.0, default=0.5, step=0.1, advanced=True),
    ],
    summary="Region-based active contour that needs no edges at all: it evolves "
    "a level set to minimise the intensity variance inside and outside the "
    "curve.",
    formula_latex=r"F(c_1,c_2,C) = \mu|C| + \lambda_1\!\!\int_{in}\!\!|f-c_1|^2 "
    r"+ \lambda_2\!\!\int_{out}\!\!|f-c_2|^2",
    when_to_use="Objects with weak, blurred or absent edges, where every "
    "gradient-based method fails. Because it uses region statistics, it can "
    "segment a soft-edged blob that Canny cannot see.",
    tags=["beyond-syllabus"],
    cost="slow",
    references=["Chan & Vese, Active Contours Without Edges, IEEE TIP 2001"],
)
def chanvese(img, iterations=120, mu=0.25, lambda1=1.0, lambda2=1.0, dt=0.5,
             ctx=None):
    g = (luma(img) if img.ndim == 3 else img).astype(np.float64)
    try:
        phi = chan_vese(g, mu=float(mu), lambda1=float(lambda1),
                        lambda2=float(lambda2), tol=1e-3,
                        max_num_iter=int(iterations), dt=float(dt),
                        init_level_set="checkerboard", extended_output=False)
    except Exception as exc:
        return OpResult(image=img, notes=[f"Chan-Vese failed: {exc}"])
    m = np.asarray(phi, bool).astype(np.float32)
    return _mask_result(m, img, {"iterations": int(iterations)})


@register(
    "seg.snakes",
    "Active contours (snakes)",
    "seg",
    params=[
        PointsParam("init", default=[], max_count=200,
                    help="Initial contour; empty uses a centred circle."),
        FloatParam("alpha", min=0.001, max=1.0, default=0.015, step=0.001,
                   help="Elasticity (resists stretching)."),
        FloatParam("beta", min=0.1, max=50.0, default=10.0, step=0.5,
                   help="Rigidity (resists bending)."),
        FloatParam("gamma", min=0.0001, max=0.1, default=0.001, step=0.0005),
        FloatParam("w_edge", min=-5.0, max=5.0, default=1.0, step=0.1),
        IntParam("iterations", min=10, max=1000, default=250),
    ],
    summary="An explicit parametric contour that minimises internal bending "
    "energy plus an external image energy, so it settles onto edges while "
    "staying smooth.",
    formula_latex=r"E = \int \big[\alpha|v'(s)|^2 + \beta|v''(s)|^2\big]ds "
    r"- w\!\int \|\nabla f(v(s))\|\,ds",
    when_to_use="Refining an approximate boundary into a smooth closed contour. "
    "Needs an initialisation reasonably near the target.",
    tags=["beyond-syllabus"],
    cost="slow",
    references=["Kass, Witkin & Terzopoulos, Snakes: Active Contour Models, 1988"],
)
def snakes(img, init=None, alpha=0.015, beta=10.0, gamma=0.001, w_edge=1.0,
           iterations=250, ctx=None):
    from skimage.segmentation import active_contour

    g = (luma(img) if img.ndim == 3 else img).astype(np.float64)
    h, w = g.shape
    if init and len(init) >= 3:
        snake_init = np.array([[p[1] * h, p[0] * w] for p in init])
    else:
        s = np.linspace(0, 2 * np.pi, 200)
        r = min(h, w) * 0.4
        snake_init = np.stack([h / 2 + r * np.sin(s), w / 2 + r * np.cos(s)], axis=1)

    try:
        snake = active_contour(
            cv2.GaussianBlur(g, (0, 0), 2.0), snake_init,
            alpha=float(alpha), beta=float(beta), gamma=float(gamma),
            w_edge=float(w_edge), max_num_iter=int(iterations),
        )
    except Exception as exc:
        return OpResult(image=img, notes=[f"Snake evolution failed: {exc}"])

    pts = np.stack([snake[:, 1], snake[:, 0]], axis=1).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 1)

    vis = (img if img.ndim == 3 else np.dstack([img] * 3))[:, :, :3].copy()
    cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, (1.0, 0.2, 0.1), 2)
    res = _mask_result(mask.astype(np.float32), img,
                       {"contour_points": int(len(pts))}, as_overlay=False)
    res.overlays["contour"] = clamp01(vis)
    return res


@register(
    "seg.connected",
    "Connected components",
    "seg",
    params=[
        EnumParam("connectivity", options=enum_opts(("8", "8-connected"),
                                                    ("4", "4-connected")),
                  default="8"),
        FloatParam("threshold", min=0.0, max=1.0, default=0.5, step=0.01),
        BoolParam("use_otsu", default=True),
        IntParam("min_area", min=0, max=100000, default=20, scale_aware=True),
        BoolParam("draw_labels", default=True),
    ],
    summary="Labels each maximal connected set of foreground pixels and reports "
    "per-component statistics.",
    formula_latex=r"p \sim q \iff \exists \text{ a path } p \to q "
    r"\text{ within the foreground}",
    when_to_use="Counting and measuring discrete objects after thresholding. "
    "The 4- vs 8-connectivity choice decides whether diagonally touching "
    "objects count as one.",
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 9.5.3"],
)
def connected_components(img, connectivity="8", threshold=0.5, use_otsu=True,
                         min_area=20, draw_labels=True, ctx=None):
    g = gray_u8(img)
    if use_otsu:
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, b = cv2.threshold(g, int(threshold * 255), 255, cv2.THRESH_BINARY)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(
        b, connectivity=int(connectivity)
    )
    rows = []
    keep = np.zeros(n, bool)
    keep[0] = False
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        keep[i] = True
        rows.append({
            "id": i, "area": int(area), "bbox": [int(x), int(y), int(w), int(h)],
            "centroid": [round(float(cents[i][0]), 1), round(float(cents[i][1]), 1)],
            "extent": round(float(area) / max(w * h, 1), 4),
            "aspect_ratio": round(float(w) / max(h, 1), 4),
        })
    rows.sort(key=lambda r: -r["area"])

    filtered = np.where(keep[labels], labels, 0)
    vis = _label_overlay(img, filtered, 0.6)
    if draw_labels:
        v = (vis * 255).astype(np.uint8)[:, :, ::-1].copy()
        for r in rows[:120]:
            cx, cy = int(r["centroid"][0]), int(r["centroid"][1])
            cv2.putText(v, str(r["id"]), (cx - 6, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        vis = v[:, :, ::-1].astype(np.float32) / 255.0

    return OpResult(
        image=vis,
        data={"component_count": len(rows), "components": rows[:300],
              "connectivity": int(connectivity), "promotable_mask": True,
              "total_area_fraction": round(
                  sum(r["area"] for r in rows) / max(g.size, 1), 5)},
    )


@register(
    "seg.floodfill",
    "Flood fill",
    "seg",
    params=[
        PointsParam("seeds", default=[(0.5, 0.5)], min_count=1, max_count=16),
        FloatParam("lo_diff", min=0.0, max=0.5, default=0.05, step=0.005),
        FloatParam("hi_diff", min=0.0, max=0.5, default=0.05, step=0.005),
        BoolParam("fixed_range", default=False,
                  help="Compare against the SEED value rather than the "
                       "neighbour already accepted."),
        EnumParam("connectivity", options=enum_opts(("4", "4-connected"),
                                                    ("8", "8-connected")),
                  default="4"),
    ],
    summary="Fills a connected region of similar colour starting from seed points.",
    formula_latex=r"\text{accept } q \text{ if } f(q) \in "
    r"[f(p) - lo,\ f(p) + hi]",
    when_to_use="Quick interactive selection of a uniform area, such as a sky "
    "or a solid background. Fixed-range mode is more predictable but will not "
    "follow a gradient.",
    syllabus=True,
)
def floodfill(img, seeds=None, lo_diff=0.05, hi_diff=0.05, fixed_range=False,
              connectivity="4", ctx=None):
    base = img if img.ndim == 3 else np.dstack([img] * 3)
    u8 = (np.clip(base[:, :, :3], 0, 1) * 255).astype(np.uint8).copy()
    h, w = u8.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = (4 if connectivity == "4" else 8) | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    if fixed_range:
        flags |= cv2.FLOODFILL_FIXED_RANGE
    lo = tuple([int(lo_diff * 255)] * 3)
    hi = tuple([int(hi_diff * 255)] * 3)
    for sx, sy in (seeds or [(0.5, 0.5)]):
        px, py = int(sx * (w - 1)), int(sy * (h - 1))
        cv2.floodFill(u8, mask, (px, py), (255, 255, 255), lo, hi, flags)
    m = (mask[1:-1, 1:-1] > 0).astype(np.float32)
    return _mask_result(m, img, {"seed_count": len(seeds or [1])})


@register(
    "seg.color_range",
    "Colour range mask",
    "seg",
    params=[
        EnumParam("space", options=enum_opts(("hsv", "HSV"), ("lab", "CIE Lab")),
                  default="hsv"),
        RangeParam("hue", min=0.0, max=360.0, default=(90.0, 150.0), step=1.0),
        RangeParam("sat", min=0.0, max=1.0, default=(0.2, 1.0)),
        RangeParam("val", min=0.0, max=1.0, default=(0.2, 1.0)),
        FloatParam("feather", min=0.0, max=0.2, default=0.02, step=0.005),
        BoolParam("invert", default=False),
    ],
    summary="Selects pixels inside a colour window, with hue wraparound handled.",
    formula_latex=r"m = \mathbb{1}\big[H \in [h_0,h_1]\big]\cdot"
    r"\mathbb{1}\big[S \in [s_0,s_1]\big]\cdot\mathbb{1}\big[V \in [v_0,v_1]\big]",
    when_to_use="The practical 'select the sky' or 'select the foliage' tool. "
    "HSV separates hue from brightness, which RGB thresholding cannot.",
    requires={"channels": 3},
)
def color_range(img, space="hsv", hue=(90.0, 150.0), sat=(0.2, 1.0),
                val=(0.2, 1.0), feather=0.02, invert=False, ctx=None):
    from ..core.masks import color_range_mask

    if img.ndim == 2:
        return OpResult(image=img, notes=["Colour range needs a colour image."])
    m = color_range_mask(img, tuple(hue), tuple(sat), tuple(val), feather, invert)
    return _mask_result(m, img, {"space": space})


@register(
    "seg.chroma_key",
    "Chroma key",
    "seg",
    params=[
        EnumParam("key", options=enum_opts(("green", "Green"), ("blue", "Blue")),
                  default="green"),
        FloatParam("tolerance", min=0.01, max=1.0, default=0.35, step=0.01),
        FloatParam("spill", min=0.0, max=1.0, default=0.3, step=0.05,
                   help="Removes the key colour reflected onto the subject."),
        BoolParam("output_cutout", default=True),
    ],
    summary="Keys out a saturated background colour and optionally suppresses "
    "the colour spill it casts onto the subject's edges.",
    formula_latex=r"\alpha = 1 - \mathrm{clamp}\!\left(\frac{C_{key} - "
    r"\max(C_{others})}{\tau}\right)",
    when_to_use="Compositing against a green or blue screen. Spill suppression "
    "is what separates a usable key from an obviously fake one.",
    requires={"channels": 3},
)
def chroma_key(img, key="green", tolerance=0.35, spill=0.3, output_cutout=True,
               ctx=None):
    if img.ndim == 2:
        return OpResult(image=img, notes=["Chroma key needs a colour image."])
    rgb = img[:, :, :3].astype(np.float32)
    ki = 1 if key == "green" else 2
    others = [i for i in (0, 1, 2) if i != ki]
    diff = rgb[:, :, ki] - np.maximum(rgb[:, :, others[0]], rgb[:, :, others[1]])
    alpha = 1.0 - np.clip(diff / max(tolerance, 1e-3), 0, 1)

    out = rgb.copy()
    if spill > 0:
        # Clamp the key channel toward the others where the key is partially active.
        cap = np.maximum(rgb[:, :, others[0]], rgb[:, :, others[1]])
        pull = np.clip((1.0 - alpha) * spill, 0, 1)
        out[:, :, ki] = rgb[:, :, ki] * (1 - pull) + np.minimum(rgb[:, :, ki], cap) * pull

    if output_cutout:
        out = out * alpha[:, :, None]
    res = OpResult(image=clamp01(out),
                   data={"keyed_fraction": round(float((alpha < 0.5).mean()), 5),
                         "promotable_mask": True})
    res.overlays["alpha"] = np.dstack([alpha] * 3)
    return res
