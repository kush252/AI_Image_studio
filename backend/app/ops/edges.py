"""Edge detection. Catalogue Section 9 (edges portion)."""
from __future__ import annotations

import cv2
import numpy as np

from ..core.colorspace import from_hsv, luma
from ..core.params import BoolParam, EnumParam, FloatParam, OddIntParam, enum_opts
from ..core.registry import OpResult, register
from ._util import EPS, clamp01, colorize, gray_u8, normalize01


def _auto_canny(img, **_) -> dict:
    """Two standard automatic threshold rules, so the UI can offer both."""
    g = gray_u8(img)
    otsu, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    med = float(np.median(g))
    return {
        "t1": round(float(otsu) * 0.5 / 255.0, 4),
        "t2": round(float(otsu) / 255.0, 4),
        "_alternatives": {
            "otsu": {"t1": round(float(otsu) * 0.5 / 255.0, 4),
                     "t2": round(float(otsu) / 255.0, 4)},
            "median": {"t1": round(max(0.0, 0.66 * med) / 255.0, 4),
                       "t2": round(min(255.0, 1.33 * med) / 255.0, 4)},
        },
    }


@register(
    "edge.canny",
    "Canny",
    "edge",
    params=[
        FloatParam("t1", min=0.0, max=1.0, default=0.1, step=0.005, auto=True,
                   help="Lower hysteresis threshold."),
        FloatParam("t2", min=0.0, max=1.0, default=0.25, step=0.005, auto=True,
                   help="Upper hysteresis threshold."),
        OddIntParam("aperture", min=3, max=7, default=3),
        BoolParam("l2_gradient", default=True,
                  help="Use the true L2 gradient magnitude instead of |gx|+|gy|."),
        FloatParam("presmooth", min=0.0, max=5.0, default=1.0, step=0.1,
                   scale_aware=True),
        EnumParam("auto_mode", options=enum_opts(("off", "Manual"),
                                                 ("otsu", "Otsu-derived"),
                                                 ("median", "Median heuristic")),
                  default="off"),
    ],
    summary="The optimal edge detector: Gaussian smoothing, gradient, "
    "non-maximum suppression to one-pixel width, then hysteresis thresholding "
    "that keeps weak edges only where they connect to strong ones.",
    formula_latex=r"\text{NMS}(\|\nabla(f * G_\sigma)\|) \text{ then hysteresis }"
    r"(T_{low}, T_{high})",
    when_to_use="The default edge detector. Hysteresis is what makes it robust: "
    "a single threshold either fragments real edges or admits noise, while two "
    "connected thresholds do neither.",
    syllabus=True,
    auto_fn=_auto_canny,
    references=["Canny, A Computational Approach to Edge Detection, IEEE PAMI 1986"],
)
def canny(img, t1=0.1, t2=0.25, aperture=3, l2_gradient=True, presmooth=1.0,
          auto_mode="off", ctx=None):
    g = gray_u8(img)
    if presmooth > 0:
        g = cv2.GaussianBlur(g, (0, 0), float(presmooth))
    if auto_mode == "otsu":
        otsu, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lo, hi = otsu * 0.5, otsu
    elif auto_mode == "median":
        med = float(np.median(g))
        lo, hi = max(0.0, 0.66 * med), min(255.0, 1.33 * med)
    else:
        lo, hi = t1 * 255.0, t2 * 255.0
    if hi < lo:
        lo, hi = hi, lo
    e = cv2.Canny(g, float(lo), float(hi), apertureSize=int(aperture),
                  L2gradient=bool(l2_gradient))
    density = float((e > 0).mean())
    return OpResult(image=(e.astype(np.float32) / 255.0),
                    data={"t_low": round(lo / 255.0, 4), "t_high": round(hi / 255.0, 4),
                          "edge_density": round(density, 5),
                          "edge_pixels": int((e > 0).sum())})


@register(
    "edge.log",
    "Marr-Hildreth (LoG)",
    "edge",
    params=[
        FloatParam("sigma", min=0.3, max=20.0, default=2.0, step=0.1, scale_aware=True),
        BoolParam("zero_crossings", default=True,
                  help="Extract the zero crossings, which are the actual edges."),
        FloatParam("threshold", min=0.0, max=0.2, default=0.01, step=0.001,
                   help="Minimum local slope at a crossing, to reject noise."),
    ],
    summary="Laplacian of Gaussian: smooth, then take the second derivative. "
    "Edges are where the response crosses zero, not where it peaks.",
    formula_latex=r"\nabla^2 G(x,y) = \frac{x^2+y^2-2\sigma^2}{\sigma^4}"
    r"e^{-\frac{x^2+y^2}{2\sigma^2}}",
    when_to_use="Scale-selective edge detection. Zero crossings give closed "
    "contours of exactly one pixel width, which gradient maxima do not.",
    syllabus=True,
    references=["Marr & Hildreth, Theory of Edge Detection, 1980"],
)
def log_edges(img, sigma=2.0, zero_crossings=True, threshold=0.01, ctx=None):
    g = luma(img) if img.ndim == 3 else img
    blurred = cv2.GaussianBlur(g, (0, 0), float(sigma))
    lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
    if not zero_crossings:
        return clamp01(normalize01(lap))

    # A zero crossing exists where a pixel and its neighbour differ in sign.
    sign = np.sign(lap)
    zc = np.zeros(lap.shape, bool)
    zc[:-1, :] |= sign[:-1, :] * sign[1:, :] < 0
    zc[:, :-1] |= sign[:, :-1] * sign[:, 1:] < 0
    # Require a minimum slope, otherwise flat noisy regions crackle with crossings.
    slope = cv2.magnitude(cv2.Sobel(lap, cv2.CV_32F, 1, 0, ksize=3),
                          cv2.Sobel(lap, cv2.CV_32F, 0, 1, ksize=3))
    out = zc & (slope > threshold)
    return OpResult(image=out.astype(np.float32),
                    data={"edge_density": round(float(out.mean()), 5),
                          "sigma": round(float(sigma), 3)})


@register(
    "edge.dog",
    "Difference of Gaussians",
    "edge",
    params=[
        FloatParam("sigma", min=0.3, max=20.0, default=1.2, step=0.1, scale_aware=True),
        FloatParam("ratio", min=1.1, max=6.0, default=1.6, step=0.05,
                   help="1.6 is the standard approximation to the LoG."),
        BoolParam("normalize", default=True),
        BoolParam("absolute", default=True),
    ],
    summary="Difference of two Gaussian blurs, which approximates the Laplacian "
    "of Gaussian at a fraction of the cost.",
    formula_latex=r"DoG = f * G_{\sigma_1} - f * G_{\sigma_2},\quad "
    r"\sigma_2 = k\sigma_1,\ k \approx 1.6",
    when_to_use="Band-pass filtering and blob detection; the basis of SIFT's "
    "scale space. A ratio of 1.6 best matches the LoG shape.",
    syllabus=True,
    references=["Lowe, Distinctive Image Features from Scale-Invariant Keypoints, 2004"],
)
def dog(img, sigma=1.2, ratio=1.6, normalize=True, absolute=True, ctx=None):
    g = luma(img) if img.ndim == 3 else img
    g1 = cv2.GaussianBlur(g, (0, 0), float(sigma))
    g2 = cv2.GaussianBlur(g, (0, 0), float(sigma * ratio))
    d = g1 - g2
    if absolute:
        d = np.abs(d)
    out = normalize01(d) if normalize else clamp01(d + 0.5)
    return OpResult(image=out.astype(np.float32),
                    data={"sigma1": round(float(sigma), 3),
                          "sigma2": round(float(sigma * ratio), 3),
                          "response_max": round(float(np.abs(d).max()), 5)})


@register(
    "edge.gradient_vis",
    "Gradient magnitude / orientation",
    "edge",
    params=[
        EnumParam("operator", options=enum_opts(("sobel", "Sobel"),
                                                ("scharr", "Scharr"),
                                                ("prewitt", "Prewitt")),
                  default="sobel"),
        OddIntParam("ksize", min=3, max=7, default=3),
        EnumParam("display", options=enum_opts(("magnitude", "Magnitude"),
                                               ("hsv_wheel", "Orientation as hue"),
                                               ("quiver", "Magnitude + orientation map")),
                  default="hsv_wheel"),
    ],
    summary="Visualises the gradient field: magnitude as brightness and "
    "orientation as hue, so edge direction becomes directly readable.",
    formula_latex=r"\|\nabla f\| = \sqrt{G_x^2 + G_y^2},\quad "
    r"\theta = \arctan\frac{G_y}{G_x}",
    when_to_use="Understanding what a gradient operator actually produces before "
    "feeding it into Canny, HOG or Hough.",
    is_view=True,
    tags=["teaching"],
)
def gradient_vis(img, operator="sobel", ksize=3, display="hsv_wheel", ctx=None):
    g = luma(img) if img.ndim == 3 else img
    if operator == "scharr":
        gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    elif operator == "prewitt":
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
        gx = cv2.filter2D(g, cv2.CV_32F, kx)
        gy = cv2.filter2D(g, cv2.CV_32F, kx.T)
    else:
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=int(ksize))
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=int(ksize))

    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.mod(np.degrees(np.arctan2(gy, gx)), 360.0)
    magn = normalize01(mag)

    data = {
        "mean_magnitude": round(float(mag.mean()), 5),
        "max_magnitude": round(float(mag.max()), 5),
        "orientation_histogram": np.histogram(
            ang, bins=36, range=(0, 360), weights=mag
        )[0].round(3).tolist(),
    }
    if display == "magnitude":
        return OpResult(image=magn, data=data)
    hsv = np.dstack([ang, np.ones_like(magn), magn]).astype(np.float32)
    wheel = from_hsv(hsv)
    if display == "hsv_wheel":
        return OpResult(image=wheel, data=data)
    return OpResult(image=magn, data=data, overlays={"orientation": wheel})
