"""Sharpening and detail restoration. Catalogue Section 5."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import signal
from skimage.restoration import richardson_lucy, wiener as sk_wiener

from ..core.colorspace import join_alpha, luma, split_alpha
from ..core.params import (
    BoolParam,
    EnumParam,
    FloatParam,
    IntParam,
    MatrixParam,
    OddIntParam,
    enum_opts,
)
from ..core.registry import OpResult, register
from ._util import EPS, border_flag, clamp01, colorize, normalize01, on_luma

KERNEL_PRESETS = [
    {"label": "Identity", "value": [[0, 0, 0], [0, 1, 0], [0, 0, 0]]},
    {"label": "Sharpen (4-N)", "value": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]},
    {"label": "Sharpen (8-N)", "value": [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]},
    {"label": "Laplacian (4-N)", "value": [[0, 1, 0], [1, -4, 1], [0, 1, 0]]},
    {"label": "Laplacian (8-N)", "value": [[1, 1, 1], [1, -8, 1], [1, 1, 1]]},
    {"label": "Sobel X", "value": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]},
    {"label": "Sobel Y", "value": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]},
    {"label": "Prewitt X", "value": [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]},
    {"label": "Emboss", "value": [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]},
    {"label": "Box blur", "value": [[1, 1, 1], [1, 1, 1], [1, 1, 1]]},
    {"label": "Gaussian 3x3", "value": [[1, 2, 1], [2, 4, 2], [1, 2, 1]]},
    {"label": "Outline", "value": [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]},
]


@register(
    "sharpen.unsharp",
    "Unsharp mask",
    "sharpen",
    params=[
        FloatParam("radius", min=0.1, max=50.0, default=1.5, step=0.1, scale_aware=True),
        FloatParam("amount", min=0.0, max=5.0, default=0.8, step=0.05),
        FloatParam("threshold", min=0.0, max=0.5, default=0.0, step=0.002,
                   help="Edges weaker than this are left alone -- which is how "
                        "you sharpen detail without amplifying noise."),
    ],
    summary="Adds a scaled copy of the difference between the image and a "
    "blurred version of itself.",
    formula_latex=r"g = f + k\,(f - f * G_\sigma)",
    when_to_use="The standard sharpener. Set the radius to match the scale of "
    "the blur you are fighting, and raise the threshold when the image is noisy.",
    references=["Gonzalez & Woods, Ch. 3.6.3"],
)
def unsharp(img, radius=1.5, amount=0.8, threshold=0.0, ctx=None):
    if amount <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), float(radius))
    detail = img - blur
    if threshold > 0:
        # Soft gate so the threshold does not itself create edges.
        mag = np.abs(detail)
        gate = np.clip((mag - threshold) / max(threshold, 1e-3), 0, 1)
        detail = detail * gate
    return clamp01(img + amount * detail)


@register(
    "sharpen.highboost",
    "High-boost",
    "sharpen",
    params=[
        FloatParam("A", min=1.0, max=5.0, default=1.5, step=0.05),
        OddIntParam("k", min=3, max=31, default=3, scale_aware=True),
    ],
    summary="Scales the original before subtracting its low-pass, so the result "
    "retains the original DC level rather than just the edges.",
    formula_latex=r"g = A\,f - \mathrm{LP}(f) = (A-1)f + \mathrm{HP}(f)",
    when_to_use="The classical sharpening formulation. A = 1 gives a pure "
    "high-pass; larger A progressively re-adds the original image.",
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 3.6.3"],
)
def highboost(img, A=1.5, k=3, ctx=None):
    lp = cv2.blur(img, (int(k), int(k)))
    return clamp01(A * img - lp)


@register(
    "sharpen.laplacian",
    "Laplacian sharpen",
    "sharpen",
    params=[
        EnumParam("kernel", options=enum_opts(("4", "4-neighbour"), ("8", "8-neighbour")),
                  default="4"),
        FloatParam("amount", min=0.0, max=3.0, default=1.0, step=0.05),
        OddIntParam("ksize", min=1, max=31, default=1, scale_aware=True, advanced=True),
    ],
    summary="Subtracts the second derivative, which is positive on one side of "
    "an edge and negative on the other, steepening the transition.",
    formula_latex=r"\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + "
    r"\frac{\partial^2 f}{\partial y^2},\quad g = f - c\nabla^2 f",
    when_to_use="Isotropic sharpening. The 8-neighbour kernel includes diagonals "
    "and so responds to more edge orientations.",
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 3.6.2"],
)
def laplacian_sharpen(img, kernel="4", amount=1.0, ksize=1, ctx=None):
    if kernel == "8":
        k = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32)
        lap = cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT_101)
    else:
        lap = cv2.Laplacian(img, cv2.CV_32F, ksize=int(ksize))
    return clamp01(img - amount * lap)


@register(
    "sharpen.kernel",
    "Custom convolution",
    "sharpen",
    params=[
        MatrixParam("matrix", rows=3, cols=3,
                    default=((0, -1, 0), (-1, 5, -1), (0, -1, 0)),
                    min=-20.0, max=20.0, presets=KERNEL_PRESETS),
        BoolParam("normalize", default=False,
                  help="Divides by the sum of the kernel, preserving brightness."),
        FloatParam("amount", min=0.0, max=3.0, default=1.0, step=0.05),
        FloatParam("offset", min=-1.0, max=1.0, default=0.0, step=0.01),
        EnumParam("border", options=enum_opts("reflect", "replicate", "wrap", "constant"),
                  default="reflect"),
    ],
    summary="Applies any user-entered convolution kernel, and reports the "
    "kernel's own frequency response so you can see what it will do before "
    "looking at the image.",
    formula_latex=r"g(x,y) = \sum_{s}\sum_{t} w(s,t)\,f(x+s, y+t)",
    when_to_use="Exploring filter design directly. Every classical kernel in the "
    "preset list is here to be compared.",
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 3.4"],
)
def custom_kernel(img, matrix=None, normalize=False, amount=1.0, offset=0.0,
                  border="reflect", ctx=None):
    k = np.array(matrix, np.float32)
    s = float(k.sum())
    if normalize and abs(s) > 1e-6:
        k = k / s
    out = cv2.filter2D(img, -1, k, borderType=border_flag(border)) + offset
    if amount != 1.0:
        out = img + amount * (out - img)

    # Frequency response of the kernel: |DFT| on a zero-padded 64x64 grid.
    pad = np.zeros((64, 64), np.float32)
    kh, kw = k.shape
    pad[:kh, :kw] = k
    resp = np.abs(np.fft.fftshift(np.fft.fft2(pad)))
    return OpResult(
        image=clamp01(out),
        data={
            "kernel_sum": round(s, 4),
            "kernel_shape": [int(kh), int(kw)],
            "is_highpass": bool(abs(s) < 0.5),
            "frequency_response": np.round(normalize01(resp), 4).tolist(),
            "dc_gain": round(float(resp[32, 32]), 4),
        },
    )


def _gradient_op(name, label, kernels, syllabus=True, doc="", formula="", refs=None):
    def fn(img, direction="magnitude", amount=1.0, ctx=None):
        g = luma(img) if img.ndim == 3 else img
        kx, ky = kernels
        gx = cv2.filter2D(g, cv2.CV_32F, kx, borderType=cv2.BORDER_REFLECT_101)
        gy = cv2.filter2D(g, cv2.CV_32F, ky, borderType=cv2.BORDER_REFLECT_101)
        if direction == "x":
            out = gx
        elif direction == "y":
            out = gy
        elif direction == "angle":
            out = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
            return clamp01(out)
        else:
            out = np.sqrt(gx * gx + gy * gy)
        return OpResult(
            image=clamp01(np.abs(out) * amount),
            data={"max_gradient": round(float(np.abs(out).max()), 5),
                  "mean_gradient": round(float(np.abs(out).mean()), 5)},
        )

    register(
        name, label, "sharpen",
        params=[
            EnumParam("direction", options=enum_opts(
                ("magnitude", "Magnitude"), ("x", "Horizontal"), ("y", "Vertical"),
                ("angle", "Orientation")), default="magnitude"),
            FloatParam("amount", min=0.1, max=10.0, default=1.0, step=0.1),
        ],
        summary=doc, formula_latex=formula, syllabus=syllabus,
        when_to_use="First-derivative edge detection and gradient visualisation.",
        references=refs or ["Gonzalez & Woods, Ch. 3.6.4"],
    )(fn)


_SOBEL = (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32),
          np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32))
_PREWITT = (np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32),
            np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32))
_ROBERTS = (np.array([[1, 0], [0, -1]], np.float32),
            np.array([[0, 1], [-1, 0]], np.float32))
_SCHARR = (np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], np.float32),
           np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], np.float32))

_gradient_op("sharpen.sobel", "Sobel", _SOBEL,
             doc="First-derivative gradient with a built-in smoothing weight, "
                 "which makes it noticeably less noise-sensitive than Prewitt.",
             formula=r"G_x = \begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix} * f")
_gradient_op("sharpen.prewitt", "Prewitt", _PREWITT,
             doc="First-derivative gradient with uniform weights.",
             formula=r"G_x = \begin{bmatrix}-1&0&1\\-1&0&1\\-1&0&1\end{bmatrix} * f")
_gradient_op("sharpen.roberts", "Roberts cross", _ROBERTS,
             doc="2x2 diagonal difference operator: the cheapest possible "
                 "gradient, and the most noise-sensitive.",
             formula=r"G_1 = \begin{bmatrix}1&0\\0&-1\end{bmatrix} * f")
_gradient_op("sharpen.scharr", "Scharr", _SCHARR, syllabus=False,
             doc="An optimised 3x3 gradient with far better rotational symmetry "
                 "than Sobel at the same kernel size.",
             formula=r"G_x = \begin{bmatrix}-3&0&3\\-10&0&10\\-3&0&3\end{bmatrix} * f",
             refs=["Scharr, Optimal Operators in Digital Image Processing, 2000"])


@register(
    "sharpen.emboss",
    "Emboss",
    "sharpen",
    params=[
        FloatParam("angle", min=0.0, max=360.0, default=45.0, step=5.0, unit="deg"),
        FloatParam("depth", min=0.1, max=5.0, default=1.0, step=0.1),
        BoolParam("keep_color", default=False),
    ],
    summary="Directional difference filter plus a mid-gray offset, producing a "
    "relief appearance lit from a chosen angle.",
    formula_latex=r"g = f * k_\theta + 0.5",
    when_to_use="Visualising surface texture and the direction of local gradients.",
)
def emboss(img, angle=45.0, depth=1.0, keep_color=False, ctx=None):
    rad = np.deg2rad(angle)
    dx, dy = np.cos(rad), -np.sin(rad)
    k = np.array([[-dx - dy, -dy, dy - dx],
                  [-dx, 0, dx],
                  [dx - dy, dy, dx + dy]], np.float32) * depth
    src = img if keep_color else luma(img)
    return clamp01(cv2.filter2D(src, -1, k, borderType=cv2.BORDER_REFLECT_101) + 0.5)


@register(
    "sharpen.clarity",
    "Clarity / local contrast",
    "sharpen",
    params=[
        FloatParam("amount", min=-1.0, max=2.0, default=0.4, step=0.02),
        FloatParam("radius", min=5.0, max=200.0, default=40.0, step=1.0,
                   scale_aware=True),
        FloatParam("midtone_bias", min=0.0, max=1.0, default=0.7, step=0.05,
                   help="Concentrates the effect in the midtones, protecting "
                        "shadows and highlights."),
    ],
    summary="Midtone local-contrast enhancement using an edge-preserving base "
    "layer, so it adds 'punch' without the halos an unsharp mask of this radius "
    "would produce.",
    formula_latex=r"g = f + a\,w(f)\,(f - \mathrm{bilateral}(f))",
    when_to_use="Giving an image presence without visible sharpening artifacts. "
    "Operates at a much larger radius than unsharp masking.",
    tags=["beyond-syllabus"],
    cost="fast",
)
def clarity(img, amount=0.4, radius=40.0, midtone_bias=0.7, ctx=None):
    y = clamp01(luma(img))
    r = int(max(2, radius))
    try:
        base = cv2.ximgproc.guidedFilter(guide=y, src=y, radius=r, eps=1e-2)
    except Exception:
        base = cv2.bilateralFilter(y, 0, 0.15, r)
    detail = y - base
    w = 1.0 - midtone_bias * np.abs(y - 0.5) * 2.0
    w = np.clip(w, 0, 1)
    y2 = clamp01(y + amount * detail * w)
    gain = (y2 + EPS) / (y + EPS)
    rgb, alpha = split_alpha(img)
    if rgb.ndim == 2:
        return join_alpha(y2, alpha)
    return join_alpha(clamp01(rgb * gain[:, :, None]), alpha)


@register(
    "sharpen.detail_split",
    "Base / detail decomposition",
    "sharpen",
    params=[
        FloatParam("radius", min=2.0, max=100.0, default=15.0, step=1.0,
                   scale_aware=True),
        FloatParam("base_gain", min=0.0, max=2.0, default=1.0, step=0.02),
        FloatParam("detail_gain", min=0.0, max=5.0, default=1.6, step=0.05),
        BoolParam("show_detail_only", default=False),
    ],
    summary="Splits the image into a smooth base and a detail residual, lets you "
    "scale each independently, then recombines them.",
    formula_latex=r"f = B + D,\quad g = \alpha B + \beta D",
    when_to_use="Frequency separation: amplify texture while flattening tone, or "
    "the reverse. Also the honest way to show that detail lost to over-smoothing "
    "cannot be recovered -- amplifying D amplifies nothing where D is empty.",
    tags=["beyond-syllabus"],
    cost="fast",
)
def detail_split(img, radius=15.0, base_gain=1.0, detail_gain=1.6,
                 show_detail_only=False, ctx=None):
    r = int(max(2, radius))
    try:
        base = cv2.ximgproc.guidedFilter(
            guide=np.ascontiguousarray(img, np.float32),
            src=np.ascontiguousarray(img, np.float32), radius=r, eps=1e-2)
    except Exception:
        base = cv2.bilateralFilter(np.ascontiguousarray(img, np.float32), 0, 0.15, r)
    detail = img - base
    if show_detail_only:
        return clamp01(detail * detail_gain + 0.5)
    return clamp01(base * base_gain + detail * detail_gain)


@register(
    "sharpen.structure_aware",
    "Masked (structure-aware) sharpen",
    "sharpen",
    params=[
        FloatParam("amount", min=0.0, max=3.0, default=1.0, step=0.05),
        FloatParam("radius", min=0.3, max=10.0, default=1.2, step=0.1, scale_aware=True),
        FloatParam("edge_threshold", min=0.0, max=0.5, default=0.04, step=0.005),
        FloatParam("noise_guard", min=0.0, max=1.0, default=0.5, step=0.05,
                   help="Suppresses sharpening in low-gradient (flat) areas, "
                        "which is where noise lives."),
        BoolParam("show_mask", default=False),
    ],
    summary="Unsharp masking gated by a gradient-magnitude mask, so only genuine "
    "structure is sharpened and flat regions keep their noise unamplified.",
    formula_latex=r"g = f + a\,m(\|\nabla f\|)\,(f - f * G_\sigma)",
    when_to_use="Sharpening a noisy image. This is the operator the Advisor emits "
    "whenever measured noise is non-trivial.",
    tags=["beyond-syllabus"],
)
def structure_aware(img, amount=1.0, radius=1.2, edge_threshold=0.04,
                    noise_guard=0.5, ctx=None, show_mask=False):
    y = luma(img) if img.ndim == 3 else img
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = cv2.GaussianBlur(mag, (0, 0), max(radius, 1.0))
    mask = np.clip((mag - edge_threshold) / max(edge_threshold, 1e-3), 0, 1)
    mask = mask * noise_guard + (1.0 - noise_guard)
    mask = np.clip(mask, 0, 1).astype(np.float32)

    blur = cv2.GaussianBlur(img, (0, 0), float(radius))
    detail = img - blur
    m = mask[:, :, None] if img.ndim == 3 else mask
    out = clamp01(img + amount * detail * m)
    res = OpResult(image=out, data={"mask_mean": round(float(mask.mean()), 4)})
    if show_mask:
        res.overlays["sharpen_mask"] = colorize(mask, "magma")
    return res


def _psf(kind: str, size: float, angle: float) -> np.ndarray:
    """Build a point spread function to deconvolve against."""
    if kind == "motion":
        n = max(3, int(round(size)) | 1)
        k = np.zeros((n, n), np.float32)
        rad = np.deg2rad(angle)
        dx, dy = np.cos(rad), -np.sin(rad)
        c = n // 2
        for t in np.linspace(-size / 2.0, size / 2.0, int(size * 4) + 2):
            x, y = int(round(c + dx * t)), int(round(c + dy * t))
            if 0 <= x < n and 0 <= y < n:
                k[y, x] = 1.0
        return k / max(k.sum(), EPS)
    if kind == "disc":
        r = max(1.0, size / 2.0)
        n = int(np.ceil(r * 2)) | 1
        yy, xx = np.mgrid[:n, :n].astype(np.float32)
        c = (n - 1) / 2.0
        k = (((xx - c) ** 2 + (yy - c) ** 2) <= r * r).astype(np.float32)
        return k / max(k.sum(), EPS)
    sigma = max(size / 3.0, 0.3)
    n = int(sigma * 6) | 1
    k1 = cv2.getGaussianKernel(n, sigma).astype(np.float32)
    k = k1 @ k1.T
    return k / max(k.sum(), EPS)


@register(
    "sharpen.deconv",
    "Deconvolution",
    "sharpen",
    params=[
        EnumParam("method", options=enum_opts(("wiener", "Wiener"),
                                              ("richardson_lucy", "Richardson-Lucy")),
                  default="wiener"),
        EnumParam("psf", options=enum_opts(("gaussian", "Gaussian (defocus)"),
                                           ("motion", "Linear motion"),
                                           ("disc", "Disc (aperture)")),
                  default="gaussian"),
        FloatParam("psf_size", min=1.0, max=60.0, default=5.0, step=0.5,
                   scale_aware=True),
        FloatParam("angle", min=0.0, max=180.0, default=0.0, step=1.0, unit="deg",
                   depends_on={"param": "psf", "equals": "motion"}),
        FloatParam("noise", min=1e-4, max=0.5, default=0.01, step=1e-3, log_scale=True,
                   depends_on={"param": "method", "equals": "wiener"}),
        IntParam("iterations", min=1, max=100, default=20,
                 depends_on={"param": "method", "equals": "richardson_lucy"}),
    ],
    summary="Inverts a known blur kernel rather than merely exaggerating edges. "
    "Wiener regularises the inverse by the noise-to-signal ratio; "
    "Richardson-Lucy iterates a non-negative maximum-likelihood estimate.",
    formula_latex=r"\hat{F} = \frac{H^*}{|H|^2 + K}\,G \quad\text{(Wiener)}",
    when_to_use="The CORRECT fix for real blur, where unsharp masking only "
    "exaggerates the edges that survived. Pairs with the Advisor's motion-blur "
    "detector, which estimates the angle and length from the spectrum.",
    tags=["beyond-syllabus"],
    cost="medium",
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 5.7-5.8",
                "Richardson, Bayesian-Based Iterative Method of Image Restoration, 1972"],
)
def deconvolve(img, method="wiener", psf="gaussian", psf_size=5.0, angle=0.0,
               noise=0.01, iterations=20, ctx=None):
    k = _psf(psf, float(psf_size), float(angle))

    def one(ch):
        x = np.clip(ch.astype(np.float64), 1e-6, 1.0)
        if method == "richardson_lucy":
            try:
                out = richardson_lucy(x, k.astype(np.float64),
                                      num_iter=int(iterations), clip=True)
            except TypeError:  # older skimage keyword
                out = richardson_lucy(x, k.astype(np.float64),
                                      iterations=int(iterations), clip=True)
            return out
        return _wiener_deconv(x, k.astype(np.float64), float(noise))

    def preserve_dc(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Restore the input's mean. The PSF sums to 1, so deconvolution should
        not change overall level; RL's clipping and Wiener's regularisation both
        let it drift, which reads downstream as an exposure change."""
        a, b = float(np.mean(src)), float(np.mean(dst))
        if b <= 1e-6:
            return dst
        return dst * (a / b)

    rgb, alpha = split_alpha(img)
    if rgb.ndim == 2:
        res = preserve_dc(rgb, one(rgb).astype(np.float32))
        return join_alpha(clamp01(res), alpha)
    out = np.dstack([
        preserve_dc(rgb[:, :, i], one(rgb[:, :, i]).astype(np.float32))
        for i in range(3)
    ]).astype(np.float32)
    return join_alpha(clamp01(out), alpha)


def _wiener_deconv(img: np.ndarray, psf: np.ndarray, k: float) -> np.ndarray:
    """Frequency-domain Wiener deconvolution with a constant NSR term."""
    h, w = img.shape
    # Pad the PSF to the image size and centre it on the origin, so the
    # inverse transform does not shift the result.
    pad = np.zeros((h, w), np.float64)
    kh, kw = psf.shape
    pad[:kh, :kw] = psf
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))

    H = np.fft.rfft2(pad)
    G = np.fft.rfft2(img)
    F = np.conj(H) / (np.abs(H) ** 2 + k) * G
    return np.fft.irfft2(F, s=(h, w))
