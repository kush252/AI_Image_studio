"""Smoothing and denoising. Catalogue Section 4."""
from __future__ import annotations

import cv2
import numpy as np
import pywt
from scipy import ndimage, signal
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet

from ..core.colorspace import from_ycrcb, join_alpha, split_alpha, to_ycrcb
from ..core.params import (
    BoolParam,
    EnumParam,
    FloatParam,
    IntParam,
    OddIntParam,
    enum_opts,
)
from ..core.registry import OpResult, register
from ._util import EPS, border_flag, clamp01, from_u8, structuring_element, to_u8

BORDER_OPTS = enum_opts(
    ("reflect", "Reflect"), ("replicate", "Replicate"), ("wrap", "Wrap"),
    ("constant", "Constant (zero)"),
)

CHANNEL_SCOPE = enum_opts(
    ("all", "All channels"), ("luma", "Luma only"), ("chroma", "Chroma only"),
    help_map={
        "chroma": "Denoises Cb/Cr only, preserving every bit of luma detail. The "
                  "biggest quality win available on high-ISO phone JPEGs.",
    },
)


def _scoped(img, scope, fn):
    """Apply a filter to all channels, luma only, or chroma only."""
    rgb, alpha = split_alpha(img)
    if rgb.ndim == 2 or scope == "all":
        return join_alpha(clamp01(fn(rgb)), alpha)
    ycc = to_ycrcb(rgb)
    if scope == "luma":
        ycc[:, :, 0] = fn(ycc[:, :, 0])
    else:
        ycc[:, :, 1] = fn(ycc[:, :, 1])
        ycc[:, :, 2] = fn(ycc[:, :, 2])
    return join_alpha(from_ycrcb(ycc), alpha)


@register(
    "smooth.box",
    "Averaging (box)",
    "smooth",
    params=[
        OddIntParam("k", min=1, max=99, default=5, scale_aware=True),
        BoolParam("normalize", default=True),
        EnumParam("border", options=BORDER_OPTS, default="reflect"),
    ],
    summary="Replaces each pixel with the unweighted mean of its neighbourhood.",
    formula_latex=r"g(x,y) = \frac{1}{mn}\sum_{(s,t)\in S_{xy}} f(s,t)",
    when_to_use="The simplest smoother. Cheap, but its frequency response has "
    "sidelobes, so it rings; prefer Gaussian for anything but a demo.",
    syllabus=True,
    domain="linear",
    references=["Gonzalez & Woods, Ch. 3.5.1"],
)
def box(img, k=5, normalize=True, border="reflect", ctx=None):
    if k <= 1:
        return img
    return clamp01(
        cv2.boxFilter(img, -1, (k, k), normalize=normalize, borderType=border_flag(border))
    )


@register(
    "smooth.gaussian",
    "Gaussian blur",
    "smooth",
    params=[
        FloatParam("sigma", min=0.0, max=50.0, default=2.0, step=0.1, scale_aware=True),
        OddIntParam("k", min=0, max=199, default=0, scale_aware=True, advanced=True,
                    help="0 derives the kernel size from sigma."),
        EnumParam("border", options=BORDER_OPTS, default="reflect"),
    ],
    summary="Convolution with a Gaussian. Separable, isotropic, and the only "
    "smoother with no ringing in its impulse response.",
    formula_latex=r"G(x,y) = \frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}",
    when_to_use="The default smoother, and the building block of DoG, LoG, "
    "unsharp masking and scale space. Runs in linear light so it models real "
    "optical defocus.",
    syllabus=True,
    domain="linear",
    references=["Gonzalez & Woods, Ch. 3.5.1"],
)
def gaussian(img, sigma=2.0, k=0, border="reflect", ctx=None):
    if sigma <= 0 and k <= 1:
        return img
    ks = (int(k) | 1, int(k) | 1) if k and k > 1 else (0, 0)
    return clamp01(
        cv2.GaussianBlur(img, ks, float(sigma), borderType=border_flag(border))
    )


@register(
    "smooth.median",
    "Median",
    "smooth",
    params=[
        OddIntParam("k", min=3, max=99, default=5, scale_aware=True),
        EnumParam("scope", options=CHANNEL_SCOPE, default="all"),
    ],
    summary="Replaces each pixel with the median of its neighbourhood: an "
    "order-statistic, not a convolution.",
    formula_latex=r"g(x,y) = \mathrm{median}\{f(s,t) : (s,t) \in S_{xy}\}",
    when_to_use="Salt-and-pepper noise, where it is dramatically better than any "
    "linear filter -- an impulse is an outlier, and the median ignores outliers "
    "while the mean averages them in.",
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 3.5.2"],
)
def median(img, k=5, scope="all", ctx=None):
    def f(a):
        if k <= 7 and a.dtype == np.float32:
            return cv2.medianBlur(a, int(k))
        return cv2.medianBlur(to_u8(a), int(k)).astype(np.float32) / 255.0

    return _scoped(img, scope, f)


@register(
    "smooth.median_weighted",
    "Weighted median",
    "smooth",
    params=[
        OddIntParam("k", min=3, max=15, default=3, scale_aware=True),
        IntParam("center_weight", min=1, max=9, default=3),
    ],
    summary="Median of a neighbourhood in which the centre pixel is repeated "
    "several times, trading impulse rejection for detail retention.",
    formula_latex=r"g = \mathrm{median}\{\dots, \underbrace{f(x,y),\dots}_{w}\}",
    when_to_use="When a plain median is softening fine detail you want to keep.",
    cost="medium",
)
def median_weighted(img, k=3, center_weight=3, ctx=None):
    def f(ch):
        pad = k // 2
        p = np.pad(ch, pad, mode="reflect")
        wins = np.lib.stride_tricks.sliding_window_view(p, (k, k))
        flat = wins.reshape(*wins.shape[:2], -1)
        c = (k * k) // 2
        extra = np.repeat(flat[:, :, c : c + 1], center_weight - 1, axis=2)
        return np.median(np.concatenate([flat, extra], axis=2), axis=2)

    rgb, alpha = split_alpha(img)
    if rgb.ndim == 2:
        return join_alpha(clamp01(f(rgb)), alpha)
    return join_alpha(
        clamp01(np.dstack([f(rgb[:, :, i]) for i in range(rgb.shape[2])])), alpha
    )


@register(
    "smooth.median_adaptive",
    "Adaptive median",
    "smooth",
    params=[OddIntParam("k_max", min=3, max=15, default=7, scale_aware=True)],
    summary="Grows the window at each pixel until the local median is not itself "
    "an extreme value, then replaces only pixels that are impulses.",
    formula_latex=r"\text{grow } S_{xy} \text{ until } z_{min} < z_{med} < z_{max};"
    r"\ g = z_{xy} \text{ if } z_{min} < z_{xy} < z_{max}, \text{ else } z_{med}",
    when_to_use="High-density salt-and-pepper noise. Strictly better than a plain "
    "median there, because it leaves non-impulse pixels completely untouched -- "
    "which is why the Advisor routes impulse noise here.",
    tags=["beyond-syllabus"],
    cost="medium",
    references=["Gonzalez & Woods, Ch. 5.3.3"],
)
def median_adaptive(img, k_max=7, ctx=None):
    def f(ch):
        out = ch.copy()
        undecided = np.ones(ch.shape, bool)
        for k in range(3, int(k_max) + 1, 2):
            if not undecided.any():
                break
            med = cv2.medianBlur(to_u8(ch), k).astype(np.float32) / 255.0
            se = np.ones((k, k), np.uint8)
            zmin = cv2.erode(ch, se)
            zmax = cv2.dilate(ch, se)
            # Stage A: is the median itself a valid (non-impulse) value?
            med_ok = (med > zmin) & (med < zmax)
            act = undecided & med_ok
            if act.any():
                # Stage B: keep the pixel unless it is itself an extreme.
                pix_ok = (ch > zmin) & (ch < zmax)
                out[act] = np.where(pix_ok[act], ch[act], med[act])
                undecided[act] = False
        if undecided.any():
            med = cv2.medianBlur(to_u8(ch), int(k_max) | 1).astype(np.float32) / 255.0
            out[undecided] = med[undecided]
        return out

    rgb, alpha = split_alpha(img)
    if rgb.ndim == 2:
        return join_alpha(clamp01(f(rgb)), alpha)
    return join_alpha(
        clamp01(np.dstack([f(rgb[:, :, i]) for i in range(rgb.shape[2])])), alpha
    )


@register(
    "smooth.rank",
    "Min / Max / Midpoint",
    "smooth",
    params=[
        OddIntParam("k", min=3, max=51, default=3, scale_aware=True),
        EnumParam("mode", options=enum_opts(("min", "Minimum (erode)"),
                                            ("max", "Maximum (dilate)"),
                                            ("midpoint", "Midpoint")),
                  default="midpoint"),
    ],
    summary="Order-statistic filters taking the minimum, maximum or their mean.",
    formula_latex=r"g = \min S_{xy},\ \max S_{xy},\ \tfrac{1}{2}(\min + \max)",
    when_to_use="Minimum removes salt (bright impulses); maximum removes pepper "
    "(dark impulses); midpoint suits uniform noise.",
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 5.3.2"],
)
def rank_filter(img, k=3, mode="midpoint", ctx=None):
    se = np.ones((k, k), np.uint8)
    if mode == "min":
        return clamp01(cv2.erode(img, se))
    if mode == "max":
        return clamp01(cv2.dilate(img, se))
    return clamp01((cv2.erode(img, se) + cv2.dilate(img, se)) / 2.0)


@register(
    "smooth.alpha_trimmed",
    "Alpha-trimmed mean",
    "smooth",
    params=[
        OddIntParam("k", min=3, max=15, default=5, scale_aware=True),
        IntParam("d", min=0, max=48, default=4,
                 help="Total number of extreme values discarded, half from each end."),
    ],
    summary="Discards the d/2 lowest and d/2 highest neighbours, then averages "
    "the rest.",
    formula_latex=r"g = \frac{1}{mn-d}\sum_{r\in S_{xy}^{trimmed}} r",
    when_to_use="Mixed noise: d=0 is the arithmetic mean, d=mn-1 is the median, "
    "and intermediate values handle Gaussian and impulse noise together.",
    syllabus=True,
    cost="medium",
    references=["Gonzalez & Woods, Ch. 5.3.3"],
)
def alpha_trimmed(img, k=5, d=4, ctx=None):
    n = k * k
    d = min(int(d), n - 1)
    half = d // 2

    def f(ch):
        pad = k // 2
        p = np.pad(ch, pad, mode="reflect")
        wins = np.lib.stride_tricks.sliding_window_view(p, (k, k))
        flat = np.sort(wins.reshape(*wins.shape[:2], -1), axis=2)
        hi = n - half if half else n
        return flat[:, :, half:hi].mean(axis=2)

    rgb, alpha = split_alpha(img)
    if rgb.ndim == 2:
        return join_alpha(clamp01(f(rgb)), alpha)
    return join_alpha(
        clamp01(np.dstack([f(rgb[:, :, i]) for i in range(rgb.shape[2])])), alpha
    )


@register(
    "smooth.bilateral",
    "Bilateral",
    "smooth",
    params=[
        IntParam("d", min=0, max=25, default=7, scale_aware=True,
                 help="Spatial diameter; 0 derives it from sigma_space."),
        FloatParam("sigma_color", min=0.005, max=1.0, default=0.1, step=0.005),
        FloatParam("sigma_space", min=1.0, max=50.0, default=7.0, step=0.5,
                   scale_aware=True),
        EnumParam("scope", options=CHANNEL_SCOPE, default="all"),
    ],
    summary="Weights neighbours by both spatial distance AND intensity "
    "similarity, so averaging stops at edges.",
    formula_latex=r"g(x) = \frac{1}{W}\sum_{y} f(y)\,"
    r"e^{-\frac{\|x-y\|^2}{2\sigma_s^2}}\,e^{-\frac{|f(x)-f(y)|^2}{2\sigma_r^2}}",
    when_to_use="The first genuinely edge-preserving denoiser, and the Advisor's "
    "choice for mild Gaussian noise.",
    tags=["beyond-syllabus"],
    cost="fast",
    references=["Tomasi & Manduchi, Bilateral Filtering for Gray and Color Images, 1998"],
)
def bilateral(img, d=7, sigma_color=0.1, sigma_space=7.0, scope="all", ctx=None):
    def f(a):
        return cv2.bilateralFilter(
            np.ascontiguousarray(a, np.float32), int(d), float(sigma_color),
            float(sigma_space)
        )

    return _scoped(img, scope, f)


@register(
    "smooth.guided",
    "Guided filter",
    "smooth",
    params=[
        IntParam("radius", min=1, max=64, default=8, scale_aware=True),
        FloatParam("eps", min=1e-5, max=0.1, default=0.01, step=1e-4, log_scale=True),
    ],
    summary="Edge-preserving smoothing by local linear regression against a "
    "guide image.",
    formula_latex=r"q_i = \bar{a}_i I_i + \bar{b}_i,\quad "
    r"a_k = \frac{\mathrm{cov}(I,p)}{\mathrm{var}(I)+\epsilon}",
    when_to_use="Faster than bilateral, with no gradient-reversal halos. It is "
    "also the engine behind shadows/highlights recovery and dehaze "
    "transmission refinement.",
    tags=["beyond-syllabus"],
    cost="fast",
    references=["He, Sun & Tang, Guided Image Filtering, ECCV 2010"],
)
def guided(img, radius=8, eps=0.01, ctx=None):
    a = np.ascontiguousarray(img, np.float32)
    try:
        return clamp01(
            cv2.ximgproc.guidedFilter(guide=a, src=a, radius=int(radius), eps=float(eps))
        )
    except Exception:
        return _guided_numpy(a, int(radius), float(eps))


def _guided_numpy(img: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Self-guided filter via box means. Fallback when ximgproc is unavailable."""
    def box(x):
        return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True)

    def one(p):
        mean_i = box(p)
        corr_i = box(p * p)
        var_i = corr_i - mean_i * mean_i
        a = var_i / (var_i + eps)
        b = mean_i - a * mean_i
        return box(a) * p + box(b)

    if img.ndim == 2:
        return clamp01(one(img))
    return clamp01(np.dstack([one(img[:, :, i]) for i in range(img.shape[2])]))


@register(
    "smooth.nlm",
    "Non-local means",
    "smooth",
    params=[
        FloatParam("h", min=1.0, max=40.0, default=8.0, step=0.5,
                   help="Filter strength, in 8-bit intensity units."),
        OddIntParam("template", min=3, max=11, default=7, scale_aware=True),
        OddIntParam("search", min=5, max=35, default=21, scale_aware=True),
        EnumParam("scope", options=CHANNEL_SCOPE, default="all"),
    ],
    summary="Averages pixels weighted by the similarity of their surrounding "
    "PATCHES, searching non-locally rather than only in a neighbourhood.",
    formula_latex=r"g(i) = \frac{1}{Z(i)}\sum_j e^{-\frac{\|P_i - P_j\|^2}{h^2}} f(j)",
    when_to_use="The best detail retention available for Gaussian noise, because "
    "repeated structure anywhere in the image contributes. Slow -- the Advisor "
    "selects it for moderate noise only.",
    tags=["beyond-syllabus"],
    cost="slow",
    references=["Buades, Coll & Morel, A non-local algorithm for image denoising, 2005"],
)
def nlm(img, h=8.0, template=7, search=21, scope="all", ctx=None):
    rgb, alpha = split_alpha(img)
    u8 = to_u8(rgb)
    if u8.ndim == 2:
        out = cv2.fastNlMeansDenoising(u8, None, float(h), int(template), int(search))
    elif scope == "chroma":
        # Denoise colour only: h_luma = 0 leaves Y untouched entirely.
        out = cv2.fastNlMeansDenoisingColored(
            u8, None, 0.0, float(h), int(template), int(search)
        )
    elif scope == "luma":
        out = cv2.fastNlMeansDenoisingColored(
            u8, None, float(h), 0.0, int(template), int(search)
        )
    else:
        out = cv2.fastNlMeansDenoisingColored(
            u8, None, float(h), float(h), int(template), int(search)
        )
    return join_alpha(from_u8(out), alpha)


@register(
    "smooth.anisotropic",
    "Anisotropic diffusion",
    "smooth",
    params=[
        IntParam("iterations", min=1, max=120, default=20),
        FloatParam("kappa", min=0.005, max=0.5, default=0.05, step=0.005,
                   help="Gradient threshold: below it the pixel diffuses, above "
                        "it the edge is preserved."),
        FloatParam("gamma", min=0.01, max=0.25, default=0.15, step=0.01,
                   help="Time step. Above 0.25 the explicit scheme is unstable."),
        EnumParam("conduction", options=enum_opts(("exp", "Exponential (c1)"),
                                                  ("quad", "Quadratic (c2)")),
                  default="exp"),
    ],
    summary="Iteratively solves a diffusion PDE whose conductivity falls where "
    "the gradient is high, so smoothing happens within regions but not across "
    "their boundaries.",
    formula_latex=r"\frac{\partial I}{\partial t} = \nabla\cdot\big(c(\|\nabla I\|)"
    r"\nabla I\big),\quad c(s) = e^{-(s/\kappa)^2}",
    when_to_use="Noise reduction that actively sharpens edges as it runs. "
    "Excellent on medical and scientific images.",
    tags=["beyond-syllabus"],
    cost="slow",
    references=["Perona & Malik, Scale-space and edge detection using anisotropic "
                "diffusion, IEEE PAMI 1990"],
)
def anisotropic(img, iterations=20, kappa=0.05, gamma=0.15, conduction="exp", ctx=None):
    def f(ch):
        u = ch.astype(np.float32).copy()
        for _ in range(int(iterations)):
            # Four nearest-neighbour differences (Perona-Malik discretisation)
            dn = np.vstack([u[:1], u[:-1]]) - u
            ds = np.vstack([u[1:], u[-1:]]) - u
            de = np.hstack([u[:, :1], u[:, :-1]]) - u
            dw = np.hstack([u[:, 1:], u[:, -1:]]) - u
            if conduction == "exp":
                c = lambda d: np.exp(-((d / kappa) ** 2))  # noqa: E731
            else:
                c = lambda d: 1.0 / (1.0 + (d / kappa) ** 2)  # noqa: E731
            u += gamma * (c(dn) * dn + c(ds) * ds + c(de) * de + c(dw) * dw)
        return u

    rgb, alpha = split_alpha(img)
    if rgb.ndim == 2:
        return join_alpha(clamp01(f(rgb)), alpha)
    return join_alpha(
        clamp01(np.dstack([f(rgb[:, :, i]) for i in range(rgb.shape[2])])), alpha
    )


@register(
    "smooth.tv",
    "Total variation (Chambolle)",
    "smooth",
    params=[
        FloatParam("weight", min=0.005, max=0.5, default=0.08, step=0.005),
        IntParam("iterations", min=20, max=400, default=100, advanced=True),
    ],
    summary="Minimises total variation subject to data fidelity, producing "
    "piecewise-smooth output with sharp discontinuities.",
    formula_latex=r"\min_u \ \|u - f\|_2^2 + \lambda \int \|\nabla u\|\,dx",
    when_to_use="Documents, graphics, cartoons and any content that genuinely is "
    "piecewise-constant. Produces characteristic flat 'staircase' regions.",
    tags=["beyond-syllabus"],
    cost="slow",
    references=["Chambolle, An Algorithm for Total Variation Minimization, 2004"],
)
def tv_denoise(img, weight=0.08, iterations=100, ctx=None):
    rgb, alpha = split_alpha(img)
    out = denoise_tv_chambolle(
        rgb.astype(np.float32),
        weight=float(weight),
        max_num_iter=int(iterations),
        channel_axis=-1 if rgb.ndim == 3 else None,
    )
    return join_alpha(clamp01(out.astype(np.float32)), alpha)


@register(
    "smooth.wavelet",
    "Wavelet denoise",
    "smooth",
    params=[
        EnumParam("wavelet", options=enum_opts("db1", "db2", "db4", "db8", "sym4",
                                               "coif1", "bior2.2"), default="db4"),
        IntParam("levels", min=1, max=6, default=3),
        EnumParam("method", options=enum_opts(("BayesShrink", "BayesShrink"),
                                              ("VisuShrink", "VisuShrink")),
                  default="BayesShrink"),
        FloatParam("sigma", min=0.0, max=0.3, default=0.0, step=0.001, auto=True,
                   help="0 estimates the noise level automatically."),
        EnumParam("mode", options=enum_opts(("soft", "Soft threshold"),
                                            ("hard", "Hard threshold")), default="soft"),
    ],
    summary="Thresholds wavelet detail coefficients, which concentrate signal "
    "into a few large values while noise stays spread thinly across many.",
    formula_latex=r"\hat{w} = \mathrm{sign}(w)\max(|w| - \tau, 0),\quad "
    r"\tau = \frac{\sigma^2}{\sigma_w}",
    when_to_use="Heavy noise, where it preserves edges better than any spatial "
    "filter. The Advisor's choice above sigma = 14.",
    tags=["beyond-syllabus"],
    cost="medium",
    references=["Chang, Yu & Vetterli, Adaptive wavelet thresholding, IEEE TIP 2000"],
)
def wavelet_denoise(img, wavelet="db4", levels=3, method="BayesShrink",
                    sigma=0.0, mode="soft", ctx=None):
    rgb, alpha = split_alpha(img)
    out = denoise_wavelet(
        rgb.astype(np.float32),
        sigma=float(sigma) if sigma > 0 else None,
        wavelet=wavelet,
        mode=mode,
        wavelet_levels=int(levels),
        method=method,
        rescale_sigma=True,
        channel_axis=-1 if rgb.ndim == 3 else None,
    )
    return join_alpha(clamp01(out.astype(np.float32)), alpha)


@register(
    "smooth.wiener",
    "Adaptive Wiener",
    "smooth",
    params=[
        OddIntParam("k", min=3, max=31, default=5, scale_aware=True),
        FloatParam("noise_var", min=0.0, max=0.1, default=0.0, step=0.0005, auto=True,
                   help="0 estimates the noise variance from the image."),
    ],
    summary="Locally adaptive linear filter: smooths heavily where local variance "
    "is near the noise variance, and barely at all where it is much higher.",
    formula_latex=r"g = \bar{f} + \frac{\sigma_f^2 - \sigma_n^2}{\sigma_f^2}"
    r"\,(f - \bar{f})",
    when_to_use="The classical minimum-mean-square-error restoration filter. "
    "Adapts to local structure without the cost of bilateral or NLM.",
    syllabus=True,
    references=["Lim, Two-Dimensional Signal and Image Processing, 1990",
                "Gonzalez & Woods, Ch. 5.3.4"],
)
def wiener(img, k=5, noise_var=0.0, ctx=None):
    def f(ch):
        local_mean = cv2.boxFilter(ch, -1, (k, k), normalize=True)
        local_sqr = cv2.boxFilter(ch * ch, -1, (k, k), normalize=True)
        local_var = np.maximum(local_sqr - local_mean**2, 0.0)
        nv = float(noise_var) if noise_var > 0 else float(np.mean(local_var))
        return local_mean + np.maximum(local_var - nv, 0.0) / np.maximum(
            local_var, nv + EPS
        ) * (ch - local_mean)

    rgb, alpha = split_alpha(img)
    if rgb.ndim == 2:
        return join_alpha(clamp01(f(rgb)), alpha)
    return join_alpha(
        clamp01(np.dstack([f(rgb[:, :, i]) for i in range(rgb.shape[2])])), alpha
    )


@register(
    "smooth.morph_denoise",
    "Morphological denoise",
    "smooth",
    params=[
        EnumParam("op", options=enum_opts(("open", "Opening (removes bright specks)"),
                                          ("close", "Closing (fills dark gaps)"),
                                          ("open_close", "Opening then closing")),
                  default="open_close"),
        EnumParam("shape", options=enum_opts("ellipse", "rect", "cross", "diamond"),
                  default="ellipse"),
        OddIntParam("size", min=3, max=31, default=3, scale_aware=True),
    ],
    summary="Removes small bright or dark structures by morphological opening "
    "and/or closing.",
    formula_latex=r"f \circ b = (f \ominus b) \oplus b,\quad f \bullet b = "
    r"(f \oplus b) \ominus b",
    when_to_use="Binary and near-binary images, and impulse noise whose specks "
    "are smaller than the structuring element.",
    syllabus=True,
)
def morph_denoise(img, op="open_close", shape="ellipse", size=3, ctx=None):
    se = structuring_element(shape, size)
    x = img
    if op in ("open", "open_close"):
        x = cv2.morphologyEx(x, cv2.MORPH_OPEN, se)
    if op in ("close", "open_close"):
        x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, se)
    return clamp01(x)


@register(
    "smooth.regrain",
    "Re-grain (dither)",
    "smooth",
    params=[
        FloatParam("sigma", min=0.0, max=8.0, default=1.0, step=0.1,
                   help="Grain amplitude in 8-bit levels."),
        BoolParam("monochrome", default=True,
                  help="One grain pattern shared by all channels, as film "
                       "grain behaves. Per-channel grain reads as colour "
                       "noise."),
        IntParam("seed", min=0, max=9999, default=0,
                 help="Fixed so the same image always yields the same grain."),
    ],
    summary="Adds a calibrated trace of grain to restore the dither that "
    "denoising removed.",
    formula_latex=r"g(x,y) = f(x,y) + \mathcal{N}(0,\sigma^2)",
    when_to_use="After denoising. A denoiser removes the noise that was "
    "dithering the quantisation steps, which uncovers banding that was "
    "latent in the image all along; a trace of grain below the visible "
    "threshold restores the dither without restoring the noise.",
)
def regrain(img, sigma=1.0, monochrome=True, seed=0, ctx=None):
    if sigma <= 0.0:
        return img
    base, alpha = split_alpha(img)
    rng = np.random.default_rng(seed)
    s = float(sigma) / 255.0
    if monochrome or base.ndim == 2:
        g = rng.normal(0.0, s, base.shape[:2]).astype(np.float32)
        if base.ndim == 3:
            g = g[:, :, None]
    else:
        g = rng.normal(0.0, s, base.shape).astype(np.float32)
    return OpResult(join_alpha(clamp01(base + g), alpha))
