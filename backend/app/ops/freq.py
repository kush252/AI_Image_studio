"""Frequency-domain filtering and transforms. Catalogue Section 6 (major syllabus block)."""
from __future__ import annotations

import cv2
import numpy as np
import pywt
from scipy.fft import dct, idct

from ..core.colorspace import join_alpha, luma, split_alpha
from ..core.params import (
    BoolParam,
    EnumParam,
    FloatParam,
    FreqMaskParam,
    IntParam,
    PointsParam,
    enum_opts,
)
from ..core.registry import OpResult, register
from ._util import EPS, clamp01, colorize, normalize01

KIND_OPTS = enum_opts(
    ("ideal", "Ideal (brick wall)"),
    ("butterworth", "Butterworth"),
    ("gaussian", "Gaussian"),
    help_map={
        "ideal": "Sharp cutoff. Produces strong ringing (the Gibbs phenomenon) "
                 "because its spatial kernel is a sinc with infinite support.",
        "butterworth": "Order n controls the transition steepness: low orders "
                       "approach Gaussian, high orders approach Ideal.",
        "gaussian": "No ringing at all, because the transform of a Gaussian is "
                    "a Gaussian -- strictly positive, with no sidelobes.",
    },
)

PAD_OPTS = enum_opts(
    ("optimal", "Optimal DFT size"), ("none", "None"), ("power2", "Power of two")
)

WINDOW_OPTS = enum_opts(
    ("none", "None"), ("hann", "Hann"), ("hamming", "Hamming")
)


# ------------------------------------------------------------------ plumbing

def _pad_size(n: int, mode: str) -> int:
    if mode == "none":
        return n
    if mode == "power2":
        return 1 << int(np.ceil(np.log2(max(n, 1))))
    return cv2.getOptimalDFTSize(n)


def _window2d(shape: tuple[int, int], kind: str) -> np.ndarray | None:
    """Tapering the border suppresses the ringing caused by the implicit
    discontinuity between opposite image edges."""
    if kind == "none":
        return None
    h, w = shape
    if kind == "hamming":
        wy, wx = np.hamming(h), np.hamming(w)
    else:
        wy, wx = np.hanning(h), np.hanning(w)
    return np.outer(wy, wx).astype(np.float32)


def _radius_grid(h: int, w: int) -> np.ndarray:
    """Normalised radial frequency, 0 at DC and 1 at the Nyquist corner."""
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return d / max(np.sqrt(cx**2 + cy**2), EPS)


def _transfer(kind: str, d: np.ndarray, cutoff: float, order: int,
              highpass: bool) -> np.ndarray:
    c = max(float(cutoff), 1e-4)
    if kind == "ideal":
        h = (d <= c).astype(np.float32)
    elif kind == "gaussian":
        h = np.exp(-(d**2) / (2.0 * c**2)).astype(np.float32)
    else:
        h = (1.0 / (1.0 + (d / c) ** (2 * int(order)))).astype(np.float32)
    return (1.0 - h) if highpass else h


def _apply_filter_channel(ch: np.ndarray, H: np.ndarray, pad: str,
                          window: str) -> np.ndarray:
    """Pad, window, transform, multiply by a (fftshifted) transfer function."""
    h, w = ch.shape
    win = _window2d((h, w), window)
    x = ch * win if win is not None else ch
    ph, pw = _pad_size(h, pad), _pad_size(w, pad)
    if (ph, pw) != (h, w):
        x = cv2.copyMakeBorder(x, 0, ph - h, 0, pw - w, cv2.BORDER_REFLECT_101)
    F = np.fft.fftshift(np.fft.fft2(x))
    if H.shape != F.shape:
        H = cv2.resize(H, (F.shape[1], F.shape[0]), interpolation=cv2.INTER_LINEAR)
    out = np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))[:h, :w]
    if win is not None:
        # Undo the window where it is safely non-zero; near the border we keep
        # the windowed result rather than dividing by ~0 and exploding.
        safe = win > 0.05
        out = np.where(safe, out / np.maximum(win, 0.05), out)
    return out.astype(np.float32)


def _filter_op(op_id: str, label: str, highpass: bool, doc: str, formula: str):
    def fn(img, kind="gaussian", cutoff=0.2, order=2, pad="optimal",
           window="none", scope="luma", ctx=None):
        rgb, alpha = split_alpha(img)
        base = luma(rgb) if (scope == "luma" and rgb.ndim == 3) else rgb
        shape = base.shape[:2]
        ph, pw = _pad_size(shape[0], pad), _pad_size(shape[1], pad)
        d = _radius_grid(ph, pw)
        H = _transfer(kind, d, cutoff, order, highpass)

        if base.ndim == 2:
            out = _apply_filter_channel(base, H, pad, window)
            if highpass:
                out = out + 0.5  # recentre: a high-pass result is zero-mean
            if scope == "luma" and rgb.ndim == 3:
                y = luma(rgb)
                gain = (clamp01(out) + EPS) / (y + EPS)
                res = clamp01(rgb * gain[:, :, None])
            else:
                res = clamp01(out)
        else:
            chans = [_apply_filter_channel(base[:, :, i], H, pad, window)
                     for i in range(base.shape[2])]
            out = np.dstack(chans)
            if highpass:
                out = out + 0.5
            res = clamp01(out)

        return OpResult(
            image=join_alpha(res, alpha),
            data={
                "transfer_profile": np.round(
                    _transfer(kind, np.linspace(0, 1, 128), cutoff, order, highpass), 4
                ).tolist(),
                "cutoff": round(float(cutoff), 4),
                "kind": kind,
            },
        )

    register(
        op_id, label, "freq",
        params=[
            EnumParam("kind", options=KIND_OPTS, default="gaussian"),
            FloatParam("cutoff", min=0.005, max=1.0, default=0.2, step=0.005,
                       scale_aware=True, scale_inverse=True,
                       help="Normalised cutoff radius: 1.0 is the Nyquist corner."),
            IntParam("order", min=1, max=10, default=2,
                     depends_on={"param": "kind", "equals": "butterworth"}),
            EnumParam("pad", options=PAD_OPTS, default="optimal", advanced=True),
            EnumParam("window", options=WINDOW_OPTS, default="none", advanced=True),
            EnumParam("scope", options=enum_opts(("luma", "Luma only"),
                                                 ("all", "All channels")),
                      default="luma"),
        ],
        summary=doc, formula_latex=formula, syllabus=True,
        when_to_use="Comparing the three filter families is the point: Ideal "
        "rings, Gaussian does not, and Butterworth interpolates between them.",
        references=["Gonzalez & Woods, Ch. 4.8-4.9"],
        cost="medium",
    )(fn)


_filter_op(
    "freq.lowpass", "Low-pass filter", False,
    "Attenuates high spatial frequencies, which blurs the image. Available as "
    "Ideal, Butterworth or Gaussian.",
    r"H_{ideal} = \begin{cases}1 & D \le D_0\\0 & D > D_0\end{cases},\quad "
    r"H_{btw} = \frac{1}{1+(D/D_0)^{2n}},\quad H_{gauss} = e^{-D^2/2D_0^2}",
)
_filter_op(
    "freq.highpass", "High-pass filter", True,
    "Attenuates low frequencies, keeping edges and fine texture. H_hp = 1 - H_lp.",
    r"H_{hp}(u,v) = 1 - H_{lp}(u,v)",
)


@register(
    "freq.bandpass",
    "Band-pass / Band-reject",
    "freq",
    params=[
        EnumParam("kind", options=KIND_OPTS, default="butterworth"),
        FloatParam("c_low", min=0.005, max=1.0, default=0.05, step=0.005,
                   scale_aware=True, scale_inverse=True),
        FloatParam("c_high", min=0.005, max=1.0, default=0.35, step=0.005,
                   scale_aware=True, scale_inverse=True),
        IntParam("order", min=1, max=10, default=2),
        BoolParam("reject", default=False,
                  help="Inverts the band: keeps everything EXCEPT the band."),
        EnumParam("pad", options=PAD_OPTS, default="optimal", advanced=True),
    ],
    summary="Passes (or rejects) an annulus of radial frequencies between two "
    "cutoffs.",
    formula_latex=r"H_{bp} = H_{lp}(D_1) \cdot \big(1 - H_{lp}(D_0)\big)",
    when_to_use="Isolating a specific scale of detail; band-reject removes "
    "periodic interference whose frequency is known.",
    syllabus=True,
    cost="medium",
    references=["Gonzalez & Woods, Ch. 4.10"],
)
def bandpass(img, kind="butterworth", c_low=0.05, c_high=0.35, order=2,
             reject=False, pad="optimal", ctx=None):
    lo, hi = min(c_low, c_high), max(c_low, c_high)
    rgb, alpha = split_alpha(img)
    base = luma(rgb) if rgb.ndim == 3 else rgb
    ph, pw = _pad_size(base.shape[0], pad), _pad_size(base.shape[1], pad)
    d = _radius_grid(ph, pw)
    H = _transfer(kind, d, hi, order, False) * (1.0 - _transfer(kind, d, lo, order, False))
    if reject:
        H = 1.0 - H
    out = _apply_filter_channel(base, H, pad, "none")
    if not reject:
        out = out + 0.5
    if rgb.ndim == 3:
        y = luma(rgb)
        gain = (clamp01(out) + EPS) / (y + EPS)
        res = clamp01(rgb * gain[:, :, None])
    else:
        res = clamp01(out)
    return join_alpha(res, alpha)


@register(
    "freq.notch",
    "Notch filter",
    "freq",
    params=[
        PointsParam("notches", default=[], max_count=32,
                    help="Normalised spectrum coordinates to suppress."),
        FloatParam("radius", min=0.005, max=0.3, default=0.03, step=0.005),
        EnumParam("kind", options=KIND_OPTS, default="gaussian"),
        IntParam("order", min=1, max=8, default=2),
        BoolParam("symmetric", default=True,
                  help="Also notch the conjugate point, which a real-valued "
                       "image always requires."),
    ],
    summary="Suppresses small regions of the spectrum, each notch removing one "
    "periodic interference component.",
    formula_latex=r"H(u,v) = \prod_k \left[1 - e^{-D_k^2/2r^2}\right]",
    when_to_use="Periodic noise: scanner banding, halftone patterns, power-line "
    "interference. A sinusoid in the image is a point pair in the spectrum.",
    syllabus=True,
    cost="fast",
    references=["Gonzalez & Woods, Ch. 4.10.2"],
)
def notch(img, notches=None, radius=0.03, kind="gaussian", order=2,
          symmetric=True, ctx=None):
    notches = notches or []
    rgb, alpha = split_alpha(img)
    base = luma(rgb) if rgb.ndim == 3 else rgb
    h, w = base.shape[:2]
    if not notches:
        return img

    H = np.ones((h, w), np.float32)
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rad = radius * max(h, w) / 2.0
    for nx, ny in notches:
        pts = [(nx, ny)]
        if symmetric:
            pts.append((1.0 - nx, 1.0 - ny))
        for px, py in pts:
            px_abs, py_abs = px * w, py * h
            d = np.sqrt((xx - px_abs) ** 2 + (yy - py_abs) ** 2) / max(rad, EPS)
            H *= _transfer(kind, d, 1.0, order, True)

    out = _apply_filter_channel(base, H, "none", "none")
    if rgb.ndim == 3:
        y = luma(rgb)
        gain = (clamp01(out) + EPS) / (y + EPS)
        res = clamp01(rgb * gain[:, :, None])
    else:
        res = clamp01(out)
    return OpResult(image=join_alpha(res, alpha),
                    data={"notch_count": len(notches)})


def detect_spectral_peaks(gray: np.ndarray, sensitivity: float = 4.5,
                          max_peaks: int = 8) -> list[dict]:
    """Find periodic-noise peaks by subtracting the radial average.

    Natural images have a smooth, monotonically decaying radial power spectrum.
    A periodic interference adds a localised spike, so anything standing far
    above the radial mean at its own radius is interference, not content.
    (advisor-spec.md 2.5)
    """
    h, w = gray.shape[:2]
    # Window first: the implicit wrap-around discontinuity between opposite
    # image edges otherwise lays a bright cross along the fx=0 and fy=0 axes
    # of the spectrum, and the top-hat below reads points off its arms as
    # interference. This showed up as phantom peaks on every blurred image.
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    g = (gray - float(gray.mean())) * win
    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.log1p(np.abs(F))

    d = _radius_grid(h, w)

    # Compare each bin against a LOCAL 2-D background rather than its radial
    # ring. A sinusoid is a delta in the spectrum, so it stands above its own
    # neighbourhood; a blurred image has a smooth spectrum that stands above
    # nothing. Ring statistics fail here because a strong blur empties the
    # high-frequency rings, collapsing their standard deviation so that any
    # residual scores as a huge outlier -- which made every blurred image
    # report phantom interference.
    # The background estimate is a grayscale morphological OPENING of the
    # magnitude, so the residual is a white top-hat: exactly the operator for
    # isolating small bright spikes on an arbitrary smooth background.
    k = int(np.clip(int(0.025 * max(h, w)) | 1, 5, 31))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    local_bg = cv2.morphologyEx(mag.astype(np.float32), cv2.MORPH_OPEN, se)
    resid = np.maximum(mag - local_bg, 0.0)
    # Scale by the spread of the top-hat itself, which is near zero for a smooth
    # spectrum and large only where genuine spikes exist.
    # Score each bin by how ISOLATED it is: a periodic component is a delta in
    # the spectrum, so it must stand above its own immediate neighbourhood, not
    # merely above a global or radial average. Measured as the ratio of the
    # residual to the mean residual in a surrounding annulus, computed from two
    # box means. On the fixtures this separates a planted sinusoid (ratio ~5.8)
    # from ordinary image structure and from blur (both ~2.1-2.4), which no
    # radial or global threshold managed.
    k_in, k_out = 5, 17
    m_in = cv2.blur(resid, (k_in, k_in))
    m_out = cv2.blur(resid, (k_out, k_out))
    n_in, n_out = k_in * k_in, k_out * k_out
    annulus = (m_out * n_out - m_in * n_in) / float(n_out - n_in)
    z = resid / np.maximum(annulus, 1e-3)

    # A peak must ALSO be absolutely significant against the spectrum's own
    # dynamic range. Without this, a blurred image whose top-hat is ~0
    # everywhere still produces huge ratios from numerical dust.
    dyn = float(np.percentile(mag, 99.9)) - float(np.median(mag))
    z[resid < 0.10 * max(dyn, 1e-6)] = 0.0

    # The peak must also sit in a LIVE part of the spectrum. After a strong
    # blur the high-frequency region is dead, and numerical dust there still
    # looks isolated relative to its equally-dead neighbourhood; requiring the
    # magnitude itself to be above the spectrum's median rejects that, while a
    # real interference line is always among the stronger components.
    z[mag < float(np.median(mag))] = 0.0

    # Ignore DC and its immediate neighbourhood, where image content lives.
    z[d < 0.03] = 0.0
    peaks: list[dict] = []
    zc = z.copy()
    rad_px = max(3, int(0.01 * max(h, w)))
    for _ in range(max_peaks):
        idx = int(np.argmax(zc))
        py, px = divmod(idx, w)
        if zc[py, px] < sensitivity:
            break
        peaks.append({
            "x": round(px / w, 4),
            "y": round(py / h, 4),
            "fx": round((px - w / 2) / w, 4),
            "fy": round((py - h / 2) / h, 4),
            "prominence": round(float(zc[py, px]), 3),
        })
        y0, y1 = max(0, py - rad_px), min(h, py + rad_px + 1)
        x0, x1 = max(0, px - rad_px), min(w, px + rad_px + 1)
        zc[y0:y1, x0:x1] = 0.0
        # Also suppress the conjugate so it is not reported as a second finding.
        my, mx = h - py, w - px
        if 0 <= my < h and 0 <= mx < w:
            zc[max(0, my - rad_px) : my + rad_px + 1,
               max(0, mx - rad_px) : mx + rad_px + 1] = 0.0
    return peaks


@register(
    "freq.notch_auto",
    "Auto periodic-noise removal",
    "freq",
    params=[
        FloatParam("sensitivity", min=1.5, max=12.0, default=4.5, step=0.1,
                   help="Isolation ratio a peak must exceed: its residual "
                        "against the mean residual of the surrounding annulus."),
        FloatParam("radius", min=0.005, max=0.2, default=0.025, step=0.005),
        IntParam("max_peaks", min=1, max=16, default=6),
        BoolParam("show_spectrum", default=True),
    ],
    summary="Detects spectral peaks by subtracting the radial average power, "
    "places notches at them automatically, and reports what it removed.",
    formula_latex=r"z(u,v) = \frac{\log|F| - \mu_r(D)}{\sigma_r(D)} > \tau",
    when_to_use="Scanner banding and halftone patterns, where finding the "
    "frequencies by hand is tedious. Feeds the Advisor's NSE_PERIODIC rule.",
    tags=["beyond-syllabus"],
    cost="medium",
)
def notch_auto(img, sensitivity=4.0, radius=0.025, max_peaks=6,
               show_spectrum=True, ctx=None):
    g = luma(img) if img.ndim == 3 else img
    peaks = detect_spectral_peaks(g, sensitivity, max_peaks)
    if not peaks:
        return OpResult(image=img, data={"peaks": [], "removed": 0},
                        notes=["No periodic interference detected."])
    pts = [(p["x"], p["y"]) for p in peaks]
    out = notch(img, pts, radius, "gaussian", 2, True, ctx=ctx)
    image = out.image if isinstance(out, OpResult) else out

    res = OpResult(image=image, data={"peaks": peaks, "removed": len(peaks)})
    if show_spectrum:
        F = np.fft.fftshift(np.fft.fft2(g - g.mean()))
        res.overlays["spectrum"] = colorize(normalize01(np.log1p(np.abs(F))), "turbo")
    return res


@register(
    "freq.mask_paint",
    "Painted frequency mask",
    "freq",
    params=[
        FreqMaskParam("mask", help="Binary mask painted directly on the spectrum."),
        BoolParam("soft_edges", default=True),
        FloatParam("feather", min=0.0, max=0.1, default=0.01, step=0.002),
        BoolParam("invert", default=False),
    ],
    summary="Applies an arbitrary mask painted by hand onto the magnitude "
    "spectrum, then inverse-transforms.",
    formula_latex=r"g = \mathcal{F}^{-1}\{M(u,v)\,F(u,v)\}",
    when_to_use="The most direct way to build intuition about what each region "
    "of the spectrum contributes: erase the centre and only edges survive; keep "
    "only the centre and the image becomes a blur.",
    tags=["beyond-syllabus", "teaching"],
    cost="fast",
)
def mask_paint(img, mask=None, soft_edges=True, feather=0.01, invert=False, ctx=None):
    if not mask:
        return img
    rgb, alpha = split_alpha(img)
    base = luma(rgb) if rgb.ndim == 3 else rgb
    h, w = base.shape[:2]

    from ..core.masks import _rle_decode

    mw, mh = int(mask.get("width", 0)), int(mask.get("height", 0))
    data = mask.get("data")
    if not (mw and mh and data):
        return img
    M = _rle_decode(data, mw * mh).reshape(mh, mw).astype(np.float32)
    M = cv2.resize(M, (w, h), interpolation=cv2.INTER_LINEAR)
    if invert:
        M = 1.0 - M
    if soft_edges and feather > 0:
        sigma = max(0.5, feather * max(h, w))
        M = cv2.GaussianBlur(M, (0, 0), sigma)

    out = _apply_filter_channel(base, M, "none", "none")
    if rgb.ndim == 3:
        y = luma(rgb)
        gain = (clamp01(out) + EPS) / (y + EPS)
        res = clamp01(rgb * gain[:, :, None])
    else:
        res = clamp01(out)
    return join_alpha(res, alpha)


@register(
    "freq.phase_only",
    "Phase-only reconstruction",
    "freq",
    summary="Discards the magnitude spectrum, keeping only phase, then inverse "
    "transforms.",
    formula_latex=r"g = \mathcal{F}^{-1}\left\{e^{\,j\angle F(u,v)}\right\}",
    when_to_use="The classic demonstration that PHASE carries image structure. "
    "Edges and object outlines remain perfectly recognisable with all magnitude "
    "information destroyed.",
    syllabus=True,
    tags=["teaching"],
    references=["Oppenheim & Lim, The importance of phase in signals, 1981"],
)
def phase_only(img, ctx=None):
    g = luma(img) if img.ndim == 3 else img
    F = np.fft.fft2(g)
    out = np.real(np.fft.ifft2(np.exp(1j * np.angle(F))))
    return normalize01(out)


@register(
    "freq.mag_only",
    "Magnitude-only reconstruction",
    "freq",
    summary="Discards phase, keeping only the magnitude spectrum.",
    formula_latex=r"g = \mathcal{F}^{-1}\left\{|F(u,v)|\right\}",
    when_to_use="The complement of the phase-only demo: the result is an "
    "unrecognisable blob, proving magnitude alone carries almost no structure.",
    syllabus=True,
    tags=["teaching"],
)
def mag_only(img, ctx=None):
    g = luma(img) if img.ndim == 3 else img
    F = np.fft.fft2(g)
    out = np.real(np.fft.ifft2(np.abs(F)))
    return normalize01(np.fft.fftshift(out))


@register(
    "freq.homomorphic",
    "Homomorphic filter",
    "freq",
    params=[
        FloatParam("gamma_low", min=0.0, max=1.5, default=0.4, step=0.05,
                   help="Gain for illumination (low frequencies). Below 1 "
                        "compresses the illumination range."),
        FloatParam("gamma_high", min=0.5, max=3.0, default=1.8, step=0.05,
                   help="Gain for reflectance (high frequencies)."),
        FloatParam("cutoff", min=0.005, max=0.6, default=0.08, step=0.005,
                   scale_aware=True, scale_inverse=True),
        IntParam("order", min=1, max=8, default=2),
    ],
    summary="Separates illumination from reflectance by working in the log "
    "domain, where their product becomes a sum, then attenuates low frequencies "
    "and boosts high ones.",
    formula_latex=r"\ln f = \ln i + \ln r,\quad "
    r"H(u,v) = (\gamma_H - \gamma_L)\left[1 - e^{-cD^2/D_0^2}\right] + \gamma_L",
    when_to_use="The principled fix for uneven illumination: a shadow across the "
    "frame is multiplicative, so no additive correction can remove it properly.",
    syllabus=True,
    cost="medium",
    references=["Gonzalez & Woods, Ch. 4.9.6"],
)
def homomorphic(img, gamma_low=0.4, gamma_high=1.8, cutoff=0.08, order=2, ctx=None):
    rgb, alpha = split_alpha(img)
    y = np.maximum(luma(rgb) if rgb.ndim == 3 else rgb, 1e-4)
    log_y = np.log(y)

    h, w = y.shape
    d = _radius_grid(h, w)
    hp = 1.0 / (1.0 + (cutoff / np.maximum(d, 1e-6)) ** (2 * int(order)))
    H = (gamma_high - gamma_low) * hp + gamma_low

    F = np.fft.fftshift(np.fft.fft2(log_y))
    out = np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))
    out = np.exp(out)
    out = normalize01(out)

    if rgb.ndim == 3:
        gain = (out + EPS) / (y + EPS)
        return join_alpha(clamp01(rgb * gain[:, :, None]), alpha)
    return join_alpha(clamp01(out), alpha)


@register(
    "freq.dct_compress",
    "DCT compression demo",
    "freq",
    params=[
        EnumParam("mode", options=enum_opts(("block8", "8x8 blocks (JPEG-style)"),
                                            ("global", "Whole image")),
                  default="block8"),
        FloatParam("keep_ratio", min=0.005, max=1.0, default=0.15, step=0.005,
                   help="Fraction of coefficients retained, lowest-frequency first."),
        BoolParam("show_blocks", default=False),
    ],
    summary="Forward DCT, zeroing of the smallest/highest coefficients, then "
    "inverse DCT -- with the resulting PSNR and compression ratio reported.",
    formula_latex=r"C(u,v) = \alpha(u)\alpha(v)\sum_x\sum_y f(x,y)"
    r"\cos\frac{(2x{+}1)u\pi}{2N}\cos\frac{(2y{+}1)v\pi}{2N}",
    when_to_use="Showing why JPEG works: the DCT concentrates nearly all energy "
    "into a handful of low-frequency coefficients, so most can be discarded "
    "with little visible loss -- until blocking artifacts appear.",
    tags=["beyond-syllabus", "teaching"],
    cost="slow",
    references=["Ahmed, Natarajan & Rao, Discrete Cosine Transform, 1974"],
)
def dct_compress(img, mode="block8", keep_ratio=0.15, show_blocks=False, ctx=None):
    rgb, alpha = split_alpha(img)
    work = rgb if rgb.ndim == 2 else rgb[:, :, :3]

    def compress(ch):
        if mode == "global":
            C = dct(dct(ch.T, norm="ortho").T, norm="ortho")
            flat = np.abs(C).ravel()
            k = max(1, int(len(flat) * keep_ratio))
            thresh = np.partition(flat, -k)[-k]
            C2 = np.where(np.abs(C) >= thresh, C, 0.0)
            return idct(idct(C2.T, norm="ortho").T, norm="ortho")
        # 8x8 block DCT, keeping a zig-zag prefix of each block
        h, w = ch.shape
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        p = np.pad(ch, ((0, ph), (0, pw)), mode="edge")
        H, W = p.shape
        keep_n = max(1, int(round(64 * keep_ratio)))
        # Zig-zag order as a frequency ranking
        order = sorted(
            ((u, v) for u in range(8) for v in range(8)),
            key=lambda uv: (uv[0] + uv[1], uv[1] if (uv[0] + uv[1]) % 2 else uv[0]),
        )
        keepmask = np.zeros((8, 8), np.float32)
        for u, v in order[:keep_n]:
            keepmask[u, v] = 1.0
        blocks = p.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3)
        out = np.empty_like(blocks)
        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                b = blocks[i, j] - 0.5
                C = dct(dct(b.T, norm="ortho").T, norm="ortho") * keepmask
                out[i, j] = idct(idct(C.T, norm="ortho").T, norm="ortho") + 0.5
        rec = out.transpose(0, 2, 1, 3).reshape(H, W)
        return rec[:h, :w]

    if work.ndim == 2:
        res = compress(work)
    else:
        res = np.dstack([compress(work[:, :, i]) for i in range(work.shape[2])])
    res = clamp01(res)

    mse = float(np.mean((res - work) ** 2))
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-12))
    out = OpResult(
        image=join_alpha(res, alpha),
        data={
            "keep_ratio": round(float(keep_ratio), 4),
            "compression_ratio": round(1.0 / max(keep_ratio, 1e-6), 2),
            "psnr_db": round(psnr, 2),
            "mse": round(mse, 8),
            "mode": mode,
        },
    )
    if show_blocks:
        g = luma(res) if res.ndim == 3 else res
        grid = g.copy()
        grid[::8, :] = 1.0
        grid[:, ::8] = 1.0
        out.overlays["block_grid"] = np.dstack([grid] * 3)
    return out


# ----------------------------------------------- Hadamard / Walsh (syllabus)

def _hadamard_matrix(n: int) -> np.ndarray:
    """Sylvester construction. n must be a power of two."""
    H = np.ones((1, 1), np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]]).astype(np.float32)
    return H


def _sequency_order(n: int) -> np.ndarray:
    """Walsh (sequency) ordering of Hadamard rows.

    Sequency = number of sign changes in a row. Sorting the natural (Hadamard)
    order by sign changes gives the Walsh order, which is the frequency-ordered
    analogue of the DFT.
    """
    H = _hadamard_matrix(n)
    changes = np.sum(np.abs(np.diff(np.sign(H), axis=1)) > 0, axis=1)
    return np.argsort(changes, kind="stable")


def _walsh_hadamard_op(op_id: str, label: str, sequency: bool):
    def fn(img, keep_ratio=1.0, show_spectrum=True, ctx=None):
        g = luma(img) if img.ndim == 3 else img
        h, w = g.shape
        n = 1 << int(np.ceil(np.log2(max(h, w))))
        p = np.zeros((n, n), np.float32)
        p[:h, :w] = g

        H = _hadamard_matrix(n)
        if sequency:
            H = H[_sequency_order(n)]
        # Orthonormal scaling so the inverse is the same matrix.
        Hn = H / np.sqrt(n)
        C = Hn @ p @ Hn.T

        if keep_ratio < 1.0:
            flat = np.abs(C).ravel()
            k = max(1, int(len(flat) * keep_ratio))
            thresh = np.partition(flat, -k)[-k]
            C = np.where(np.abs(C) >= thresh, C, 0.0)

        rec = (Hn.T @ C @ Hn)[:h, :w]
        err = float(np.max(np.abs(rec - g))) if keep_ratio >= 1.0 else None

        res = OpResult(
            image=clamp01(rec),
            data={
                "transform_size": int(n),
                "keep_ratio": round(float(keep_ratio), 4),
                "roundtrip_max_error": round(err, 9) if err is not None else None,
                "energy_retained": round(
                    float(np.sum(C**2) / max(np.sum((Hn @ p @ Hn.T) ** 2), EPS)), 5
                ),
                "ordering": "sequency (Walsh)" if sequency else "natural (Hadamard)",
            },
        )
        if show_spectrum:
            res.overlays["spectrum"] = colorize(
                normalize01(np.log1p(np.abs(C))), "turbo"
            )
        return res

    register(
        op_id, label, "freq",
        params=[
            FloatParam("keep_ratio", min=0.005, max=1.0, default=1.0, step=0.005,
                       help="1.0 performs a lossless round-trip; lower values "
                            "demonstrate coefficient truncation."),
            BoolParam("show_spectrum", default=True),
        ],
        summary=(
            "Walsh transform: Hadamard basis reordered by SEQUENCY (number of "
            "sign changes), making it the frequency-ordered analogue of the DFT."
            if sequency else
            "Hadamard transform: projection onto a basis of +1/-1 square waves, "
            "computed by the Sylvester construction. Requires no multiplications "
            "at all, only additions and subtractions."
        ),
        formula_latex=(
            r"W = \frac{1}{N} H_{seq}\, f\, H_{seq}^T" if sequency else
            r"H_{2n} = \begin{bmatrix}H_n & H_n\\ H_n & -H_n\end{bmatrix},\quad "
            r"T = \frac{1}{N}H f H^T"
        ),
        when_to_use="A multiplication-free orthogonal transform. Historically "
        "important for hardware implementations, and a clean illustration that "
        "the DFT is only one choice of basis among many.",
        syllabus=True,
        cost="medium",
        references=["Gonzalez & Woods, Ch. 7 (Walsh-Hadamard transforms)"],
    )(fn)


_walsh_hadamard_op("freq.hadamard", "Hadamard transform", False)
_walsh_hadamard_op("freq.walsh", "Walsh transform", True)


@register(
    "freq.dwt",
    "Wavelet transform",
    "freq",
    params=[
        EnumParam("wavelet", options=enum_opts("haar", "db2", "db4", "db8", "sym4",
                                               "coif1", "bior2.2"), default="haar"),
        IntParam("levels", min=1, max=6, default=2),
        FloatParam("gain_ll", min=0.0, max=3.0, default=1.0, step=0.05),
        FloatParam("gain_lh", min=0.0, max=5.0, default=1.0, step=0.05),
        FloatParam("gain_hl", min=0.0, max=5.0, default=1.0, step=0.05),
        FloatParam("gain_hh", min=0.0, max=5.0, default=1.0, step=0.05),
        BoolParam("show_subbands", default=True),
    ],
    summary="Multi-level discrete wavelet decomposition with independent gain on "
    "each subband, then reconstruction.",
    formula_latex=r"f = \sum_{j,k} c_{j,k}\,\phi_{j,k} + "
    r"\sum_{j,k} d_{j,k}\,\psi_{j,k}",
    when_to_use="Scale-specific editing: boost HH to sharpen fine texture only, "
    "or cut it to denoise. Unlike the DFT, the wavelet transform is localised in "
    "space as well as frequency.",
    tags=["beyond-syllabus"],
    cost="fast",
    references=["Mallat, A Theory for Multiresolution Signal Decomposition, 1989"],
)
def dwt(img, wavelet="haar", levels=2, gain_ll=1.0, gain_lh=1.0, gain_hl=1.0,
        gain_hh=1.0, show_subbands=True, ctx=None):
    rgb, alpha = split_alpha(img)
    work = luma(rgb) if rgb.ndim == 3 else rgb

    coeffs = pywt.wavedec2(work, wavelet, level=int(levels))
    new = [coeffs[0] * gain_ll]
    for (cH, cV, cD) in coeffs[1:]:
        new.append((cH * gain_lh, cV * gain_hl, cD * gain_hh))
    rec = pywt.waverec2(new, wavelet)[: work.shape[0], : work.shape[1]]

    err = None
    if all(abs(g - 1.0) < 1e-9 for g in (gain_ll, gain_lh, gain_hl, gain_hh)):
        err = float(np.max(np.abs(rec - work)))

    if rgb.ndim == 3:
        gain = (clamp01(rec) + EPS) / (work + EPS)
        out = clamp01(rgb * gain[:, :, None])
    else:
        out = clamp01(rec)

    res = OpResult(
        image=join_alpha(out, alpha),
        data={
            "levels": int(levels),
            "wavelet": wavelet,
            "roundtrip_max_error": round(err, 9) if err is not None else None,
            "subband_energy": [
                round(float(np.sum(np.asarray(c) ** 2)), 4)
                for c in ([coeffs[0]] + [x for t in coeffs[1:] for x in t])
            ],
        },
    )
    if show_subbands:
        arr, _ = pywt.coeffs_to_array(coeffs)
        res.overlays["subbands"] = colorize(
            normalize01(np.log1p(np.abs(arr))), "viridis"
        )
    return res
