"""Augmentation operators. Catalogue Section 11.

Every operator takes a `seed` and parameter RANGES rather than fixed values, so
a recipe is a distribution that can be sampled reproducibly. Degradation
operators model real sensor and codec physics rather than just adding noise.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core.colorspace import (
    from_hsv,
    join_alpha,
    linear_to_srgb,
    luma,
    split_alpha,
    srgb_to_linear,
    to_hsv,
)
from ..core.params import (
    BoolParam,
    EnumParam,
    FloatParam,
    ImageRefParam,
    IntParam,
    OddIntParam,
    RangeParam,
    SeedParam,
    enum_opts,
)
from ..core.registry import OpResult, register
from ._util import EPS, border_flag, clamp01, interp_flag, resolve_ref_image

P_PARAM = FloatParam("p", min=0.0, max=1.0, default=1.0, step=0.05,
                     help="Probability this stage is applied at all.")


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) & 0x7FFFFFFF)


def _skip(rng: np.random.Generator, p: float) -> bool:
    return p < 1.0 and rng.random() >= p


def _uniform(rng, rng_pair) -> float:
    lo, hi = float(rng_pair[0]), float(rng_pair[1])
    return float(rng.uniform(min(lo, hi), max(lo, hi)))


def _aug(op_id, label, fn, params, doc, formula, when, tags=None, cost="fast",
         requires=None, domain="srgb", refs=None):
    def wrapper(img, seed=0, p=1.0, ctx=None, **kw):
        rng = _rng(seed)
        if _skip(rng, p):
            return OpResult(image=img, data={"applied": False})
        out = fn(img, rng, kw, ctx or {})
        if isinstance(out, OpResult):
            out.data.setdefault("applied", True)
            return out
        if isinstance(out, tuple):
            image, data = out
            data = dict(data or {})
            data["applied"] = True
            return OpResult(image=image, data=data)
        return OpResult(image=out, data={"applied": True})

    register(
        op_id, label, "aug",
        params=list(params) + [SeedParam("seed", default=0), P_PARAM],
        summary=doc, formula_latex=formula, when_to_use=when,
        tags=(tags or []) + ["augmentation"], cost=cost, requires=requires or {},
        domain=domain, references=refs or [],
    )(wrapper)
    return wrapper


# ------------------------------------------------------------------ geometric

def _rand_flip(img, rng, kw, ctx):
    did_h = rng.random() < float(kw.get("p_h", 0.5))
    did_v = rng.random() < float(kw.get("p_v", 0.0))
    out = img
    if did_h:
        out = out[:, ::-1]
    if did_v:
        out = out[::-1, :]
    return np.ascontiguousarray(out), {"horizontal": did_h, "vertical": did_v}


_aug("aug.flip", "Random flip", _rand_flip,
     [FloatParam("p_h", min=0.0, max=1.0, default=0.5, step=0.05),
      FloatParam("p_v", min=0.0, max=1.0, default=0.0, step=0.05)],
     "Randomly mirrors horizontally and/or vertically.",
     r"f'(x,y) = f(W-1-x, y)",
     "The cheapest and only lossless augmentation. Vertical flips are usually "
     "wrong for natural photographs (gravity is a real prior) but fine for "
     "satellite or microscopy imagery.")


def _rand_rotate(img, rng, kw, ctx):
    ang = _uniform(rng, kw.get("angle_range", (-15.0, 15.0)))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
    out = cv2.warpAffine(img, M, (w, h), flags=interp_flag(kw.get("interp", "bilinear")),
                         borderMode=border_flag(kw.get("border", "reflect")))
    return clamp01(out), {"angle": round(ang, 2)}


_aug("aug.rotate", "Random rotation", _rand_rotate,
     [RangeParam("angle_range", min=-180.0, max=180.0, default=(-15.0, 15.0), step=1.0),
      EnumParam("interp", options=enum_opts("bilinear", "bicubic", "nearest"),
                default="bilinear"),
      EnumParam("border", options=enum_opts("reflect", "replicate", "constant"),
                default="reflect")],
     "Rotates by a random angle within a range.",
     r"\theta \sim U(\theta_{min}, \theta_{max})",
     "Rotation invariance. Keep the range small for images with a meaningful "
     "up direction.")


def _rand_crop(img, rng, kw, ctx):
    h, w = img.shape[:2]
    scale = _uniform(rng, kw.get("scale_range", (0.7, 1.0)))
    ch, cw = max(8, int(h * scale)), max(8, int(w * scale))
    y = int(rng.integers(0, max(1, h - ch + 1)))
    x = int(rng.integers(0, max(1, w - cw + 1)))
    patch = img[y : y + ch, x : x + cw]
    if kw.get("resize_back", True):
        patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)
    return clamp01(patch), {"scale": round(scale, 3), "origin": [x, y]}


_aug("aug.random_crop", "Random crop", _rand_crop,
     [RangeParam("scale_range", min=0.1, max=1.0, default=(0.7, 1.0)),
      BoolParam("resize_back", default=True)],
     "Crops a random sub-region, optionally resizing back to the original size.",
     r"s \sim U(s_{min}, s_{max});\ (x,y) \sim U(\text{valid origins})",
     "Translation invariance and scale robustness. The single most effective "
     "augmentation for classification.")


def _scale_jitter(img, rng, kw, ctx):
    f = _uniform(rng, kw.get("scale_range", (0.8, 1.25)))
    h, w = img.shape[:2]
    nh, nw = max(2, int(h * f)), max(2, int(w * f))
    small = cv2.resize(img, (nw, nh),
                       interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
    if not kw.get("keep_size", True):
        return clamp01(small), {"factor": round(f, 3)}
    out = np.zeros_like(img)
    if f < 1:
        y, x = (h - nh) // 2, (w - nw) // 2
        out[y : y + nh, x : x + nw] = small
    else:
        y, x = (nh - h) // 2, (nw - w) // 2
        out = small[y : y + h, x : x + w]
    return clamp01(out), {"factor": round(f, 3)}


_aug("aug.scale_jitter", "Scale jitter", _scale_jitter,
     [RangeParam("scale_range", min=0.25, max=4.0, default=(0.8, 1.25)),
      BoolParam("keep_size", default=True)],
     "Resizes by a random factor.",
     r"s \sim U(s_{min}, s_{max})",
     "Scale robustness without changing the frame.")


def _rand_translate(img, rng, kw, ctx):
    h, w = img.shape[:2]
    tx = _uniform(rng, kw.get("x_range", (-0.1, 0.1))) * w
    ty = _uniform(rng, kw.get("y_range", (-0.1, 0.1))) * h
    M = np.array([[1, 0, tx], [0, 1, ty]], np.float32)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=border_flag(kw.get("border", "reflect")))
    return clamp01(out), {"tx": round(tx, 1), "ty": round(ty, 1)}


_aug("aug.translate", "Random translate", _rand_translate,
     [RangeParam("x_range", min=-0.5, max=0.5, default=(-0.1, 0.1)),
      RangeParam("y_range", min=-0.5, max=0.5, default=(-0.1, 0.1)),
      EnumParam("border", options=enum_opts("reflect", "replicate", "constant"),
                default="reflect")],
     "Shifts the image by a random offset.",
     r"t \sim U(t_{min}, t_{max})",
     "Translation invariance when random cropping is not wanted.")


def _rand_shear(img, rng, kw, ctx):
    sx = _uniform(rng, kw.get("x_range", (-0.1, 0.1)))
    sy = _uniform(rng, kw.get("y_range", (0.0, 0.0)))
    h, w = img.shape[:2]
    M = np.array([[1, sx, -sx * h / 2], [sy, 1, -sy * w / 2]], np.float32)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return clamp01(out), {"shx": round(sx, 4), "shy": round(sy, 4)}


_aug("aug.shear", "Random shear", _rand_shear,
     [RangeParam("x_range", min=-0.5, max=0.5, default=(-0.1, 0.1)),
      RangeParam("y_range", min=-0.5, max=0.5, default=(0.0, 0.0))],
     "Slants the image by a random amount.",
     r"sh \sim U(sh_{min}, sh_{max})",
     "Mild viewpoint variation; common in OCR pipelines.")


def _persp_jitter(img, rng, kw, ctx):
    h, w = img.shape[:2]
    m = float(kw.get("magnitude", 0.06))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-m, m, (4, 2)).astype(np.float32) * np.float32([w, h])
    H = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
    return clamp01(out), {"magnitude": m}


_aug("aug.perspective_jitter", "Perspective jitter", _persp_jitter,
     [FloatParam("magnitude", min=0.0, max=0.3, default=0.06, step=0.005)],
     "Randomly displaces the four corners, producing a random homography.",
     r"\Delta c_i \sim U(-m, m)",
     "Simulating viewpoint change -- far more realistic than affine jitter, "
     "since real cameras produce projective distortion.")


def _elastic(img, rng, kw, ctx):
    h, w = img.shape[:2]
    alpha = float(kw.get("alpha", 40.0))
    sigma = float(kw.get("sigma", 6.0))
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma)
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma)
    # Normalise so alpha is the displacement in pixels regardless of sigma.
    dx = dx / max(float(np.abs(dx).max()), EPS) * alpha
    dy = dy / max(float(np.abs(dy).max()), EPS) * alpha
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = cv2.remap(img, xx + dx, yy + dy, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101)
    return clamp01(out), {"alpha": alpha, "sigma": sigma}


_aug("aug.elastic", "Elastic deformation", _elastic,
     [FloatParam("alpha", min=1.0, max=200.0, default=40.0, step=1.0, scale_aware=True),
      FloatParam("sigma", min=1.0, max=40.0, default=6.0, step=0.5, scale_aware=True)],
     "Displaces pixels by a smoothed random vector field.",
     r"(x', y') = (x + \alpha\,\hat{d}_x,\ y + \alpha\,\hat{d}_y),\quad "
     r"d = G_\sigma * U(-1,1)",
     "Non-rigid variation. The standard augmentation for handwriting and "
     "biomedical images, where real specimens genuinely deform.",
     cost="fast",
     refs=["Simard, Steinkraus & Platt, Best Practices for CNNs Applied to "
           "Visual Document Analysis, 2003"])


def _grid_distort(img, rng, kw, ctx):
    h, w = img.shape[:2]
    n = int(kw.get("steps", 5))
    m = float(kw.get("magnitude", 0.1))
    xs = np.linspace(0, w, n + 1)
    ys = np.linspace(0, h, n + 1)
    xs_j = xs + rng.uniform(-m, m, n + 1) * (w / n)
    ys_j = ys + rng.uniform(-m, m, n + 1) * (h / n)
    xs_j[0], xs_j[-1], ys_j[0], ys_j[-1] = 0, w, 0, h
    mapx = np.interp(np.arange(w), xs, xs_j).astype(np.float32)
    mapy = np.interp(np.arange(h), ys, ys_j).astype(np.float32)
    MX = np.tile(mapx, (h, 1))
    MY = np.tile(mapy[:, None], (1, w))
    out = cv2.remap(img, MX, MY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return clamp01(out), {"steps": n, "magnitude": m}


_aug("aug.grid_distort", "Grid distortion", _grid_distort,
     [IntParam("steps", min=2, max=16, default=5),
      FloatParam("magnitude", min=0.0, max=0.5, default=0.1, step=0.01)],
     "Perturbs a coarse grid of control points and interpolates between them.",
     r"\text{grid}_i' = \text{grid}_i + U(-m, m)",
     "Smoother and more structured than elastic deformation; good for document "
     "and texture augmentation.")


def _optical_distort(img, rng, kw, ctx):
    k = _uniform(rng, kw.get("k_range", (-0.2, 0.2)))
    h, w = img.shape[:2]
    f = max(w, h) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (xx - w / 2.0) / f
    y = (yy - h / 2.0) / f
    r2 = x * x + y * y
    radial = 1.0 + k * r2
    mapx = (x * radial * f + w / 2.0).astype(np.float32)
    mapy = (y * radial * f + h / 2.0).astype(np.float32)
    out = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101)
    return clamp01(out), {"k1": round(k, 4)}


_aug("aug.optical_distort", "Optical distortion", _optical_distort,
     [RangeParam("k_range", min=-0.5, max=0.5, default=(-0.2, 0.2))],
     "Applies random barrel or pincushion radial distortion.",
     r"r' = r(1 + k_1 r^2)",
     "Simulating different lenses, so a model does not learn one camera's "
     "specific geometry.")


# --------------------------------------------------------------- photometric

def _jitter_op(op_id, label, apply_fn, default_range, doc, formula, when,
               range_min=-1.0, range_max=2.0, requires=None):
    def fn(img, rng, kw, ctx):
        v = _uniform(rng, kw.get("amount_range", default_range))
        return clamp01(apply_fn(img, v)), {"amount": round(v, 4)}

    _aug(op_id, label, fn,
         [RangeParam("amount_range", min=range_min, max=range_max,
                     default=default_range)],
         doc, formula, when, requires=requires)


_jitter_op("aug.brightness_jitter", "Brightness jitter",
           lambda img, v: img + v, (-0.2, 0.2),
           "Adds a random constant offset.", r"b \sim U(b_{min}, b_{max})",
           "Illumination robustness.", -1.0, 1.0)

_jitter_op("aug.contrast_jitter", "Contrast jitter",
           lambda img, v: (img - 0.5) * (1 + v) + 0.5, (-0.3, 0.3),
           "Scales deviation from mid-gray by a random factor.",
           r"s = (r - 0.5)(1 + a) + 0.5", "Contrast robustness.", -1.0, 2.0)

_jitter_op("aug.gamma_jitter", "Gamma jitter",
           lambda img, v: np.power(np.clip(img, 0, 1), np.exp(v)), (-0.4, 0.4),
           "Applies a random power-law curve, parameterised in log space so the "
           "distribution is symmetric about gamma = 1.",
           r"\gamma = e^{v},\ v \sim U(v_{min}, v_{max})",
           "Tonal robustness, and a better model of display differences than a "
           "linear brightness shift.", -2.0, 2.0)


def _sat_jitter(img, rng, kw, ctx):
    if img.ndim == 2:
        return img, {}
    v = _uniform(rng, kw.get("amount_range", (-0.3, 0.3)))
    rgb, alpha = split_alpha(img)
    hsv = to_hsv(rgb)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + v), 0, 1)
    return join_alpha(from_hsv(hsv), alpha), {"amount": round(v, 4)}


_aug("aug.saturation_jitter", "Saturation jitter", _sat_jitter,
     [RangeParam("amount_range", min=-1.0, max=2.0, default=(-0.3, 0.3))],
     "Scales saturation by a random factor.", r"S' = S(1 + a)",
     "Colour robustness.", requires={"channels": 3})


def _hue_jitter(img, rng, kw, ctx):
    if img.ndim == 2:
        return img, {}
    v = _uniform(rng, kw.get("degrees_range", (-15.0, 15.0)))
    rgb, alpha = split_alpha(img)
    hsv = to_hsv(rgb)
    hsv[:, :, 0] = np.mod(hsv[:, :, 0] + v, 360.0)
    return join_alpha(from_hsv(hsv), alpha), {"degrees": round(v, 2)}


_aug("aug.hue_jitter", "Hue jitter", _hue_jitter,
     [RangeParam("degrees_range", min=-180.0, max=180.0, default=(-15.0, 15.0),
                 step=1.0)],
     "Rotates hue by a random angle.", r"H' = (H + \Delta) \bmod 360",
     "Colour-invariance. Use small ranges: a large hue shift can change an "
     "object's identity (a green banana is not a banana).",
     requires={"channels": 3})


def _rand_clahe(img, rng, kw, ctx):
    clip = _uniform(rng, kw.get("clip_range", (1.0, 4.0)))
    tiles = int(rng.integers(4, 13))
    c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tiles, tiles))
    g = luma(img) if img.ndim == 3 else img
    eq = c.apply((np.clip(g, 0, 1) * 255).astype(np.uint8)).astype(np.float32) / 255.0
    if img.ndim == 2:
        return eq, {"clip": round(clip, 2), "tiles": tiles}
    gain = (eq + EPS) / (g + EPS)
    rgb, alpha = split_alpha(img)
    return join_alpha(clamp01(rgb * gain[:, :, None]), alpha), \
        {"clip": round(clip, 2), "tiles": tiles}


_aug("aug.random_clahe", "Random CLAHE", _rand_clahe,
     [RangeParam("clip_range", min=0.5, max=8.0, default=(1.0, 4.0))],
     "Applies CLAHE with a random clip limit and tile count.",
     r"\beta \sim U(\beta_{min}, \beta_{max})",
     "Local-contrast robustness; common in medical imaging pipelines.")


def _posterize_aug(img, rng, kw, ctx):
    bits = int(rng.integers(int(kw.get("bits_range", (3, 7))[0]),
                            int(kw.get("bits_range", (3, 7))[1]) + 1))
    levels = 2**bits
    out = np.floor(np.clip(img, 0, 1) * (levels - 1) + 0.5) / (levels - 1)
    return clamp01(out), {"bits": bits}


_aug("aug.posterize", "Posterize", _posterize_aug,
     [RangeParam("bits_range", min=1.0, max=8.0, default=(3.0, 7.0), step=1.0)],
     "Reduces bit depth to a random number of bits.",
     r"s = \frac{\lfloor r(2^b - 1) + 0.5\rfloor}{2^b - 1}",
     "Robustness to quantisation and limited colour depth.")


def _solarize_aug(img, rng, kw, ctx):
    t = _uniform(rng, kw.get("threshold_range", (0.5, 1.0)))
    rgb, alpha = split_alpha(img)
    return join_alpha(np.where(rgb >= t, 1.0 - rgb, rgb).astype(np.float32), alpha), \
        {"threshold": round(t, 3)}


_aug("aug.solarize", "Solarize", _solarize_aug,
     [RangeParam("threshold_range", min=0.0, max=1.0, default=(0.5, 1.0))],
     "Inverts pixels above a random threshold.",
     r"s = 1 - r \text{ if } r \ge T",
     "A strong photometric perturbation used in AutoAugment-style policies.")


def _equalize_aug(img, rng, kw, ctx):
    g = luma(img) if img.ndim == 3 else img
    eq = cv2.equalizeHist((np.clip(g, 0, 1) * 255).astype(np.uint8)
                          ).astype(np.float32) / 255.0
    if img.ndim == 2:
        return eq, {}
    gain = (eq + EPS) / (g + EPS)
    rgb, alpha = split_alpha(img)
    return join_alpha(clamp01(rgb * gain[:, :, None]), alpha), {}


_aug("aug.equalize", "Equalize", _equalize_aug, [],
     "Applies global histogram equalisation.",
     r"s = \text{CDF}(r)", "A deterministic but strong tonal perturbation.")


def _channel_shuffle(img, rng, kw, ctx):
    if img.ndim == 2:
        return img, {}
    rgb, alpha = split_alpha(img)
    perm = rng.permutation(3)
    return join_alpha(rgb[:, :, perm].astype(np.float32), alpha), \
        {"permutation": perm.tolist()}


_aug("aug.channel_shuffle", "Channel shuffle", _channel_shuffle, [],
     "Randomly permutes the colour channels.",
     r"(R,G,B) \to \pi(R,G,B)",
     "Forces reliance on structure rather than absolute colour. Aggressive -- "
     "unsuitable where colour is semantically meaningful.",
     requires={"channels": 3})


def _to_gray_aug(img, rng, kw, ctx):
    if img.ndim == 2:
        return img, {}
    g = luma(img)
    rgb, alpha = split_alpha(img)
    return join_alpha(np.repeat(g[:, :, None], 3, axis=2), alpha), {}


_aug("aug.to_gray", "Random grayscale", _to_gray_aug, [],
     "Converts to grayscale (replicated to three channels).",
     r"Y = 0.2126R + 0.7152G + 0.0722B",
     "Prevents over-reliance on colour; typically applied with p around 0.1.",
     requires={"channels": 3})


def _invert_aug(img, rng, kw, ctx):
    rgb, alpha = split_alpha(img)
    return join_alpha(clamp01(1.0 - rgb), alpha), {}


_aug("aug.invert", "Invert", _invert_aug, [],
     "Inverts intensity.", r"s = 1 - r",
     "A very strong perturbation; rarely appropriate above p = 0.05.")


# --------------------------------------------------------------- degradation

def _gauss_noise(img, rng, kw, ctx):
    sigma = _uniform(rng, kw.get("sigma_range", (0.01, 0.06)))
    per_ch = bool(kw.get("per_channel", True))
    shape = img.shape if per_ch else img.shape[:2]
    n = rng.normal(0.0, sigma, shape).astype(np.float32)
    if not per_ch and img.ndim == 3:
        n = n[:, :, None]
    return clamp01(img + n), {"sigma": round(sigma, 5),
                              "sigma_8bit": round(sigma * 255, 2)}


_aug("aug.gaussian_noise", "Gaussian noise", _gauss_noise,
     [RangeParam("sigma_range", min=0.0, max=0.3, default=(0.01, 0.06)),
      BoolParam("per_channel", default=True)],
     "Adds zero-mean Gaussian noise.",
     r"g = f + n,\quad n \sim \mathcal{N}(0, \sigma^2)",
     "The standard noise model, and the one the Advisor's sigma estimator is "
     "calibrated against.")


def _poisson_noise(img, rng, kw, ctx):
    scale = _uniform(rng, kw.get("scale_range", (20.0, 200.0)))
    lin = srgb_to_linear(np.clip(img, 0, 1))
    noisy = rng.poisson(lin * scale).astype(np.float32) / scale
    return clamp01(linear_to_srgb(noisy)), {"photon_scale": round(scale, 1)}


_aug("aug.poisson_noise", "Poisson / shot noise", _poisson_noise,
     [RangeParam("scale_range", min=5.0, max=1000.0, default=(20.0, 200.0), step=5.0)],
     "Samples each pixel from a Poisson distribution whose mean is the photon "
     "count, so the noise is signal-DEPENDENT -- brighter areas are noisier in "
     "absolute terms, quieter in relative terms.",
     r"g \sim \mathrm{Poisson}(\lambda f) / \lambda",
     "The physically correct noise model for photon-limited imaging. Applied in "
     "linear light, because that is where photon counts live.",
     cost="medium")


def _salt_pepper(img, rng, kw, ctx):
    amount = _uniform(rng, kw.get("amount_range", (0.005, 0.03)))
    ratio = float(kw.get("salt_ratio", 0.5))
    out = img.copy()
    h, w = img.shape[:2]
    n = int(amount * h * w)
    if n > 0:
        ys = rng.integers(0, h, n)
        xs = rng.integers(0, w, n)
        is_salt = rng.random(n) < ratio
        out[ys[is_salt], xs[is_salt]] = 1.0
        out[ys[~is_salt], xs[~is_salt]] = 0.0
    return clamp01(out), {"amount": round(amount, 5), "pixels": n}


_aug("aug.salt_pepper", "Salt-and-pepper", _salt_pepper,
     [RangeParam("amount_range", min=0.0, max=0.2, default=(0.005, 0.03)),
      FloatParam("salt_ratio", min=0.0, max=1.0, default=0.5, step=0.05)],
     "Sets a random fraction of pixels to pure black or pure white.",
     r"P(g = 1) = p_s,\ P(g = 0) = p_p",
     "Impulse noise from transmission errors and dead sensor pixels. Use this "
     "to demonstrate why the median filter beats the mean.")


def _speckle(img, rng, kw, ctx):
    sigma = _uniform(rng, kw.get("sigma_range", (0.05, 0.25)))
    n = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return clamp01(img + img * n), {"sigma": round(sigma, 4)}


_aug("aug.speckle", "Speckle (multiplicative)", _speckle,
     [RangeParam("sigma_range", min=0.0, max=1.0, default=(0.05, 0.25))],
     "Multiplicative noise proportional to the signal.",
     r"g = f + f \cdot n,\quad n \sim \mathcal{N}(0, \sigma^2)",
     "Coherent-imaging noise: ultrasound, SAR and laser imaging all produce "
     "speckle rather than additive noise.")


def _iso_sim(img, rng, kw, ctx):
    iso = _uniform(rng, kw.get("iso_range", (400.0, 6400.0)))
    # Map ISO to a photon scale and a read-noise floor; both empirical but
    # monotonic and physically shaped.
    photons = 12000.0 / max(iso, 50.0)
    read = 0.002 + iso / 6400.0 * 0.012
    lin = srgb_to_linear(np.clip(img, 0, 1))
    shot = rng.poisson(np.maximum(lin, 0) * photons) / photons
    out = shot + rng.normal(0.0, read, lin.shape)
    # Chroma noise exceeds luma noise at high ISO because of demosaicing.
    if img.ndim == 3:
        chroma = rng.normal(0.0, read * 1.6, (lin.shape[0], lin.shape[1], 1))
        out = out + chroma * np.array([1.0, -0.5, 1.0], np.float32)[None, None, :]
    return clamp01(linear_to_srgb(np.clip(out, 0, 1))), \
        {"iso": int(iso), "read_noise": round(read, 5)}


_aug("aug.iso_simulation", "Sensor / ISO simulation", _iso_sim,
     [RangeParam("iso_range", min=50.0, max=25600.0, default=(400.0, 6400.0),
                 step=50.0)],
     "Simulates a camera sensor at a given ISO: Poisson shot noise plus "
     "Gaussian read noise in linear light, with extra chroma noise, then "
     "re-encoded to sRGB.",
     r"g = \mathrm{OETF}\!\left(\frac{\mathrm{Poisson}(\lambda \cdot "
     r"\mathrm{EOTF}(f))}{\lambda} + \mathcal{N}(0,\sigma_r^2)\right)",
     "Far more realistic than adding Gaussian noise to the encoded image, "
     "because real noise is generated before gamma encoding and is "
     "signal-dependent. This is the generator the Advisor's noise metrics are "
     "validated against.",
     cost="medium")


def _jpeg_artifacts(img, rng, kw, ctx):
    q = int(rng.integers(int(kw.get("quality_range", (20, 70))[0]),
                         int(kw.get("quality_range", (20, 70))[1]) + 1))
    rgb, alpha = split_alpha(img)
    u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    bgr = u8[:, :, ::-1] if u8.ndim == 3 else u8
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return img, {}
    dec = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    if dec.ndim == 3:
        dec = dec[:, :, ::-1]
    out = dec.astype(np.float32) / 255.0
    return join_alpha(clamp01(out), alpha), {"quality": q,
                                             "bytes": int(enc.size)}


_aug("aug.jpeg_artifacts", "JPEG compression artifacts", _jpeg_artifacts,
     [RangeParam("quality_range", min=1.0, max=100.0, default=(20.0, 70.0),
                 step=1.0)],
     "Performs a real JPEG encode/decode round-trip at a random quality, "
     "producing genuine 8x8 blocking and chroma subsampling artifacts.",
     r"g = \mathrm{JPEG}^{-1}\big(\mathrm{JPEG}_q(f)\big)",
     "Essential when a model will see web images. Synthesising blockiness by "
     "hand never matches a real codec, so we use the codec.")


def _motion_blur(img, rng, kw, ctx):
    length = _uniform(rng, kw.get("length_range", (5.0, 20.0)))
    angle = _uniform(rng, kw.get("angle_range", (0.0, 180.0)))
    n = max(3, int(round(length)) | 1)
    k = np.zeros((n, n), np.float32)
    rad = np.deg2rad(angle)
    dx, dy = np.cos(rad), -np.sin(rad)
    c = n // 2
    for t in np.linspace(-length / 2, length / 2, int(length * 4) + 2):
        x, y = int(round(c + dx * t)), int(round(c + dy * t))
        if 0 <= x < n and 0 <= y < n:
            k[y, x] = 1.0
    k /= max(k.sum(), EPS)
    out = cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT_101)
    return clamp01(out), {"length": round(length, 1), "angle": round(angle, 1)}


_aug("aug.motion_blur", "Motion blur", _motion_blur,
     [RangeParam("length_range", min=3.0, max=60.0, default=(5.0, 20.0), step=1.0),
      RangeParam("angle_range", min=0.0, max=180.0, default=(0.0, 180.0), step=1.0)],
     "Convolves with a line kernel, modelling camera or subject motion during "
     "the exposure.",
     r"h(x,y) = \frac{1}{L}\,\mathbb{1}\big[(x,y) \in "
     r"\text{line}(L, \theta)\big]",
     "Motion robustness, and the generator used to validate the Advisor's "
     "motion-blur detector, which recovers this angle and length from the "
     "spectrum.",
     domain="linear",
     cost="medium")


def _defocus_blur(img, rng, kw, ctx):
    r = _uniform(rng, kw.get("radius_range", (2.0, 8.0)))
    n = int(np.ceil(r * 2)) | 1
    yy, xx = np.mgrid[:n, :n].astype(np.float32)
    c = (n - 1) / 2.0
    k = (((xx - c) ** 2 + (yy - c) ** 2) <= r * r).astype(np.float32)
    k /= max(k.sum(), EPS)
    out = cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT_101)
    return clamp01(out), {"radius": round(r, 2)}


_aug("aug.defocus_blur", "Defocus blur", _defocus_blur,
     [RangeParam("radius_range", min=1.0, max=30.0, default=(2.0, 8.0))],
     "Convolves with a disc, which is the true point spread function of a "
     "circular aperture out of focus -- not a Gaussian.",
     r"h(x,y) = \frac{1}{\pi r^2}\,\mathbb{1}\big[x^2 + y^2 \le r^2\big]",
     "Realistic focus error. A disc PSF has sharp edges and spectral nulls, "
     "which a Gaussian does not, and those nulls are what make the blur "
     "identifiable.",
     domain="linear",
     cost="medium")


def _zoom_blur(img, rng, kw, ctx):
    mx = _uniform(rng, kw.get("max_factor_range", (1.02, 1.12)))
    steps = int(kw.get("steps", 8))
    h, w = img.shape[:2]
    acc = np.zeros_like(img, np.float32)
    for f in np.linspace(1.0, mx, steps):
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0, f)
        acc += cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
    return clamp01(acc / steps), {"max_factor": round(mx, 4), "steps": steps}


_aug("aug.zoom_blur", "Zoom blur", _zoom_blur,
     [RangeParam("max_factor_range", min=1.0, max=1.5, default=(1.02, 1.12)),
      IntParam("steps", min=3, max=24, default=8)],
     "Averages progressively scaled copies, modelling a zoom during exposure.",
     r"g = \frac{1}{N}\sum_{i} f\big(s_i (x - x_0) + x_0\big)",
     "Radial motion robustness.", cost="medium", domain="linear")


def _haze(img, rng, kw, ctx):
    strength = _uniform(rng, kw.get("strength_range", (0.2, 0.6)))
    A = float(kw.get("airlight", 0.85))
    h, w = img.shape[:2]
    # Depth proxy: a vertical gradient plus smoothed noise, so the haze is
    # spatially varying rather than a flat wash.
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    noise = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0),
                             max(h, w) / 12.0)
    depth = np.clip(0.55 * (1 - yy) + 0.45 * noise, 0, 1)
    t = np.exp(-strength * 3.0 * depth)  # transmission
    if img.ndim == 3:
        t = t[:, :, None]
    out = img * t + A * (1 - t)
    return clamp01(out), {"strength": round(strength, 3), "airlight": A}


_aug("aug.haze", "Haze / fog", _haze,
     [RangeParam("strength_range", min=0.0, max=1.0, default=(0.2, 0.6)),
      FloatParam("airlight", min=0.4, max=1.0, default=0.85, step=0.01)],
     "Applies the atmospheric scattering model over a synthetic depth proxy.",
     r"I = J\,t + A(1 - t),\quad t = e^{-\beta d}",
     "Weather robustness, and the generator used to validate the Advisor's "
     "dark-channel haze detection -- the same physical model, run forwards.",
     cost="medium",
     refs=["He, Sun & Tang, Single Image Haze Removal Using Dark Channel "
           "Prior, CVPR 2009"])


def _streaks(img, rng, kw, ctx, snow=False):
    count = int(kw.get("count", 400))
    h, w = img.shape[:2]
    layer = np.zeros((h, w), np.float32)
    if snow:
        for _ in range(count):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            r = int(rng.integers(1, max(2, int(kw.get("size", 3)))))
            cv2.circle(layer, (x, y), r, float(rng.uniform(0.5, 1.0)), -1)
        layer = cv2.GaussianBlur(layer, (0, 0), 0.8)
    else:
        angle = float(kw.get("angle", 80.0))
        length = int(kw.get("length", 18))
        rad = np.deg2rad(angle)
        dx, dy = np.cos(rad) * length, -np.sin(rad) * length
        for _ in range(count):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            cv2.line(layer, (x, y), (int(x + dx), int(y + dy)),
                     float(rng.uniform(0.4, 0.9)), 1)
        layer = cv2.GaussianBlur(layer, (0, 0), 1.0)
    alpha = float(kw.get("opacity", 0.5))
    l3 = layer[:, :, None] if img.ndim == 3 else layer
    out = img * (1 - l3 * alpha) + l3 * alpha
    return clamp01(out), {"count": count}


_aug("aug.rain", "Rain streaks",
     lambda img, rng, kw, ctx: _streaks(img, rng, kw, ctx, snow=False),
     [IntParam("count", min=10, max=4000, default=400, scale_aware=True),
      IntParam("length", min=3, max=80, default=18, scale_aware=True),
      FloatParam("angle", min=0.0, max=180.0, default=80.0, step=1.0),
      FloatParam("opacity", min=0.0, max=1.0, default=0.5, step=0.05)],
     "Overlays semi-transparent motion-blurred streaks.",
     r"g = f(1 - \alpha s) + \alpha s",
     "Adverse-weather robustness for outdoor vision.", cost="fast")

_aug("aug.snow", "Snow",
     lambda img, rng, kw, ctx: _streaks(img, rng, kw, ctx, snow=True),
     [IntParam("count", min=10, max=4000, default=600, scale_aware=True),
      IntParam("size", min=1, max=12, default=3, scale_aware=True),
      FloatParam("opacity", min=0.0, max=1.0, default=0.6, step=0.05)],
     "Overlays bright blurred particles.",
     r"g = f(1 - \alpha s) + \alpha s",
     "Adverse-weather robustness.", cost="fast")


def _vignette_aug(img, rng, kw, ctx):
    amount = _uniform(rng, kw.get("amount_range", (-0.5, -0.15)))
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / np.sqrt(cx**2 + cy**2)
    gain = np.clip(1.0 + amount * r**2, 0, 2)
    out = img * (gain[:, :, None] if img.ndim == 3 else gain)
    return clamp01(out), {"amount": round(amount, 3)}


_aug("aug.vignette", "Vignette", _vignette_aug,
     [RangeParam("amount_range", min=-1.0, max=1.0, default=(-0.5, -0.15))],
     "Darkens or brightens the corners radially.",
     r"g = f(1 + a r^2)",
     "Lens falloff robustness.", domain="linear",
     cost="medium")


def _ca_aug(img, rng, kw, ctx):
    if img.ndim == 2:
        return img, {}
    mag = _uniform(rng, kw.get("magnitude_range", (0.001, 0.004)))
    h, w = img.shape[:2]
    rgb, alpha = split_alpha(img)

    def rescale(ch, s):
        M = np.array([[s, 0, (1 - s) * (w - 1) / 2.0],
                      [0, s, (1 - s) * (h - 1) / 2.0]], np.float32)
        return cv2.warpAffine(ch, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REFLECT_101)

    out = np.dstack([rescale(rgb[:, :, 0], 1.0 + mag), rgb[:, :, 1],
                     rescale(rgb[:, :, 2], 1.0 - mag)])
    return join_alpha(clamp01(out), alpha), {"magnitude": round(mag, 5)}


_aug("aug.chromatic_aberration", "Chromatic aberration", _ca_aug,
     [RangeParam("magnitude_range", min=0.0, max=0.02, default=(0.001, 0.004),
                 step=0.0005)],
     "Scales the red and blue channels apart, producing colour fringing that "
     "grows toward the corners.",
     r"I'_c(x) = I_c\big(s_c(x - x_0) + x_0\big)",
     "Lens-defect robustness.", requires={"channels": 3})


def _downscale_up(img, rng, kw, ctx):
    f = _uniform(rng, kw.get("factor_range", (0.25, 0.6)))
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(2, int(w * f)), max(2, int(h * f))),
                       interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h),
                    interpolation=interp_flag(kw.get("up_interp", "bilinear")))
    return clamp01(up), {"factor": round(f, 3)}


_aug("aug.downscale_upscale", "Resolution loss", _downscale_up,
     [RangeParam("factor_range", min=0.1, max=0.9, default=(0.25, 0.6)),
      EnumParam("up_interp", options=enum_opts("bilinear", "nearest", "bicubic"),
                default="bilinear")],
     "Downscales then upscales back, destroying high-frequency detail "
     "irreversibly.",
     r"g = \mathrm{up}\big(\mathrm{down}(f)\big)",
     "Low-resolution-source robustness. Also the honest demonstration that "
     "upsampling cannot recover what downsampling removed.")


def _cutout(img, rng, kw, ctx):
    n = int(kw.get("count", 4))
    size = _uniform(rng, kw.get("size_range", (0.05, 0.15)))
    h, w = img.shape[:2]
    out = img.copy()
    fill = float(kw.get("fill", 0.0))
    boxes = []
    for _ in range(n):
        bh, bw = int(size * h), int(size * w)
        y = int(rng.integers(0, max(1, h - bh)))
        x = int(rng.integers(0, max(1, w - bw)))
        out[y : y + bh, x : x + bw] = fill
        boxes.append([x, y, bw, bh])
    return clamp01(out), {"boxes": boxes}


_aug("aug.cutout", "Cutout", _cutout,
     [IntParam("count", min=1, max=32, default=4),
      RangeParam("size_range", min=0.01, max=0.5, default=(0.05, 0.15)),
      FloatParam("fill", min=0.0, max=1.0, default=0.0, step=0.05)],
     "Erases rectangular patches, filling them with a constant.",
     r"g(x) = c \text{ for } x \in \bigcup_i B_i",
     "Occlusion robustness; forces a model to use the whole object rather than "
     "one discriminative part.",
     refs=["DeVries & Taylor, Improved Regularization of CNNs with Cutout, 2017"])


def _random_erase(img, rng, kw, ctx):
    area = _uniform(rng, kw.get("area_range", (0.02, 0.2)))
    ar = _uniform(rng, kw.get("aspect_range", (0.3, 3.3)))
    h, w = img.shape[:2]
    target = area * h * w
    bh = int(np.sqrt(target / max(ar, EPS)))
    bw = int(np.sqrt(target * ar))
    bh, bw = max(1, min(bh, h - 1)), max(1, min(bw, w - 1))
    y = int(rng.integers(0, max(1, h - bh)))
    x = int(rng.integers(0, max(1, w - bw)))
    out = img.copy()
    patch_shape = (bh, bw, img.shape[2]) if img.ndim == 3 else (bh, bw)
    out[y : y + bh, x : x + bw] = rng.random(patch_shape).astype(np.float32)
    return clamp01(out), {"box": [x, y, bw, bh], "area": round(area, 4)}


_aug("aug.random_erase", "Random erasing", _random_erase,
     [RangeParam("area_range", min=0.01, max=0.5, default=(0.02, 0.2)),
      RangeParam("aspect_range", min=0.1, max=5.0, default=(0.3, 3.3))],
     "Replaces one random rectangle with random noise.",
     r"g(x) = U(0,1) \text{ for } x \in B",
     "Occlusion robustness. Noise fill rather than a constant prevents the "
     "model learning to detect the patch itself.",
     refs=["Zhong et al., Random Erasing Data Augmentation, 2017"])


def _grid_mask(img, rng, kw, ctx):
    d = int(kw.get("unit", 40))
    ratio = float(kw.get("ratio", 0.5))
    h, w = img.shape[:2]
    off_x, off_y = int(rng.integers(0, max(1, d))), int(rng.integers(0, max(1, d)))
    keep = int(d * ratio)
    mask = np.ones((h, w), np.float32)
    for y in range(-d + off_y, h, d):
        for x in range(-d + off_x, w, d):
            y0, x0 = max(0, y), max(0, x)
            mask[y0 : y + keep, x0 : x + keep] = 0.0
    m = mask[:, :, None] if img.ndim == 3 else mask
    return clamp01(img * m), {"unit": d, "ratio": ratio}


_aug("aug.grid_mask", "Grid mask", _grid_mask,
     [IntParam("unit", min=8, max=200, default=40, scale_aware=True),
      FloatParam("ratio", min=0.1, max=0.9, default=0.5, step=0.05)],
     "Masks a regular grid of squares with a random offset.",
     r"g = f \cdot \mathbb{1}\big[(x,y) \notin \text{grid}\big]",
     "A structured alternative to cutout: it guarantees that some part of every "
     "region survives, which random boxes do not.",
     refs=["Chen et al., GridMask Data Augmentation, 2020"])


def _mixup(img, rng, kw, ctx):
    other = resolve_ref_image(ctx, kw.get("reference"), img)
    lam = _uniform(rng, kw.get("alpha_range", (0.2, 0.5)))
    a = img if img.ndim == 3 else np.dstack([img] * 3)
    b = other if other.ndim == 3 else np.dstack([other] * 3)
    out = a[:, :, : b.shape[2]] * (1 - lam) + b[:, :, : a.shape[2]] * lam
    return clamp01(out), {"lambda": round(lam, 4)}


_aug("aug.mixup", "Mixup", _mixup,
     [ImageRefParam("reference"),
      RangeParam("alpha_range", min=0.0, max=1.0, default=(0.2, 0.5))],
     "Linearly blends two images.",
     r"\tilde{x} = \lambda x_i + (1-\lambda) x_j",
     "Strong regulariser; in supervised training the labels are mixed by the "
     "same lambda.",
     requires={"second_image": True, "second_image_param": "reference"},
     refs=["Zhang et al., mixup: Beyond Empirical Risk Minimization, ICLR 2018"])


def _cutmix(img, rng, kw, ctx):
    other = resolve_ref_image(ctx, kw.get("reference"), img)
    lam = _uniform(rng, kw.get("area_range", (0.1, 0.4)))
    h, w = img.shape[:2]
    bh, bw = int(h * np.sqrt(lam)), int(w * np.sqrt(lam))
    y = int(rng.integers(0, max(1, h - bh)))
    x = int(rng.integers(0, max(1, w - bw)))
    out = img.copy()
    src = other if other.ndim == img.ndim else (
        np.dstack([other] * img.shape[2]) if img.ndim == 3 else luma(other))
    out[y : y + bh, x : x + bw] = src[y : y + bh, x : x + bw]
    return clamp01(out), {"box": [x, y, bw, bh], "area": round(lam, 4)}


_aug("aug.cutmix", "CutMix", _cutmix,
     [ImageRefParam("reference"),
      RangeParam("area_range", min=0.02, max=0.7, default=(0.1, 0.4))],
     "Pastes a rectangular patch from a second image.",
     r"\tilde{x} = M \odot x_i + (1 - M) \odot x_j",
     "Combines mixup's regularisation with cutout's occlusion, while keeping "
     "every pixel a real pixel rather than a blend.",
     requires={"second_image": True, "second_image_param": "reference"},
     refs=["Yun et al., CutMix, ICCV 2019"])
