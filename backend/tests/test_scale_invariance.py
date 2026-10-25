"""Scale-aware parameters. PLAN.md D3 and Phase 1 acceptance.

An export runs the same recipe at native resolution. If a kernel size or blur
sigma does not scale with the image, the exported file looks different from the
preview the user approved -- the most common bug in this class of application.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from skimage.metrics import structural_similarity

from app.core import registry
from app.core.colorspace import luma
from app.core.imageio import make_proxy
from app.core.params import FloatParam, IntParam, OddIntParam
from app.core.pipeline import OpInvocation, execute

# Recipes exercising the scale-sensitive families, with parameters chosen to
# have a clearly visible effect (an invisible effect would pass trivially).
# (name, ops, min_ssim). Linear and frequency-domain operators commute with
# resampling and must agree closely. Nonlinear RANK and MORPHOLOGICAL
# operators do not: a median or an erosion of a downscaled image is not the
# downscale of the median or erosion, whatever the kernel size, because order
# statistics do not commute with averaging. For those the test asserts that
# proxy and full-resolution stay visually equivalent rather than identical.
LINEAR_MIN = 0.97
NONLINEAR_MIN = 0.88

RECIPES: list[tuple[str, list[tuple[str, dict]], float]] = [
    ("gaussian", [("smooth.gaussian", {"sigma": 3.0})], LINEAR_MIN),
    ("box", [("smooth.box", {"k": 9})], LINEAR_MIN),
    ("unsharp", [("sharpen.unsharp", {"radius": 3.0, "amount": 1.0})], LINEAR_MIN),
    ("clarity", [("sharpen.clarity", {"amount": 0.6, "radius": 40.0})], LINEAR_MIN),
    ("shadows_highlights", [("tone.shadows_highlights",
                             {"shadows": 0.5, "highlights": -0.3, "radius": 40.0})],
     LINEAR_MIN),
    ("lowpass", [("freq.lowpass", {"kind": "gaussian", "cutoff": 0.15})], LINEAR_MIN),
    ("highpass", [("freq.highpass",
                   {"kind": "butterworth", "cutoff": 0.1, "order": 2})], LINEAR_MIN),
    ("homomorphic", [("freq.homomorphic", {"cutoff": 0.08})], LINEAR_MIN),
    ("bilateral", [("smooth.bilateral",
                    {"d": 9, "sigma_color": 0.15, "sigma_space": 9.0})], 0.95),
    ("clahe", [("tone.clahe", {"clip": 3.0, "tiles_x": 8, "tiles_y": 8})], 0.95),
    # Nonlinear: see the note above.
    ("median", [("smooth.median", {"k": 7})], NONLINEAR_MIN),
    ("morph_open", [("morph.open", {"size": 9})], NONLINEAR_MIN),
    ("tophat", [("morph.tophat", {"size": 21})], NONLINEAR_MIN),
    ("bg_divide", [("morph.bg_divide", {"se_size": 61})], NONLINEAR_MIN),
    ("combo", [
        ("smooth.bilateral", {"d": 7, "sigma_color": 0.1, "sigma_space": 7.0}),
        ("tone.clahe", {"clip": 2.5, "tiles_x": 8, "tiles_y": 8}),
        ("sharpen.unsharp", {"radius": 2.0, "amount": 0.8}),
    ], 0.95),
]


def _render(img, ops, scale_factor):
    invocations = [OpInvocation(op_id, dict(params)) for op_id, params in ops]
    return execute(img, invocations, image_id="", use_cache=False,
                   scale_factor=scale_factor, ctx={"original": img}).image


@pytest.mark.parametrize("name,ops,min_ssim", RECIPES, ids=[r[0] for r in RECIPES])
def test_recipe_agrees_across_resolutions(plate, name, ops, min_ssim):
    """Render at proxy and at 2.5x, downscale the big one, and compare."""
    big = plate(0, h=640, w=640)
    proxy, _ = make_proxy(big, 256)
    factor = max(big.shape[:2]) / max(proxy.shape[:2])

    small_out = _render(proxy, ops, 1.0)
    big_out = _render(big, ops, factor)

    # Bring the full-resolution result back to proxy size to compare.
    big_down = cv2.resize(big_out, (small_out.shape[1], small_out.shape[0]),
                          interpolation=cv2.INTER_AREA)

    a, b = luma(small_out), luma(big_down)
    ssim = structural_similarity(a, b, data_range=1.0)
    assert ssim > min_ssim, (
        f"{name}: proxy and full-resolution renders disagree "
        f"(SSIM {ssim:.4f} < {min_ssim}), so an export would not match the "
        f"preview the user approved"
    )


def test_scale_aware_params_actually_change(plate):
    """A sanity check on the test above: the rescaling must do something.

    Without this, a recipe whose parameters were silently ignored would pass
    the agreement test for the wrong reason.
    """
    big = plate(0, h=640, w=640)
    unscaled = _render(big, [("smooth.gaussian", {"sigma": 3.0})], 1.0)
    scaled = _render(big, [("smooth.gaussian", {"sigma": 3.0})], 2.5)
    assert not np.allclose(unscaled, scaled, atol=1e-3), \
        "scale_factor had no effect, so the agreement test proves nothing"


def test_every_length_param_is_marked_scale_aware():
    """Catch a new operator that forgets the flag.

    Heuristic but effective: a parameter whose name reads like a length, a
    radius or a kernel size, and whose range extends well past a few pixels,
    almost certainly needs to scale with the image.
    """
    LENGTH_HINTS = (
        "radius", "sigma_space", "se_size", "kernel", "ksize", "window",
        "block", "patch", "min_dist", "psf_size", "template", "search",
        "spatial_radius", "cell", "max_size", "marker_size", "unit",
    )
    # Parameters that look like lengths but genuinely are not, with the reason.
    EXEMPT = {
        ("freq.dct_compress", "keep_ratio"),
        ("seg.slic", "n_segments"),        # a count, not a length
        ("feat.hog", "block"),             # measured in cells, not pixels
        ("seg.kmeans", "k"),               # a cluster count
        # A derivative aperture, not a spatial extent: scaling it changes the
        # Sobel kernel's normalisation and so the response magnitude, rather
        # than the size of the neighbourhood being measured.
        ("feat.harris", "ksize"),
        # A distance in COLOUR space, unrelated to pixel dimensions.
        ("seg.meanshift", "color_radius"),
    }
    problems = []
    for spec in registry.all_ops():
        for p in spec.params:
            if not isinstance(p, (FloatParam, IntParam, OddIntParam)):
                continue
            if (spec.id, p.name) in EXEMPT:
                continue
            looks_like_length = any(h in p.name for h in LENGTH_HINTS)
            big_range = float(p.max) >= 15
            if looks_like_length and big_range and not p.scale_aware:
                problems.append(f"{spec.id}.{p.name} (max={p.max})")
    assert not problems, (
        "these look like pixel lengths but are not marked scale_aware, so a "
        "full-resolution export will not match the preview:\n  "
        + "\n  ".join(problems)
    )


def test_odd_int_rescale_stays_odd_and_in_range():
    for spec in registry.all_ops():
        for p in spec.params:
            if isinstance(p, OddIntParam) and p.scale_aware:
                for factor in (0.3, 1.0, 1.7, 2.5, 4.0, 8.0):
                    v = p.rescale(p.default, factor)
                    assert v % 2 == 1, f"{spec.id}.{p.name} -> {v} is even"
                    assert p.min <= v <= p.max, f"{spec.id}.{p.name} -> {v} out of range"
