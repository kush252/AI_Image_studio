"""Core engine: params, registry, pipeline, cache, imageio. PLAN.md Phase 1/10."""
from __future__ import annotations

import numpy as np
import pytest

from app.core import imageio as iio
from app.core import registry
from app.core.cache import ArrayCache, canonical_json, prefix_key
from app.core.errors import ParamInvalid, ParamOutOfRange, UnknownOperation
from app.core.params import (
    EnumParam,
    FloatParam,
    IntParam,
    MatrixParam,
    OddIntParam,
    enum_opts,
    rescale_params,
    validate_params,
)
from app.core.pipeline import OpInvocation, execute


# ---------------------------------------------------------------- params

def test_float_param_clamps_and_rejects():
    p = FloatParam("x", min=0.0, max=1.0, default=0.5)
    assert p.coerce(0.25) == 0.25
    with pytest.raises(ParamOutOfRange):
        p.coerce(1.5)
    with pytest.raises(ParamInvalid):
        p.coerce("not a number")
    with pytest.raises(ParamInvalid):
        p.coerce(float("nan"))


def test_odd_int_param_always_odd():
    p = OddIntParam("k", min=3, max=31, default=5)
    assert p.coerce(4) == 5
    assert p.coerce(8) == 9
    assert p.coerce(3) == 3
    # Rescaling for full-res export must preserve oddness, or OpenCV raises.
    p2 = OddIntParam("k", min=3, max=99, default=5, scale_aware=True)
    for factor in (1.5, 2.0, 2.7, 3.3):
        assert p2.rescale(5, factor) % 2 == 1


def test_enum_param_rejects_unknown():
    p = EnumParam("mode", options=enum_opts("a", "b"), default="a")
    assert p.coerce("b") == "b"
    with pytest.raises(ParamInvalid):
        p.coerce("c")


def test_matrix_param_requires_odd_dimensions():
    p = MatrixParam("m", rows=3, cols=3, default=((0, 0, 0), (0, 1, 0), (0, 0, 0)))
    assert p.coerce([[1, 0, 1], [0, 1, 0], [1, 0, 1]])[1][1] == 1
    with pytest.raises(ParamInvalid):
        p.coerce([[1, 0], [0, 1]])  # even: no centre tap
    with pytest.raises(ParamInvalid):
        p.coerce([[1, 0, 1], [0, 1]])  # ragged


def test_validate_params_fills_defaults_and_rejects_unknown():
    specs = [FloatParam("a", default=0.3), IntParam("b", min=0, max=9, default=2)]
    out = validate_params(specs, {"a": 0.8})
    assert out == {"a": 0.8, "b": 2}
    with pytest.raises(ParamInvalid):
        validate_params(specs, {"nope": 1})


def test_rescale_only_touches_scale_aware():
    specs = [
        FloatParam("sigma", min=0, max=50, default=2, scale_aware=True),
        FloatParam("amount", min=0, max=5, default=1),
    ]
    out = rescale_params(specs, {"sigma": 2.0, "amount": 1.0}, 3.0)
    assert out["sigma"] == pytest.approx(6.0)
    assert out["amount"] == 1.0, "amount is not a length; it must not scale"


# -------------------------------------------------------------- registry

def test_registry_loaded_and_consistent():
    ops = registry.all_ops()
    assert len(ops) > 200, f"expected the full catalogue, got {len(ops)}"
    cat = registry.catalogue()
    assert cat["total"] == len(ops)
    assert cat["syllabus_count"] > 50
    assert sum(c["count"] for c in cat["categories"]) == len(ops)


def test_registry_unknown_op_raises():
    with pytest.raises(UnknownOperation):
        registry.get("nope.missing")


def test_schema_completeness():
    """Every operator must document itself. PLAN.md Phase 10 acceptance.

    This is what keeps Learn mode honest as the catalogue grows: an operator
    with no summary or formula would appear in the UI with an empty Explain
    drawer.
    """
    problems: list[str] = []
    for spec in registry.all_ops():
        if not spec.summary:
            problems.append(f"{spec.id}: no summary")
        if not spec.when_to_use:
            problems.append(f"{spec.id}: no when_to_use")
        if not spec.formula_latex:
            problems.append(f"{spec.id}: no formula")
        if spec.category not in registry.CATEGORIES:
            problems.append(f"{spec.id}: unknown category")
        for p in spec.params:
            if isinstance(p, (FloatParam, IntParam)) and p.min >= p.max:
                problems.append(f"{spec.id}.{p.name}: min >= max")
            if p.default is None and p.kind in ("float", "int", "odd_int"):
                problems.append(f"{spec.id}.{p.name}: numeric param with no default")
    assert not problems, "schema problems:\n" + "\n".join(problems[:40])


def test_every_op_serialises_to_json():
    import json

    payload = json.dumps(registry.catalogue())
    assert len(payload) > 10_000


# -------------------------------------------------------------- pipeline

def test_pipeline_runs_and_preserves_dtype(clean):
    ops = [
        OpInvocation("tone.gamma", {"gamma": 0.8}),
        OpInvocation("tone.clahe", {"clip": 2.0}),
    ]
    res = execute(clean, ops, image_id="t")
    assert res.image.dtype == np.float32
    assert res.image.shape == clean.shape
    assert 0.0 <= float(res.image.min()) and float(res.image.max()) <= 1.0
    assert not res.notes


def test_disabled_op_is_a_noop(clean):
    a = execute(clean, [OpInvocation("tone.gamma", {"gamma": 0.4}, enabled=False)],
                image_id="t2")
    assert np.allclose(a.image, clean)


def test_prefix_cache_reuses_work(clean):
    ops = [
        OpInvocation("smooth.gaussian", {"sigma": 2.0}),
        OpInvocation("tone.clahe", {"clip": 2.0}),
        OpInvocation("sharpen.unsharp", {"amount": 0.5}),
    ]
    first = execute(clean, ops, image_id="cache-test")
    assert first.cache_hits == 0
    second = execute(clean, ops, image_id="cache-test")
    assert second.cache_hits == len(ops), "the whole prefix should be cached"
    assert np.allclose(first.image, second.image)

    # Changing only the LAST operator must reuse the first two stages.
    ops[-1] = OpInvocation("sharpen.unsharp", {"amount": 0.9})
    third = execute(clean, ops, image_id="cache-test")
    assert third.cache_hits == 2


def test_failing_operator_does_not_kill_the_stack(clean):
    """An operator that raises should be reported, not abort the render."""
    from app.core.registry import register

    @register("test.explode", "Explodes", "tone",
              summary="Raises on purpose.", formula_latex="x",
              when_to_use="Testing only.")
    def _explode(img, ctx=None):
        raise RuntimeError("boom")

    res = execute(clean, [
        OpInvocation("tone.gamma", {"gamma": 0.9}),
        OpInvocation("test.explode", {}),
        OpInvocation("tone.brightness", {"offset": 0.05}),
    ], image_id="t3")
    assert any("boom" in n for n in res.notes)
    # The operators either side of the failure still applied.
    assert not np.allclose(res.image, clean)


def test_opacity_blends_toward_input(clean):
    full = execute(clean, [OpInvocation("tone.negative", {})], image_id="t4").image
    half = execute(clean, [OpInvocation("tone.negative", {}, opacity=0.5)],
                   image_id="t5").image
    assert np.allclose(half, (clean + full) / 2.0, atol=2e-3)


def test_mask_confines_effect_exactly(clean):
    """Phase 9 acceptance: zero effect where the mask is zero."""
    mask = {"kind": "rect", "rect": [0.0, 0.0, 0.5, 1.0], "feather": 0.0}
    res = execute(clean, [OpInvocation("tone.negative", {}, mask=mask)],
                  image_id="t6").image
    w = clean.shape[1]
    right = slice(int(w * 0.62), w)
    assert np.allclose(res[:, right], clean[:, right], atol=1e-6), \
        "outside the mask the image must be bit-identical"
    left = slice(0, int(w * 0.38))
    assert not np.allclose(res[:, left], clean[:, left])


# ----------------------------------------------------------------- cache

def test_canonical_json_is_order_independent():
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


def test_prefix_key_changes_with_params():
    a = prefix_key("img", "proxy", [{"op": "tone.gamma", "p": {"gamma": 1.0}}])
    b = prefix_key("img", "proxy", [{"op": "tone.gamma", "p": {"gamma": 1.1}}])
    assert a != b


def test_array_cache_evicts_by_bytes(tmp_path):
    cache = ArrayCache(max_bytes=4 * 1024, spill_dir=tmp_path)
    arrays = {f"k{i}": np.zeros(512, np.float32) for i in range(8)}  # 2 KB each
    for k, v in arrays.items():
        cache.put(k, v)
    stats = cache.stats()
    assert stats["bytes"] <= cache.max_bytes
    # Evicted entries spill to disk rather than vanishing, so they come back.
    got = cache.get("k0")
    assert got is not None and got.shape == (512,)


# --------------------------------------------------------------- imageio

@pytest.mark.parametrize("fmt", ["png", "jpeg", "webp", "tiff"])
def test_encode_decode_roundtrip(clean, fmt):
    payload = iio.encode(clean, fmt=fmt, quality=98)
    decoded = iio.decode(payload)
    assert decoded.width == clean.shape[1]
    assert decoded.height == clean.shape[0]
    assert decoded.data.dtype == np.float32
    if fmt in ("png", "tiff"):
        assert np.allclose(decoded.data, clean, atol=1.2 / 255), \
            f"{fmt} should be lossless to 8-bit precision"


def test_16_bit_png_roundtrip_is_more_precise(clean):
    p8 = iio.decode(iio.encode(clean, "png", bit_depth=8)).data
    p16 = iio.decode(iio.encode(clean, "png", bit_depth=16)).data
    assert np.abs(p16 - clean).max() < np.abs(p8 - clean).max()


def test_grayscale_and_alpha_survive(clean):
    gray = clean[:, :, 0]
    assert iio.decode(iio.encode(gray, "png")).channels == 1
    rgba = np.dstack([clean, np.full(clean.shape[:2], 0.5, np.float32)])
    out = iio.decode(iio.encode(rgba, "png"))
    assert out.channels == 4 and out.had_alpha


def test_decode_rejects_oversized(monkeypatch, clean):
    from app.config import settings
    from app.core.errors import ImageTooLarge

    payload = iio.encode(clean, "png")
    monkeypatch.setattr(settings, "max_megapixels", 0.001)
    with pytest.raises(ImageTooLarge):
        iio.decode(payload)


def test_decode_rejects_non_image():
    from app.core.errors import UnsupportedFormat

    with pytest.raises(UnsupportedFormat):
        iio.decode(b"this is definitely not an image file")


def test_make_proxy_respects_long_edge(clean):
    proxy, scale = iio.make_proxy(clean, 128)
    assert max(proxy.shape[:2]) == 128
    assert scale == pytest.approx(128 / max(clean.shape[:2]))
    # An image already smaller than the target is returned untouched.
    same, s2 = iio.make_proxy(clean, 10_000)
    assert s2 == 1.0 and same is clean
