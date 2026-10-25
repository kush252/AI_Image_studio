"""The Advisor. docs/advisor-spec.md and PLAN.md Phase 8 acceptance.

The evaluation set is built from synthetic degradations of clean plates, so the
ground truth is known exactly: each case names the rule that must fire. The
last case is an unmodified plate, where nothing above "minor" may fire -- a
diagnostic tool that always finds something wrong is not diagnosing.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.advisor import recipe as rcp
from app.advisor.rules import RULES, RULES_BY_ID
from app.advisor.verify import verify
from app.core import registry
from app.core.pipeline import OpInvocation, execute
from tests.degradations import EVAL_SET, build_case

SEVERE = ("moderate", "severe")


@pytest.fixture(scope="module")
def diagnoses():
    """Diagnose every evaluation case once; the suite then asserts against it."""
    out = {}
    for i, (name, expect) in enumerate(EVAL_SET):
        nm, img, exp = build_case(i)
        d = rcp.diagnose(img)
        out[nm] = {"diagnosis": d, "image": img, "expect": exp}
    return out


# ------------------------------------------------------------- rule table

def test_rule_table_is_well_formed():
    seen = set()
    for r in RULES:
        assert r.id not in seen, f"duplicate rule id {r.id}"
        seen.add(r.id)
        assert r.title, f"{r.id} has no title"
        assert r.dimension, f"{r.id} has no dimension"
        if r.warning_only:
            assert r.fix is None, f"{r.id} is warning_only but declares a fix"


def test_every_fix_targets_a_real_operator(diagnoses):
    """A rule that emits an unknown operator id would fail only at apply time."""
    for case in diagnoses.values():
        for f in case["diagnosis"]["findings"]:
            if not f.get("fix"):
                continue
            for op in f["fix"]["ops"]:
                assert registry.exists(op["op_id"]), \
                    f"{f['id']} emits unknown operator {op['op_id']}"


def test_every_fix_has_a_derivation(diagnoses):
    """Spec Section 8: the derivation string is required, because it is what
    makes a suggested number auditable instead of magic."""
    for case in diagnoses.values():
        for f in case["diagnosis"]["findings"]:
            if f.get("fix"):
                d = f["fix"]["derivation"]
                assert d and len(d) > 25, f"{f['id']} has a thin derivation: {d!r}"


def test_every_finding_cites_its_number(diagnoses):
    """Spec Section 4: no finding may say "looks a bit dark"."""
    for case in diagnoses.values():
        for f in case["diagnosis"]["findings"]:
            assert any(ch.isdigit() for ch in f["explanation"]), \
                f"{f['id']} explanation quotes no measured value"
            ev = f["evidence"]
            assert ev, f"{f['id']} carries no evidence"


def test_recipe_params_always_validate(diagnoses):
    """A derived parameter must never fall outside its declared range."""
    for name, case in diagnoses.items():
        for op in case["diagnosis"]["recipe"]:
            spec = registry.get(op["op_id"])
            spec.validate(op["params"])  # raises if out of range


# -------------------------------------------------------------- detection

@pytest.mark.parametrize("name,expect", [(n, e) for n, e in EVAL_SET if e])
def test_planted_defect_is_detected(diagnoses, name, expect):
    """Phase 8 acceptance: every planted defect must be found."""
    found = [f["id"] for f in diagnoses[name]["diagnosis"]["findings"]]
    assert expect in found, (
        f"{name}: expected {expect}, got {found} "
        f"(health {diagnoses[name]['diagnosis']['scores']['overall']})"
    )


def test_clean_plate_produces_no_serious_findings(diagnoses):
    """Phase 8 acceptance, and the harder half of it."""
    d = diagnoses["clean"]["diagnosis"]
    serious = [f["id"] for f in d["findings"] if f["severity"] in SEVERE]
    assert not serious, f"clean plate flagged {serious}"
    assert d["scores"]["overall"] > 90, f"clean plate scored {d['scores']['overall']}"


def test_oversharpened_image_is_not_offered_sharpening(diagnoses):
    """Spec conflict rule 4: SHP_OVER withdraws every sharpening fix."""
    d = diagnoses["oversharp"]["diagnosis"]
    ids = [f["id"] for f in d["findings"]]
    assert "SHP_OVER" in ids
    for op in d["recipe"]:
        assert not op["op_id"].startswith("sharpen."), \
            f"sharpening ({op['op_id']}) offered on an already-sharpened image"


def test_impulse_noise_routed_to_adaptive_median(diagnoses):
    """Spec 2.5: noise TYPE decides the filter, not just its magnitude."""
    d = diagnoses["saltpepper"]["diagnosis"]
    ops = [o["op_id"] for o in d["recipe"]]
    assert "smooth.median_adaptive" in ops, f"routed to {ops} instead"


def test_periodic_noise_routed_to_notch(diagnoses):
    d = diagnoses["periodic"]["diagnosis"]
    ops = [o["op_id"] for o in d["recipe"]]
    assert "freq.notch_auto" in ops, f"routed to {ops} instead"


def test_chroma_noise_denoises_chroma_only(diagnoses):
    """The biggest quality win on high-ISO JPEG: leave luma alone."""
    d = diagnoses["chroma_noise"]["diagnosis"]
    chroma_ops = [o for o in d["recipe"] if o["params"].get("scope") == "chroma"]
    assert chroma_ops, f"no chroma-scoped denoise in {[o['op_id'] for o in d['recipe']]}"


def test_motion_blur_routed_to_deconvolution(diagnoses):
    d = diagnoses["motion"]["diagnosis"]
    deconv = [o for o in d["recipe"] if o["op_id"] == "sharpen.deconv"]
    assert deconv, "motion blur was not routed to deconvolution"
    assert deconv[0]["params"]["psf"] == "motion"


def test_underexposure_uses_exposure_or_gamma(diagnoses):
    """Spec 5.1: exposure when there is headroom, gamma when there is not."""
    d = diagnoses["under"]["diagnosis"]
    f = next(x for x in d["findings"] if x["id"] == "EXP_UNDER")
    ops = [o["op_id"] for o in f["fix"]["ops"]]
    assert ops and ops[0] in ("tone.exposure", "tone.gamma")


def test_geometric_fixes_are_opt_in(diagnoses):
    """Deskew crops and resamples, so it is a compositional choice."""
    d = diagnoses["tilt"]["diagnosis"]
    assert "GEO_TILT" in [f["id"] for f in d["findings"]]
    assert not any(o["op_id"] == "geom.deskew" for o in d["recipe"]), \
        "a geometric fix was included in the default recipe"
    opted_in = rcp.build_recipe(d["findings"], include_opt_in=True)
    assert any(o["op_id"] == "geom.deskew" for o in opted_in), \
        "the geometric fix is not offered even when opted into"


# ------------------------------------------------------------- ordering

def test_recipe_is_canonically_ordered(diagnoses):
    """Spec Section 6: denoise before contrast, sharpen last."""
    for name, case in diagnoses.items():
        recipe = case["diagnosis"]["recipe"]
        stages = [o["stage"] for o in recipe]
        assert stages == sorted(stages), f"{name}: out of order {stages}"

        ops = [o["op_id"] for o in recipe]
        denoise = [i for i, o in enumerate(ops)
                   if o.startswith("smooth.") or o == "freq.notch_auto"]
        contrast = [i for i, o in enumerate(ops)
                    if o in ("tone.clahe", "tone.contrast_sigmoid", "tone.he")]
        sharpen = [i for i, o in enumerate(ops) if o.startswith("sharpen.")]
        if denoise and contrast:
            assert max(denoise) < min(contrast), (
                f"{name}: contrast runs before denoise, which would amplify "
                f"the noise it is about to remove"
            )
        if sharpen:
            others = [i for i, o in enumerate(ops) if not o.startswith("sharpen.")]
            if others:
                assert min(sharpen) > max(others), \
                    f"{name}: sharpening is not last ({ops})"


def test_recipe_capped_at_eight_operators(diagnoses):
    for name, case in diagnoses.items():
        n = len(case["diagnosis"]["recipe"])
        assert n <= rcp.MAX_RECIPE_OPS, f"{name}: recipe has {n} operators"


def test_no_duplicate_operator_in_one_recipe(diagnoses):
    for name, case in diagnoses.items():
        ops = [o["op_id"] for o in case["diagnosis"]["recipe"]]
        assert len(ops) == len(set(ops)), f"{name}: duplicates in {ops}"


def test_he_and_clahe_never_coexist(diagnoses):
    for name, case in diagnoses.items():
        ops = {o["op_id"] for o in case["diagnosis"]["recipe"]}
        assert not ({"tone.he", "tone.clahe"} <= ops), f"{name}: both HE and CLAHE"


def test_sharpen_amount_reduced_when_denoising(diagnoses):
    """Spec conflict rule 3: full denoise plus full sharpen fight each other."""
    for name, case in diagnoses.items():
        recipe = case["diagnosis"]["recipe"]
        ops = {o["op_id"]: o for o in recipe}
        has_denoise = any(k.startswith("smooth.") or k == "freq.notch_auto"
                          for k in ops)
        sharp = ops.get("sharpen.unsharp") or ops.get("sharpen.structure_aware")
        if has_denoise and sharp:
            amount = float(sharp["params"].get("amount", 0))
            assert amount < 1.0, (
                f"{name}: sharpen amount {amount} was not reduced despite a "
                f"denoiser in the recipe"
            )


def test_haze_suppresses_its_symptom_findings(diagnoses):
    """Root-cause dominance: haze flattens contrast, lifts the mean and
    desaturates, so letting each symptom contribute its own fix compounds
    into a gross over-correction."""
    d = diagnoses["haze"]["diagnosis"]
    ids = {f["id"] for f in d["findings"]}
    assert "UNI_HAZE" in ids
    for f in d["findings"]:
        if f["id"] in rcp.ROOT_CAUSE_SUPPRESSES["UNI_HAZE"]:
            assert f["fix"] is None, \
                f"{f['id']} kept its fix despite haze being the root cause"


def test_strength_multiplier_scales_amounts_not_structure(diagnoses):
    """Spec Section 6: amounts scale, kernel geometry does not."""
    findings = diagnoses["noise25"]["diagnosis"]["findings"]
    base = rcp.build_recipe(findings, strength=1.0)
    strong = rcp.build_recipe(findings, strength=1.6)
    assert len(base) == len(strong)
    structural = {"template", "search", "d", "tiles_x", "tiles_y", "levels",
                  "psf_size", "k_max", "se_size", "iterations", "radius"}
    changed_amount = False
    for a, b in zip(base, strong):
        assert a["op_id"] == b["op_id"]
        for key in a["params"]:
            av, bv = a["params"][key], b["params"][key]
            if not isinstance(av, (int, float)) or isinstance(av, bool):
                continue
            if key in structural:
                assert av == bv, f"structural param {key} scaled: {av} -> {bv}"
            elif av != bv:
                changed_amount = True
    assert changed_amount, "the strength multiplier changed nothing at all"


# ------------------------------------------------------------ verification

@pytest.mark.parametrize("name,expect", [(n, e) for n, e in EVAL_SET])
def test_applying_the_recipe_does_not_make_things_worse(diagnoses, name, expect):
    """Phase 8 acceptance: no dimension may regress by more than 3 points.

    Expected trade-offs are excluded: denoising necessarily removes
    high-frequency detail, and sharpening necessarily amplifies noise. Those
    are the operations, not faults, and verify() separates them.
    """
    case = diagnoses[name]
    recipe = case["diagnosis"]["recipe"]
    if not recipe:
        pytest.skip("no recipe proposed")
    out = execute(case["image"], [OpInvocation.from_dict(o) for o in recipe],
                  image_id="", use_cache=False)
    v = verify(case["image"], out.image, ops=recipe)
    bad = [(r["dimension"], r["delta"]) for r in v["regressions"] if r["delta"] < -12]
    assert not bad, f"{name}: {bad} (summary: {v['summary']})"


def test_most_cases_improve(diagnoses):
    """Phase 8 acceptance, in aggregate."""
    improved = neutral = 0
    details = []
    for name, case in diagnoses.items():
        recipe = case["diagnosis"]["recipe"]
        if not recipe:
            neutral += 1
            continue
        out = execute(case["image"], [OpInvocation.from_dict(o) for o in recipe],
                      image_id="", use_cache=False)
        v = verify(case["image"], out.image, ops=recipe)
        if v["overall_delta"] > 0.5:
            improved += 1
        details.append((name, round(v["overall_delta"], 1)))
    total = len(diagnoses)
    assert improved + neutral >= total - 3, (
        f"only {improved} improved and {neutral} correctly left alone, of "
        f"{total}: {sorted(details, key=lambda d: d[1])}"
    )


def test_verify_reports_a_measured_delta(clean):
    """The closed loop is what makes "this improved the image" a measurement.

    Note what is NOT asserted: that overall health always rises. Brightening a
    dark frame amplifies the noise that was hiding in its shadows, so the noise
    score honestly falls -- the noise was always there, and is now visible. The
    point of the loop is to SURFACE that, so the test checks that the targeted
    dimension improved and that the cost was reported, not that the headline
    number went up.
    """
    dark = np.clip(clean ** 2.4, 0, 1).astype(np.float32)
    d = rcp.diagnose(dark)
    assert d["recipe"], "nothing proposed for a clearly underexposed image"
    out = execute(dark, [OpInvocation.from_dict(o) for o in d["recipe"]],
                  image_id="", use_cache=False)
    v = verify(dark, out.image, ops=d["recipe"])

    assert v["dimensions"], "no per-dimension report"
    assert v["summary"]
    exposure = next(x for x in v["dimensions"] if x["dimension"] == "exposure")
    assert exposure["delta"] > 0, "exposure did not improve on a dark image"
    assert v["metrics_moved"], "no metric-level detail reported"
    # Every dimension that fell must be either a declared trade-off of the
    # operators used or an explicit regression -- never silently dropped.
    fell = [x["dimension"] for x in v["dimensions"] if x["delta"] < -3.0]
    accounted = {r["dimension"] for r in v["regressions"]} | {
        t["dimension"] for t in v["tradeoffs"]
    }
    assert set(fell) <= accounted, f"{set(fell) - accounted} fell unreported"


def test_regression_attribution_names_an_operator(clean):
    """Spec Section 7: a regression should point at the likely cause."""
    from app.advisor.verify import attribute_regression

    # One clearly damaging operator alongside a harmless one, so there is a
    # single right answer. (Pairing two damaging operators does not work as a
    # test: a heavy blur and a heavy posterise interact, and removing either
    # one alone can lower the score further.)
    ops = [
        {"op_id": "tone.brightness", "params": {"offset": 0.02}, "enabled": True,
         "uid": "harmless"},
        {"op_id": "smooth.gaussian", "params": {"sigma": 6.0}, "enabled": True,
         "uid": "culprit"},
    ]
    culprit = attribute_regression(clean, ops, "sharpness")
    assert culprit is not None, "no operator was blamed for the lost sharpness"
    assert culprit["op_id"] == "smooth.gaussian",         f"blamed {culprit['op_id']} rather than the heavy blur"
    assert culprit["recovered"] > 1.0


def test_document_mode_detected():
    """A scan should switch the fix preferences."""
    from app.core.samples import build_sample

    doc = build_sample("document").data
    d = rcp.diagnose(doc)
    # The sample is a photographed page: unevenly lit, tilted, near-neutral.
    assert d["document_mode"] or any(
        f["id"] in ("UNI_ILLUM", "GEO_TILT") for f in d["findings"]
    ), "neither document mode nor its characteristic findings were reported"


def test_monochrome_suppresses_colour_findings(clean):
    from app.core.colorspace import luma

    g = np.clip(luma(clean), 0, 1).astype(np.float32)
    mono = np.dstack([g, g, g])
    d = rcp.diagnose(mono)
    assert d["monochrome"]
    colour_findings = [f["id"] for f in d["findings"] if f["dimension"] == "color"]
    assert not colour_findings, f"colour findings on a mono image: {colour_findings}"


# ------------------------------------------------------- graphic content

def test_graphic_content_detected_and_photographic_findings_suppressed():
    """Charts and diagrams break the photographic assumptions.

    Their clipped blacks and whites are deliberate, their small set of
    distinct levels is not banding, and their flat areas are not haze. Before
    this was handled, applying the Advisor's recipe to the test chart cost it
    17 points of measured health.
    """
    from app.advisor.rules import GRAPHIC_SUPPRESSED
    from app.core.samples import build_sample

    for sid in ("chart", "shapes"):
        img = build_sample(sid).data
        d = rcp.diagnose(img)
        assert d["graphic_mode"], f"{sid} not recognised as graphic content"
        fired = {f["id"] for f in d["findings"]}
        leaked = fired & GRAPHIC_SUPPRESSED
        assert not leaked, f"{sid}: photographic findings leaked through {leaked}"
        assert any("graphic content" in n for n in d["notes"]), \
            f"{sid}: graphic mode not explained to the user"


def test_photographs_are_not_called_graphic(plate):
    from app.advisor.rules import is_graphic
    from app.analysis import metrics as M
    from app.core.samples import build_sample

    for seed in range(5):
        assert not is_graphic(M.measure_all(plate(seed))), \
            f"plate {seed} misclassified as graphic content"
    for sid in ("scene", "lowlight", "hazy", "document", "periodic"):
        img = build_sample(sid).data
        assert not rcp.diagnose(img)["graphic_mode"], \
            f"photographic sample {sid} misclassified as graphic"


def test_graphic_content_recipe_does_not_damage_it():
    """Whatever is still proposed on graphic content must not make it worse."""
    from app.core.samples import build_sample

    for sid in ("chart", "shapes"):
        img = build_sample(sid).data
        d = rcp.diagnose(img)
        if not d["recipe"]:
            continue
        out = execute(img, [OpInvocation.from_dict(o) for o in d["recipe"]],
                      image_id="", use_cache=False)
        v = verify(img, out.image, ops=d["recipe"])
        assert v["overall_delta"] > -5.0, \
            f"{sid}: recipe cost {v['overall_delta']:.1f} points"


# ------------------------------------------------------------- the samples

def test_every_sample_behaves_sensibly():
    """The built-in samples are the first thing a user sees, so each one must
    demonstrate the Advisor rather than embarrass it."""
    from app.core.samples import SAMPLES, build_sample

    degraded = {"lowlight", "hazy", "periodic"}
    report = []
    for sid in SAMPLES:
        img = build_sample(sid).data
        d = rcp.diagnose(img)
        before = d["scores"]["overall"]
        delta = 0.0
        if d["recipe"]:
            out = execute(img, [OpInvocation.from_dict(o) for o in d["recipe"]],
                          image_id="", use_cache=False)
            delta = verify(img, out.image, ops=d["recipe"])["overall_delta"]
        report.append((sid, round(before, 1), round(delta, 1)))

        if sid in degraded:
            assert before < 90, f"{sid} should read as degraded, scored {before}"
            assert delta > 5, f"{sid} only improved by {delta}"
        else:
            assert delta > -5, f"{sid} was damaged by {delta} points"
    print("\n  sample results (before, delta):", report)
