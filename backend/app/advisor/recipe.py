"""Diagnose, order and resolve conflicts. docs/advisor-spec.md Sections 4 and 6."""
from __future__ import annotations

import inspect as pyinspect
from typing import Any

import numpy as np

from ..analysis import metrics as M
from ..analysis.score import OVERALL_WEIGHTS, score_all, severity_for
from ..core.pipeline import OpInvocation, execute
from . import rules as R

# Canonical pipeline order (spec 6). The index is the sort key.
STAGE_ORDER = {
    # 1. Geometry -- before anything resamples detail
    "geom.deskew": 10,
    "geom.perspective4": 11,
    "geom.lens_distort": 12,
    "color.ca_correct": 13,
    "geom.vignette": 14,
    # 2. White balance -- before tone, so tone sees true luminance
    "color.channel_gains": 20,
    "color.wb_grayworld": 21,
    "color.wb_shades_of_gray": 21,
    "color.wb_whitepatch": 21,
    "color.temp_tint": 22,
    # 3. Denoise -- before ANY contrast amplification
    "freq.notch_auto": 30,
    "smooth.median_adaptive": 31,
    "smooth.wavelet": 32,
    "smooth.nlm": 33,
    "smooth.bilateral": 34,
    # 4. Exposure / tone
    "tone.exposure": 40,
    "tone.gamma": 41,
    "tone.autolevels": 42,
    "tone.levels": 43,
    "tone.curves": 44,
    # 5. Local tone / dehaze / illumination
    "tone.shadows_highlights": 50,
    "tone.retinex": 51,
    "freq.homomorphic": 52,
    "morph.bg_divide": 53,
    # 6. Contrast
    "tone.clahe": 60,
    "tone.contrast_sigmoid": 61,
    "tone.he": 62,
    # 7. Colour
    "color.vibrance": 70,
    "color.hsv_shift": 71,
    "color.gamut_clip": 72,
    # 8. Sharpen -- always last of the pixel operators
    "sharpen.structure_aware": 80,
    "sharpen.unsharp": 81,
    "sharpen.deconv": 82,
    "sharpen.detail_split": 83,
    # 9. Cleanup
    "smooth.morph_denoise": 90,
}

DEFAULT_STAGE = 55
MAX_RECIPE_OPS = 8

SEVERITY_RANK = {"none": 0, "info": 1, "minor": 2, "moderate": 3, "severe": 4}

# Operators whose magnitude should be scaled by the strength multiplier.
AMOUNT_PARAMS = {
    "stops", "amount", "shadows", "highlights", "midtone_contrast", "clip",
    "strength", "gain", "detail_gain", "base_contrast", "h", "sigma_color",
    "weight", "A", "saturation", "value",
}


def _pct(metrics: dict, group: str, key: str) -> str:
    v = metrics.get(group, {}).get(key)
    return f"{float(v) * 100:.0f}%" if isinstance(v, (int, float)) else "much"


def _stage(op_id: str) -> int:
    return STAGE_ORDER.get(op_id, DEFAULT_STAGE)


def clamp_to_spec(op_id: str, params: dict) -> dict:
    """Clamp derived values into each parameter's declared range.

    A derivation formula can legitimately produce a value beyond an operator's
    bounds (a 15 px motion blur measured on a large image implies a PSF wider
    than the slider allows). Clamping here rather than raising means the Advisor
    always emits a runnable recipe, and it guarantees that no derived parameter
    can ever fail validation.
    """
    from ..core import registry
    from ..core.params import FloatParam, IntParam

    try:
        spec = registry.get(op_id)
    except Exception:
        return params
    by_name = {p.name: p for p in spec.params}
    out = dict(params)
    for name, val in params.items():
        p = by_name.get(name)
        if p is None or not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        if isinstance(p, (FloatParam, IntParam)):
            lo, hi = float(p.min), float(p.max)
            clamped = min(max(float(val), lo), hi)
            out[name] = int(round(clamped)) if isinstance(p, IntParam) else round(
                clamped, 6)
    return out


def _call_fix(rule: R.Rule, metrics: dict, long_edge: int):
    """Invoke a rule's fix, passing long_edge only if it accepts it."""
    fn = rule.fix
    if fn is None:
        return [], ""
    try:
        sig = pyinspect.signature(fn)
        if "long_edge" in sig.parameters:
            return fn(metrics, long_edge=long_edge)
    except (TypeError, ValueError):
        pass
    return fn(metrics)


# How far a declared trade-off is excused before it is charged in full.
TRADEOFF_ALLOWANCE = 10.0


def refine_recipe(img: np.ndarray, findings: list[dict], recipe: list[dict],
                  *, strength: float = 1.0, include_opt_in: bool = False,
                  max_rounds: int = 3) -> tuple[list[dict], dict]:
    """Test the recipe before offering it, and back off until it helps.

    Without this the Advisor proposes advice it has never tried. On real
    photographs that went wrong often: measured against six real images, three
    recipes made the image worse, one by 21 points. Deriving a parameter from a
    formula is not evidence that applying it helps.

    The loop is: apply, measure, and if overall health fell or a dimension
    regressed badly, either halve the strength or drop the single worst
    operator, then try again. Returns whatever measured best -- possibly the
    empty recipe, which is the right answer when nothing helps.
    """
    from .verify import verify

    if not recipe:
        return recipe, {"rounds": 0, "accepted": True, "reason": "empty recipe"}

    targeted = {f["dimension"] for f in findings if f.get("fix")}

    def score(ops: list[dict]) -> tuple[float, dict]:
        """Net benefit: the composite, with declared trade-offs discounted.

        The raw composite cannot represent a correct trade-off. Noise carries
        a weight of 0.14 while sharpness and detail together carry 0.20, so
        denoising a visibly noisy image always measures as a net loss and the
        right treatment gets withdrawn. But crediting ONLY the targeted
        dimensions is equally wrong in the other direction -- it throws away
        recipes that measurably improved the picture somewhere we had not
        named, and it lets a recipe serve one metric while flattening the
        rest.

        So: score every dimension, but charge a regression at full weight only
        when it is NOT an expected cost of the operators in the recipe, and
        add a bonus for moving the dimensions the findings actually named.
        """
        if not ops:
            return 0.0, {}
        res = execute(img, [OpInvocation.from_dict(o) for o in ops],
                      image_id="", use_cache=False, collect_views=False)
        v = verify(img, res.image, ops=ops)
        total = 0.0
        for d in v["dimensions"]:
            delta = d["delta"]
            if delta < 0 and d["expected_tradeoff"]:
                # Discounted only up to TRADEOFF_ALLOWANCE. A denoise costing
                # three points of sharpness is the operation working; one
                # costing thirty is over-treatment, and calling that
                # "expected" let a recipe through that wrecked the picture.
                excess = max(-delta - TRADEOFF_ALLOWANCE, 0.0)
                delta = -(min(-delta, TRADEOFF_ALLOWANCE) * 0.25 + excess)
            total += OVERALL_WEIGHTS.get(d["dimension"], 0.0) * delta
        # Kept for the guard below, which has to judge the recipe by the same
        # discounted arithmetic the search used. Judging it on the RAW
        # composite vetoed brightening a dark frame, because the noise that
        # was hiding in its shadows counted against the fix at full weight.
        v["adjusted_delta"] = total
        bonus = 0.5 * sum(d["delta"] for d in v["dimensions"]
                          if d["dimension"] in targeted and d["delta"] > 0)
        return total + bonus, v

    best_delta, best_v = score(recipe)
    best_ops = recipe
    trail = [{"round": 0, "ops": len(recipe), "delta": round(best_delta, 2),
              "action": "initial"}]

    current = recipe
    cur_strength = strength
    for rnd in range(1, max_rounds + 1):
        delta, v = score(current)
        hard = [r for r in v.get("regressions", []) if r["delta"] < -6.0]
        if delta > 0.5 and not hard:
            break  # good enough: it helps and nothing broke badly

        if hard:
            # Drop the operator most responsible for the worst regression.
            worst = min(hard, key=lambda r: r["delta"])
            culprit = worst.get("likely_cause") or _blame(img, current,
                                                          worst["dimension"])
            if culprit is not None and len(current) > 1:
                current = [o for i, o in enumerate(current)
                           if i != culprit["index"]]
                trail.append({"round": rnd, "ops": len(current),
                              "action": f"dropped {culprit['op_id']} "
                                        f"(hurt {worst['dimension']})"})
            else:
                current = []
                trail.append({"round": rnd, "ops": 0,
                              "action": "withdrew the recipe"})
        else:
            # It simply does not help; try a gentler version before giving up.
            cur_strength *= 0.5
            if cur_strength < 0.2:
                current = []
                trail.append({"round": rnd, "ops": 0,
                              "action": "withdrew the recipe"})
            else:
                current = build_recipe(findings, include_opt_in=include_opt_in,
                                       strength=cur_strength)
                trail.append({"round": rnd, "ops": len(current),
                              "action": f"reduced strength to {cur_strength:.2f}"})

        new_delta, new_v = score(current)
        trail[-1]["delta"] = round(new_delta, 2)
        if new_delta > best_delta:
            best_ops, best_delta, best_v = current, new_delta, new_v
        if not current:
            break

    # Never hand back something that measures worse than doing nothing.
    # A recipe may spend a little overall health to fix what it targets (a
    # denoise always costs some sharpness), but not much. Without this cap the
    # targeted-dimension score above will happily accept a recipe that wrecks
    # the image to serve one metric.
    # Measured over the dimensions the recipe was NOT expected to cost. Using
    # the raw composite here was wrong in both directions: it vetoed
    # brightening a dark frame (which legitimately uncovers the noise that was
    # hiding in its shadows) while still letting through a recipe that served
    # one metric at the expense of the picture.
    # How much overall health this image may spend to be fixed. An image that
    # is already in good shape may spend NOTHING: the allowance exists so a
    # real defect can be treated at a real cost, not so a near-perfect picture
    # can be traded away. Two of the photographs this was tested against were
    # a deliberate panning shot and a screenshot of a well-exposed photo, and
    # both were being "corrected" into something measurably worse.
    # Measured on the WORST dimension the findings actually named, not on the
    # overall average. Overall health was the wrong test in both directions:
    # it refused to deblock a JPEG that was fine apart from being blocky, and
    # an average can stay high while one dimension is genuinely bad.
    worst_targeted = 100.0
    if best_v:
        named = [d["before"] for d in best_v["dimensions"]
                 if d["dimension"] in targeted]
        if named:
            worst_targeted = min(named)
    allowance = (0.0 if worst_targeted >= 80.0
                 else min(6.0, (80.0 - worst_targeted) * 0.3))

    collateral = best_v.get("adjusted_delta", 0.0) if best_v else 0.0
    if best_ops and collateral < -allowance:
        return [], {"rounds": len(trail) - 1, "accepted": False,
                    "best_delta": round(best_delta, 2), "trail": trail,
                    "collateral": round(collateral, 2),
                    "reason": f"the best recipe improved the dimensions it "
                              f"targets but cost {-collateral:.1f} points "
                              f"net, more than the {allowance:.1f} justified "
                              f"by its worst affected dimension, which was "
                              f"already scoring {worst_targeted:.0f}/100, so "
                              f"nothing is suggested"}

    if best_delta <= 0.0:
        return [], {"rounds": len(trail) - 1, "accepted": False,
                    "best_delta": round(best_delta, 2), "trail": trail,
                    "reason": "no variant of the recipe improved the image, so "
                              "none is offered"}
    return best_ops, {"rounds": len(trail) - 1, "accepted": True,
                      "best_delta": round(best_delta, 2), "trail": trail,
                      "reason": f"accepted a {len(best_ops)}-operator recipe "
                                f"measuring {best_delta:+.1f}"}


def _blame(img: np.ndarray, ops: list[dict], dimension: str) -> dict | None:
    from .verify import attribute_regression

    try:
        return attribute_regression(img, ops, dimension)
    except Exception:
        return None


def diagnose(img: np.ndarray, long_edge: int | None = None,
             groups: list[str] | None = None,
             self_verify: bool = True) -> dict:
    """Measure -> score -> diagnose -> derive -> order -> SELF-VERIFY."""
    metrics = M.measure_all(img, groups)
    scores = score_all(metrics)
    if long_edge is None:
        long_edge = int(max(img.shape[0], img.shape[1]))

    graphic = R.is_graphic(metrics)

    findings: list[dict] = []
    for rule in R.RULES:
        try:
            if not rule.when(metrics):
                continue
        except Exception:
            continue
        # Photographic findings do not apply to charts, diagrams or
        # screenshots: their clipped blacks and whites are deliberate, their
        # few distinct levels are not banding, and their flat areas are not
        # haze.
        if graphic and rule.id in R.GRAPHIC_SUPPRESSED:
            continue

        dim = scores["dimensions"].get(rule.dimension, {})
        severity = rule.severity_override or dim.get("severity", "minor")
        # A rule that fired but whose dimension scored clean is at least "info":
        # it found something specific the dimensional average smoothed over.
        if severity == "none":
            severity = "info"

        try:
            confidence = float(np.clip(rule.confidence(metrics), 0.0, 1.0))
        except Exception:
            confidence = 1.0

        ops, derivation = ([], "")
        if rule.fix is not None and not rule.warning_only:
            try:
                ops, derivation = _call_fix(rule, metrics, long_edge)
            except Exception as exc:
                ops, derivation = [], f"Fix derivation failed: {exc}"

        try:
            evidence = rule.evidence(metrics)
        except Exception:
            evidence = {}

        findings.append({
            "id": rule.id,
            "dimension": rule.dimension,
            "severity": severity,
            "confidence": round(confidence, 3),
            "title": rule.title,
            "explanation": rule.explain(metrics),
            "evidence": evidence,
            "warning_only": rule.warning_only,
            "opt_in": rule.opt_in,
            "tags": rule.tags,
            "fix": ({
                "label": rule.fix_label(metrics),
                "ops": ops,
                "derivation": derivation,
            } if ops else None),
        })

    findings = _resolve_conflicts(findings, metrics)
    findings.sort(key=lambda f: (-SEVERITY_RANK.get(f["severity"], 0),
                                 -f["confidence"]))

    recipe = build_recipe(findings, include_opt_in=False)

    # Test the recipe before offering it. A derived parameter is a prediction,
    # not evidence; this is where it gets checked.
    refinement: dict = {"rounds": 0, "accepted": True, "reason": "not checked"}
    if self_verify:
        recipe, refinement = refine_recipe(img, findings, recipe)

    return {
        "metrics": metrics,
        "scores": scores,
        "findings": findings,
        "recipe": recipe,
        "document_mode": R.is_document(metrics),
        "monochrome": R.is_monochrome(metrics),
        "graphic_mode": graphic,
        "refinement": refinement,
        "notes": _global_notes(findings, metrics),
    }


# Findings that are DOWNSTREAM SYMPTOMS of another finding's root cause.
# Haze is the clear case: scattering simultaneously flattens contrast, lifts
# the mean, compresses the histogram and desaturates, so five rules fire on one
# physical cause. Letting each contribute its own fix compounds them into a
# gross over-correction, which measured as a 52-point loss on the evaluation
# set. The dehaze recipe already restores contrast and saturation.
ROOT_CAUSE_SUPPRESSES = {
    # Haze also lays a smooth depth-dependent brightness gradient across the
    # frame, which is exactly what the illumination metric fits, so UNI_ILLUM
    # is another symptom of it rather than a separate defect.
    "UNI_HAZE": {"CON_FLAT", "CON_LOW_ENTROPY", "COL_DULL", "EXP_NARROW",
                 "EXP_OVER", "UNI_ILLUM"},
}


def _resolve_conflicts(findings: list[dict], metrics: dict) -> list[dict]:
    """Spec 6 conflict rules."""
    ids = {f["id"] for f in findings}
    out = list(findings)

    # 0. Root-cause dominance: drop the fixes of findings that are merely
    #    symptoms of a deeper finding that is also present.
    for root, suppressed in ROOT_CAUSE_SUPPRESSES.items():
        if root not in ids:
            continue
        root_title = next((f["title"] for f in out if f["id"] == root), root)
        for f in out:
            if f["id"] in suppressed and f["fix"]:
                f["fix"] = None
                f["warning_only"] = True
                f["explanation"] += (
                    f" Its fix was withdrawn because \"{root_title}\" is also "
                    f"present and is the underlying cause of this symptom; "
                    f"correcting that addresses this too, whereas applying both "
                    f"would over-correct.")

    # 4. SHP_OVER suppresses every sharpening fix, whatever else fired.
    if "SHP_OVER" in ids:
        for f in out:
            if f["fix"]:
                f["fix"]["ops"] = [
                    o for o in f["fix"]["ops"]
                    if not o["op_id"].startswith("sharpen.")
                ]
                if not f["fix"]["ops"]:
                    f["fix"] = None
                    f["warning_only"] = True
                    f["explanation"] += (
                        " Its sharpening fix was withdrawn because the image is "
                        "already over-sharpened.")

    # 2. autolevels vs exposure/gamma: keep whichever the evidence favours.
    if "EXP_NARROW" in ids and ("EXP_UNDER" in ids or "EXP_OVER" in ids):
        range_usage = R._g(metrics, "exposure", "range_usage")
        drop = "EXP_NARROW" if range_usage >= 0.60 else None
        if drop is None:
            drop = "EXP_UNDER" if "EXP_UNDER" in ids else "EXP_OVER"
        for f in out:
            if f["id"] == drop and f["fix"]:
                f["fix"] = None
                f["warning_only"] = True
                f["explanation"] += (
                    f" Its fix was withdrawn in favour of the other exposure "
                    f"finding (range usage = {range_usage:.2f}), since applying "
                    f"both would double-correct.")

    # 1. Global HE and CLAHE are mutually exclusive -- keep CLAHE.
    has_clahe = any(o["op_id"] == "tone.clahe"
                    for f in out if f["fix"] for o in f["fix"]["ops"])
    if has_clahe:
        for f in out:
            if f["fix"]:
                f["fix"]["ops"] = [o for o in f["fix"]["ops"]
                                   if o["op_id"] != "tone.he"]

    # SHP_NOISE_LIMITED and the noise rules both emit a denoiser: keep one.
    if "SHP_NOISE_LIMITED" in ids and ids & {"NSE_GAUSS", "NSE_CHROMA",
                                             "NSE_IMPULSE", "NSE_PERIODIC"}:
        for f in out:
            if f["id"] == "SHP_NOISE_LIMITED" and f["fix"]:
                f["fix"] = None
                f["explanation"] += (
                    " The denoiser is contributed by the noise finding below, so "
                    "it is not duplicated here.")

    # 5. ART_PRESTRETCHED caps every tone fix at 60%.
    if "ART_PRESTRETCHED" in ids:
        for f in out:
            if f["fix"] and f["dimension"] in ("exposure", "contrast"):
                for o in f["fix"]["ops"]:
                    _scale_op(o, 0.6)
                f["fix"]["derivation"] += (
                    " All magnitudes capped at 60% because the histogram shows "
                    "the image was already stretched.")

    # 3. Denoise + sharpen: scale the sharpen amount down.
    has_denoise = any(
        o["op_id"].startswith("smooth.") or o["op_id"] == "freq.notch_auto"
        for f in out if f["fix"] for o in f["fix"]["ops"])
    if has_denoise:
        for f in out:
            if f["fix"]:
                for o in f["fix"]["ops"]:
                    if o["op_id"] in ("sharpen.unsharp", "sharpen.structure_aware"):
                        before = o["params"].get("amount")
                        _scale_op(o, 0.55)
                        if before is not None:
                            f["fix"]["derivation"] += (
                                f" Sharpen amount reduced {before:.2f} -> "
                                f"{o['params']['amount']:.2f} because the recipe "
                                f"also contains a denoiser.")
    return out


def _scale_op(op: dict, factor: float) -> None:
    """Scale amount-like params only; structural params must not move (spec 6)."""
    for k, v in list(op.get("params", {}).items()):
        if k in AMOUNT_PARAMS and isinstance(v, (int, float)) \
                and not isinstance(v, bool):
            if k == "clip":  # CLAHE clip is a limit, scale toward its neutral 1.0
                op["params"][k] = round(1.0 + (float(v) - 1.0) * factor, 4)
            else:
                op["params"][k] = round(float(v) * factor, 4)


def build_recipe(findings: list[dict], include_opt_in: bool = False,
                 strength: float = 1.0) -> list[dict]:
    """Collect fixes, order canonically, cap, and apply the strength multiplier."""
    collected: list[dict] = []
    for f in findings:
        if not f.get("fix"):
            continue
        if f.get("opt_in") and not include_opt_in and f["severity"] != "severe":
            continue
        for o in f["fix"]["ops"]:
            collected.append({
                **{k: v for k, v in o.items()},
                "_from": f["id"],
                "_severity": f["severity"],
                "_confidence": f["confidence"],
                "_priority": SEVERITY_RANK.get(f["severity"], 0) * f["confidence"],
            })

    # 6. At most one operator per op_id -- keep the highest-priority instance.
    best: dict[str, dict] = {}
    for o in collected:
        prev = best.get(o["op_id"])
        if prev is None or o["_priority"] > prev["_priority"]:
            best[o["op_id"]] = o
    chosen = list(best.values())

    # 7. Cap at MAX_RECIPE_OPS by severity x confidence.
    if len(chosen) > MAX_RECIPE_OPS:
        chosen.sort(key=lambda o: -o["_priority"])
        chosen = chosen[:MAX_RECIPE_OPS]

    chosen.sort(key=lambda o: _stage(o["op_id"]))

    if abs(strength - 1.0) > 1e-6:
        for o in chosen:
            _scale_op(o, strength)

    out = []
    for i, o in enumerate(chosen):
        entry = {
            "op_id": o["op_id"],
            "params": clamp_to_spec(o["op_id"], o.get("params", {})),
            "enabled": True,
            "uid": f"advisor-{i}-{o['op_id']}",
            "from_finding": o["_from"],
            "stage": _stage(o["op_id"]),
        }
        if "mask" in o:
            entry["mask"] = o["mask"]
        out.append(entry)
    return out


def _global_notes(findings: list[dict], metrics: dict) -> list[str]:
    notes = []
    if R.is_graphic(metrics):
        notes.append(
            "This looks like synthetic graphic content (a chart, diagram, "
            "screenshot or render) rather than a photograph: "
            f"{_pct(metrics, 'texture', 'graphic_frac')} of pixels sit in "
            "perfectly uniform neighbourhoods, which a camera never produces. "
            "Photographic findings are suppressed, because clipped blacks and "
            "whites, a small set of distinct levels and large flat areas are "
            "all deliberate here rather than defects.")
    if R.is_document(metrics):
        notes.append(
            "Document mode: the image looks like a scan (high edge density, "
            "near-neutral, bimodal histogram), so Sauvola/background-division "
            "fixes are preferred over CLAHE and saturation findings are "
            "suppressed.")
    if R.is_monochrome(metrics):
        notes.append(
            "The image is effectively monochrome, so all colour findings are "
            "suppressed rather than 'correcting' a black-and-white photograph.")
    actionable = [f for f in findings if f.get("fix")]
    if not actionable and not findings:
        notes.append(
            "No defects crossed their thresholds. Every measured dimension is "
            "inside its expected range.")
    low_conf = [f for f in findings if f["confidence"] < 0.45]
    if low_conf:
        notes.append(
            f"{len(low_conf)} finding(s) are below 0.45 confidence and are "
            f"grouped separately rather than shown by default.")
    return notes
