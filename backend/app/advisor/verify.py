"""Closed-loop verification. docs/advisor-spec.md Section 7.

Re-measures after a recipe is applied and reports per-dimension deltas. This is
what makes "this improved the image" a measurement rather than an assertion.
"""
from __future__ import annotations

import numpy as np

from ..analysis import metrics as M
from ..analysis.score import SCORING, score_all
from ..core.pipeline import OpInvocation, execute

REGRESSION_TOLERANCE = 3.0


# Dimensions a given operator family is EXPECTED to cost. Denoising trades
# high-frequency detail for a lower noise floor; sharpening trades noise for
# acuity. Reporting those as faults would drown the genuine regressions.
EXPECTED_TRADEOFFS = {
    "smooth.": ("sharpness", "detail", "color"),
    "freq.notch": ("sharpness", "detail"),
    "sharpen.": ("noise", "artifacts"),
    # Anything that lifts shadows amplifies the noise that was hiding in them.
    # A dark frame's noise floor is scaled up along with its signal, so a noise
    # score falling after an exposure lift is the operation working, not
    # failing.
    "tone.clahe": ("noise",),
    "tone.he": ("noise",),
    "tone.exposure": ("noise",),
    "tone.gamma": ("noise",),
    "tone.shadows_highlights": ("noise",),
    "tone.retinex": ("noise",),
    "tone.autolevels": ("noise",),
    "freq.homomorphic": ("noise",),
}


def expected_tradeoff_dims(ops: list[dict] | None) -> set[str]:
    """Dimensions the recipe is expected to cost, given the operators in it."""
    dims: set[str] = set()
    for o in ops or []:
        oid = str(o.get("op_id", ""))
        for prefix, affected in EXPECTED_TRADEOFFS.items():
            if oid.startswith(prefix):
                dims.update(affected)
    return dims


def verify(before_img: np.ndarray, after_img: np.ndarray,
           groups: list[str] | None = None,
           ops: list[dict] | None = None) -> dict:
    """Compare the scores of two images, separating trade-offs from faults."""
    m_before = M.measure_all(before_img, groups)
    m_after = M.measure_all(after_img, groups)
    s_before = score_all(m_before)
    s_after = score_all(m_after)

    tradeoff_dims = expected_tradeoff_dims(ops)
    dims = []
    regressions = []
    tradeoffs = []
    for dim in s_before["dimensions"]:
        b = s_before["dimensions"][dim]["score"]
        a = s_after["dimensions"].get(dim, {}).get("score", b)
        delta = a - b
        row = {
            "dimension": dim,
            "before": b,
            "after": a,
            "delta": round(delta, 1),
            "severity_before": s_before["dimensions"][dim]["severity"],
            "severity_after": s_after["dimensions"].get(dim, {}).get("severity"),
        }
        row["expected_tradeoff"] = dim in tradeoff_dims
        dims.append(row)
        if delta < -REGRESSION_TOLERANCE:
            (tradeoffs if dim in tradeoff_dims else regressions).append(row)

    moved = _metric_deltas(m_before, m_after)
    overall_delta = s_after["overall"] - s_before["overall"]

    return {
        "overall_before": s_before["overall"],
        "overall_after": s_after["overall"],
        "overall_delta": round(overall_delta, 1),
        "improved": overall_delta > 0.5,
        "dimensions": dims,
        "regressions": regressions,
        "tradeoffs": tradeoffs,
        "metrics_moved": moved,
        "scores_before": s_before,
        "scores_after": s_after,
        "summary": _summarise(overall_delta, dims, regressions, tradeoffs),
    }


def _metric_deltas(before: dict, after: dict) -> list[dict]:
    """Every scored metric that moved, with whether the move was an improvement."""
    rows = []
    for dim, specs in SCORING.items():
        for group, name, _w, ilo, ihi, _flo, _fhi in specs:
            if name.startswith("_"):
                continue
            b = before.get(group, {}).get(name)
            a = after.get(group, {}).get(name)
            if not isinstance(b, (int, float)) or not isinstance(a, (int, float)):
                continue
            if abs(float(a) - float(b)) < 1e-6:
                continue
            in_before = ilo <= float(b) <= ihi
            in_after = ilo <= float(a) <= ihi
            rows.append({
                "dimension": dim,
                "metric": name,
                "before": round(float(b), 6),
                "after": round(float(a), 6),
                "delta": round(float(a) - float(b), 6),
                "expected": [ilo, ihi],
                "entered_range": (not in_before) and in_after,
                "left_range": in_before and (not in_after),
            })
    rows.sort(key=lambda r: (not r["left_range"], not r["entered_range"]))
    return rows


def _summarise(overall_delta: float, dims: list[dict],
               regressions: list[dict],
               tradeoffs: list[dict] | None = None) -> str:
    gains = [d for d in dims if d["delta"] > 1.0]
    best = max(dims, key=lambda d: d["delta"]) if dims else None
    parts = []
    if overall_delta > 0.5:
        parts.append(f"Overall image health improved by {overall_delta:+.1f} points")
    elif overall_delta < -0.5:
        parts.append(f"Overall image health fell by {overall_delta:.1f} points")
    else:
        parts.append("Overall image health is essentially unchanged")
    if best and best["delta"] > 1.0:
        parts.append(f"{best['dimension']} gained {best['delta']:+.1f}")
    if len(gains) > 1:
        parts.append(f"{len(gains)} dimensions improved")
    if regressions:
        names = ", ".join(r["dimension"] for r in regressions)
        parts.append(f"but {names} regressed by more than "
                     f"{REGRESSION_TOLERANCE:.0f} points")
    if tradeoffs:
        names = ", ".join(t["dimension"] for t in tradeoffs)
        parts.append(f"{names} fell as an expected cost of the operators used")
    return "; ".join(parts) + "."


def attribute_regression(base_img: np.ndarray, ops: list[dict], dimension: str,
                         image_id: str = "", target: str = "proxy") -> dict | None:
    """Find which single operator is most responsible for a regression.

    Re-runs the recipe with each operator disabled in turn and reports the one
    whose removal recovers the most. Cheap, thanks to the prefix cache (D2).
    """
    if not ops:
        return None
    from ..analysis.score import score_all as _score

    def dim_score(img: np.ndarray) -> float:
        return _score(M.measure_all(img, [_group_for(dimension)]))[
            "dimensions"].get(dimension, {}).get("score", 0.0)

    full = execute(base_img, [OpInvocation.from_dict(o) for o in ops],
                   image_id=image_id, target=target)
    baseline = dim_score(full.image)

    best = None
    for i in range(len(ops)):
        trial = [dict(o) for o in ops]
        trial[i]["enabled"] = False
        res = execute(base_img, [OpInvocation.from_dict(o) for o in trial],
                      image_id=image_id, target=target)
        s = dim_score(res.image)
        gain = s - baseline
        if best is None or gain > best["recovered"]:
            best = {
                "index": i,
                "op_id": ops[i].get("op_id"),
                "uid": ops[i].get("uid"),
                "recovered": round(gain, 2),
                "score_without": round(s, 1),
                "score_with": round(baseline, 1),
            }
    if best and best["recovered"] > 1.0:
        return best
    return None


def _group_for(dimension: str) -> str:
    return {
        "exposure": "exposure",
        "contrast": "contrast",
        "color": "color",
        "sharpness": "sharpness",
        "noise": "noise",
        "detail": "texture",
        "artifacts": "artifacts",
        "geometry": "geometry",
        "uniformity": "uniformity",
    }.get(dimension, dimension)
