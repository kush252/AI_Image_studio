"""Dimension scoring. Implements docs/advisor-spec.md Section 3.

Every threshold lives in SCORING below -- one table, so the curves can be tuned
and cited rather than being buried in branches.
"""
from __future__ import annotations

from typing import Any


def piecewise(value: float, ideal_lo: float, ideal_hi: float,
              fail_lo: float, fail_hi: float) -> float:
    """100 inside [ideal_lo, ideal_hi], falling linearly to 0 at the fail bounds."""
    v = float(value)
    if ideal_lo <= v <= ideal_hi:
        return 100.0
    if v < ideal_lo:
        if v <= fail_lo:
            return 0.0
        span = ideal_lo - fail_lo
        return 100.0 * (v - fail_lo) / span if span > 1e-9 else 0.0
    if v >= fail_hi:
        return 0.0
    span = fail_hi - ideal_hi
    return 100.0 * (fail_hi - v) / span if span > 1e-9 else 0.0


# (group, metric, weight, ideal_lo, ideal_hi, fail_lo, fail_hi)
#
# Bounds are calibrated against backend/tests/fixtures/synth.py: the clean
# plates sit inside every ideal band, and each synthetic degradation pushes the
# metric it targets toward its fail bound. Re-run tests/test_advisor_eval.py
# after touching any of these.
SCORING: dict[str, list[tuple]] = {
    "exposure": [
        ("exposure", "mean_y", 0.40, 0.35, 0.60, 0.05, 0.92),
        ("exposure", "clip_shadow", 0.20, 0.0, 0.010, -1.0, 0.12),
        ("exposure", "clip_highlight", 0.25, 0.0, 0.005, -1.0, 0.08),
        ("exposure", "range_usage", 0.15, 0.45, 1.01, 0.15, 2.0),
    ],
    "contrast": [
        ("contrast", "rms_contrast", 0.40, 0.10, 0.30, 0.03, 0.46),
        ("contrast", "entropy", 0.25, 6.30, 8.10, 3.20, 9.0),
        ("contrast", "local_contrast", 0.25, 0.022, 0.25, 0.005, 0.42),
        ("contrast", "occupancy", 0.10, 0.45, 1.01, 0.10, 2.0),
    ],
    "color": [
        ("color", "cast_magnitude", 0.45, 0.0, 0.115, -1.0, 0.40),
        ("color", "sat_mean", 0.30, 0.12, 0.60, 0.02, 0.90),
        ("color", "oversat_frac", 0.15, 0.0, 0.020, -1.0, 0.22),
        ("exposure", "clip_channel_hi", 0.10, 0.0, 0.012, -1.0, 0.14),
    ],
    "sharpness": [
        ("sharpness", "edge_width", 0.40, 0.0, 3.40, -1.0, 9.0),
        ("sharpness", "hf_ratio", 0.30, 0.0010, 0.035, 0.00005, 0.12),
        ("sharpness", "sharp_roi", 0.20, 0.25, 6.0, 0.02, 20.0),
        ("sharpness", "ringing", 0.10, 0.0, 0.11, -1.0, 0.40),
    ],
    "noise": [
        ("noise", "sigma_luma_8bit", 0.45, 0.0, 3.00, -1.0, 26.0),
        ("noise", "impulse_frac", 0.20, 0.0, 0.0010, -1.0, 0.035),
        ("noise", "chroma_ratio", 0.15, 0.0, 1.60, -1.0, 5.0),
        ("noise", "_periodic_count", 0.20, 0.0, 0.0, -1.0, 3.0),
    ],
    "detail": [
        ("texture", "detail_loss", 0.60, 0.0, 0.35, -1.0, 0.95),
        ("texture", "edge_density", 0.40, 0.0035, 0.35, 0.0004, 0.70),
    ],
    "artifacts": [
        ("artifacts", "blockiness", 0.45, 0.0, 1.12, -1.0, 2.20),
        ("artifacts", "banding_score", 0.35, 0.0, 0.090, -1.0, 0.45),
        ("artifacts", "hist_comb", 0.20, 0.0, 0.060, -1.0, 0.35),
    ],
    "geometry": [
        ("geometry", "_abs_tilt", 0.60, 0.0, 0.40, -1.0, 6.0),
        ("geometry", "ca_shift_px", 0.25, 0.0, 0.60, -1.0, 3.5),
        ("geometry", "border_frac", 0.15, 0.0, 0.005, -1.0, 0.10),
    ],
    "uniformity": [
        ("uniformity", "illum_nonuniformity", 0.40, 0.0, 0.80, -1.0, 1.80),
        ("uniformity", "vignette_falloff", 0.30, -0.02, 0.12, -0.6, 0.55),
        ("uniformity", "dark_channel_mean", 0.30, 0.0, 0.40, -1.0, 0.85),
    ],
}

OVERALL_WEIGHTS = {
    "exposure": 0.18,
    "contrast": 0.15,
    "sharpness": 0.15,
    "noise": 0.14,
    "color": 0.13,
    "uniformity": 0.09,
    "artifacts": 0.07,
    "detail": 0.05,
    "geometry": 0.04,
}

# Dimension score -> severity label (spec 3).
SEVERITY_BANDS = [
    (85.0, "none"),
    (70.0, "info"),
    (55.0, "minor"),
    (35.0, "moderate"),
    (0.0, "severe"),
]


def severity_for(score: float) -> str:
    for threshold, label in SEVERITY_BANDS:
        if score >= threshold:
            return label
    return "severe"


def _derived(metrics: dict, group: str, name: str):
    """Metrics that are computed from others rather than measured directly."""
    if name == "_periodic_count":
        return len(metrics.get("noise", {}).get("periodic_peaks", []) or [])
    if name == "_abs_tilt":
        return abs(float(metrics.get("geometry", {}).get("tilt_angle", 0.0) or 0.0))
    return metrics.get(group, {}).get(name)


def score_all(metrics: dict) -> dict:
    """Collapse the measurement dict into dimension scores plus an overall."""
    dims: dict[str, Any] = {}
    for dim, rows in SCORING.items():
        total_w = 0.0
        acc = 0.0
        breakdown = []
        for group, name, weight, ilo, ihi, flo, fhi in rows:
            val = _derived(metrics, group, name)
            if val is None or isinstance(val, (dict, list, str)):
                continue
            s = piecewise(float(val), ilo, ihi, flo, fhi)
            acc += s * weight
            total_w += weight
            breakdown.append({
                "metric": name,
                "value": round(float(val), 6),
                "score": round(s, 1),
                "weight": weight,
                "expected": [ilo, ihi],
            })
        score = (acc / total_w) if total_w > 1e-9 else 100.0
        dims[dim] = {
            "score": round(score, 1),
            "severity": severity_for(score),
            "breakdown": breakdown,
        }

    # Monochrome images must not be penalised on colour.
    if metrics.get("color", {}).get("monochrome") or \
            float(metrics.get("color", {}).get("gray_frac", 0.0) or 0.0) > 0.95:
        dims["color"] = {"score": 100.0, "severity": "none", "breakdown": [],
                         "note": "Monochrome image; colour scoring suppressed."}

    present = {k: v for k, v in dims.items() if k in OVERALL_WEIGHTS}
    wsum = sum(OVERALL_WEIGHTS[k] for k in present)
    overall = (sum(present[k]["score"] * OVERALL_WEIGHTS[k] for k in present)
               / max(wsum, 1e-9))

    # A single severe defect must not be averaged away (spec 3).
    worst = min((v["score"] for v in present.values()), default=100.0)
    capped = min(overall, worst + 25.0)

    return {
        "dimensions": dims,
        "overall": round(capped, 1),
        "overall_uncapped": round(overall, 1),
        "worst_dimension": min(present, key=lambda k: present[k]["score"])
        if present else None,
        "weights": OVERALL_WEIGHTS,
    }
