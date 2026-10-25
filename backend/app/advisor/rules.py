"""Declarative rule table. Implements docs/advisor-spec.md Section 4.

Each Rule maps a predicate over the measured metrics to a Finding plus a Fix.
Every finding MUST interpolate the triggering value -- no "looks a bit dark".
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import derive
from ..analysis.metrics import is_impulse_noise


@dataclass
class Rule:
    id: str
    dimension: str
    title: str
    when: Callable[[dict], bool]
    explain: Callable[[dict], str]
    fix: Callable[..., tuple[list[dict], str]] | None
    evidence: Callable[[dict], dict]
    # Severity is normally inherited from the dimension score; a rule may
    # override it when the defect is specific rather than dimensional.
    severity_override: str | None = None
    confidence: Callable[[dict], float] = lambda m: 1.0
    fix_label: Callable[[dict], str] = lambda m: "Apply fix"
    opt_in: bool = False  # excluded from "Apply all" unless severe
    warning_only: bool = False
    tags: list[str] = field(default_factory=list)


def _g(m: dict, group: str, key: str, default: Any = 0.0) -> float:
    v = m.get(group, {}).get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _ev(group: str, metric: str, expected: list, extra: dict | None = None):
    def fn(m: dict) -> dict:
        d = {
            "metric": metric,
            "value": _g(m, group, metric),
            "expected": expected,
        }
        if extra:
            d["secondary"] = {k: _g(m, g, k) for k, g in extra.items()}
        return d
    return fn


def _cast_confidence(m: dict) -> float:
    """How much to trust a colour-cast finding.

    Three things undermine the estimate, and each is measurable:

    * a single dominant hue (a sunset, neon, a studio gel) is often the
      photographer's intent rather than a cast;
    * illuminant estimation reads the tonal distribution, so a badly exposed
      frame gives an unreliable answer -- fix exposure and re-measure;
    * the three estimators (gray-world, white-patch, shades-of-gray)
      DISAGREEING relative to the size of the cast they claim means the scene
      breaks their shared assumption. On the fixtures this separates cleanly:
      a real cast gives a spread/magnitude ratio near 0.2, a spurious one
      near 1.9.
    """
    cast = _g(m, "color", "cast_magnitude")
    spread = _g(m, "color", "illuminant_spread")
    conf = min(1.0, 0.65 + cast * 3.0)

    if cast > 1e-6:
        disagreement = spread / cast
        if disagreement > 0.9:
            conf = min(conf, 0.35)
        elif disagreement > 0.5:
            conf = min(conf, 0.6)

    if _g(m, "color", "hue_entropy") < 1.5:
        conf = min(conf, 0.55)
    if not (0.22 <= _g(m, "exposure", "mean_y") <= 0.70):
        conf = min(conf, 0.45)
    return float(max(0.05, conf))


def is_document(m: dict) -> bool:
    """Document-mode detection (spec 4). Switches fix preferences."""
    edge_density = _g(m, "texture", "edge_density")
    gray_frac = _g(m, "color", "gray_frac")
    zh = m.get("exposure", {}).get("zone_hist") or []
    bimodal = False
    if len(zh) >= 11:
        ends = float(zh[0] + zh[1] + zh[-1] + zh[-2])
        bimodal = ends > 0.55
    return edge_density > 0.06 and gray_frac > 0.80 and bimodal


def is_graphic(m: dict) -> bool:
    """Synthetic graphic content rather than a photograph.

    Charts, diagrams, screenshots and renders break the assumptions behind
    most of the photographic metrics: their clipped blacks and whites are
    deliberate, their small set of distinct levels is not banding, and their
    flat regions are not haze. Detected from the fraction of pixels sitting in
    perfectly uniform neighbourhoods -- a real photograph has sensor noise
    everywhere, so that fraction is near zero for imagery and near one for
    graphics.
    """
    return _g(m, "texture", "graphic_frac") > 0.35


# Findings that assume photographic content and are therefore meaningless on
# charts, diagrams and screenshots.
GRAPHIC_SUPPRESSED = {
    # Pure black and white patches, a handful of distinct levels and large
    # flat areas are all deliberate in a chart.
    "ART_BAND", "ART_PRESTRETCHED", "ART_BLOCK",
    "EXP_DUAL_CLIP", "EXP_SHADOW_CLIP", "EXP_NARROW",
    "CON_LOW_ENTROPY", "CON_FLAT", "COL_DULL", "DET_PLASTIC",
    "UNI_HAZE", "UNI_ILLUM",
    # A chart's converging line pairs are strongly oriented, which makes its
    # spectrum anisotropic and reads as motion blur.
    "SHP_MOTION", "SHP_NOISE_LIMITED", "SHP_SOFT", "SHP_OVER",
    "NSE_IMPULSE",
    # EXP_UNDER, EXP_OVER and NSE_GAUSS are deliberately NOT here: a
    # screenshot can genuinely be too dark, too bright or noisy, and saying so
    # is useful. Suppressing them hid a real underexposure on a photograph
    # that was misdetected as graphic content.
}


def is_monochrome(m: dict) -> bool:
    return bool(m.get("color", {}).get("monochrome")) or _g(m, "color", "gray_frac") > 0.95


# ============================================================== exposure rules

RULES: list[Rule] = [
    Rule(
        id="EXP_UNDER",
        dimension="exposure",
        title="Underexposed",
        when=lambda m: _g(m, "exposure", "mean_y") < 0.35
        and _g(m, "exposure", "clip_highlight") < 0.002,
        explain=lambda m: (
            f"Mean luminance is {_g(m, 'exposure', 'mean_y'):.3f}, below the "
            f"0.35-0.60 range expected of a well-exposed image. Only "
            f"{_g(m, 'exposure', 'clip_highlight') * 100:.2f}% of pixels are near "
            f"white, so there is headroom to raise exposure without clipping "
            f"highlights."
        ),
        fix=derive.exposure_fix,
        evidence=_ev("exposure", "mean_y", [0.35, 0.60],
                     {"clip_highlight": "exposure", "range_usage": "exposure"}),
        fix_label=lambda m: "Raise exposure",
    ),
    Rule(
        id="EXP_OVER",
        dimension="exposure",
        title="Overexposed / blown highlights",
        when=lambda m: _g(m, "exposure", "mean_y") > 0.62
        or _g(m, "exposure", "clip_highlight") > 0.010,
        explain=lambda m: (
            f"Mean luminance is {_g(m, 'exposure', 'mean_y'):.3f} and "
            f"{_g(m, 'exposure', 'clip_highlight') * 100:.2f}% of pixels sit at or "
            f"above 253/255. Pixels that are fully clipped carry no recoverable "
            f"detail, so the fix reduces exposure and reshapes what remains "
            f"rather than restoring lost information."
        ),
        fix=derive.overexposure_fix,
        evidence=_ev("exposure", "clip_highlight", [0.0, 0.005],
                     {"mean_y": "exposure"}),
        fix_label=lambda m: "Reduce exposure",
    ),
    Rule(
        id="EXP_DUAL_CLIP",
        dimension="exposure",
        title="High dynamic range: both ends clipping",
        when=lambda m: _g(m, "exposure", "clip_shadow") > 0.010
        and _g(m, "exposure", "clip_highlight") > 0.005,
        explain=lambda m: (
            f"{_g(m, 'exposure', 'clip_shadow') * 100:.2f}% of pixels are crushed "
            f"to black AND {_g(m, 'exposure', 'clip_highlight') * 100:.2f}% are "
            f"blown to white. The scene's dynamic range exceeds what the file "
            f"holds, so no single global curve can fix one end without "
            f"worsening the other -- this needs a local (base-layer) operator."
        ),
        fix=derive.dual_clip_fix,
        evidence=_ev("exposure", "clip_shadow", [0.0, 0.010],
                     {"clip_highlight": "exposure"}),
        severity_override="moderate",
        fix_label=lambda m: "Recover both ends locally",
    ),
    Rule(
        id="EXP_SHADOW_CLIP",
        dimension="exposure",
        title="Shadows crushed",
        when=lambda m: _g(m, "exposure", "clip_shadow") > 0.020
        and _g(m, "exposure", "clip_highlight") <= 0.005,
        explain=lambda m: (
            f"{_g(m, 'exposure', 'clip_shadow') * 100:.2f}% of pixels sit at or "
            f"below 2/255, where no detail survives. Lifting the shadow region "
            f"will reveal what is above the floor, but genuinely clipped pixels "
            f"cannot be recovered by any operator."
        ),
        fix=derive.shadow_lift_fix,
        evidence=_ev("exposure", "clip_shadow", [0.0, 0.010]),
        fix_label=lambda m: "Lift shadows",
    ),
    Rule(
        id="EXP_REGIONAL",
        dimension="exposure",
        title="Part of the frame is badly exposed",
        when=lambda m: (
            _g(m, "exposure", "too_dark_region_frac") > 0.12
            or _g(m, "exposure", "too_bright_region_frac") > 0.12
            # Or: the darkest regions fall well short of the 0.45 target while
            # the frame as a whole is unevenly lit. An absolute floor alone
            # missed an image whose dark half averaged 0.26.
            or (_g(m, "exposure", "dark_region_mean") < 0.28
                and _g(m, "exposure", "region_spread") > 0.30)
        ) and 0.28 <= _g(m, "exposure", "mean_y") <= 0.72,
        explain=lambda m: (
            f"Overall exposure looks fine (mean luminance "
            f"{_g(m, 'exposure', 'mean_y'):.2f}), but it is not evenly "
            f"distributed: "
            + (f"{_g(m, 'exposure', 'too_dark_region_frac') * 100:.0f}% of the "
               f"frame sits below 0.18, averaging "
               f"{_g(m, 'exposure', 'dark_region_mean'):.2f} in its darkest "
               f"regions. "
               if _g(m, "exposure", "too_dark_region_frac") > 0.12 else "")
            + (f"{_g(m, 'exposure', 'too_bright_region_frac') * 100:.0f}% sits "
               f"above 0.82. "
               if _g(m, "exposure", "too_bright_region_frac") > 0.12 else "")
            + f"The spread between the darkest and brightest regions is "
              f"{_g(m, 'exposure', 'region_spread'):.2f}. A global brightness "
              f"or exposure change cannot fix this, because the average is "
              f"already correct; it needs a LOCAL adjustment."
        ),
        fix=derive.regional_exposure_fix,
        # Telling a deliberately moody photograph from an unevenly exposed one
        # needs semantics we do not have, so this is offered at moderate
        # confidence as a suggestion the verification step has already shown to
        # measure better -- not as a claim about the photographer's intent.
        confidence=lambda m: float(
            min(0.75, 0.40 + _g(m, "exposure", "region_spread") * 0.5)
        ),
        evidence=_ev("exposure", "too_dark_region_frac", [0.0, 0.12],
                     {"dark_region_mean": "exposure",
                      "bright_region_mean": "exposure",
                      "region_spread": "exposure"}),
        fix_label=lambda m: "Even out the exposure",
    ),
    Rule(
        id="EXP_NARROW",
        dimension="exposure",
        title="Tonal range underused",
        when=lambda m: _g(m, "exposure", "range_usage") < 0.45,
        explain=lambda m: (
            f"The 0.5-99.5 percentile span covers only "
            f"{_g(m, 'exposure', 'range_usage'):.2f} of the available range "
            f"(0.45 or more is normal), so the image never reaches true black or "
            f"true white and looks washed out."
        ),
        fix=derive.autolevels_fix,
        evidence=_ev("exposure", "range_usage", [0.45, 1.0]),
        fix_label=lambda m: "Stretch to full range",
    ),

    # ============================================================ contrast
    Rule(
        id="CON_FLAT",
        dimension="contrast",
        title="Flat / low contrast",
        when=lambda m: _g(m, "contrast", "rms_contrast") < 0.10
        and _g(m, "contrast", "global_local_ratio") < 8.0,
        explain=lambda m: (
            f"RMS contrast is {_g(m, 'contrast', 'rms_contrast'):.3f} against an "
            f"expected 0.10-0.30, and histogram entropy is "
            f"{_g(m, 'contrast', 'entropy'):.2f} bits. The deficit is global "
            f"rather than local, so a range stretch plus a gentle S-curve is the "
            f"right tool."
        ),
        fix=lambda m: (
            derive.autolevels_fix(m)[0]
            + [{"op_id": "tone.contrast_sigmoid",
                "params": {"strength": 2.5, "pivot": 0.5}}],
            derive.autolevels_fix(m)[1]
            + " Followed by a sigmoid curve (strength 2.5) rather than linear "
              "contrast, so the shadows and highlights roll off instead of "
              "hard-clipping."
        ),
        evidence=_ev("contrast", "rms_contrast", [0.14, 0.30],
                     {"entropy": "contrast", "local_contrast": "contrast"}),
        fix_label=lambda m: "Expand contrast",
    ),
    Rule(
        id="CON_LOCAL_FLAT",
        dimension="contrast",
        title="Locally flat (global contrast is fine)",
        when=lambda m: _g(m, "contrast", "local_contrast") < 0.022
        and _g(m, "contrast", "rms_contrast") >= 0.10,
        explain=lambda m: (
            f"Global RMS contrast is adequate at "
            f"{_g(m, 'contrast', 'rms_contrast'):.3f}, but mean local contrast is "
            f"only {_g(m, 'contrast', 'local_contrast'):.3f} (expected 0.022+). "
            f"The image spans a wide range overall while individual regions are "
            f"flat -- which is precisely CLAHE's job, not a curve's."
        ),
        fix=lambda m, long_edge=1600: derive.clahe_fix(m, long_edge),
        evidence=_ev("contrast", "local_contrast", [0.022, 0.25],
                     {"rms_contrast": "contrast", "global_local_ratio": "contrast"}),
        fix_label=lambda m: "Apply CLAHE",
    ),
    Rule(
        id="CON_LOW_ENTROPY",
        dimension="contrast",
        title="Few distinct tones",
        when=lambda m: _g(m, "contrast", "entropy") < 6.0
        and _g(m, "contrast", "rms_contrast") >= 0.10,
        explain=lambda m: (
            f"Histogram entropy is {_g(m, 'contrast', 'entropy'):.2f} bits with "
            f"only {_g(m, 'contrast', 'occupancy') * 100:.0f}% of the 256 "
            f"intensity bins occupied, so tonal transitions are coarse."
        ),
        fix=lambda m, long_edge=1600: derive.clahe_fix(m, long_edge),
        evidence=_ev("contrast", "entropy", [6.5, 8.1],
                     {"occupancy": "contrast"}),
        fix_label=lambda m: "Apply CLAHE (mild)",
    ),
    Rule(
        id="CON_OVER",
        dimension="contrast",
        title="Over-contrasted",
        when=lambda m: _g(m, "contrast", "rms_contrast") > 0.32
        and (_g(m, "exposure", "clip_shadow") > 0.02
             or _g(m, "exposure", "clip_highlight") > 0.01),
        explain=lambda m: (
            f"RMS contrast is {_g(m, 'contrast', 'rms_contrast'):.3f} (above the "
            f"0.30 ceiling) and both ends of the histogram are clipping "
            f"({_g(m, 'exposure', 'clip_shadow') * 100:.1f}% shadow, "
            f"{_g(m, 'exposure', 'clip_highlight') * 100:.1f}% highlight). Adding "
            f"contrast would destroy more detail; no automatic fix is offered."
        ),
        fix=None,
        evidence=_ev("contrast", "rms_contrast", [0.14, 0.30]),
        warning_only=True,
    ),

    # =============================================================== colour
    Rule(
        id="COL_CAST",
        dimension="color",
        title="Colour cast",
        when=lambda m: not is_monochrome(m)
        and _g(m, "color", "cast_magnitude") > 0.13,
        explain=lambda m: (
            f"The combined illuminant estimate implies a "
            f"{m.get('color', {}).get('cast_direction', 'colour')} cast of "
            f"magnitude {_g(m, 'color', 'cast_magnitude'):.3f} (above the 0.115 "
            f"threshold), equivalent to roughly "
            f"{m.get('color', {}).get('cct_estimate', '?')} K "
            f"({m.get('color', {}).get('cct_label', '')}). Lab a*/b* means are "
            f"{m.get('color', {}).get('ab_offset', [0, 0])}, which corroborates it."
        ),
        fix=derive.white_balance_fix,
        evidence=_ev("color", "cast_magnitude", [0.0, 0.115],
                     {"ab_magnitude": "color", "hue_entropy": "color"}),
        confidence=lambda m: _cast_confidence(m),
        fix_label=lambda m: "Neutralise cast",
    ),
    Rule(
        id="COL_DULL",
        dimension="color",
        title="Colours look muted",
        when=lambda m: not is_monochrome(m) and not is_document(m)
        and _g(m, "color", "sat_mean") < 0.15,
        explain=lambda m: (
            f"Mean saturation over non-neutral pixels is "
            f"{_g(m, 'color', 'sat_mean'):.3f}, below the 0.18-0.55 range. "
            f"{_g(m, 'color', 'gray_frac') * 100:.0f}% of the image is "
            f"effectively neutral."
        ),
        fix=derive.vibrance_fix,
        evidence=_ev("color", "sat_mean", [0.18, 0.55],
                     {"gray_frac": "color", "sat_p95": "color"}),
        fix_label=lambda m: "Add vibrance",
    ),
    Rule(
        id="COL_OVERSAT",
        dimension="color",
        title="Oversaturated",
        when=lambda m: not is_monochrome(m)
        and _g(m, "color", "oversat_frac") > 0.040,
        explain=lambda m: (
            f"{_g(m, 'color', 'oversat_frac') * 100:.1f}% of pixels are at "
            f"saturation 0.95 or above (expected under 2%), where hue detail is "
            f"flattened into solid blocks of colour."
        ),
        fix=derive.desaturate_fix,
        evidence=_ev("color", "oversat_frac", [0.0, 0.020],
                     {"sat_mean": "color"}),
        fix_label=lambda m: "Reduce saturation",
    ),
    Rule(
        id="COL_CHANNEL_CLIP",
        dimension="color",
        title="Single-channel clipping is shifting highlight hues",
        when=lambda m: _g(m, "exposure", "clip_channel_hi") > 0.010,
        explain=lambda m: (
            f"On {_g(m, 'exposure', 'clip_channel_hi') * 100:.2f}% of pixels at "
            f"least one channel is at 254/255 while overall luminance is still "
            f"below 0.94. Those pixels are not 'blown' but their hue is wrong, "
            f"because one channel saturated before the others."
        ),
        fix=lambda m: ([{"op_id": "color.gamut_clip",
                         "params": {"mode": "soft", "knee": 0.85}}],
                       "A soft knee above 0.85 compresses the approach to white "
                       "smoothly, so channels stop saturating independently and "
                       "the hue shift disappears."),
        evidence=_ev("exposure", "clip_channel_hi", [0.0, 0.010]),
        fix_label=lambda m: "Soft-clip highlights",
    ),

    # ============================================================ sharpness
    Rule(
        id="SHP_MOTION",
        dimension="sharpness",
        title="Motion blur",
        when=lambda m: _g(m, "sharpness", "aniso") > 0.45
        and _g(m, "sharpness", "edge_width") > 2.8
        and _g(m, "sharpness", "motion_length") >= 2.0,
        explain=lambda m: (
            f"The log-magnitude spectrum is anisotropic (anisotropy = "
            f"{_g(m, 'sharpness', 'aniso'):.2f}), which indicates directional "
            f"rather than symmetric blur. The spacing of the spectral nulls "
            f"gives a motion length of about "
            f"{_g(m, 'sharpness', 'motion_length'):.1f} px at "
            f"{_g(m, 'sharpness', 'motion_angle'):.0f} deg."
        ),
        fix=derive.motion_deblur_fix,
        evidence=_ev("sharpness", "aniso", [0.0, 0.35],
                     {"motion_length": "sharpness", "motion_angle": "sharpness",
                      "edge_width": "sharpness"}),
        confidence=lambda m: min(1.0, 0.5 + _g(m, "sharpness", "aniso")),
        fix_label=lambda m: "Deconvolve motion blur",
    ),
    Rule(
        id="SHP_NOISE_LIMITED",
        dimension="sharpness",
        title="Looks soft, but noise is the limiting factor",
        when=lambda m: _g(m, "sharpness", "edge_width") > 3.8
        and _g(m, "noise", "sigma_luma_8bit") >= 8.0,
        explain=lambda m: (
            f"Edge width is {_g(m, 'sharpness', 'edge_width'):.2f} px, but noise "
            f"sigma is {_g(m, 'noise', 'sigma_luma_8bit'):.1f}/255, which "
            f"corrupts the sharpness measurement itself. Sharpening now would "
            f"amplify the noise; the recipe denoises first and the sharpness "
            f"should be re-measured afterwards."
        ),
        fix=derive.denoise_fix,
        evidence=_ev("sharpness", "edge_width", [0.0, 3.4],
                     {"sigma_luma_8bit": "noise"}),
        fix_label=lambda m: "Denoise first",
    ),
    Rule(
        id="SHP_SOFT",
        dimension="sharpness",
        title="Soft focus",
        when=lambda m: _g(m, "sharpness", "edge_width") > 3.8
        and _g(m, "sharpness", "aniso") <= 0.35
        and _g(m, "noise", "sigma_luma_8bit") < 8.0
        and _g(m, "sharpness", "ringing") < 0.12,
        explain=lambda m: (
            f"Mean edge width is {_g(m, 'sharpness', 'edge_width'):.2f} px "
            f"(3.4 or below is sharp) and high-frequency energy is only "
            f"{_g(m, 'sharpness', 'hf_ratio') * 100:.2f}% of the total. The blur "
            f"is isotropic (anisotropy {_g(m, 'sharpness', 'aniso'):.2f}), so it "
            f"is defocus or soft optics rather than motion."
        ),
        fix=lambda m: derive.sharpen_fix(m, denoiser_present=False),
        evidence=_ev("sharpness", "edge_width", [0.0, 3.4],
                     {"hf_ratio": "sharpness", "spectrum_slope": "sharpness",
                      "aniso": "sharpness"}),
        fix_label=lambda m: "Sharpen",
    ),
    Rule(
        id="SHP_REGIONAL",
        dimension="sharpness",
        title="Part of the frame is blurred",
        when=lambda m: _g(m, "sharpness", "edge_width_worst_region") > 3.6
        and _g(m, "sharpness", "edge_width_spread") > 1.0
        and _g(m, "sharpness", "edge_width") <= 3.8
        and _g(m, "sharpness", "ringing") < 0.12,
        explain=lambda m: (
            f"Measured over the whole frame the image looks acceptably sharp "
            f"(edge width {_g(m, 'sharpness', 'edge_width'):.2f} px), but that "
            f"average hides an uneven result: the softest regions measure "
            f"{_g(m, 'sharpness', 'edge_width_worst_region'):.2f} px against a "
            f"typical {_g(m, 'sharpness', 'edge_width_median_region'):.2f} px "
            f"elsewhere. That pattern is a moving subject in an otherwise sharp "
            f"frame, a missed focus point, or soft lens corners -- but it is "
            f"also exactly what a deliberate panning shot or a shallow "
            f"depth-of-field portrait looks like, and those are right as taken. "
            f"Nothing measurable distinguishes the two, so this is offered as a "
            f"suggestion rather than a defect. The fix is confined to the soft "
            f"regions by a gradient mask, so the already-sharp parts of the "
            f"frame are not over-sharpened."
        ),
        fix=derive.regional_sharpen_fix,
        evidence=_ev("sharpness", "edge_width_worst_region", [0.0, 3.6],
                     {"edge_width": "sharpness",
                      "edge_width_median_region": "sharpness",
                      "edge_width_spread": "sharpness"}),
        confidence=lambda m: float(
            min(0.55, 0.30 + _g(m, "sharpness", "edge_width_spread") * 0.10)
        ),
        fix_label=lambda m: "Sharpen the soft areas",
    ),
    Rule(
        id="SHP_OVER",
        dimension="sharpness",
        title="Already over-sharpened (haloing)",
        when=lambda m: _g(m, "sharpness", "ringing") > 0.15,
        explain=lambda m: (
            f"Gradient overshoot adjacent to strong edges measures "
            f"{_g(m, 'sharpness', 'ringing'):.2f} and high-frequency energy is "
            f"{_g(m, 'sharpness', 'hf_ratio') * 100:.2f}% of the total, both above "
            f"normal. This image has already been sharpened; sharpening it "
            f"again would deepen the halos. No sharpening is offered."
        ),
        fix=None,
        evidence=_ev("sharpness", "ringing", [0.0, 0.11],
                     {"hf_ratio": "sharpness"}),
        warning_only=True,
        tags=["suppresses_sharpen"],
    ),

    # ================================================================ noise
    Rule(
        id="NSE_IMPULSE",
        dimension="noise",
        title="Salt-and-pepper impulses",
        # Impulse noise is identified by spikes standing over a CLEAN noise
        # floor, not merely by the number of outliers: ordinary heavy sensor
        # noise produces plenty of those too, and routing that to a median
        # filter is the wrong treatment.
        when=lambda m: is_impulse_noise(m),
        explain=lambda m: (
            f"{_g(m, 'noise', 'impulse_frac') * 100:.3f}% of pixels are local "
            f"extrema differing from their 3x3 median by more than 0.22. That is "
            f"impulse noise, not Gaussian: a linear filter would smear each "
            f"impulse over its neighbourhood instead of removing it."
        ),
        fix=derive.denoise_fix,
        evidence=_ev("noise", "impulse_frac", [0.0, 0.0010]),
        fix_label=lambda m: "Remove impulses (adaptive median)",
    ),
    Rule(
        id="NSE_PERIODIC",
        dimension="noise",
        title="Periodic interference",
        when=lambda m: len(m.get("noise", {}).get("periodic_peaks") or []) > 0
        # Not motion blur, whose line PSF produces real periodic spectral
        # ripples, and not an image whose high frequencies have been destroyed
        # by heavy blur -- there is no interference to find in a dead spectrum.
        and _g(m, "sharpness", "aniso") < 0.30
        and _g(m, "sharpness", "spectrum_slope") > -5.0,
        explain=lambda m: (
            f"{len(m.get('noise', {}).get('periodic_peaks') or [])} isolated "
            f"peak(s) stand above the radial mean of the log-spectrum by more "
            f"than 4 sigma. Natural images have a smooth radial spectrum, so "
            f"localised spikes are interference -- scanner banding, a halftone "
            f"screen or electrical pickup."
        ),
        fix=derive.denoise_fix,
        evidence=lambda m: {
            "metric": "periodic_peaks",
            "value": len(m.get("noise", {}).get("periodic_peaks") or []),
            "expected": [0, 0],
            "peaks": m.get("noise", {}).get("periodic_peaks") or [],
        },
        fix_label=lambda m: "Notch out interference",
    ),
    Rule(
        id="NSE_CHROMA",
        dimension="noise",
        title="Colour noise exceeds luma noise",
        when=lambda m: _g(m, "noise", "chroma_ratio") > 1.6
        and _g(m, "noise", "sigma_luma_8bit") > 2.0,
        explain=lambda m: (
            f"Chroma noise is {_g(m, 'noise', 'sigma_chroma_8bit'):.1f}/255 "
            f"against luma noise of {_g(m, 'noise', 'sigma_luma_8bit'):.1f}/255, "
            f"a ratio of {_g(m, 'noise', 'chroma_ratio'):.1f}. This is typical of "
            f"high-ISO JPEG. Denoising the chroma channels alone removes the "
            f"blotches while preserving every bit of luminance detail."
        ),
        fix=derive.denoise_fix,
        evidence=_ev("noise", "chroma_ratio", [0.0, 1.6],
                     {"sigma_chroma_8bit": "noise", "sigma_luma_8bit": "noise"}),
        fix_label=lambda m: "Denoise colour only",
    ),
    Rule(
        id="NSE_GAUSS",
        dimension="noise",
        title="Sensor noise",
        when=lambda m: _g(m, "noise", "sigma_luma_8bit") > 3.0
        and not is_impulse_noise(m)
        and _g(m, "noise", "chroma_ratio") <= 1.6
        and not (m.get("noise", {}).get("periodic_peaks") or []),
        explain=lambda m: (
            f"Noise standard deviation is "
            f"{_g(m, 'noise', 'sigma_luma_8bit'):.1f}/255 (measured over the "
            f"flattest 5% of tiles, which avoids counting texture as noise), "
            f"against an expected 3.0 or below. SNR is "
            f"{_g(m, 'noise', 'snr_db'):.1f} dB."
        ),
        fix=derive.denoise_fix,
        evidence=_ev("noise", "sigma_luma_8bit", [0.0, 3.0],
                     {"snr_db": "noise", "sigma_fast_8bit": "noise"}),
        confidence=lambda m: max(0.4, _g(m, "noise", "estimator_agreement")),
        fix_label=lambda m: "Denoise",
    ),
    Rule(
        id="NSE_SHADOW",
        dimension="noise",
        title="Noise concentrated in the shadows",
        when=lambda m: _g(m, "noise", "noise_in_shadows_8bit")
        > 2.2 * max(_g(m, "noise", "sigma_luma_8bit"), 0.5)
        and _g(m, "noise", "noise_in_shadows_8bit") > 4.0,
        explain=lambda m: (
            f"Noise measured only in the darkest quarter of the tonal range is "
            f"{_g(m, 'noise', 'noise_in_shadows_8bit'):.1f}/255, against "
            f"{_g(m, 'noise', 'sigma_luma_8bit'):.1f}/255 overall. Denoising "
            f"globally would soften the well-exposed areas unnecessarily, so "
            f"the fix is masked to the shadows."
        ),
        fix=derive.shadow_denoise_fix,
        evidence=_ev("noise", "noise_in_shadows_8bit", [0.0, 5.0],
                     {"sigma_luma_8bit": "noise"}),
        fix_label=lambda m: "Denoise shadows only",
    ),

    # ============================================================ artifacts
    Rule(
        id="ART_BLOCK",
        dimension="artifacts",
        title="JPEG blocking",
        when=lambda m: _g(m, "artifacts", "blockiness") > 1.18,
        explain=lambda m: (
            f"Pixel differences across the 8-pixel grid are "
            f"{_g(m, 'artifacts', 'blockiness'):.2f}x larger than elsewhere "
            f"(1.0 means no blocking), implying a JPEG quality near "
            f"{int(_g(m, 'artifacts', 'jpeg_quality_est'))}. The detail the codec "
            f"discarded is gone; only the visibility of the artifacts can be "
            f"reduced."
        ),
        fix=derive.deblock_fix,
        evidence=_ev("artifacts", "blockiness", [0.0, 1.12],
                     {"jpeg_quality_est": "artifacts"}),
        fix_label=lambda m: "Soften block edges",
    ),
    Rule(
        id="ART_BAND",
        dimension="artifacts",
        title="Banding / posterization",
        when=lambda m: _g(m, "artifacts", "banding_score") > 0.14,
        explain=lambda m: (
            f"Banding score is {_g(m, 'artifacts', 'banding_score'):.2f}, from "
            f"{int(_g(m, 'artifacts', 'empty_bin_runs'))} runs of empty histogram "
            f"bins together with quantisation steps detected inside smooth "
            f"regions. Gradients will show visible contours."
        ),
        fix=derive.debanding_fix,
        evidence=_ev("artifacts", "banding_score", [0.0, 0.09],
                     {"empty_bin_runs": "artifacts", "hist_comb": "artifacts"}),
        fix_label=lambda m: "Reduce banding",
    ),
    Rule(
        id="ART_PRESTRETCHED",
        dimension="artifacts",
        title="Already heavily edited",
        when=lambda m: _g(m, "artifacts", "hist_comb") > 0.22,
        explain=lambda m: (
            f"{_g(m, 'artifacts', 'hist_comb') * 100:.0f}% of histogram bins are "
            f"empty, which is the signature of an image that has already been "
            f"stretched. Further tonal expansion will spread the gaps wider and "
            f"posterize, so all tone fixes in this recipe are capped at 60% "
            f"strength."
        ),
        fix=None,
        evidence=_ev("artifacts", "hist_comb", [0.0, 0.06]),
        warning_only=True,
        tags=["caps_tone_fixes"],
    ),

    # ============================================================= geometry
    Rule(
        id="GEO_TILT",
        dimension="geometry",
        title="Image is tilted",
        when=lambda m: abs(_g(m, "geometry", "tilt_angle")) > 0.40
        and _g(m, "geometry", "tilt_confidence") > 0.30,
        explain=lambda m: (
            f"The dominant line orientation is "
            f"{_g(m, 'geometry', 'tilt_angle'):+.2f} deg off axis, supported by "
            f"{_g(m, 'geometry', 'tilt_confidence') * 100:.0f}% of the total "
            f"detected line length. Correcting this resamples the image and "
            f"crops the corners, so it is offered rather than applied "
            f"automatically."
        ),
        fix=derive.deskew_fix,
        evidence=_ev("geometry", "tilt_angle", [-0.40, 0.40],
                     {"tilt_confidence": "geometry"}),
        confidence=lambda m: min(1.0, _g(m, "geometry", "tilt_confidence") * 2.0),
        opt_in=True,
        fix_label=lambda m: f"Rotate {-_g(m, 'geometry', 'tilt_angle'):+.2f} deg",
    ),
    Rule(
        id="GEO_CA",
        dimension="geometry",
        title="Chromatic aberration",
        when=lambda m: _g(m, "geometry", "ca_shift_px") > 0.80,
        explain=lambda m: (
            f"The red and blue channels are misregistered against green by "
            f"{_g(m, 'geometry', 'ca_shift_px'):.2f} px at the frame edge, which "
            f"shows as colour fringing on high-contrast edges toward the corners."
        ),
        fix=derive.ca_fix,
        evidence=_ev("geometry", "ca_shift_px", [0.0, 0.60]),
        fix_label=lambda m: "Correct fringing",
    ),

    # =========================================================== uniformity
    Rule(
        id="UNI_HAZE",
        dimension="uniformity",
        title="Haze / atmospheric veiling",
        when=lambda m: _g(m, "uniformity", "dark_channel_mean") > 0.44
        and _g(m, "uniformity", "contrast_vs_depth") > 0.10,
        explain=lambda m: (
            f"The dark channel averages "
            f"{_g(m, 'uniformity', 'dark_channel_mean'):.3f}; haze-free outdoor "
            f"images sit below 0.40, because most patches contain something dark "
            f"in at least one channel. Local contrast also correlates with the "
            f"transmission estimate at r = "
            f"{_g(m, 'uniformity', 'contrast_vs_depth'):.2f}, which distinguishes "
            f"real haze from a scene that is simply low in contrast."
        ),
        fix=lambda m, long_edge=1600: derive.dehaze_fix(m, long_edge),
        evidence=_ev("uniformity", "dark_channel_mean", [0.0, 0.40],
                     {"contrast_vs_depth": "uniformity",
                      "mean_transmission": "uniformity"}),
        confidence=lambda m: min(
            1.0, 0.45 + _g(m, "uniformity", "contrast_vs_depth")),
        fix_label=lambda m: "Cut through haze",
    ),
    Rule(
        id="UNI_VIGNETTE",
        dimension="uniformity",
        title="Vignetting",
        when=lambda m: _g(m, "uniformity", "vignette_falloff") > 0.15,
        explain=lambda m: (
            f"A radial fit to the low-passed luminance gives corners "
            f"{_g(m, 'uniformity', 'vignette_falloff') * 100:.0f}% darker than the "
            f"centre (12% is the tolerance). The fitted coefficients are "
            f"{m.get('uniformity', {}).get('vignette_fit')}."
        ),
        fix=derive.vignette_fix,
        evidence=_ev("uniformity", "vignette_falloff", [-0.02, 0.12]),
        fix_label=lambda m: "Remove vignette",
    ),
    Rule(
        id="UNI_ILLUM",
        dimension="uniformity",
        title="Uneven illumination",
        when=lambda m: _g(m, "uniformity", "illum_nonuniformity") > 0.78,
        explain=lambda m: (
            f"A fitted low-order surface accounts for "
            f"{_g(m, 'uniformity', 'illum_nonuniformity') * 100:.0f}% of the "
            f"image's contrast (80% is the tolerance), explaining "
            f"{_g(m, 'uniformity', 'illum_fit_r2') * 100:.0f}% of its variance, "
            f"with the ramp running toward "
            f"{_g(m, 'uniformity', 'illum_gradient_dir'):.0f} deg. Illumination is "
            f"multiplicative, so correcting it needs a division or a log-domain "
            f"filter, not a brightness offset. Note that a smooth brightness "
            f"gradient is inherently ambiguous: it cannot be distinguished from "
            f"a scene that genuinely is darker on one side, so this finding is "
            f"reported at reduced confidence and is worth checking by eye."
        ),
        fix=lambda m, long_edge=1600: derive.illumination_fix(
            m, long_edge, document=is_document(m)),
        evidence=_ev("uniformity", "illum_nonuniformity", [0.0, 0.80],
                     {"illum_gradient_mag": "uniformity",
                      "illum_fit_r2": "uniformity"}),
        confidence=lambda m: float(
            min(0.75, 0.30 + _g(m, "uniformity", "illum_fit_r2") * 0.5)
        ),
        fix_label=lambda m: "Flatten illumination",
    ),

    # =============================================================== detail
    Rule(
        id="DET_PLASTIC",
        dimension="detail",
        title="Over-smoothed, texture lost",
        when=lambda m: _g(m, "texture", "detail_loss") > 0.55
        and _g(m, "texture", "edge_density") > 0.003,
        explain=lambda m: (
            f"High-frequency energy is only "
            f"{_g(m, 'texture', 'hf_ratio') * 100:.2f}% of the total, far below "
            f"what the edge density of "
            f"{_g(m, 'texture', 'edge_density') * 100:.1f}% implies "
            f"(detail-loss index {_g(m, 'texture', 'detail_loss'):.2f}). The image "
            f"has been smoothed. No automatic fix is offered: amplifying the "
            f"detail layer cannot recreate texture that is gone, and measuring "
            f"the result shows it costs more than it recovers. Use "
            f"Sharpen > Base/detail decomposition manually if you want to try."
        ),
        # No fix is offered. Amplifying the detail layer cannot recreate
        # detail that was destroyed, and measurement bears that out: on the
        # evaluation set it left the detail-loss index slightly worse while
        # costing overall image health. Reporting the finding is useful;
        # offering a remedy for it would not be honest.
        fix=None,
        evidence=_ev("texture", "detail_loss", [0.0, 0.35],
                     {"hf_ratio": "texture", "edge_density": "texture"}),
        warning_only=True,
    ),
]


RULES_BY_ID = {r.id: r for r in RULES}
