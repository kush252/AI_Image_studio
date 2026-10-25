"""Metric -> operator parameters. Implements docs/advisor-spec.md Section 5.

Every function returns (ops, derivation_string). The derivation string is
REQUIRED: it is what makes a suggested number auditable instead of magic.
"""
from __future__ import annotations

import math

import numpy as np
from ..analysis.metrics import is_impulse_noise


def clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _op(op_id: str, **params) -> dict:
    return {"op_id": op_id, "params": params}


# ----------------------------------------------------------------- 5.1 exposure

def exposure_fix(m: dict) -> tuple[list[dict], str]:
    ex = m["exposure"]
    mean_y = float(ex["mean_y"])
    clip_hi = float(ex["clip_highlight"])
    target = 0.45

    if clip_hi < 0.002:
        # Headroom exists, so a multiplicative exposure change is correct.
        stops = clamp(math.log2(target / max(mean_y, 0.02)), -2.5, 2.5)
        # Cap the gain by the actual headroom above the 99.5th percentile, so
        # the correction cannot itself create clipping. Without this, raising a
        # dark image by ~1 EV after a white-balance gain clips whichever
        # channel the balance boosted.
        p995 = float(ex["percentiles"].get("p99.9", 1.0))
        headroom_stops = math.log2(0.98 / max(p995, 0.02))
        note_cap = ""
        if stops > headroom_stops:
            note_cap = (f" Capped at {headroom_stops:.2f} EV by the available "
                        f"headroom (the 99.9th percentile sits at {p995:.3f}), "
                        f"so the correction cannot itself clip highlights.")
            stops = max(min(stops, headroom_stops), 0.0)
        return ([_op("tone.exposure", stops=round(stops, 3))],
                f"stops = log2({target} / {mean_y:.3f}) = {stops:.2f}, "
                f"clamped to +/-2.5. Exposure (a linear-light multiply) is "
                f"chosen because only {clip_hi * 100:.2f}% of pixels are near "
                f"white, so there is headroom to raise it without clipping."
                + note_cap)

    # Highlights are already at risk: gamma leaves white fixed, so it cannot
    # blow them further.
    gamma = clamp(math.log(target) / math.log(max(mean_y, 0.02)), 0.4, 2.5)
    return ([_op("tone.gamma", gamma=round(gamma, 3))],
            f"gamma = log({target}) / log({mean_y:.3f}) = {gamma:.2f}, clamped "
            f"to [0.4, 2.5]. Gamma is chosen over exposure because "
            f"{clip_hi * 100:.2f}% of pixels are already near white; a power "
            f"law holds white fixed while lifting shadows.")


def overexposure_fix(m: dict) -> tuple[list[dict], str]:
    ex = m["exposure"]
    mean_y = float(ex["mean_y"])
    clip_hi = float(ex["clip_highlight"])
    target = 0.45
    stops = clamp(math.log2(target / max(mean_y, 0.02)), -2.5, 0.0)
    ops = [_op("tone.exposure", stops=round(stops, 3))]
    note = (f"stops = log2({target} / {mean_y:.3f}) = {stops:.2f}. ")
    if clip_hi > 0.005:
        recover = clamp(-0.25 - clip_hi * 8.0, -0.9, -0.15)
        ops.append(_op("tone.shadows_highlights",
                       highlights=round(recover, 3), shadows=0.0, radius=40.0))
        note += (f"Highlight recovery = {recover:.2f} because {clip_hi * 100:.2f}% "
                 f"of pixels are clipped; note that fully clipped detail is "
                 f"unrecoverable and recovery only reshapes what remains.")
    return ops, note


def dual_clip_fix(m: dict) -> tuple[list[dict], str]:
    ex = m["exposure"]
    lo, hi = float(ex["clip_shadow"]), float(ex["clip_highlight"])
    shadows = clamp(0.25 + lo * 6.0, 0.15, 0.85)
    highlights = clamp(-(0.20 + hi * 8.0), -0.85, -0.12)
    return ([_op("tone.shadows_highlights", shadows=round(shadows, 3),
                 highlights=round(highlights, 3), radius=48.0,
                 midtone_contrast=0.08)],
            f"shadows = +{shadows:.2f} from {lo * 100:.2f}% crushed pixels; "
            f"highlights = {highlights:.2f} from {hi * 100:.2f}% blown pixels. "
            f"A local (base-layer) operator is required here: no single global "
            f"curve can lift the shadows without pushing the highlights "
            f"further into clipping.")


def shadow_lift_fix(m: dict) -> tuple[list[dict], str]:
    lo = float(m["exposure"]["clip_shadow"])
    shadows = clamp(0.20 + lo * 7.0, 0.12, 0.80)
    return ([_op("tone.shadows_highlights", shadows=round(shadows, 3),
                 highlights=0.0, radius=40.0)],
            f"shadows = +{shadows:.2f}, scaled from the {lo * 100:.2f}% of "
            f"pixels sitting at or below 2/255.")


# ------------------------------------------------------------- 5.2 auto levels

def autolevels_fix(m: dict) -> tuple[list[dict], str]:
    art = m.get("artifacts", {})
    col = m.get("color", {})
    ex = m["exposure"]
    comb = float(art.get("hist_comb", 0.0) or 0.0)
    low_pct = 0.5 if comb < 0.15 else 1.5
    high_pct = 99.5 if float(ex["clip_highlight"]) < 0.002 else 99.9
    per_channel = float(col.get("cast_magnitude", 0.0) or 0.0) > 0.08
    return ([_op("tone.autolevels", low_pct=low_pct, high_pct=high_pct,
                 per_channel=per_channel)],
            f"Percentiles {low_pct}/{high_pct}. "
            + (f"Gentler low percentile because hist_comb = {comb:.2f} shows the "
               f"image was already stretched. " if comb >= 0.15 else "")
            + (f"Per-channel enabled because cast_magnitude = "
               f"{col.get('cast_magnitude'):.3f} > 0.08, so the stretch also "
               f"neutralises the cast." if per_channel
               else "Luma-only to avoid shifting hue."))


# ------------------------------------------------------------------ 5.3 CLAHE

def clahe_fix(m: dict, long_edge: int = 1600) -> tuple[list[dict], str]:
    con = m["contrast"]
    nse = m.get("noise", {})
    local = float(con["local_contrast"])
    sigma8 = float(nse.get("sigma_luma_8bit", 0.0) or 0.0)

    deficit = max(0.0, 0.14 - local)
    clip = clamp(1.5 + deficit * 45.0, 1.0, 4.0)
    note_noise = ""
    if sigma8 > 6.0:
        clip *= 0.65
        note_noise = (f" Reduced by 35% because the measured noise sigma is "
                      f"{sigma8:.1f}/255 and CLAHE amplifies noise in flat tiles.")
    tiles = int(clamp(round(long_edge / 180.0), 4, 12))
    return ([_op("tone.clahe", clip=round(clip, 2), tiles_x=tiles, tiles_y=tiles,
                 space="lab_l")],
            f"clip = 1.5 + (0.14 - {local:.3f}) * 45 = {clip:.2f}; tiles = "
            f"{tiles}x{tiles} from a {long_edge} px long edge. Operating on "
            f"CIE L* only, because per-channel CLAHE shifts hue."
            + note_noise)


# ----------------------------------------------------------- 5.4 white balance

def white_balance_fix(m: dict) -> tuple[list[dict], str]:
    col = m["color"]
    cast = float(col["cast_magnitude"])
    gains = np.array(col["gains"], dtype=float)
    hue_entropy = float(col.get("hue_entropy", 5.0) or 5.0)
    mean_y = float(m.get("exposure", {}).get("mean_y", 0.45) or 0.45)

    strength = clamp(cast / 0.25, 0.3, 1.0)
    note = (f"strength = cast_magnitude / 0.25 = {cast:.3f} / 0.25 = "
            f"{strength:.2f}, so a subtle cast gets a subtle correction.")
    if not (0.22 <= mean_y <= 0.70):
        strength *= 0.5
        note += (f" Halved because mean luminance is {mean_y:.2f}: both the "
                 f"gray-world and white-patch estimators depend on the tonal "
                 f"distribution, so the cast estimate from a badly exposed "
                 f"frame is unreliable. Fix exposure and re-measure.")
    if hue_entropy < 1.5:
        strength *= 0.5
        note += (f" Halved because hue entropy is only {hue_entropy:.2f} bits: "
                 f"the scene is dominated by a single hue, which is often a "
                 f"deliberate light (a sunset, neon, a studio gel) rather than "
                 f"a cast. Neutralising that would be the classic auto-white-"
                 f"balance failure.")

    applied = 1.0 + (gains - 1.0) * strength
    applied = np.clip(applied, 0.75, 1.35)
    return ([_op("color.channel_gains", r=round(float(applied[0]), 4),
                 g=round(float(applied[1]), 4), b=round(float(applied[2]), 4),
                 linear_light=True)],
            note + f" Gains {[round(float(x), 3) for x in applied]} applied in "
                   f"linear light, where a channel gain physically models an "
                   f"illuminant change, and hard-clamped to [0.75, 1.35].")


# --------------------------------------------------------------- 5.5 vibrance

def vibrance_fix(m: dict) -> tuple[list[dict], str]:
    sat = float(m["color"]["sat_mean"])
    amount = clamp((0.30 - sat) * 2.2, 0.0, 0.55)
    return ([_op("color.vibrance", amount=round(amount, 3), protect_skin=True)],
            f"amount = (0.30 - {sat:.3f}) * 2.2 = {amount:.2f}. Vibrance rather "
            f"than flat saturation: the (1-S) weighting lifts muted colours "
            f"while leaving already-saturated pixels and skin tones alone.")


def desaturate_fix(m: dict) -> tuple[list[dict], str]:
    over = float(m["color"]["oversat_frac"])
    amount = clamp(-(over * 3.0), -0.5, -0.05)
    return ([_op("color.vibrance", amount=round(amount, 3), protect_skin=False)],
            f"amount = -{over:.3f} * 3 = {amount:.2f}, from the "
            f"{over * 100:.1f}% of pixels at saturation >= 0.95 where hue "
            f"detail is already lost.")


# ------------------------------------------------------------- 5.6 sharpening

def sharpen_fix(m: dict, denoiser_present: bool = False) -> tuple[list[dict], str]:
    sh = m["sharpness"]
    nse = m.get("noise", {})
    ew = float(sh["edge_width"])
    sigma8 = float(nse.get("sigma_luma_8bit", 0.0) or 0.0)

    radius = clamp(ew / 2.2, 0.6, 3.0)
    gap = clamp(2.2 - min(ew, 2.2), 0.0, 2.2)
    amount = clamp(0.35 + gap * 0.55, 0.0, 1.4)

    notes = [f"radius = edge_width / 2.2 = {ew:.2f} / 2.2 = {radius:.2f} px, "
             f"matching the blur's own scale"]
    # Edge width ABOVE the target means blur; the amount should grow with it.
    if ew > 2.2:
        amount = clamp(0.35 + (ew - 2.2) * 0.35, 0.35, 1.4)
        notes.append(f"amount = 0.35 + ({ew:.2f} - 2.2) * 0.35 = {amount:.2f}")
    else:
        notes.append(f"amount = {amount:.2f}")

    if denoiser_present:
        amount *= 0.55
        notes.append("scaled by 0.55 because a denoiser is also in the recipe; "
                     "full denoise plus full sharpen fight each other")
    noise_scale = clamp(1.0 - (sigma8 - 3.0) / 12.0, 0.3, 1.0)
    if noise_scale < 1.0:
        amount *= noise_scale
        notes.append(f"scaled by {noise_scale:.2f} for measured noise "
                     f"sigma = {sigma8:.1f}/255")
    threshold = clamp(sigma8 / 255.0 * 2.5, 0.0, 0.05)

    op_id = "sharpen.structure_aware" if sigma8 > 3.0 else "sharpen.unsharp"
    if op_id == "sharpen.structure_aware":
        ops = [_op(op_id, amount=round(amount, 3), radius=round(radius, 2),
                   edge_threshold=round(max(threshold, 0.02), 4), noise_guard=0.6)]
        notes.append("using the gradient-masked sharpener so flat (noisy) "
                     "regions are left alone")
    else:
        ops = [_op(op_id, radius=round(radius, 2), amount=round(amount, 3),
                   threshold=round(threshold, 4))]
    return ops, "; ".join(notes) + "."


def motion_deblur_fix(m: dict) -> tuple[list[dict], str]:
    sh = m["sharpness"]
    length = float(sh.get("motion_length", 0.0) or 0.0)
    angle = float(sh.get("motion_angle", 0.0) or 0.0)
    if length < 2.0:
        length = clamp(float(sh["edge_width"]) * 1.5, 3.0, 20.0)
    iters = int(clamp(round(6 + length * 0.6), 8, 22))
    sigma8 = float(m.get("noise", {}).get("sigma_luma_8bit", 0.0) or 0.0)
    # Regularisation from the measured noise, with a floor: the PSF is only an
    # estimate, and under-regularised deconvolution amplifies the mismatch into
    # ringing rather than recovering detail.
    noise_term = clamp(0.004 + (sigma8 / 255.0) ** 2 * 40, 0.004, 0.2)
    return ([_op("sharpen.deconv", method="wiener", psf="motion",
                 psf_size=round(length, 1), angle=round(angle, 1),
                 noise=round(noise_term, 4), iterations=iters)],
            f"Motion PSF of {length:.1f} px at {angle:.0f} deg, recovered from "
            f"the spectrum: anisotropy = {sh['aniso']:.2f} indicates directional "
            f"blur, and the spacing of the spectral nulls along that direction "
            f"gives the length. Wiener regularisation K = {noise_term:.4f} from "
            f"the measured noise of {sigma8:.1f}/255, with a floor: the PSF is "
            f"only estimated, and under-regularised deconvolution turns that "
            f"mismatch into ringing instead of detail. Deconvolution, not "
            f"unsharp masking, "
            f"because the blur has a known kernel to invert.")


def defocus_deblur_fix(m: dict) -> tuple[list[dict], str]:
    ew = float(m["sharpness"]["edge_width"])
    sigma8 = float(m.get("noise", {}).get("sigma_luma_8bit", 0.0) or 0.0)
    radius = clamp(ew / 2.0, 1.5, 12.0)
    noise_term = clamp(0.002 + (sigma8 / 255.0) ** 2 * 40, 0.002, 0.2)
    return ([_op("sharpen.deconv", method="wiener", psf="disc",
                 psf_size=round(radius * 2, 1), noise=round(noise_term, 4))],
            f"Disc PSF of diameter {radius * 2:.1f} px from the measured edge "
            f"width of {ew:.2f} px. Wiener regularisation K = {noise_term:.4f}, "
            f"derived from noise sigma {sigma8:.1f}/255 -- too small a K would "
            f"amplify that noise enormously.")


# ---------------------------------------------------------------- 5.7 denoise

def denoise_fix(m: dict) -> tuple[list[dict], str]:
    nse = m["noise"]
    sigma8 = float(nse["sigma_luma_8bit"])
    impulse = float(nse["impulse_frac"])
    chroma_ratio = float(nse.get("chroma_ratio", 0.0) or 0.0)
    peaks = nse.get("periodic_peaks") or []

    if is_impulse_noise(m):
        k = int(clamp(3 + 2 * math.ceil(40 * impulse), 3, 9))
        if k % 2 == 0:
            k += 1
        return ([_op("smooth.median_adaptive", k_max=k)],
                f"Impulse (salt-and-pepper) noise on {impulse * 100:.2f}% of "
                f"pixels. k_max = 3 + 2*ceil(40 * {impulse:.4f}) = {k}. The "
                f"adaptive median is used rather than a plain median because it "
                f"replaces ONLY pixels identified as impulses, leaving all "
                f"others bit-identical.")

    if peaks:
        return ([_op("freq.notch_auto", sensitivity=4.0, radius=0.025,
                     max_peaks=min(len(peaks) + 1, 8))],
                f"{len(peaks)} periodic interference peak(s) detected in the "
                f"spectrum at "
                + ", ".join(f"({p['fx']:.3f}, {p['fy']:.3f}) cyc/px"
                            for p in peaks[:3])
                + ". Notch filtering removes these without touching the rest of "
                  "the spectrum, which a spatial denoiser cannot do.")

    if chroma_ratio > 1.6 and sigma8 > 2.0:
        h = clamp(1.2 * float(nse["sigma_chroma_8bit"]), 3.0, 20.0)
        return ([_op("smooth.nlm", h=round(h, 1), template=7, search=21,
                     scope="chroma")],
                f"Chroma noise ({nse['sigma_chroma_8bit']:.1f}/255) exceeds luma "
                f"noise ({sigma8:.1f}/255) by {chroma_ratio:.1f}x, which is "
                f"typical of high-ISO JPEG. Denoising Cb/Cr only with h = "
                f"1.2 * {nse['sigma_chroma_8bit']:.1f} = {h:.1f} preserves every "
                f"bit of luma detail.")

    # Measured on real photographs rather than synthetic plates. Three things
    # the synthetic calibration had wrong:
    #  * bilateral at sigma_color = 1.8*sigma/255 is very nearly a no-op --
    #    it moved sigma 5.06 -> 5.04 and 11.03 -> 10.78 -- so the "cheap
    #    option below 9/255" was buying nothing at all;
    #  * h = 0.8*sigma under-treats; h = 1.0*sigma removed roughly twice as
    #    much noise with no measured cost in sharpness or detail;
    #  * a denoiser removes the noise that was dithering the quantisation
    #    steps, uncovering banding that was latent in the image all along.
    #    That showed up as a 35-point artifacts regression at a strength
    #    that had barely denoised anything, which vetoed the whole recipe.
    #    A trace of grain afterwards restores the dither and takes histogram
    #    combing back to zero.
    regrain_sigma = round(clamp(0.22 * sigma8, 0.6, 2.0), 2)
    regrain = _op("smooth.regrain", sigma=regrain_sigma, monochrome=True)

    if sigma8 <= 16.0:
        h = clamp(1.0 * sigma8, 2.0, 16.0)
        return ([_op("smooth.nlm", h=round(h, 1), template=7, search=21,
                     scope="all"), regrain],
                f"Gaussian noise, sigma = {sigma8:.1f}/255. h = 1.0 * "
                f"{sigma8:.1f} = {h:.1f}: non-local means retains detail far "
                f"better than a spatial filter, and matching h to sigma was "
                f"measured to remove about twice the noise of the h = 0.8*sigma "
                f"this used to apply, at no cost in sharpness. A bilateral "
                f"pass was tried here first and is not used because at this "
                f"sigma it removed essentially nothing. Followed by "
                f"{regrain_sigma}/255 of grain, which restores the dither the "
                f"denoiser removed -- without it the denoised image shows "
                f"banding that the noise had been masking."
                )
    return ([_op("smooth.wavelet", wavelet="db4", levels=3, method="BayesShrink",
                 sigma=round(sigma8 / 255.0, 5), mode="soft"),
             _op("smooth.bilateral", d=5, sigma_color=0.06, sigma_space=5.0),
             regrain],
            f"Heavy noise, sigma = {sigma8:.1f}/255. Wavelet BayesShrink "
            f"(db4, 3 levels, sigma = {sigma8 / 255:.4f}) followed by a mild "
            f"bilateral pass to clean the residual, then {regrain_sigma}/255 "
            f"of grain to restore the dither. Above sigma = 16 a spatial "
            f"filter would have to blur away real detail to compete.")


def shadow_denoise_fix(m: dict) -> tuple[list[dict], str]:
    nse = m["noise"]
    s_sh = float(nse["noise_in_shadows_8bit"])
    h = clamp(1.1 * s_sh, 3.0, 20.0)
    return ([{"op_id": "smooth.nlm",
              "params": {"h": round(h, 1), "template": 7, "search": 21,
                         "scope": "all"},
              "mask": {"kind": "luminosity", "range": [0.0, 0.30],
                       "feather": 0.06}}],
            f"Noise in the shadows is {s_sh:.1f}/255 against "
            f"{nse['sigma_luma_8bit']:.1f}/255 overall, so the denoiser is "
            f"confined to a luminosity mask over the lower 30% of the tonal "
            f"range. h = 1.1 * {s_sh:.1f} = {h:.1f}.")


# ------------------------------------------------------- 5.8 dehaze/uniformity

def dehaze_fix(m: dict, long_edge: int = 1600) -> tuple[list[dict], str]:
    uni = m["uniformity"]
    dc = float(uni["dark_channel_mean"])
    # Dehazing is a tone/contrast recovery; the dark-channel measurement drives
    # a shadows/highlights + local-contrast pair, since we expose no single
    # dehaze operator.
    strength = clamp((dc - 0.30) * 2.2, 0.15, 0.85)
    clip = clamp(1.5 + (dc - 0.30) * 2.0, 1.3, 2.6)
    tiles = int(clamp(round(long_edge / 200.0), 4, 10))
    return ([_op("tone.clahe", clip=round(clip, 2), tiles_x=tiles, tiles_y=tiles,
                 space="lab_l"),
             _op("tone.shadows_highlights", shadows=0.0,
                 highlights=round(-strength * 0.35, 3), radius=60.0,
                 midtone_contrast=round(strength * 0.45, 3)),
             _op("color.vibrance", amount=round(strength * 0.75, 3))],
            f"Dark-channel mean is {dc:.3f} (haze-free outdoor images sit below "
            f"0.35), and local contrast correlates with the transmission "
            f"estimate at r = {uni['contrast_vs_depth']:.2f}, which corroborates "
            f"haze rather than a genuinely low-contrast scene. Recovery "
            f"strength = ({dc:.3f} - 0.30) * 2.2 = {strength:.2f}; CLAHE clip = "
            f"{clip:.2f}. Saturation is also lifted because airlight desaturates "
            f"the scene as well as flattening it.")


def illumination_fix(m: dict, long_edge: int = 1600,
                     document: bool = False) -> tuple[list[dict], str]:
    uni = m["uniformity"]
    nonuni = float(uni["illum_nonuniformity"])
    if document:
        se = int(clamp(round(long_edge / 12.0), 15, 121))
        if se % 2 == 0:
            se += 1
        return ([_op("morph.bg_divide", method="open", se_size=se,
                     combine="divide", renormalize=True)],
                f"Illumination non-uniformity = {nonuni:.3f}. Document mode: a "
                f"morphological background estimate with a {se} px structuring "
                f"element, divided out. Division (not subtraction) because "
                f"illumination is multiplicative.")
    cutoff = clamp(0.06 + nonuni * 0.2, 0.05, 0.25)
    return ([_op("freq.homomorphic", gamma_low=0.45, gamma_high=1.6,
                 cutoff=round(cutoff, 4), order=2)],
            f"Illumination non-uniformity = {nonuni:.3f} with the ramp running "
            f"at {uni['illum_gradient_dir']:.0f} deg. Homomorphic filtering with "
            f"cutoff = 0.06 + {nonuni:.3f} * 0.2 = {cutoff:.3f}: taking the log "
            f"turns the multiplicative illumination/reflectance product into a "
            f"sum, so a high-pass can separate them.")


def vignette_fix(m: dict) -> tuple[list[dict], str]:
    f = float(m["uniformity"]["vignette_falloff"])
    return ([_op("geom.vignette", auto_remove=True)],
            f"Measured corner falloff is {f * 100:.1f}%, fitted as "
            f"L(r) = a + b r^2 + c r^4 with coefficients "
            f"{m['uniformity']['vignette_fit']}. The correction is the inverse "
            f"of that fitted model, so it is derived from the image rather than "
            f"assumed.")


# ------------------------------------------------------------- geometry fixes

def deskew_fix(m: dict) -> tuple[list[dict], str]:
    geo = m["geometry"]
    a = float(geo["tilt_angle"])
    c = float(geo["tilt_confidence"])
    return ([_op("geom.deskew", max_angle=20.0, auto_crop=True, interp="bicubic")],
            f"Dominant line orientation is {a:+.2f} deg off axis, supported by "
            f"{c * 100:.0f}% of total detected line length. Rotating by "
            f"{-a:+.2f} deg. Note this resamples the image and crops the "
            f"corners, so it is a compositional choice rather than a pure "
            f"correction.")


def ca_fix(m: dict) -> tuple[list[dict], str]:
    s = float(m["geometry"]["ca_shift_px"])
    return ([_op("color.ca_correct", auto=True)],
            f"Lateral chromatic aberration of {s:.2f} px measured between the "
            f"red/blue channels and green. The auto mode re-measures and "
            f"applies the per-channel radial rescale that maximises edge "
            f"alignment.")


# ------------------------------------------------------------ artifact fixes

def deblock_fix(m: dict) -> tuple[list[dict], str]:
    art = m["artifacts"]
    b = float(art["blockiness"])
    q = int(art["jpeg_quality_est"])
    strength = clamp((b - 1.10) * 0.35, 0.02, 0.15)
    return ([_op("smooth.bilateral", d=5, sigma_color=round(strength, 4),
                 sigma_space=4.0, scope="all")],
            f"Blockiness ratio = {b:.2f} (1.0 means no blocking), implying a "
            f"JPEG quality near {q}. A mild bilateral pass with sigma_color = "
            f"({b:.2f} - 1.10) * 0.35 = {strength:.3f} softens the block "
            f"boundaries. Be aware this cannot restore the detail the codec "
            f"discarded -- it only makes the remaining artifacts less visible.")


def debanding_fix(m: dict) -> tuple[list[dict], str]:
    b = float(m["artifacts"]["banding_score"])
    return ([_op("smooth.bilateral", d=7, sigma_color=round(clamp(b * 0.12, 0.01,
                                                                  0.08), 4),
                 sigma_space=9.0, scope="luma")],
            f"Banding score = {b:.2f}, from quantisation steps detected inside "
            f"smooth regions. Gentle edge-preserving smoothing confined to luma "
            f"reduces the visible contours without blurring real edges.")


def detail_recovery_fix(m: dict) -> tuple[list[dict], str]:
    d = float(m["texture"]["detail_loss"])
    gain = clamp(1.0 + d * 1.4, 1.1, 2.4)
    return ([_op("sharpen.detail_split", radius=12.0, base_gain=1.0,
                 detail_gain=round(gain, 2))],
            f"Detail-loss index = {d:.2f}: high-frequency energy is far below "
            f"what the edge density implies, so the image has been "
            f"over-smoothed. Amplifying the detail layer by {gain:.2f}x makes "
            f"the remaining texture more visible, but detail that was destroyed "
            f"cannot be recreated -- amplifying an empty detail layer yields "
            f"nothing.")

def regional_exposure_fix(m: dict) -> tuple[list[dict], str]:
    """Local tone mapping for a frame whose average is fine but whose parts
    are not. A global curve cannot help here by definition: it would move the
    correct regions as much as the incorrect ones."""
    ex = m["exposure"]
    dark_frac = float(ex.get("too_dark_region_frac", 0.0) or 0.0)
    bright_frac = float(ex.get("too_bright_region_frac", 0.0) or 0.0)
    dark_mean = float(ex.get("dark_region_mean", 0.3) or 0.3)
    spread = float(ex.get("region_spread", 0.0) or 0.0)

    shadows = clamp((0.30 - dark_mean) * 2.0 + dark_frac, 0.0, 0.85)
    highlights = -clamp(bright_frac * 1.5, 0.0, 0.7)
    radius = clamp(30.0 + spread * 60.0, 25.0, 90.0)

    notes = [
        f"Shadow lift = (0.30 - {dark_mean:.2f}) * 2 + {dark_frac:.2f} = "
        f"{shadows:.2f}",
    ]
    if highlights < -0.01:
        notes.append(f"highlight recovery = -{bright_frac:.2f} * 1.5 = "
                     f"{highlights:.2f}")
    notes.append(
        f"applied through a guided-filter base layer of radius {radius:.0f} px, "
        f"scaled from the {spread:.2f} spread between the darkest and brightest "
        f"regions. A local operator is required: the frame's average exposure "
        f"is already correct, so any global curve would damage the regions that "
        f"are right in order to fix the ones that are not"
    )
    return ([_op("tone.shadows_highlights", shadows=round(shadows, 3),
                 highlights=round(highlights, 3), radius=round(radius, 1),
                 midtone_contrast=0.05)],
            "; ".join(notes) + ".")


def regional_sharpen_fix(m: dict) -> tuple[list[dict], str]:
    """Sharpen only where the frame is soft.

    Gated by a gradient mask so the already-sharp regions are left alone: a
    global unsharp pass would over-sharpen them to fix the soft ones.
    """
    sh = m["sharpness"]
    worst = float(sh.get("edge_width_worst_region", 4.0) or 4.0)
    spread = float(sh.get("edge_width_spread", 1.0) or 1.0)
    sigma8 = float(m.get("noise", {}).get("sigma_luma_8bit", 0.0) or 0.0)

    radius = clamp(worst / 2.4, 0.7, 3.0)
    amount = clamp(0.3 + spread * 0.28, 0.2, 1.1)
    amount *= clamp(1.0 - (sigma8 - 3.0) / 12.0, 0.3, 1.0)
    return ([_op("sharpen.structure_aware", amount=round(amount, 3),
                 radius=round(radius, 2), edge_threshold=0.03,
                 noise_guard=0.65)],
            f"radius = {worst:.2f} / 2.4 = {radius:.2f} px, from the softest "
            f"regions' edge width; amount = 0.3 + {spread:.2f} * 0.28 = "
            f"{amount:.2f}, scaled by the measured noise of {sigma8:.1f}/255. "
            f"The structure-aware sharpener is used rather than a plain unsharp "
            f"mask because its gradient mask confines the effect to real edges, "
            f"leaving the already-sharp regions and the flat (noisy) areas "
            f"untouched.")
