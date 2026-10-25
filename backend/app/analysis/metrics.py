"""Measurement suite. Implements docs/advisor-spec.md Section 2.

Every function returns a flat dict of plain-Python numbers. No learned models:
each value is a statistic of the pixels, computed here and traceable to a
formula in the spec.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core.colorspace import (
    linear_to_srgb,
    luma,
    rgb_to_cct,
    cct_label,
    srgb_to_linear,
    to_lab,
    to_hsv,
    to_ycrcb,
)

EPS = 1e-8


def _f(x, n=5):
    return round(float(x), n)


# =========================================================== 2.1 Exposure

def exposure(img: np.ndarray) -> dict:
    y = np.clip(luma(img), 0, 1)
    y_lin = np.clip(luma(srgb_to_linear(img)) if img.ndim == 3
                    else srgb_to_linear(img), 0, 1)
    pcts = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
    p = {f"p{v}": _f(np.percentile(y, v)) for v in pcts}

    mean_y = float(y.mean())
    mean_lin = float(y_lin.mean())
    clip_hi = float((y >= 253 / 255).mean())
    clip_lo = float((y <= 2 / 255).mean())

    ch_hi = 0.0
    if img.ndim == 3:
        any_hot = np.any(img[:, :, :3] >= 254 / 255, axis=2)
        ch_hi = float((any_hot & (y < 0.94)).mean())

    n_zones = 11
    z = np.clip((y * (n_zones - 1) + 0.5).astype(np.int32), 0, n_zones - 1)
    zone_hist = np.bincount(z.ravel(), minlength=n_zones).astype(np.float64)
    zone_hist /= max(zone_hist.sum(), 1)

    # REGIONAL exposure. A global mean hides the most common real complaint
    # there is: part of the frame is badly exposed while the average looks
    # fine. A backlit subject against a window, or a side-by-side comparison,
    # both average out to a perfectly healthy 0.45.
    h, w = y.shape
    tile = max(32, min(h, w) // 6)
    nh, nw = max(1, h // tile), max(1, w // tile)
    region_means: list[float] = []
    for ty in range(nh):
        for tx in range(nw):
            patch = y[ty * tile : (ty + 1) * tile, tx * tile : (tx + 1) * tile]
            if patch.size:
                region_means.append(float(patch.mean()))
    region_means.sort()
    if region_means:
        k = max(1, len(region_means) // 8)      # darkest and brightest eighths
        dark_region = float(np.mean(region_means[:k]))
        bright_region = float(np.mean(region_means[-k:]))
    else:
        dark_region = bright_region = mean_y
    # How much of the frame is badly exposed, regardless of the average.
    arr = np.array(region_means) if region_means else np.array([mean_y])
    too_dark_frac = float((arr < 0.18).mean())
    too_bright_frac = float((arr > 0.82).mean())

    return {
        "mean_y": _f(mean_y),
        "mean_y_linear": _f(mean_lin),
        "dark_region_mean": _f(dark_region),
        "bright_region_mean": _f(bright_region),
        "region_spread": _f(bright_region - dark_region),
        "too_dark_region_frac": _f(too_dark_frac, 4),
        "too_bright_region_frac": _f(too_bright_frac, 4),
        "median_y": _f(np.median(y)),
        "percentiles": p,
        "clip_shadow": _f(clip_lo, 6),
        "clip_highlight": _f(clip_hi, 6),
        "clip_channel_hi": _f(ch_hi, 6),
        "range_usage": _f(np.percentile(y, 99.5) - np.percentile(y, 0.5)),
        "zone_hist": [_f(v) for v in zone_hist],
        # EV offset to an 18% linear mid-gray target.
        "ev_offset": _f(np.log2(0.18 / max(mean_lin, 1e-4)), 4),
    }


# =========================================================== 2.2 Contrast

def contrast(img: np.ndarray) -> dict:
    y = np.clip(luma(img), 0, 1)
    hist, _ = np.histogram(y, bins=256, range=(0, 1))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    nz = p[p > 0]
    entropy = float(-np.sum(nz * np.log2(nz)))
    occupancy = float(np.count_nonzero(hist) / 256.0)

    p1, p99 = np.percentile(y, 1), np.percentile(y, 99)
    michelson = float((p99 - p1) / max(p99 + p1, EPS))

    # Local contrast: mean of per-tile standard deviation over 16x16 tiles.
    h, w = y.shape
    t = 16
    hh, ww = (h // t) * t, (w // t) * t
    if hh >= t and ww >= t:
        tiles = y[:hh, :ww].reshape(hh // t, t, ww // t, t).transpose(0, 2, 1, 3)
        local = float(tiles.reshape(-1, t * t).std(axis=1).mean())
    else:
        local = float(y.std())

    rms = float(y.std())
    third = n_third = 11 // 3
    zh = np.array(exposure(img)["zone_hist"])
    return {
        "rms_contrast": _f(rms),
        "michelson": _f(michelson),
        "entropy": _f(entropy, 4),
        "occupancy": _f(occupancy, 4),
        "local_contrast": _f(local),
        # The discriminator between a global stretch and CLAHE (spec 2.2).
        "global_local_ratio": _f(rms / max(local, EPS), 4),
        "tonal_balance": {
            "shadows": _f(zh[:third].sum()),
            "mids": _f(zh[third : 11 - third].sum()),
            "highlights": _f(zh[11 - third :].sum()),
        },
    }


# =========================================================== 2.3 Color

def _illuminant_estimates(img: np.ndarray) -> dict:
    lin = srgb_to_linear(img[:, :, :3])
    flat = lin.reshape(-1, 3)
    gw = flat.mean(axis=0)
    wp = np.percentile(flat, 99.0, axis=0)
    pnorm = 6.0
    sg = np.power(np.mean(np.power(np.maximum(flat, 0), pnorm), axis=0), 1.0 / pnorm)

    def norm(e):
        e = np.maximum(e, 1e-6)
        return e / e.mean()

    gw, wp, sg = norm(gw), norm(wp), norm(sg)
    combined = norm(np.median(np.stack([gw, wp, sg]), axis=0))
    return {"gray_world": gw, "white_patch": wp, "shades_of_gray": sg,
            "combined": combined}


def color(img: np.ndarray) -> dict:
    if img.ndim == 2 or img.shape[2] < 3:
        return {"monochrome": True, "gray_frac": 1.0, "cast_magnitude": 0.0,
                "sat_mean": 0.0, "oversat_frac": 0.0, "hue_entropy": 0.0,
                "ab_offset": [0.0, 0.0], "cast_direction": "none",
                "cct_estimate": None, "gains": [1.0, 1.0, 1.0]}

    est = _illuminant_estimates(img)
    e = est["combined"]
    gains = np.maximum(e.mean() / e, 1e-6)
    cast_mag = float(np.max(np.abs(gains - 1.0)))

    hsv = to_hsv(img)
    s, v, hue = hsv[:, :, 1], hsv[:, :, 2], hsv[:, :, 0]
    gray_frac = float((s < 0.05).mean())
    colored = s >= 0.05
    sat_mean = float(s[colored].mean()) if colored.any() else 0.0

    lab = to_lab(img)
    ab = [float(lab[:, :, 1].mean()), float(lab[:, :, 2].mean())]

    hw = np.histogram(hue, bins=36, range=(0, 360), weights=s)[0]
    hp = hw / max(hw.sum(), EPS)
    hnz = hp[hp > 0]
    hue_entropy = float(-np.sum(hnz * np.log2(hnz))) if hnz.size else 0.0

    # Describe the cast in words from the gain vector's dominant axis.
    gi = int(np.argmax(np.abs(gains - 1.0)))
    axis = ["red", "green", "blue"][gi]
    direction = "neutral"
    if cast_mag > 0.02:
        too_much = gains[gi] < 1.0  # a gain below 1 means that channel is excessive
        if axis == "red":
            direction = "warm / red" if too_much else "cool / cyan"
        elif axis == "blue":
            direction = "cool / blue" if too_much else "warm / yellow"
        else:
            direction = "green" if too_much else "magenta"

    cct = rgb_to_cct(e)
    return {
        "monochrome": False,
        "gains": [_f(g, 4) for g in gains],
        "illuminant": [_f(x, 4) for x in e],
        "illuminant_spread": _f(float(np.max([
            np.abs(est["gray_world"] - est["white_patch"]).max(),
            np.abs(est["gray_world"] - est["shades_of_gray"]).max(),
        ])), 4),
        "cast_magnitude": _f(cast_mag, 4),
        "cast_direction": direction,
        "cct_estimate": (None if not np.isfinite(cct) else int(round(cct))),
        "cct_label": cct_label(cct),
        "ab_offset": [_f(ab[0], 3), _f(ab[1], 3)],
        "ab_magnitude": _f(np.hypot(*ab), 3),
        "sat_mean": _f(sat_mean),
        "sat_p95": _f(np.percentile(s, 95)),
        "oversat_frac": _f((s >= 0.95).mean(), 6),
        "gray_frac": _f(gray_frac),
        "hue_entropy": _f(hue_entropy, 4),
    }


# =========================================================== 2.4 Sharpness

def _edge_width(gray: np.ndarray) -> float:
    """Edge width in pixels, measured across both axes (see the worker)."""
    return _edge_width_axis(gray)


def _edge_width_axis(gray: np.ndarray, _axis: int = 0) -> float:
    """Mean edge width in pixels, measured as the 10-90% rise distance.

    Only pixels Canny accepts as edges are measured, and the walk is bounded by
    the local extrema either side of the edge. Measuring the 10-90% rise rather
    than the full monotonic extent matters: on smooth natural content a profile
    can stay monotonic for tens of pixels, which makes a full-extent walk report
    an enormous width for a perfectly sharp edge.
    """
    g = cv2.GaussianBlur(gray, (0, 0), 0.5)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    # Select edges by the image's OWN gradient distribution rather than by
    # Canny's intensity-median heuristic: a heavily blurred image has few
    # gradients above a fixed threshold, and returning 0 there would report
    # "perfectly sharp" for the blurriest input, which is backwards.
    thr = float(np.percentile(mag, 97.0))
    if thr <= 1e-5:
        return 0.0

    # Measure along BOTH axes and take the wider result. Scanning only
    # horizontally across vertical edges makes blur along the vertical axis
    # invisible -- a real motion-blurred photograph reported 1.71 px, i.e.
    # perfectly sharp, because its motion was not horizontal.
    if _axis == 0:
        other = _edge_width_axis(np.ascontiguousarray(gray.T), _axis=1)
    else:
        other = 0.0

    vertical = np.abs(gx) > np.abs(gy)
    ys, xs = np.nonzero((mag >= thr) & vertical)
    if len(xs) < 8:
        return other

    step = max(1, len(xs) // 800)
    ys, xs = ys[::step], xs[::step]
    h, w = g.shape
    span = 6  # local window; wider spans straddle several structures
    widths = []
    for y, x in zip(ys, xs):
        if x < span or x > w - span - 1:
            continue
        prof = g[y, x - span : x + span + 1].astype(np.float64)
        if prof.size < 5:
            continue
        amp = float(prof.max() - prof.min())
        if amp < 0.03:  # too faint to measure reliably
            continue
        slope = float(np.abs(np.diff(prof)).max())
        if slope < 1e-4:
            continue
        # Equivalent rise distance: amplitude divided by the steepest slope.
        # For an ideal step blurred by a Gaussian of width sigma this is
        # sigma*sqrt(2*pi) ~ 2.5*sigma, so it scales directly with the blur and
        # does not depend on how far the profile happens to stay monotonic.
        widths.append(amp / slope)
    if not widths:
        return other
    return max(float(np.median(widths)), other)


def _spectrum_stats(gray: np.ndarray) -> dict:
    g = gray - gray.mean()
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(g * win))
    power = (np.abs(F) ** 2).astype(np.float64)

    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    rad = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rmax = min(cx, cy)
    nb = 128
    idx = np.clip((rad / max(rmax, EPS) * (nb - 1)).astype(np.int32), 0, nb - 1)
    valid = rad <= rmax
    sums = np.bincount(idx[valid], weights=power[valid], minlength=nb)
    counts = np.bincount(idx[valid], minlength=nb)
    radial = sums / np.maximum(counts, 1)

    freqs = np.linspace(1e-4, 0.5, nb)
    lo_i, hi_i = max(1, int(0.04 * nb)), int(0.8 * nb)
    fx = np.log10(freqs[lo_i:hi_i])
    fy = np.log10(np.maximum(radial[lo_i:hi_i], 1e-20))
    slope = float(np.polyfit(fx, fy, 1)[0]) if hi_i > lo_i + 2 else 0.0

    # Energy fraction above a quarter of Nyquist, using the TOTAL power in each
    # annulus rather than its mean. Using the mean lets the lowest rings
    # dominate (P falls as f^-2), which drives the ratio to ~1e-5 on every
    # image and makes it useless as a discriminator. Weighting by the number of
    # modes per ring gives roughly equal energy per octave for natural images,
    # so a sharp photo lands in the 0.05-0.3 range and a blurred one near zero.
    total_e = float(sums.sum()) + EPS
    hf_ratio = float(sums[int(0.5 * nb):].sum() / total_e)

    # Spectral anisotropy from the second-moment matrix of the log spectrum.
    logp = np.log1p(power)
    logp[rad < 0.03 * rmax] = 0.0  # drop DC
    wsum = float(logp.sum()) + EPS
    dx, dy = (xx - cx), (yy - cy)
    mxx = float((logp * dx * dx).sum() / wsum)
    myy = float((logp * dy * dy).sum() / wsum)
    mxy = float((logp * dx * dy).sum() / wsum)
    tr, det = mxx + myy, mxx * myy - mxy * mxy
    disc = max(tr * tr / 4.0 - det, 0.0)
    l1 = tr / 2.0 + np.sqrt(disc)
    l2 = tr / 2.0 - np.sqrt(disc)
    aniso = float((l1 - l2) / max(l1 + l2, EPS))
    # Orientation of the spectrum's major axis. A motion blur attenuates
    # frequencies ALONG the motion, so the spectrum elongates perpendicular to
    # it -- hence the +90 degrees.
    theta = float(np.degrees(0.5 * np.arctan2(2 * mxy, mxx - myy)))
    # The spectrum elongates PERPENDICULAR to the motion (a linear PSF
    # attenuates frequencies along its own direction), and the image y-axis
    # points down while the angle convention is y-up. Both together make the
    # motion direction 90 - theta, not theta + 90: verified against planted
    # kernels at 0, 30, 60, 90, 120 and 150 degrees.
    motion_angle = float((90.0 - theta) % 180.0)

    return {
        "slope": round(slope, 4),
        "hf_ratio": round(hf_ratio, 6),
        "aniso": round(aniso, 4),
        "spectrum_angle": round(theta % 180.0, 2),
        "motion_angle": round(motion_angle, 2),
        "radial": radial,
        "freqs": freqs,
    }


def _motion_length(gray: np.ndarray, angle_deg: float) -> float:
    """Estimate motion-blur length from the spacing of spectral nulls.

    Measured against planted kernels, the ANGLE is recovered to within a few
    degrees, and the length to roughly +/-35% for lengths of about 15 px and
    up, with occasional combinations of length and orientation landing near
    0.65x because the period estimate is quantised by the profile length.
    Below ~12 px the sinc nulls are spaced so widely that only one or two fall
    inside the measurable band and the length becomes unreliable.

    That accuracy is adequate for the use it is put to: the Wiener
    regularisation in the deconvolution operator absorbs a PSF mismatch of
    this order, whereas an unregularised inverse would not.

    A box (line) PSF of length L has a sinc transform whose first null sits at
    1/L cycles per pixel along the motion direction, so measuring the null
    spacing recovers L.
    """
    g = gray - gray.mean()
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(g * win))
    logm = np.log1p(np.abs(F))

    cy, cx = h / 2.0, w / 2.0
    rad = np.deg2rad(angle_deg)
    dx, dy = np.cos(rad), -np.sin(rad)
    n = int(min(cx, cy))
    if n < 12:
        return 0.0
    # Average the profile over a STRIP of parallel lines rather than reading a
    # single one. The ripple is coherent across the strip while the image's own
    # spectral structure is not, so averaging lifts the signal well above it.
    # A single line left the period estimate latching onto the wrong bin, which
    # made the length wrong by factors of two either way.
    px_, py_ = -dy, dx  # unit vector perpendicular to the motion
    ts = np.arange(1, n)
    offsets = np.arange(-6, 7, dtype=np.float32)
    acc = np.zeros(ts.size, np.float64)
    for off in offsets:
        xs = np.clip(cx + dx * ts + px_ * off, 0, w - 1).astype(np.float32)
        ys = np.clip(cy + dy * ts + py_ * off, 0, h - 1).astype(np.float32)
        acc += cv2.remap(logm.astype(np.float32), xs.reshape(-1, 1),
                         ys.reshape(-1, 1), cv2.INTER_LINEAR).ravel()
    prof = (acc / offsets.size).astype(np.float32)
    if prof.size < 24:
        return 0.0

    # A linear PSF of length L has transfer function sinc(pi f L), whose nulls
    # sit at f = k/L. Along this profile, index t corresponds to frequency t/N,
    # so the nulls repeat every N/L samples: the profile carries a RIPPLE whose
    # period gives L directly.
    #
    # Finding the nulls as local minima was tried first and failed badly -- it
    # latched onto noise in the profile and over-reported the length roughly
    # fourfold. Estimating the ripple's period spectrally (a one-dimensional
    # cepstrum) uses every null at once and is far more stable.
    d = prof - cv2.blur(prof.reshape(-1, 1), (1, 15)).ravel()
    d = d * np.hanning(d.size).astype(np.float32)
    spec = np.abs(np.fft.rfft(d))
    if spec.size < 6:
        return 0.0
    # Period P samples <-> bin index k = len(d)/P. Only periods between ~3 and
    # ~len/3 samples are physically meaningful here.
    n = d.size
    k_lo = max(2, int(np.ceil(n / (n / 3.0))))
    k_hi = min(spec.size - 1, int(n / 3.0))
    if k_hi <= k_lo:
        return 0.0
    band = spec[k_lo : k_hi + 1]
    k = int(np.argmax(band)) + k_lo
    if band.max() <= 3.0 * float(np.median(spec[k_lo : k_hi + 1])):
        return 0.0  # no dominant ripple: probably not linear motion
    period = n / float(k)            # null spacing, in profile samples
    N = 2.0 * float(min(cx, cy))     # samples across the full frequency axis
    L = N / max(period, 1e-6)
    return float(np.clip(L, 0.0, min(h, w) / 3.0))


def _ringing_ratio(g: np.ndarray, gm2: np.ndarray | None = None) -> float:
    """Edge-profile overshoot: the signature of over-sharpening.

    Measured the way ISO 12233 defines it, as a signed pattern across the edge
    rather than an amount of local variation. An unsharp mask pushes the bright
    side of an edge ABOVE its own plateau and the dark side BELOW, so comparing
    the value 2 px from the edge against the plateau 6 px out isolates halos.

    Two simpler measures were tried first and both failed: absolute overshoot
    cannot tell a halo from noise (noise raises variation everywhere), and
    normalising by flat-area variation turns the metric into a blur detector
    (blur shrinks the denominator). Returns 0 for a clean edge and grows with
    the halo's depth as a fraction of edge amplitude.
    """
    gs = cv2.GaussianBlur(g, (0, 0), 0.5)
    u8 = (np.clip(gs, 0, 1) * 255).astype(np.uint8)
    med = float(np.median(u8))
    edges = cv2.Canny(u8, max(5.0, 0.66 * med), min(250.0, 1.33 * med))
    gx = cv2.Sobel(gs, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3)
    vertical = np.abs(gx) > np.abs(gy)
    ys, xs = np.nonzero((edges > 0) & vertical)
    if len(xs) < 8:
        return 0.0

    step = max(1, len(xs) // 600)
    ys, xs = ys[::step], xs[::step]
    h, w = gs.shape
    reach = 6
    vals = []
    for y, x in zip(ys, xs):
        if x < reach + 1 or x > w - reach - 2:
            continue
        prof = gs[y, x - reach : x + reach + 1].astype(np.float64)
        amp = float(prof.max() - prof.min())
        if amp < 0.04:
            continue
        c = reach
        left_near, left_far = float(prof[c - 2]), float(prof[c - reach])
        right_near, right_far = float(prof[c + 2]), float(prof[c + reach])
        # Whichever side is brighter should not exceed its own plateau.
        if right_far > left_far:
            over = max(0.0, right_near - right_far) + max(0.0, left_far - left_near)
        else:
            over = max(0.0, left_near - left_far) + max(0.0, right_far - right_near)
        vals.append(over / amp)
    if not vals:
        return 0.0
    return float(np.clip(np.median(vals), 0.0, 5.0))


def _regional_edge_width(gray: np.ndarray) -> tuple[float, float]:
    """Worst and median edge width across tiles.

    A single global figure averages a blurred subject together with a sharp
    background, so a moving subject in an otherwise sharp frame reads as
    perfectly sharp -- which is exactly how a real motion-blurred photograph
    slipped through. Returns (worst, median) over tiles that contain enough
    edge content to measure.
    """
    h, w = gray.shape
    tile = max(64, min(h, w) // 4)
    widths: list[float] = []
    for ty in range(0, h - tile // 2, tile):
        for tx in range(0, w - tile // 2, tile):
            patch_ = gray[ty : ty + tile, tx : tx + tile]
            if patch_.shape[0] < 32 or patch_.shape[1] < 32:
                continue
            ew = _edge_width_axis(np.ascontiguousarray(patch_))
            if ew > 0:
                widths.append(ew)
    if not widths:
        return 0.0, 0.0
    widths.sort()
    # The 85th percentile rather than the maximum, so one odd tile cannot
    # declare the whole frame blurred.
    worst = float(widths[min(len(widths) - 1, int(len(widths) * 0.85))])
    return worst, float(np.median(widths))


def sharpness(img: np.ndarray, saliency_map: np.ndarray | None = None) -> dict:
    g = np.clip(luma(img), 0, 1).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    vol = float(lap.var())
    var_g = float(g.var())

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    gm2 = gx * gx + gy * gy
    thr = float(np.percentile(np.sqrt(gm2), 70))
    sel = np.sqrt(gm2) > thr
    tenengrad = float(gm2[sel].mean()) if sel.any() else 0.0

    sp = _spectrum_stats(g)
    ew = _edge_width(g)
    ew_worst, ew_median = _regional_edge_width(g)

    ringing = _ringing_ratio(g, gm2)

    motion_len = 0.0
    if sp["aniso"] > 0.25:
        motion_len = _motion_length(g, sp["motion_angle"])

    sharp_roi = vol / max(var_g, EPS)
    if saliency_map is not None and saliency_map.shape == g.shape:
        wmap = saliency_map / max(float(saliency_map.sum()), EPS)
        sharp_roi = float(np.sum((lap**2) * wmap) / max(var_g, EPS))

    return {
        "vol": _f(vol, 8),
        "vol_norm": _f(vol / max(var_g, EPS), 6),
        "tenengrad": _f(tenengrad, 8),
        "edge_width": _f(ew, 3),
        "edge_width_worst_region": _f(ew_worst, 3),
        "edge_width_median_region": _f(ew_median, 3),
        "edge_width_spread": _f(max(ew_worst - ew_median, 0.0), 3),
        "hf_ratio": _f(sp["hf_ratio"], 6),
        "spectrum_slope": _f(sp["slope"], 4),
        "aniso": _f(sp["aniso"], 4),
        "motion_angle": _f(sp["motion_angle"], 2),
        "motion_length": _f(motion_len, 2),
        "ringing": _f(ringing, 5),
        "sharp_roi": _f(sharp_roi, 6),
    }


# =========================================================== 2.5 Noise

IMMERKAER_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def immerkaer_sigma(gray: np.ndarray) -> float:
    """Fast noise estimate. advisor-spec.md 2.5.

    The kernel annihilates any locally-linear signal, so what survives is
    almost entirely noise; the constant converts the mean absolute response to
    a standard deviation.
    """
    h, w = gray.shape
    if h < 5 or w < 5:
        return 0.0
    resp = cv2.filter2D(gray, cv2.CV_32F, IMMERKAER_K)
    resp = resp[1:-1, 1:-1]
    s = float(np.sum(np.abs(resp)))
    return float(np.sqrt(np.pi / 2.0) * s / (6.0 * (w - 2) * (h - 2)))


def robust_sigma(gray: np.ndarray, tile: int = 16, keep_frac: float = 0.08) -> float:
    """Noise sigma from the Immerkaer response over the FLATTEST tiles only.

    Immerkaer's kernel is used because its noise gain is known exactly: the
    kernel's squared coefficients sum to 36, so for white noise of standard
    deviation sigma the response has standard deviation 6*sigma, and
    E|response| = 6*sigma*sqrt(2/pi). That gives an unbiased estimator with no
    fitted constant.

    Restricting it to the flattest tiles is what makes it robust: applied to
    the whole frame the kernel also passes texture, which inflates the
    estimate on detailed images. An earlier version used the residual after a
    3x3 median instead, which has no analytic gain and under-reported the true
    sigma by about 37% at every level.
    """
    h, w = gray.shape
    if h < tile * 2 or w < tile * 2:
        return immerkaer_sigma(gray)

    resp = np.abs(cv2.filter2D(gray, cv2.CV_32F, IMMERKAER_K))
    # Choose the flat tiles from a SMOOTHED copy. Ranking tiles by the
    # gradient energy of the noisy image preferentially selects tiles whose
    # noise realisation happens to be quiet, which biases the estimate low by
    # about a quarter. Smoothing first makes the selection depend on image
    # structure alone, independent of the noise being measured.
    smooth = cv2.GaussianBlur(gray, (0, 0), 2.0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    energy = gx * gx + gy * gy

    nh, nw = h // tile, w // tile

    def tiles_of(a: np.ndarray) -> np.ndarray:
        t = a[: nh * tile, : nw * tile].reshape(nh, tile, nw, tile)
        return t.transpose(0, 2, 1, 3).reshape(nh * nw, -1)

    e = tiles_of(energy).mean(axis=1)
    r = tiles_of(resp)

    k = max(1, int(len(e) * keep_frac))
    flat_idx = np.argsort(e)[:k]
    # Trim the kernel's border rows/columns are already excluded by tiling
    # away from the frame edge; take the mean absolute response on the flat
    # tiles and invert the analytic gain.
    vals = r[flat_idx].ravel()
    if vals.size == 0:
        return immerkaer_sigma(gray)
    return float(np.sqrt(np.pi / 2.0) * float(np.mean(vals)) / 6.0)


def noise(img: np.ndarray) -> dict:
    g = np.clip(luma(img), 0, 1).astype(np.float32)
    sigma_robust = robust_sigma(g)
    sigma_fast = immerkaer_sigma(g)

    sigma_chroma = 0.0
    per_channel = []
    if img.ndim == 3 and img.shape[2] >= 3:
        ycc = to_ycrcb(img)
        sc = [robust_sigma(np.ascontiguousarray(ycc[:, :, i])) for i in (1, 2)]
        sigma_chroma = float(np.mean(sc))
        per_channel = [robust_sigma(np.ascontiguousarray(img[:, :, i]))
                       for i in range(3)]

    med3 = cv2.medianBlur(g, 3)
    se = np.ones((3, 3), np.uint8)
    # Allow a small tolerance when testing for a local extremum. Requiring the
    # pixel to be EXACTLY the neighbourhood min or max works on synthetic
    # impulses but fails on a JPEG, where ringing around each impulse leaves it
    # a hair short of extreme.
    tol = 2.0 / 255.0
    is_extreme = ((g >= cv2.dilate(g, se) - tol)
                  | (g <= cv2.erode(g, se) + tol))
    # The deviation threshold must scale with the Gaussian noise already
    # present: at sigma = 25/255 a large fraction of pixels are local extrema
    # differing from the 3x3 median by more than a fixed 0.22, so a constant
    # threshold reports heavy sensor noise as salt-and-pepper and routes it to
    # a median filter instead of a denoiser. An impulse is an outlier RELATIVE
    # to the noise floor.
    # A FIXED threshold, deliberately not scaled by the measured sigma. A
    # sigma-relative threshold is not comparable between two versions of the
    # same image: denoising lowers sigma, which lowers the threshold, which
    # counts MORE pixels as impulses -- so a successful denoise scored as if it
    # had added impulse noise, and the advisor withdrew the fix that worked.
    # Discrimination against Gaussian noise is done by impulse_ratio below,
    # which is the right place for it.
    impulse_thr = 0.10
    impulse = float(((np.abs(g - med3) > impulse_thr) & is_extreme).mean())

    # Separate the two cases properly. What distinguishes impulse noise is not
    # how many outliers there are but how EXTREME they are: salt and pepper
    # sits at the ends of the range, whereas Gaussian noise does not. Measure
    # the share of outliers that are also near pure black or pure white.
    outliers = (np.abs(g - med3) > impulse_thr) & is_extreme
    if outliers.any():
        at_rails = ((g <= 0.10) | (g >= 0.90)) & outliers
        impulse_extremity = float(at_rails.sum() / max(outliers.sum(), 1))
    else:
        impulse_extremity = 0.0

    # Spikes per unit of noise floor. Impulse noise produces many spikes
    # over a CLEAN floor, so this runs high; Gaussian noise raises the
    # floor itself, so it runs low however many outliers it throws.
    impulse_ratio = float(impulse / max(sigma_robust, 1.0 / 255.0))

    shadows = g < 0.25
    sigma_shadow = robust_sigma(g) if shadows.mean() < 0.02 else robust_sigma(
        np.where(shadows, g, float(g[shadows].mean()) if shadows.any() else 0.0)
        .astype(np.float32))

    from ..ops.freq import detect_spectral_peaks
    peaks = detect_spectral_peaks(g, sensitivity=4.5, max_peaks=6)

    mean_y = float(g.mean())
    snr = 20.0 * np.log10(max(mean_y, EPS) / max(sigma_robust, 1e-6))

    # Disagreement between the two estimators lowers the finding's confidence.
    agreement = 1.0 - min(abs(sigma_fast - sigma_robust)
                          / max(sigma_robust, 1e-4), 1.0)
    return {
        "sigma_luma": _f(sigma_robust, 6),
        "sigma_luma_8bit": _f(sigma_robust * 255, 3),
        "sigma_fast_8bit": _f(sigma_fast * 255, 3),
        "estimator_agreement": _f(max(agreement, 0.0), 4),
        "sigma_chroma": _f(sigma_chroma, 6),
        "sigma_chroma_8bit": _f(sigma_chroma * 255, 3),
        "chroma_ratio": _f(sigma_chroma / max(sigma_robust, 1e-6), 4),
        "sigma_per_channel_8bit": [_f(s * 255, 3) for s in per_channel],
        "impulse_frac": _f(impulse, 6),
        "impulse_extremity": _f(impulse_extremity, 4),
        "impulse_ratio": _f(impulse_ratio, 4),
        "impulse_threshold": _f(impulse_thr, 5),
        "noise_in_shadows_8bit": _f(sigma_shadow * 255, 3),
        "snr_db": _f(snr, 2),
        "periodic_peaks": peaks,
    }


# =========================================================== 2.6 Artifacts

def artifacts(img: np.ndarray) -> dict:
    g = np.clip(luma(img), 0, 1).astype(np.float32)
    h, w = g.shape

    # Blockiness: mean |diff| across the 8-px grid vs. elsewhere.
    dh = np.abs(np.diff(g, axis=1))
    dv = np.abs(np.diff(g, axis=0))
    bh_cols = np.arange(7, dh.shape[1], 8)
    bv_rows = np.arange(7, dv.shape[0], 8)
    on = []
    off = []
    if len(bh_cols):
        on.append(dh[:, bh_cols].mean())
        mask = np.ones(dh.shape[1], bool)
        mask[bh_cols] = False
        off.append(dh[:, mask].mean())
    if len(bv_rows):
        on.append(dv[bv_rows, :].mean())
        mask = np.ones(dv.shape[0], bool)
        mask[bv_rows] = False
        off.append(dv[mask, :].mean())
    blockiness = float(np.mean(on) / max(np.mean(off), EPS)) if on else 1.0

    hist, _ = np.histogram(g, bins=256, range=(0, 1))
    # Measure gaps only INSIDE the occupied tonal range. Counting every empty
    # bin would score a correctly-exposed image that simply does not reach pure
    # black or pure white as heavily "combed", which is wrong: a comb is a gap
    # between used levels, not an unused end of the range.
    used = np.nonzero(hist)[0]
    if used.size >= 2:
        lo_b, hi_b = int(used[0]), int(used[-1])
        inner = hist[lo_b : hi_b + 1]
        empty = inner == 0
        hist_comb = float(empty.mean())
    else:
        empty = np.zeros(1, bool)
        hist_comb = 0.0

    # Runs of >= 2 adjacent empty bins inside the range: the signature of a
    # prior stretch, which spreads levels apart and leaves regular gaps.
    runs = 0
    run = 0
    for e in empty:
        if e:
            run += 1
        else:
            if run >= 2:
                runs += 1
            run = 0
    if run >= 2:
        runs += 1

    # Banding: are the levels present in smooth regions QUANTISED? A smooth
    # gradient that only takes a few distinct values over a wide span is
    # posterised; merely having a non-zero gradient is not banding.
    local_var = cv2.blur(g * g, (9, 9)) - cv2.blur(g, (9, 9)) ** 2
    smooth = local_var < (3.0 / 255.0) ** 2
    banding = 0.0
    if smooth.mean() > 0.02:
        vals = (g[smooth] * 255.0).round().astype(np.int32)
        if vals.size > 64:
            distinct = int(np.unique(vals).size)
            span = int(vals.max() - vals.min()) + 1
            # A continuous gradient uses nearly every level in its span; a
            # posterised one uses a small fraction of them.
            if span >= 8:
                banding = float(np.clip(1.0 - distinct / span, 0.0, 1.0))
    # Gate the smooth-region term on the histogram actually having gaps. Blur
    # also produces smooth regions spanning few distinct values, but it does
    # not punch holes in the histogram, whereas quantisation does.
    banding *= float(np.clip(hist_comb / 0.02, 0.0, 1.0))
    banding_score = float(min(1.0, runs / 40.0 + banding))

    ringing = _ringing_ratio(g)

    # Rough JPEG quality estimate from blockiness (monotonic, not exact).
    q_est = int(np.clip(100 - (blockiness - 1.0) * 220, 5, 100))
    return {
        "blockiness": _f(blockiness, 4),
        "banding_score": _f(banding_score, 4),
        "empty_bin_runs": int(runs),
        "hist_comb": _f(hist_comb, 4),
        "ringing": _f(ringing, 5),
        "jpeg_quality_est": q_est,
    }


# =========================================================== 2.7 Geometry

def geometry(img: np.ndarray) -> dict:
    from ..ops.geom import _hough_tilt

    g = np.clip(luma(img), 0, 1).astype(np.float32)
    angle, conf = _hough_tilt(g, 20.0)
    # Deskewing fundamentally requires linear structure. With little or no
    # supporting line length the angle is noise, so report no tilt rather than
    # a spurious one -- otherwise the geometry score wanders between
    # measurements of the same scene.
    if conf < 0.22:
        angle = 0.0

    # Lateral chromatic aberration: best per-channel radial rescale vs green.
    ca = 0.0
    if img.ndim == 3 and img.shape[2] >= 3:
        h, w = g.shape
        gref = np.abs(cv2.Sobel(img[:, :, 1], cv2.CV_32F, 1, 1, ksize=3))
        shifts = []
        for ci in (0, 2):
            ch = np.abs(cv2.Sobel(img[:, :, ci], cv2.CV_32F, 1, 1, ksize=3))
            best, bests = 0.0, -np.inf
            for s in np.linspace(0.996, 1.004, 9):
                M = np.array([[s, 0, (1 - s) * (w - 1) / 2.0],
                              [0, s, (1 - s) * (h - 1) / 2.0]], np.float32)
                warped = cv2.warpAffine(ch, M, (w, h), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT_101)
                sc = float(np.sum(warped * gref))
                if sc > bests:
                    bests, best = sc, s
            shifts.append(abs(best - 1.0) * max(h, w) / 2.0)
        ca = float(max(shifts))

    # Uniform border detection (scan bars, letterboxing).
    border = 0.0
    for axis_slice in (g[:4, :], g[-4:, :], g[:, :4], g[:, -4:]):
        if axis_slice.size and float(axis_slice.std()) < 0.01:
            border = max(border, 0.02)

    return {
        "tilt_angle": _f(angle, 3),
        "tilt_confidence": _f(conf, 4),
        "ca_shift_px": _f(ca, 3),
        "border_frac": _f(border, 4),
    }


# =========================================================== 2.8 Uniformity

def _radial_falloff(gray: np.ndarray, rings: int = 12) -> tuple[float, float, float]:
    """Vignetting from concentric RING AVERAGES of luminance.

    Averaging over a full ring cancels scene content, which a least-squares
    polynomial over the whole frame does not: on an image with no radial
    structure that fit happily returns a large spurious falloff. A real vignette
    also declines monotonically outward, so the fraction of ring-to-ring steps
    that decrease is reported and used to gate the result.

    Returns (gated_falloff, raw_falloff, monotonicity).
    """
    h, w = gray.shape[:2]
    small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float32)
    c = 63.5
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / np.sqrt(2 * c * c)
    idx = np.clip((r * rings).astype(np.int32), 0, rings - 1)
    means = np.array([
        float(small[idx == i].mean()) if np.any(idx == i) else np.nan
        for i in range(rings)
    ])
    ok = ~np.isnan(means)
    if ok.sum() < 4:
        return 0.0, 0.0, 0.0
    means = means[ok]
    centre = float(means[: max(1, len(means) // 4)].mean())
    edge = float(means[-max(1, len(means) // 4):].mean())
    raw = float(np.clip(1.0 - edge / max(centre, 1e-4), -1.0, 1.0))

    steps = np.diff(means)
    mono = float((steps < 0).mean()) if steps.size else 0.0
    # Require a predominantly outward decline before calling it a vignette.
    # Scene content perturbs individual ring steps, so a strong falloff is
    # accepted on its magnitude alone.
    gated = raw if (raw > 0 and mono >= 0.72) else 0.0
    return gated, raw, mono


def _illumination_gradient(gray: np.ndarray) -> tuple[float, float]:
    """Uneven illumination as the amplitude of a fitted low-order surface.

    Three simpler measures were tried and each failed on natural images:

    * std/mean of the low-passed luma is inflated by a small mean, so any dark
      frame scores as unevenly lit;
    * a Michelson ratio over the low-passed luma is exposure-invariant but
      rises under any contrast stretch, so plain auto-levels "worsened" it;
    * the share of total contrast that is large-scale does not discriminate at
      all, because a 1/f image has most of its energy at low frequency by
      definition.

    What distinguishes real uneven illumination from a scene's own large-scale
    content is its SHAPE: a lighting gradient is well described by a plane or a
    gentle quadratic, while scene content is not. So fit
    L ~ a + bx + cy + dx^2 + ey^2 + fxy, take the fitted surface's amplitude
    across the frame relative to the image's own contrast, and gate it on how
    much variance that model actually explains.

    Returns (amplitude_fraction, r_squared).
    """
    small = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2.0)
    yy, xx = np.mgrid[0:48, 0:48].astype(np.float64)
    x = (xx / 47.0) - 0.5
    y = (yy / 47.0) - 0.5
    A = np.stack([np.ones_like(x).ravel(), x.ravel(), y.ravel(),
                  (x * x).ravel(), (y * y).ravel(), (x * y).ravel()], axis=1)
    z = small.ravel().astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    fit = A @ coef

    ss_res = float(np.sum((z - fit) ** 2))
    ss_tot = float(np.sum((z - z.mean()) ** 2))
    r2 = float(np.clip(1.0 - ss_res / max(ss_tot, 1e-12), 0.0, 1.0))

    # Amplitude of the illumination model relative to the image's own contrast,
    # which makes it invariant to both exposure and the tone curve.
    amp = float(np.percentile(fit, 98) - np.percentile(fit, 2))
    contrast = float(np.percentile(gray, 98) - np.percentile(gray, 2))
    frac = amp / max(contrast, 1e-3)

    # Below roughly half the variance explained, the "gradient" is just scene
    # content that least squares happened to fit.
    gated = frac if r2 >= 0.45 else frac * (r2 / 0.45) ** 2
    return float(np.clip(gated, 0.0, 3.0)), r2


def uniformity(img: np.ndarray) -> dict:
    from ..ops.geom import fit_vignette_q

    g = np.clip(luma(img), 0, 1).astype(np.float32)
    a, b, c, fit_r2 = fit_vignette_q(g)
    falloff, falloff_raw, radial_mono = _radial_falloff(g)

    nonuni, illum_r2 = _illumination_gradient(g)

    # Direction of the fitted linear illumination ramp.
    h, w = g.shape
    low = cv2.GaussianBlur(g, (0, 0), max(min(h, w) / 8.0, 2.0))
    small = cv2.resize(low, (32, 32), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0:32, 0:32].astype(np.float32)
    A = np.stack([np.ones(32 * 32), xx.ravel() / 31, yy.ravel() / 31], axis=1)
    coef, *_ = np.linalg.lstsq(A, small.ravel(), rcond=None)
    ramp_dir = float(np.degrees(np.arctan2(coef[2], coef[1])))
    ramp_mag = float(np.hypot(coef[1], coef[2]))

    base = img if img.ndim == 3 else np.dstack([img] * 3)
    mn = np.min(base[:, :, :3], axis=2)
    k = max(3, (int(0.02 * max(h, w)) | 1))
    dark = cv2.erode(mn, np.ones((k, k), np.uint8))
    dc_mean = float(dark.mean())
    flat = dark.ravel()
    n = max(1, int(0.001 * flat.size))
    airlight = float(np.mean(np.sort(flat)[-n:]))
    transmission = np.clip(1.0 - 0.95 * dark / max(airlight, 1e-3), 0, 1)

    # Corroborate haze: under real haze, local contrast tracks transmission.
    lc = cv2.blur(g * g, (17, 17)) - cv2.blur(g, (17, 17)) ** 2
    lc = np.sqrt(np.maximum(lc, 0))
    tsmall = cv2.resize(transmission, (64, 64), interpolation=cv2.INTER_AREA).ravel()
    lsmall = cv2.resize(lc, (64, 64), interpolation=cv2.INTER_AREA).ravel()
    if tsmall.std() > 1e-6 and lsmall.std() > 1e-6:
        corr = float(np.corrcoef(tsmall, lsmall)[0, 1])
    else:
        corr = 0.0

    return {
        "vignette_falloff": _f(falloff, 4),
        "vignette_falloff_raw": _f(falloff_raw, 4),
        "vignette_monotonicity": _f(radial_mono, 4),
        "vignette_fit_r2": _f(fit_r2, 4),
        "vignette_fit": [_f(a, 5), _f(b, 5), _f(c, 5)],
        "illum_nonuniformity": _f(nonuni, 4),
        "illum_fit_r2": _f(illum_r2, 4),
        "illum_gradient_dir": _f(ramp_dir, 1),
        "illum_gradient_mag": _f(ramp_mag, 4),
        "dark_channel_mean": _f(dc_mean, 5),
        "airlight_estimate": _f(airlight, 5),
        "mean_transmission": _f(float(transmission.mean()), 5),
        "contrast_vs_depth": _f(corr, 4),
    }


# =========================================================== 2.9 Texture

def texture(img: np.ndarray) -> dict:
    from skimage.feature import graycomatrix, graycoprops

    g = np.clip(luma(img), 0, 1).astype(np.float32)
    u8 = (g * 255).astype(np.uint8)

    med = float(np.median(u8))
    edges = cv2.Canny(u8, max(0, 0.66 * med), min(255, 1.33 * med))
    edge_density = float((edges > 0).mean())

    q = (u8.astype(np.uint16) * 32 // 256).astype(np.uint8)
    props: dict = {}
    try:
        m = graycomatrix(q, distances=[1], angles=[0, np.pi / 4, np.pi / 2,
                                                   3 * np.pi / 4],
                         levels=32, symmetric=True, normed=True)
        for name in ("contrast", "homogeneity", "energy", "correlation",
                     "dissimilarity"):
            props[name] = _f(float(graycoprops(m, name).mean()), 6)
    except Exception:
        pass

    sp = _spectrum_stats(g)
    # A detailed image (many edges) should have proportionate high-frequency
    # energy. A large shortfall means the detail was smoothed away.
    # Calibrated against the clean fixtures: a sharp natural image with this
    # edge density lands at roughly this high-frequency share.
    expected_hf = 0.0003 + edge_density * 0.15
    detail_loss = float(np.clip(1.0 - sp["hf_ratio"] / max(expected_hf, 1e-6), 0, 1))

    from skimage.feature import local_binary_pattern
    try:
        codes = local_binary_pattern(u8, 8, 1, method="uniform")
        hist = np.bincount(codes.astype(np.int32).ravel(), minlength=10).astype(float)
        hist /= max(hist.sum(), 1)
        lbp_uniformity = _f(float(hist[:9].sum()), 5)
    except Exception:
        lbp_uniformity = 0.0

    # Fraction of pixels whose 5x5 neighbourhood has essentially zero
    # variance. A photograph always carries some sensor noise, so this is
    # ~0 for real imagery; synthetic graphics (charts, diagrams, screenshots,
    # renders) are full of perfectly flat areas. Measured on the samples it
    # separates with an enormous margin: 0.00-0.05 photographic against
    # 0.89-0.92 graphic.
    # Scale the flatness threshold by the image's own contrast. An absolute
    # threshold mistook a very dark photograph for a chart: once the signal is
    # squeezed into the bottom of the range, absolute variance is tiny
    # everywhere, which is underexposure rather than synthetic flatness.
    span = float(np.percentile(g, 98) - np.percentile(g, 2))
    flat_thr = max(0.004 * max(span, 0.02), 0.3 / 255.0) ** 2
    local_mean = cv2.blur(g, (5, 5))
    local_var = cv2.blur(g * g, (5, 5)) - local_mean * local_mean
    flat = local_var < flat_thr

    # Synthetic graphics also use very few distinct colours. Counted on a
    # heavily downsampled copy, so JPEG block structure cannot inflate it.
    rgb = img if img.ndim == 3 else np.dstack([img] * 3)
    small = cv2.resize((np.clip(rgb[:, :, :3], 0, 1) * 255).astype(np.uint8),
                       (96, 96), interpolation=cv2.INTER_AREA)
    distinct = int(len(np.unique(small.reshape(-1, 3), axis=0)))
    few_colours = distinct < 1500     # a 96x96 photograph uses thousands

    graphic_frac = float(flat.mean()) if few_colours else 0.0

    return {
        "edge_density": _f(edge_density, 5),
        "graphic_frac": _f(graphic_frac, 5),
        "glcm": props,
        "detail_loss": _f(detail_loss, 4),
        "hf_ratio": _f(sp["hf_ratio"], 6),
        "lbp_uniformity": lbp_uniformity,
    }


# =========================================================== orchestration

GROUPS = {
    "exposure": exposure,
    "contrast": contrast,
    "color": color,
    "sharpness": sharpness,
    "noise": noise,
    "artifacts": artifacts,
    "geometry": geometry,
    "uniformity": uniformity,
    "texture": texture,
}


def is_impulse_noise(m: dict) -> bool:
    """Whether the noise is impulsive rather than Gaussian.

    Impulse noise shows as sparse spikes over an otherwise clean image.
    What identifies impulse noise is that the corrupted pixels are SPARSE and
    sit at the ends of the range, with the pixels between them left untouched.
    Dense fine-grained noise is Gaussian however coarse it looks, and belongs
    on the non-local-means route: tested against real photographs, every
    count-based or sigma-relative statistic failed to tell the two apart,
    while the extremity of the outliers separates them by a factor of six.
    """
    nse = m.get("noise", {})
    return (float(nse.get("impulse_frac", 0.0) or 0.0) > 0.004
            and float(nse.get("impulse_extremity", 0.0) or 0.0) > 0.5)


def measure_all(img: np.ndarray, groups: list[str] | None = None,
                with_saliency: bool = True) -> dict:
    """Run the measurement suite. Returns {group: {metric: value}}."""
    names = groups or list(GROUPS)
    out: dict = {}
    sal = None
    if with_saliency and "sharpness" in names:
        try:
            from ..ops.inspect import spectral_saliency
            s = spectral_saliency(np.clip(luma(img), 0, 1).astype(np.float32))
            sal = cv2.resize(s, (img.shape[1], img.shape[0]),
                             interpolation=cv2.INTER_CUBIC)
        except Exception:
            sal = None
    for n in names:
        fn = GROUPS.get(n)
        if fn is None:
            continue
        try:
            out[n] = fn(img, sal) if n == "sharpness" else fn(img)
        except Exception as exc:  # one failing metric must not kill the report
            out[n] = {"error": f"{type(exc).__name__}: {exc}"}
    return out
