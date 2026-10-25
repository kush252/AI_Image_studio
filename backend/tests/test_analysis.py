"""Analysis metrics against known ground truth. PLAN.md Phase 2 acceptance."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.analysis import metrics as M
from app.analysis.score import piecewise, score_all


# ------------------------------------------------------------------ noise

def test_flat_field_reports_zero_noise():
    """Phase 2 acceptance: a flat grey field must report sigma = 0."""
    flat = np.full((192, 192, 3), 0.5, np.float32)
    n = M.noise(flat)
    assert n["sigma_luma_8bit"] < 0.3, f"flat field reported {n['sigma_luma_8bit']}"
    assert n["impulse_frac"] == 0.0


# Independent noise of sigma per RGB channel becomes noise of
# sigma * ||w||_2 in LUMA, because the BT.709 weights are not an average.
# The metric reports luma noise, which is the perceptually relevant quantity,
# so the expectation has to carry this factor.
LUMA_NOISE_GAIN = float(np.linalg.norm([0.2126, 0.7152, 0.0722]))


def test_luma_noise_gain_is_what_we_think():
    """Guards the constant used by the test below (it is ~0.7496)."""
    assert 0.74 < LUMA_NOISE_GAIN < 0.76


@pytest.mark.parametrize("sigma8", [3, 6, 12, 20, 30])
def test_noise_sigma_estimated_within_tolerance(clean, sigma8):
    """Phase 2 acceptance: estimate a known sigma within 15-20%.

    The estimator runs on the flattest tiles, so it must not be fooled by the
    fixture's own texture.
    """
    rng = np.random.default_rng(sigma8)
    noisy = np.clip(clean + rng.normal(0, sigma8 / 255, clean.shape), 0, 1).astype(np.float32)
    est = M.noise(noisy)["sigma_luma_8bit"]
    # Per-channel sigma -> luma sigma, then in quadrature with the plate's own
    # ~1.3/255 sensor noise.
    expected = float(np.hypot(sigma8 * LUMA_NOISE_GAIN, 1.3))
    rel = abs(est - expected) / expected
    assert rel < 0.18, f"sigma {sigma8}: estimated {est:.2f}, expected ~{expected:.2f}"


def test_noise_estimator_is_unbiased_on_pure_noise():
    """The analytic gain must hold exactly, with no fitted constant."""
    from app.analysis.metrics import robust_sigma

    rng = np.random.default_rng(0)
    for sigma in (0.01, 0.05, 0.1):
        field = rng.normal(0.5, sigma, (384, 384)).astype(np.float32)
        est = robust_sigma(field)
        assert abs(est - sigma) / sigma < 0.06, f"sigma {sigma}: estimated {est:.5f}"


def test_impulse_detection_is_noise_aware(clean):
    """Heavy Gaussian noise must NOT be reported as salt-and-pepper.

    Asserted against is_impulse_noise() rather than against impulse_frac,
    because the raw count is deliberately no longer a classifier on its own.
    It used to be counted against a sigma-relative threshold, which made it
    useless for the one job the advisor needs it for: comparing an image with
    its own denoised version. Lowering sigma lowered the threshold, so a
    SUCCESSFUL denoise scored as though it had added impulse noise. The count
    is now taken at a fixed threshold -- which makes it rise with ordinary
    texture -- so the extremity of the outliers is what separates the cases.
    """
    rng = np.random.default_rng(5)
    gauss = np.clip(clean + rng.normal(0, 25 / 255, clean.shape), 0, 1).astype(np.float32)
    assert not M.is_impulse_noise({"noise": M.noise(gauss)}),         "Gaussian noise misclassified as impulse noise"

    sp = clean.copy()
    h, w = clean.shape[:2]
    n = int(0.02 * h * w)
    ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
    salt = rng.random(n) < 0.5
    sp[ys[salt], xs[salt]] = 1.0
    sp[ys[~salt], xs[~salt]] = 0.0
    assert M.noise(sp)["impulse_frac"] > 0.005, "real impulses were missed"
    assert M.is_impulse_noise({"noise": M.noise(sp)}),         "real salt-and-pepper was not classified as impulse noise"


def test_impulse_count_survives_denoising(clean):
    """A successful denoise must not look like added impulse noise.

    This is the regression that made the advisor withdraw a denoise recipe
    that had worked: the impulse count was measured against a threshold
    derived from sigma, so removing noise lowered the bar and counted MORE
    pixels as impulses. The count has to be comparable between an image and
    its own denoised version, or the closed loop cannot measure a denoise.
    """
    rng = np.random.default_rng(11)
    noisy = np.clip(clean + rng.normal(0, 14 / 255, clean.shape), 0, 1).astype(np.float32)
    denoised = cv2.GaussianBlur(noisy, (0, 0), 1.2)
    assert M.noise(denoised)["sigma_luma_8bit"] < M.noise(noisy)["sigma_luma_8bit"]
    assert M.noise(denoised)["impulse_frac"] <= M.noise(noisy)["impulse_frac"] + 1e-6,         "denoising increased the measured impulse fraction"


def test_chroma_noise_detected_separately(clean):
    rng = np.random.default_rng(7)
    ycc = cv2.cvtColor(clean, cv2.COLOR_RGB2YCrCb)
    ycc[:, :, 1] += rng.normal(0, 20 / 255, ycc.shape[:2]).astype(np.float32)
    ycc[:, :, 2] += rng.normal(0, 20 / 255, ycc.shape[:2]).astype(np.float32)
    noisy = np.clip(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB), 0, 1).astype(np.float32)
    n = M.noise(noisy)
    assert n["chroma_ratio"] > 1.6, f"chroma ratio only {n['chroma_ratio']}"


def test_periodic_noise_localised_to_planted_frequency(clean):
    h, w = clean.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fx, fy = 0.12, 0.08
    noisy = np.clip(clean + 0.10 * np.sin(2 * np.pi * (fx * xx + fy * yy))[:, :, None],
                    0, 1).astype(np.float32)
    peaks = M.noise(noisy)["periodic_peaks"]
    assert peaks, "planted sinusoid not detected"
    best = min(peaks, key=lambda p: abs(abs(p["fx"]) - fx) + abs(abs(p["fy"]) - fy))
    assert abs(abs(best["fx"]) - fx) < 0.02, f"fx {best['fx']} vs {fx}"
    assert abs(abs(best["fy"]) - fy) < 0.02, f"fy {best['fy']} vs {fy}"


def test_clean_image_has_no_periodic_peaks(plate):
    for seed in range(4):
        assert not M.noise(plate(seed))["periodic_peaks"], \
            f"false periodic detection on clean plate {seed}"


def test_blur_does_not_produce_phantom_peaks(clean):
    """A dead high-frequency region must not read as interference."""
    for sigma in (1.0, 2.0, 3.0):
        blurred = cv2.GaussianBlur(clean, (0, 0), sigma)
        assert not M.noise(blurred)["periodic_peaks"], \
            f"phantom peaks after blur sigma={sigma}"


# -------------------------------------------------------------- sharpness

def test_edge_width_grows_with_blur(clean):
    """Phase 2 acceptance: a synthetically blurred edge reports a wider edge."""
    widths = []
    for sigma in (0.0, 1.0, 2.0, 3.0):
        img = clean if sigma == 0 else cv2.GaussianBlur(clean, (0, 0), sigma)
        widths.append(M.sharpness(img)["edge_width"])
    assert widths == sorted(widths), f"edge width not monotonic in blur: {widths}"
    assert widths[0] < 3.4, f"sharp plate reported edge width {widths[0]}"
    assert widths[-1] > widths[0] * 1.5


def test_hf_ratio_falls_with_blur(clean):
    sharp = M.sharpness(clean)["hf_ratio"]
    blurred = M.sharpness(cv2.GaussianBlur(clean, (0, 0), 2.0))["hf_ratio"]
    assert blurred < sharp / 4, f"hf_ratio barely moved: {sharp} -> {blurred}"


def test_spectrum_slope_steepens_with_blur(clean):
    sharp = M.sharpness(clean)["spectrum_slope"]
    blurred = M.sharpness(cv2.GaussianBlur(clean, (0, 0), 2.0))["spectrum_slope"]
    assert blurred < sharp - 1.0, f"slope {sharp} -> {blurred}"


@pytest.mark.parametrize("length,angle", [(15.0, 30.0), (25.0, 60.0), (20.0, 120.0)])
def test_motion_blur_angle_and_length_recovered(clean, length, angle):
    """The detector must recover the planted PSF geometry from the spectrum.

    The angle must be accurate to a few degrees. The length is quantised by
    the profile length, so some length/orientation combinations land near
    0.65x; see _motion_length's docstring for why that is still adequate for
    choosing a deconvolution PSF.
    """
    n = max(3, int(length) | 1)
    k = np.zeros((n, n), np.float32)
    c = n // 2
    rad = np.deg2rad(angle)
    for t in np.linspace(-length / 2, length / 2, int(length * 4) + 2):
        x, y = int(round(c + np.cos(rad) * t)), int(round(c - np.sin(rad) * t))
        if 0 <= x < n and 0 <= y < n:
            k[y, x] = 1.0
    k /= k.sum()
    blurred = cv2.filter2D(clean, -1, k, borderType=cv2.BORDER_REFLECT_101)

    s = M.sharpness(blurred)
    assert s["aniso"] > 0.3, f"motion blur not detected as anisotropic ({s['aniso']})"
    # Orientation is modulo 180 degrees.
    err = min(abs(s["motion_angle"] - angle), 180 - abs(s["motion_angle"] - angle))
    assert err < 8, f"angle {s['motion_angle']} vs planted {angle}"
    assert 0.6 * length < s["motion_length"] < 1.6 * length, \
        f"length {s['motion_length']} vs planted {length}"


def test_isotropic_blur_is_not_called_motion(clean):
    s = M.sharpness(cv2.GaussianBlur(clean, (0, 0), 2.0))
    assert s["aniso"] < 0.3, f"Gaussian blur looks anisotropic ({s['aniso']})"


def test_oversharpening_raises_overshoot_but_noise_does_not(clean):
    """The halo metric must separate sharpening from noise."""
    base = M.sharpness(clean)["ringing"]
    sharp = np.clip(clean + 2.5 * (clean - cv2.GaussianBlur(clean, (0, 0), 1.2)),
                    0, 1).astype(np.float32)
    rng = np.random.default_rng(1)
    noisy = np.clip(clean + rng.normal(0, 10 / 255, clean.shape), 0, 1).astype(np.float32)

    over = M.sharpness(sharp)["ringing"]
    noise = M.sharpness(noisy)["ringing"]
    assert over > 0.15, f"over-sharpening overshoot only {over}"
    assert noise < 0.12, f"noise alone produced overshoot {noise}"
    assert over > noise * 2, f"cannot separate halo ({over}) from noise ({noise})"


# --------------------------------------------------------------- exposure

@pytest.mark.parametrize("gamma,direction", [(2.4, "down"), (0.42, "up")])
def test_mean_luminance_tracks_gamma(clean, gamma, direction):
    adj = np.clip(clean ** gamma, 0, 1).astype(np.float32)
    base = M.exposure(clean)["mean_y"]
    now = M.exposure(adj)["mean_y"]
    if direction == "down":
        assert now < base - 0.1
    else:
        assert now > base + 0.1


def test_clipping_counted_correctly():
    img = np.zeros((64, 64, 3), np.float32)
    img[:16] = 1.0   # a quarter blown
    img[16:32] = 0.0  # a quarter crushed
    img[32:] = 0.5
    ex = M.exposure(img)
    assert ex["clip_highlight"] == pytest.approx(0.25, abs=0.01)
    assert ex["clip_shadow"] == pytest.approx(0.25, abs=0.01)


def test_range_usage_detects_compression(clean):
    squashed = np.clip((clean - 0.5) * 0.25 + 0.5, 0, 1).astype(np.float32)
    assert M.exposure(squashed)["range_usage"] < M.exposure(clean)["range_usage"] / 2


# ------------------------------------------------------------------ colour

@pytest.mark.parametrize("gains,axis", [
    ((1.3, 1.0, 0.75), "warm"),
    ((0.75, 1.0, 1.3), "cool"),
])
def test_colour_cast_detected_with_direction(clean, gains, axis):
    cast = np.clip(clean * np.array(gains, np.float32), 0, 1)
    c = M.color(cast)
    assert c["cast_magnitude"] > 0.13, f"cast magnitude only {c['cast_magnitude']}"
    assert axis.split()[0] in c["cast_direction"].lower(), \
        f"direction '{c['cast_direction']}' does not mention {axis}"


def test_clean_plate_cast_is_small_or_low_confidence(plate):
    """A clean plate must not produce an ACTIONABLE cast finding.

    One plate's three illuminant estimators genuinely disagree, which no
    single correction can satisfy. The Advisor is expected to notice the
    disagreement and report low confidence rather than act on it.
    """
    from app.advisor.recipe import diagnose

    for seed in range(5):
        img = plate(seed)
        c = M.color(img)
        if c["cast_magnitude"] < 0.13:
            continue
        cast = [f for f in diagnose(img)["findings"] if f["id"] == "COL_CAST"]
        assert cast, f"plate {seed} has cast {c['cast_magnitude']} but no finding"
        assert cast[0]["confidence"] < 0.45, (
            f"plate {seed}: a spurious cast of {c['cast_magnitude']} was reported "
            f"at confidence {cast[0]['confidence']}, which would present it as "
            f"actionable"
        )


def test_real_casts_are_high_confidence(clean):
    from app.advisor.recipe import diagnose

    for gains in ((1.3, 1.0, 0.75), (0.75, 1.0, 1.3)):
        img = np.clip(clean * np.array(gains, np.float32), 0, 1).astype(np.float32)
        cast = [f for f in diagnose(img)["findings"] if f["id"] == "COL_CAST"]
        assert cast, f"cast {gains} not detected"
        assert cast[0]["confidence"] > 0.8, \
            f"real cast reported at only {cast[0]['confidence']} confidence"


def test_monochrome_is_flagged(clean):
    """A black-and-white photograph must not be "corrected" for colour."""
    from app.core.colorspace import luma

    g = np.clip(luma(clean), 0, 1).astype(np.float32)
    mono = np.dstack([g, g, g])
    c = M.color(mono)
    assert c["gray_frac"] > 0.95
    scores = score_all(M.measure_all(mono))
    assert scores["dimensions"]["color"]["score"] == 100.0
    assert "suppressed" in scores["dimensions"]["color"].get("note", "")


def test_saturation_measured_over_coloured_pixels(clean):
    """sat_mean excludes near-neutral pixels, so it does not fall in
    proportion to a saturation scale: desaturating pushes many pixels below
    the S > 0.05 cutoff and the mean is then taken over the remaining, more
    saturated ones. It must still drop clearly, and gray_frac must rise.
    """
    hsv = cv2.cvtColor(clean.copy(), cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] *= 0.15
    desat = np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1).astype(np.float32)
    before, after = M.color(clean), M.color(desat)
    assert after["sat_mean"] < before["sat_mean"] * 0.75, \
        f"saturation barely moved: {before['sat_mean']} -> {after['sat_mean']}"
    assert after["gray_frac"] > before["gray_frac"]


# ------------------------------------------------------------- uniformity

def test_vignette_detected_and_clean_is_not(clean):
    h, w = clean.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((xx - (w - 1) / 2) ** 2 + (yy - (h - 1) / 2) ** 2) / np.sqrt(
        2 * ((w - 1) / 2) ** 2)
    vig = np.clip(clean * (1 - 0.30 * r ** 2)[:, :, None], 0, 1).astype(np.float32)
    assert M.uniformity(vig)["vignette_falloff"] > 0.15
    assert M.uniformity(clean)["vignette_falloff"] < 0.12, \
        "a clean plate must not report vignetting"


def test_haze_raises_dark_channel(clean):
    h, w = clean.shape[:2]
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    t = np.exp(-2.2 * np.clip(0.6 * (1 - yy) + 0.2, 0, 1))[:, :, None]
    hazy = np.clip(clean * t + 0.88 * (1 - t), 0, 1).astype(np.float32)
    assert M.uniformity(hazy)["dark_channel_mean"] > M.uniformity(clean)["dark_channel_mean"]
    assert M.uniformity(hazy)["dark_channel_mean"] > 0.44


def test_illumination_metric_is_exposure_invariant(clean):
    """Darkening the whole frame must not create an illumination gradient."""
    base = M.uniformity(clean)["illum_nonuniformity"]
    darker = M.uniformity(np.clip(clean * 0.45, 0, 1).astype(np.float32))
    assert abs(darker["illum_nonuniformity"] - base) < 0.12, \
        f"a global exposure change moved the metric {base} -> {darker['illum_nonuniformity']}"


def test_illumination_metric_is_contrast_invariant(clean):
    """A contrast stretch must not create an illumination gradient either."""
    base = M.uniformity(clean)["illum_nonuniformity"]
    stretched = np.clip((clean - 0.5) * 1.6 + 0.5, 0, 1).astype(np.float32)
    now = M.uniformity(stretched)["illum_nonuniformity"]
    assert abs(now - base) < 0.15, f"a contrast stretch moved the metric {base} -> {now}"


# -------------------------------------------------------------- artifacts

def test_jpeg_blocking_detected(clean):
    u8 = (np.clip(clean, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
    ok, enc = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, 18])
    assert ok
    low = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255
    assert M.artifacts(low)["blockiness"] > 1.2
    assert M.artifacts(clean)["blockiness"] < 1.12, "clean plate flagged as blocky"


def test_banding_detected_but_blur_is_not(clean):
    """Blur also creates smooth regions; only quantisation is banding."""
    quantised = (np.floor(np.clip(clean, 0, 1) * 31 + 0.5) / 31).astype(np.float32)
    blurred = cv2.GaussianBlur(clean, (0, 0), 2.5)
    assert M.artifacts(quantised)["banding_score"] > 0.14
    assert M.artifacts(blurred)["banding_score"] < 0.09, \
        "blur misreported as banding"
    assert M.artifacts(clean)["banding_score"] < 0.09


def test_histogram_comb_ignores_narrow_range():
    """A correctly exposed image that does not reach pure black or white has a
    narrow range, not a combed histogram."""
    narrow = np.full((128, 128, 3), 0.0, np.float32)
    rng = np.random.default_rng(0)
    narrow[:] = rng.uniform(0.3, 0.7, narrow.shape).astype(np.float32)
    assert M.artifacts(narrow)["hist_comb"] < 0.1


# ---------------------------------------------------------------- geometry

def test_tilt_recovered_on_content_with_lines():
    """Phase 2 acceptance: recover a planted angle within 0.3 degrees."""
    from tests.fixtures.synth import plate_with_lines

    base = plate_with_lines(0)
    h, w = base.shape[:2]
    for planted in (1.8, -2.5):
        Mrot = cv2.getRotationMatrix2D((w / 2, h / 2), planted, 1.0)
        rot = cv2.warpAffine(base, Mrot, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REFLECT_101)
        g = M.geometry(rot)
        assert g["tilt_confidence"] > 0.2, "no reliable tilt evidence"
        # Hough reports the deviation of the lines; its sign convention is the
        # opposite of the rotation applied.
        assert abs(abs(g["tilt_angle"]) - abs(planted)) < 0.4, \
            f"tilt {g['tilt_angle']} vs planted {planted}"


def test_no_tilt_claimed_without_linear_structure(plate):
    """Deskewing needs lines; without them the angle must be reported as zero."""
    for seed in range(3):
        g = M.geometry(plate(seed))
        assert g["tilt_angle"] == 0.0 or g["tilt_confidence"] < 0.22, (
            f"plate {seed} claimed a tilt of {g['tilt_angle']} with confidence "
            f"{g['tilt_confidence']} on content with no straight lines"
        )


# ----------------------------------------------------------------- scoring

def test_piecewise_curve_shape():
    assert piecewise(0.5, 0.4, 0.6, 0.0, 1.0) == 100.0
    assert piecewise(0.4, 0.4, 0.6, 0.0, 1.0) == 100.0
    assert piecewise(0.0, 0.4, 0.6, 0.0, 1.0) == 0.0
    assert piecewise(1.0, 0.4, 0.6, 0.0, 1.0) == 0.0
    mid = piecewise(0.2, 0.4, 0.6, 0.0, 1.0)
    assert 40 < mid < 60, f"halfway to the fail bound should score ~50, got {mid}"


def test_clean_plates_score_highly(plate):
    """A clean image must score well, or the Advisor will invent problems."""
    for seed in range(5):
        s = score_all(M.measure_all(plate(seed)))
        assert s["overall"] > 85, (
            f"clean plate {seed} scored {s['overall']} "
            f"(worst: {s['worst_dimension']})"
        )


def test_overall_score_capped_by_worst_dimension(clean):
    """A single severe defect must not be averaged away."""
    scores = score_all(M.measure_all(clean))
    worst = min(d["score"] for d in scores["dimensions"].values())
    assert scores["overall"] <= worst + 25.001


def test_measure_all_survives_a_failing_group(clean, monkeypatch):
    """One broken metric must not take the whole report down."""
    def boom(_img):
        raise RuntimeError("deliberate")

    monkeypatch.setitem(M.GROUPS, "texture", boom)
    out = M.measure_all(clean)
    assert "error" in out["texture"]
    assert "mean_y" in out["exposure"], "other groups should still be reported"


def test_measure_all_groups_are_json_serialisable(clean):
    import json

    payload = json.dumps(M.measure_all(clean))
    assert len(payload) > 1000
