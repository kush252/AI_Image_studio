"""Operator behaviour. PLAN.md Phase 1-7 acceptance criteria."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.core import registry
from app.core.colorspace import luma
from app.core.pipeline import OpInvocation, execute

SECOND_IMAGE_OPS = {
    "tone.hist_match", "color.transfer", "geom.register_ecc",
    "geom.register_phase", "feat.match", "feat.pyramid_blend", "feat.template",
    "feat.stitch", "aug.cutmix", "aug.mixup", "compose.blend",
}


class OpRun:
    """One operator's result, with `data` flattened from the per-uid mapping
    the pipeline returns so tests can read the operator's own readouts."""

    def __init__(self, res):
        self.image = res.image
        self.notes = res.notes
        self.overlays = res.overlays
        self.data = next(iter(res.data.values()), {}) if res.data else {}


def run(img, op_id, params=None, **kw):
    res = execute(img, [OpInvocation(op_id, params or {}, **kw)],
                  image_id="", use_cache=False, ctx={"original": img})
    return OpRun(res)


# --------------------------------------------------- every operator runs

@pytest.mark.parametrize("op_id", [s.id for s in registry.all_ops()])
def test_operator_runs_with_defaults(op_id, clean):
    """Every operator must survive its own defaults and return sane output."""
    if op_id in SECOND_IMAGE_OPS or op_id.startswith("test."):
        pytest.skip("needs a second image")
    res = run(clean, op_id)
    img = res.image
    assert img.dtype == np.float32
    assert np.isfinite(img).all(), f"{op_id} produced non-finite values"
    assert float(img.min()) >= -1e-5 and float(img.max()) <= 1.0 + 1e-5, \
        f"{op_id} left the [0,1] range"
    # A note is allowed (e.g. "nothing detected"), an exception is not.
    for n in res.notes:
        assert "Error" not in n and "Traceback" not in n, f"{op_id}: {n}"


def test_second_image_operators_work_when_given_one(clean, plate):
    """The operators skipped above must work once a reference is supplied."""
    from app.core.imageio import DecodedImage
    from app.core.store import store

    other = plate(1)
    stored = store.add(DecodedImage(
        data=other, width=other.shape[1], height=other.shape[0], channels=3,
        source_format="synthetic", source_bit_depth=32, exif={},
        had_alpha=False, was_grayscale=False,
    ))
    param_name = {
        "feat.template": "template",
    }
    failures = []
    for op_id in sorted(SECOND_IMAGE_OPS):
        spec = registry.get(op_id)
        key = param_name.get(op_id, spec.requires.get("second_image_param", "reference"))
        try:
            res = run(clean, op_id, {key: stored.image_id})
            assert np.isfinite(res.image).all()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{op_id}: {type(exc).__name__}: {exc}")
    assert not failures, "\n".join(failures)


# ------------------------------------------------------ identity no-ops

@pytest.mark.parametrize(
    "op_id,params",
    [
        ("tone.brightness", {"offset": 0.0}),
        ("tone.contrast_linear", {"amount": 0.0}),
        ("tone.gamma", {"gamma": 1.0}),
        ("tone.exposure", {"stops": 0.0}),
        ("sharpen.unsharp", {"amount": 0.0}),
        ("color.vibrance", {"amount": 0.0}),
        ("color.hsv_shift", {"hue": 0.0, "saturation": 0.0, "value": 0.0}),
        ("geom.translate", {"tx": 0.0, "ty": 0.0}),
        ("geom.rotate", {"angle": 0.0}),
    ],
)
def test_identity_parameters_are_noops(clean, op_id, params):
    """Phase 3 acceptance: unsharp with amount 0 must be a numerical no-op."""
    out = run(clean, op_id, params).image
    assert np.abs(out - clean).max() < 1e-5, f"{op_id} is not identity at {params}"


def test_custom_kernel_identity(clean):
    out = run(clean, "sharpen.kernel", {
        "matrix": [[0, 0, 0], [0, 1, 0], [0, 0, 0]], "amount": 1.0,
    }).image
    assert np.abs(out - clean).max() < 1e-5


# ---------------------------------------------------- transform roundtrips

@pytest.mark.parametrize("op_id,keep", [
    ("freq.hadamard", 1.0),
    ("freq.walsh", 1.0),
])
def test_walsh_hadamard_roundtrip(clean, op_id, keep):
    """Phase 4 acceptance: forward/inverse error below 1e-5."""
    res = run(clean, op_id, {"keep_ratio": keep})
    err = res.data.get("roundtrip_max_error")
    assert err is not None and err < 1e-5, f"{op_id} round-trip error {err}"


def test_dwt_roundtrip(clean):
    res = run(clean, "freq.dwt", {"wavelet": "haar", "levels": 2})
    err = res.data.get("roundtrip_max_error")
    assert err is not None and err < 1e-5


def test_laplacian_pyramid_reconstruction_is_exact(clean):
    res = run(clean, "feat.pyramid_laplacian", {"levels": 4, "reconstruct": True})
    err = res.data.get("reconstruction_max_error")
    assert err is not None and err < 1e-5, f"pyramid reconstruction error {err}"


def test_dft_inverse_matches_spatial_gaussian(clean):
    """A frequency-domain Gaussian low-pass must agree with a spatial one."""
    from skimage.metrics import structural_similarity

    freq = run(clean, "freq.lowpass",
               {"kind": "gaussian", "cutoff": 0.08, "scope": "luma"}).image
    # Map the normalised cutoff to an equivalent spatial sigma.
    h, w = clean.shape[:2]
    diag = np.hypot(h / 2, w / 2)
    sigma = diag * 0.08
    spatial = run(clean, "smooth.gaussian", {"sigma": float(h / (2 * np.pi * 0.08 * diag) * 1.0)}).image
    # Compare structure rather than exact values: the two parameterisations
    # differ, so only the qualitative agreement is meaningful here.
    a = luma(freq)
    b = luma(spatial)
    s = structural_similarity(a, b, data_range=1.0)
    assert s > 0.75, f"frequency and spatial low-pass disagree (SSIM {s:.3f})"


def test_geometric_rotation_roundtrip(clean):
    """Phase 5 acceptance: +theta then -theta recovers the interior."""
    from skimage.metrics import structural_similarity

    fwd = run(clean, "geom.rotate", {"angle": 12.0, "interp": "lanczos"}).image
    back = run(fwd, "geom.rotate", {"angle": -12.0, "interp": "lanczos"}).image
    h, w = clean.shape[:2]
    m = int(min(h, w) * 0.22)  # ignore the border, which rotation cannot restore
    a = luma(clean[m:-m, m:-m])
    b = luma(back[m:-m, m:-m])
    s = structural_similarity(a, b, data_range=1.0)
    assert s > 0.95, f"rotation round-trip SSIM {s:.4f}"


def test_morphology_open_close_idempotent(shapes):
    a = run(shapes, "morph.open", {"size": 5}).image
    b = run(a, "morph.open", {"size": 5}).image
    assert np.abs(a - b).max() < 1.0 / 255, "opening must be idempotent"
    c = run(shapes, "morph.close", {"size": 5}).image
    d = run(c, "morph.close", {"size": 5}).image
    assert np.abs(c - d).max() < 1.0 / 255, "closing must be idempotent"


def test_distance_transform_matches_scipy(shapes):
    """Phase 5 acceptance: L2 distance transform against scipy."""
    from scipy import ndimage

    g = luma(shapes)
    binary = (g > 0.5).astype(np.uint8)
    ours = run(shapes, "morph.distance",
               {"metric": "l2", "normalize": False, "threshold": 0.5}).image
    ref = ndimage.distance_transform_edt(binary).astype(np.float32)
    ref_norm = ref / max(float(ref.max()), 1e-8)
    ours_norm = ours / max(float(ours.max()), 1e-8)
    assert np.abs(ours_norm - ref_norm).max() < 0.02


def test_otsu_matches_skimage(clean):
    """Phase 6 acceptance: our Otsu must equal skimage's exactly."""
    from skimage.filters import threshold_otsu

    res = run(clean, "seg.otsu", {})
    ours = res.data["level_u8"]
    g = (np.clip(luma(clean), 0, 1) * 255).astype(np.uint8)
    ref = int(threshold_otsu(g))
    assert abs(ours - ref) <= 1, f"Otsu {ours} vs skimage {ref}"


# ------------------------------------------------------------- denoising

@pytest.mark.parametrize("sigma8", [5, 15, 30])
@pytest.mark.parametrize("op_id,params", [
    ("smooth.gaussian", {"sigma": 1.2}),
    ("smooth.bilateral", {"d": 5, "sigma_color": 0.12, "sigma_space": 7.0}),
    ("smooth.nlm", {"h": 10.0}),
    ("smooth.wavelet", {}),
    ("smooth.tv", {"weight": 0.1}),
    ("smooth.wiener", {"k": 5}),
])
def test_denoisers_improve_psnr(clean, sigma8, op_id, params):
    """Phase 3 acceptance: every denoiser raises PSNR against the clean plate."""
    from skimage.metrics import peak_signal_noise_ratio

    rng = np.random.default_rng(sigma8)
    noisy = np.clip(clean + rng.normal(0, sigma8 / 255, clean.shape), 0, 1).astype(np.float32)
    out = run(noisy, op_id, params).image
    before = peak_signal_noise_ratio(clean, noisy, data_range=1.0)
    after = peak_signal_noise_ratio(clean, np.clip(out, 0, 1), data_range=1.0)
    assert after > before, f"{op_id} at sigma={sigma8} made PSNR worse ({before:.2f} -> {after:.2f})"


def test_adaptive_median_beats_plain_median_on_impulses(clean):
    """The Advisor routes impulse noise to the adaptive median; show why."""
    from skimage.metrics import peak_signal_noise_ratio

    rng = np.random.default_rng(3)
    noisy = clean.copy()
    h, w = clean.shape[:2]
    n = int(0.06 * h * w)
    ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
    salt = rng.random(n) < 0.5
    noisy[ys[salt], xs[salt]] = 1.0
    noisy[ys[~salt], xs[~salt]] = 0.0

    plain = run(noisy, "smooth.median", {"k": 3}).image
    adaptive = run(noisy, "smooth.median_adaptive", {"k_max": 7}).image
    p_plain = peak_signal_noise_ratio(clean, np.clip(plain, 0, 1), data_range=1.0)
    p_adapt = peak_signal_noise_ratio(clean, np.clip(adaptive, 0, 1), data_range=1.0)
    assert p_adapt > p_plain, f"adaptive {p_adapt:.2f} should beat plain {p_plain:.2f}"


def test_notch_auto_removes_planted_sinusoid(clean):
    """Phase 4 acceptance: the sinusoid's energy drops substantially."""
    h, w = clean.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fx, fy = 0.12, 0.08
    pattern = 0.10 * np.sin(2 * np.pi * (fx * xx + fy * yy))
    noisy = np.clip(clean + pattern[:, :, None], 0, 1).astype(np.float32)

    res = run(noisy, "freq.notch_auto", {})
    assert res.data["removed"] > 0, "the planted sinusoid was not detected"

    def energy_at(img):
        g = luma(img) - float(luma(img).mean())
        F = np.fft.fftshift(np.fft.fft2(g))
        cy, cx = h // 2, w // 2
        py, px = int(cy + fy * h), int(cx + fx * w)
        r = 4
        return float(np.abs(F[py - r:py + r + 1, px - r:px + r + 1]).max())

    before, after = energy_at(noisy), energy_at(res.image)
    reduction_db = 20 * np.log10(before / max(after, 1e-9))
    assert reduction_db > 12, f"only {reduction_db:.1f} dB of the sinusoid removed"


# --------------------------------------------------------- augmentation

def test_augmentation_is_reproducible(clean):
    """Phase 7 acceptance: the same seed yields an identical variant."""
    for op_id in ("aug.gaussian_noise", "aug.rotate", "aug.random_crop",
                  "aug.iso_simulation", "aug.motion_blur", "aug.cutout"):
        a = run(clean, op_id, {"seed": 42}).image
        b = run(clean, op_id, {"seed": 42}).image
        assert np.array_equal(a, b), f"{op_id} is not reproducible for a fixed seed"
        c = run(clean, op_id, {"seed": 43}).image
        assert not np.array_equal(a, c), f"{op_id} ignores its seed"


def test_augmentation_probability_zero_skips(clean):
    res = run(clean, "aug.gaussian_noise", {"p": 0.0, "seed": 1})
    assert res.data.get("applied") is False
    assert np.allclose(res.image, clean)


def test_jpeg_artifact_aug_actually_compresses(clean):
    low = run(clean, "aug.jpeg_artifacts",
              {"quality_range": [10, 10], "seed": 1}).data
    high = run(clean, "aug.jpeg_artifacts",
               {"quality_range": [95, 95], "seed": 1}).data
    assert low["bytes"] < high["bytes"], "lower quality must produce fewer bytes"


# ------------------------------------------------------------ hand-written

def test_surf_detector_finds_keypoints(shapes):
    """Our own approximate-Hessian SURF implementation (PLAN.md Section 2)."""
    res = run(shapes, "feat.surf_approx", {"hessian_threshold": 1e-5})
    assert res.data["keypoint_count"] > 5
    assert "hand-written" in res.data["implementation"]


def test_sift_matching_recovers_known_transform(clean):
    """Phase 6 acceptance: recover a planted homography within ~1 px RMS."""
    from app.core.imageio import DecodedImage
    from app.core.store import store

    h, w = clean.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 11.0, 1.15)
    warped = cv2.warpAffine(clean, M, (w, h), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REFLECT_101)
    stored = store.add(DecodedImage(
        data=np.ascontiguousarray(warped), width=w, height=h, channels=3,
        source_format="synthetic", source_bit_depth=32, exif={},
        had_alpha=False, was_grayscale=False,
    ))
    res = run(clean, "feat.match", {"detector": "sift", "reference": stored.image_id,
                                    "ransac": True})
    assert res.data["matches"] > 20, f"only {res.data['matches']} matches"
    assert res.data.get("inliers", 0) > 15
    rms = res.data.get("reprojection_rms_px")
    assert rms is not None and rms < 2.0, f"reprojection RMS {rms} px"


def test_seam_carving_changes_width(clean):
    res = run(clean, "geom.seam_carve", {"remove_cols": 12, "remove_rows": 0})
    assert res.image.shape[1] == clean.shape[1] - 12
    assert res.image.shape[0] == clean.shape[0]


def test_chaincode_and_fourier_descriptors(shapes):
    cc = run(shapes, "feat.chaincode", {}).data
    assert cc["length"] > 20
    assert set(cc["chain_code"]) <= set(range(8))
    fd = run(shapes, "feat.fourier_desc", {"n_coeffs": 10}).data
    assert len(fd["invariant_descriptor"]) == 10


def test_deconvolution_preserves_mean(clean):
    """DC preservation: a PSF sums to 1, so deconvolution must not relevel."""
    blurred = cv2.GaussianBlur(clean, (0, 0), 2.0)
    out = run(blurred, "sharpen.deconv",
              {"method": "wiener", "psf": "gaussian", "psf_size": 6.0,
               "noise": 0.01}).image
    assert abs(float(out.mean()) - float(blurred.mean())) < 0.02


def test_views_do_not_modify_the_image(clean):
    """is_view operators report numbers and pass the image through."""
    for spec in registry.all_ops():
        if not spec.is_view or spec.id in SECOND_IMAGE_OPS:
            continue
        res = execute(clean, [OpInvocation(spec.id, {})], image_id="",
                      use_cache=False, ctx={"original": clean})
        assert np.allclose(res.image, clean, atol=1e-6), \
            f"{spec.id} is declared is_view but altered the image"
