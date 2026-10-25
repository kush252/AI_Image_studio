"""Interest points, descriptors, matching and scale space. Catalogue Section 9.

Includes our own approximate-Hessian SURF-style detector, because SURF lives in
OpenCV's nonfree xfeatures2d module and is absent from PyPI wheels (PLAN.md
Section 2). Implementing it is also more defensible than calling a library.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog as sk_hog
from skimage.feature import local_binary_pattern

from ..core.colorspace import from_hsv, luma
from ..core.params import (
    BoolParam,
    EnumParam,
    FloatParam,
    ImageRefParam,
    IntParam,
    OddIntParam,
    enum_opts,
)
from ..core.registry import OpResult, register
from ._util import EPS, clamp01, colorize, gray_u8, normalize01, resolve_ref_image


def _draw_keypoints(img, kps, color=(1.0, 0.25, 0.1), rich=True):
    base = (np.clip(img if img.ndim == 3 else np.dstack([img] * 3), 0, 1) * 255
            ).astype(np.uint8)
    bgr = base[:, :, ::-1].copy()
    flags = (cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if rich else 0)
    c = tuple(int(v * 255) for v in color[::-1])
    out = cv2.drawKeypoints(bgr, kps, None, color=c, flags=flags)
    return (out[:, :, ::-1].astype(np.float32) / 255.0)


def _kp_list(kps, limit=600):
    return [
        {"x": round(float(k.pt[0]), 2), "y": round(float(k.pt[1]), 2),
         "size": round(float(k.size), 2), "angle": round(float(k.angle), 1),
         "response": round(float(k.response), 6), "octave": int(k.octave)}
        for k in kps[:limit]
    ]


@register(
    "feat.harris",
    "Harris corners",
    "feat",
    params=[
        IntParam("block_size", min=2, max=31, default=3, scale_aware=True),
        OddIntParam("ksize", min=1, max=31, default=3),
        FloatParam("k", min=0.01, max=0.2, default=0.04, step=0.005,
                   help="Sensitivity constant in the corner response."),
        FloatParam("threshold", min=0.0001, max=0.5, default=0.01, step=0.0005,
                   help="Fraction of the maximum response required."),
        BoolParam("show_response", default=True),
        BoolParam("subpixel", default=False),
    ],
    summary="Detects corners from the eigenvalues of the local structure "
    "tensor, using a determinant/trace response that avoids computing them "
    "explicitly.",
    formula_latex=r"M = \sum w\begin{bmatrix}I_x^2 & I_xI_y\\ I_xI_y & I_y^2"
    r"\end{bmatrix},\quad R = \det M - k\,(\mathrm{tr}\,M)^2",
    when_to_use="Corner detection for tracking and registration. R is large "
    "only when BOTH eigenvalues are large, which is what distinguishes a corner "
    "from an edge (one large) or flat region (neither).",
    syllabus=True,
    references=["Harris & Stephens, A Combined Corner and Edge Detector, 1988"],
)
def harris(img, block_size=3, ksize=3, k=0.04, threshold=0.01,
           show_response=True, subpixel=False, ctx=None):
    g = (luma(img) if img.ndim == 3 else img).astype(np.float32)
    R = cv2.cornerHarris(g, int(block_size), int(ksize), float(k))
    peak = float(R.max())
    mask = R > threshold * max(peak, EPS)

    # Non-maximum suppression so each corner reports once.
    dil = cv2.dilate(R, np.ones((3, 3), np.uint8))
    mask &= R >= dil
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    if subpixel and len(pts):
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        pts = cv2.cornerSubPix(g, pts.reshape(-1, 1, 2), (5, 5), (-1, -1), crit)
        pts = pts.reshape(-1, 2)

    kps = [cv2.KeyPoint(float(p[0]), float(p[1]), 6.0) for p in pts[:3000]]
    res = OpResult(
        image=_draw_keypoints(img, kps, (0.1, 1.0, 0.3), rich=False),
        data={"corner_count": len(kps), "max_response": round(peak, 8),
              "points": [[round(float(p[0]), 1), round(float(p[1]), 1)]
                         for p in pts[:800]]},
    )
    if show_response:
        res.overlays["response"] = colorize(normalize01(np.maximum(R, 0)), "inferno")
    return res


@register(
    "feat.shitomasi",
    "Shi-Tomasi (good features to track)",
    "feat",
    params=[
        IntParam("max_corners", min=1, max=2000, default=200),
        FloatParam("quality", min=0.001, max=0.5, default=0.01, step=0.001),
        FloatParam("min_distance", min=1.0, max=100.0, default=10.0, step=1.0,
                   scale_aware=True),
        IntParam("block_size", min=3, max=31, default=3, scale_aware=True),
        BoolParam("use_harris", default=False),
    ],
    summary="Ranks corners by the SMALLER eigenvalue of the structure tensor, "
    "which is a more direct and better-behaved criterion than Harris's response.",
    formula_latex=r"R = \min(\lambda_1, \lambda_2)",
    when_to_use="Selecting a bounded, well-spaced set of trackable points. The "
    "minimum-distance constraint gives even coverage, which Harris does not.",
    syllabus=True,
    references=["Shi & Tomasi, Good Features to Track, CVPR 1994"],
)
def shitomasi(img, max_corners=200, quality=0.01, min_distance=10.0,
              block_size=3, use_harris=False, ctx=None):
    g = (luma(img) if img.ndim == 3 else img).astype(np.float32)
    pts = cv2.goodFeaturesToTrack(
        g, int(max_corners), float(quality), float(min_distance),
        blockSize=int(block_size), useHarrisDetector=bool(use_harris), k=0.04
    )
    if pts is None:
        return OpResult(image=img, data={"corner_count": 0, "points": []})
    pts = pts.reshape(-1, 2)
    kps = [cv2.KeyPoint(float(x), float(y), 8.0) for x, y in pts]
    return OpResult(
        image=_draw_keypoints(img, kps, (1.0, 0.85, 0.1), rich=False),
        data={"corner_count": len(pts),
              "points": [[round(float(x), 1), round(float(y), 1)] for x, y in pts]},
    )


def _integral(g: np.ndarray) -> np.ndarray:
    return cv2.integral(g.astype(np.float32))


def _box_sum(ii: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Sum over a rectangle in O(1) from the integral image."""
    H, W = ii.shape
    x0, y0 = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    x1, y1 = max(0, min(x + w, W - 1)), max(0, min(y + h, H - 1))
    return float(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])


def _box_sum_grid(ii: np.ndarray, x: np.ndarray, y: np.ndarray,
                  w: int, h: int) -> np.ndarray:
    """Vectorised box sums over coordinate grids."""
    H, W = ii.shape
    x0 = np.clip(x, 0, W - 1)
    y0 = np.clip(y, 0, H - 1)
    x1 = np.clip(x + w, 0, W - 1)
    y1 = np.clip(y + h, 0, H - 1)
    return (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]).astype(np.float32)


def surf_hessian_response(ii: np.ndarray, size: int) -> np.ndarray:
    """Approximate the determinant of the Hessian with box filters.

    SURF's central idea: the second-order Gaussian derivatives Dxx, Dyy, Dxy
    are approximated by piecewise-constant box filters, each evaluable in
    constant time from an integral image, so the cost does not grow with the
    filter size. That is what makes the scale space cheap.
    """
    H, W = ii.shape[0] - 1, ii.shape[1] - 1
    L = size  # filter side length (must be odd, >= 9)
    lobe = L // 3
    border = L // 2

    yy, xx = np.mgrid[0:H, 0:W].astype(np.int32)

    # Dxx: three horizontal lobes with weights +1, -2, +1
    half_h = lobe  # vertical half-extent of the lobes
    y0 = yy - half_h
    dxx = (
        _box_sum_grid(ii, xx - (3 * lobe) // 2, y0, lobe, 2 * half_h + 1)
        - 2.0 * _box_sum_grid(ii, xx - lobe // 2, y0, lobe, 2 * half_h + 1)
        + _box_sum_grid(ii, xx + lobe // 2 + (lobe % 2), y0, lobe, 2 * half_h + 1)
    )
    # Dyy: the same, transposed
    x0 = xx - half_h
    dyy = (
        _box_sum_grid(ii, x0, yy - (3 * lobe) // 2, 2 * half_h + 1, lobe)
        - 2.0 * _box_sum_grid(ii, x0, yy - lobe // 2, 2 * half_h + 1, lobe)
        + _box_sum_grid(ii, x0, yy + lobe // 2 + (lobe % 2), 2 * half_h + 1, lobe)
    )
    # Dxy: four square lobes in a checkerboard, weights +1, -1, -1, +1
    g = lobe
    dxy = (
        _box_sum_grid(ii, xx + 1, yy - g, g, g)
        - _box_sum_grid(ii, xx - g, yy - g, g, g)
        - _box_sum_grid(ii, xx + 1, yy + 1, g, g)
        + _box_sum_grid(ii, xx - g, yy + 1, g, g)
    )

    norm = 1.0 / float(L * L)
    dxx *= norm
    dyy *= norm
    dxy *= norm
    # 0.9 compensates for the box approximation of the Gaussian (per the paper).
    det = dxx * dyy - (0.9 * dxy) ** 2
    det[:border, :] = 0
    det[-border:, :] = 0
    det[:, :border] = 0
    det[:, -border:] = 0
    return det


@register(
    "feat.surf_approx",
    "SURF (our implementation)",
    "feat",
    params=[
        FloatParam("hessian_threshold", min=1e-6, max=0.01, default=0.0002,
                   step=1e-6, log_scale=True),
        IntParam("octaves", min=1, max=4, default=3),
        IntParam("layers", min=3, max=5, default=4),
        IntParam("max_points", min=10, max=3000, default=400),
        BoolParam("show_scale_space", default=False),
    ],
    summary="Speeded-Up Robust Features detector, implemented here from the "
    "paper: an integral image plus box-filter approximations of the second-order "
    "Gaussian derivatives, scanned over a scale space and refined by 3x3x3 "
    "non-maximum suppression.",
    formula_latex=r"\det(H_{approx}) = D_{xx}D_{yy} - (0.9\,D_{xy})^2",
    when_to_use="SURF is on the syllabus but ships only in OpenCV's nonfree "
    "module, which PyPI wheels omit. Rather than substitute a different "
    "detector silently, the approximate-Hessian scheme is implemented directly "
    "-- compare it against AKAZE and SIFT on the same image.",
    syllabus=True,
    cost="slow",
    references=["Bay, Tuytelaars & Van Gool, SURF: Speeded Up Robust Features, "
                "ECCV 2006"],
)
def surf_approx(img, hessian_threshold=0.0002, octaves=3, layers=4,
                max_points=400, show_scale_space=False, ctx=None):
    g = (luma(img) if img.ndim == 3 else img).astype(np.float32)
    ii = _integral(g)

    # Filter sizes per the paper: octave o uses step 6*2^o between layers.
    responses: list[tuple[int, np.ndarray]] = []
    for o in range(int(octaves)):
        step = 6 * (2**o)
        for l in range(int(layers)):
            size = 9 + step * l
            if size > min(g.shape[:2]) // 2:
                continue
            responses.append((size, surf_hessian_response(ii, size)))
    if len(responses) < 3:
        return OpResult(image=img, data={"keypoint_count": 0},
                        notes=["Image too small for a SURF scale space."])

    # 3x3x3 non-maximum suppression across space and scale.
    kps: list[cv2.KeyPoint] = []
    for i in range(1, len(responses) - 1):
        size, cur = responses[i]
        prev, nxt = responses[i - 1][1], responses[i + 1][1]
        local_max = cv2.dilate(cur, np.ones((3, 3), np.uint8))
        cand = (cur >= local_max) & (cur > hessian_threshold)
        cand &= cur > cv2.dilate(prev, np.ones((3, 3), np.uint8))
        cand &= cur > cv2.dilate(nxt, np.ones((3, 3), np.uint8))
        ys, xs = np.nonzero(cand)
        if len(ys) == 0:
            continue
        vals = cur[ys, xs]
        order = np.argsort(-vals)[: max_points]
        scale = 1.2 * size / 9.0
        for j in order:
            kps.append(cv2.KeyPoint(float(xs[j]), float(ys[j]), float(scale * 2.0),
                                    -1.0, float(vals[j])))

    kps.sort(key=lambda k: -k.response)
    kps = kps[: int(max_points)]
    res = OpResult(
        image=_draw_keypoints(img, kps, (0.2, 0.8, 1.0)),
        data={"keypoint_count": len(kps),
              "scale_levels": [int(s) for s, _ in responses],
              "keypoints": _kp_list(kps, 400),
              "implementation": "hand-written approximate Hessian (integral image)"},
    )
    if show_scale_space and responses:
        tiles = [normalize01(np.maximum(r, 0)) for _, r in responses[:4]]
        while len(tiles) < 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
        res.overlays["scale_space"] = colorize(grid, "inferno")
    return res


@register(
    "feat.hessian",
    "Hessian / determinant-of-Hessian",
    "feat",
    params=[
        FloatParam("sigma", min=0.5, max=20.0, default=3.0, step=0.1, scale_aware=True),
        FloatParam("threshold", min=1e-7, max=0.01, default=1e-4, step=1e-6,
                   log_scale=True),
        EnumParam("measure", options=enum_opts(("determinant", "Determinant"),
                                               ("trace", "Trace (Laplacian)"),
                                               ("ridge", "Ridge (max eigenvalue)")),
                  default="determinant"),
        BoolParam("show_response", default=True),
    ],
    summary="Blob and ridge detection from the second-derivative matrix, using "
    "true Gaussian derivatives rather than box approximations.",
    formula_latex=r"H = \begin{bmatrix}L_{xx} & L_{xy}\\ L_{xy} & L_{yy}"
    r"\end{bmatrix},\quad \det H = L_{xx}L_{yy} - L_{xy}^2",
    when_to_use="Detecting blob-like structures at a chosen scale. The "
    "determinant responds to blobs, the trace to any curvature, and the maximum "
    "eigenvalue to ridges.",
    syllabus=True,
)
def hessian_detector(img, sigma=3.0, threshold=1e-4, measure="determinant",
                     show_response=True, ctx=None):
    g = (luma(img) if img.ndim == 3 else img).astype(np.float32)
    s = float(sigma)
    blurred = cv2.GaussianBlur(g, (0, 0), s)
    lxx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3) * s * s
    lyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3) * s * s
    lxy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3) * s * s

    if measure == "trace":
        R = np.abs(lxx + lyy)
    elif measure == "ridge":
        tmp = np.sqrt(np.maximum((lxx - lyy) ** 2 + 4 * lxy**2, 0))
        R = np.maximum(np.abs((lxx + lyy + tmp) / 2), np.abs((lxx + lyy - tmp) / 2))
    else:
        R = lxx * lyy - lxy**2

    dil = cv2.dilate(R, np.ones((3, 3), np.uint8))
    mask = (R >= dil) & (R > threshold)
    ys, xs = np.nonzero(mask)
    vals = R[ys, xs]
    order = np.argsort(-vals)[:1500]
    kps = [cv2.KeyPoint(float(xs[i]), float(ys[i]), float(s * 3.0), -1.0,
                        float(vals[i])) for i in order]
    res = OpResult(
        image=_draw_keypoints(img, kps, (1.0, 0.3, 0.8)),
        data={"keypoint_count": len(kps), "measure": measure,
              "max_response": round(float(R.max()), 9)},
    )
    if show_response:
        res.overlays["response"] = colorize(normalize01(np.abs(R)), "magma")
    return res


def _detector_op(op_id, label, factory, doc, formula, when, refs, syllabus=False,
                 extra_params=None, cost="medium"):
    def fn(img, max_points=500, show_descriptors=False, ctx=None, **kw):
        g = gray_u8(img)
        det = factory(kw)
        kps, desc = det.detectAndCompute(g, None)
        kps = sorted(kps, key=lambda k: -k.response)[: int(max_points)]
        data = {
            "keypoint_count": len(kps),
            "descriptor_size": int(desc.shape[1]) if desc is not None and len(desc) else 0,
            "keypoints": _kp_list(kps, 400),
        }
        if show_descriptors and desc is not None and len(desc):
            data["descriptor_sample"] = np.round(
                desc[: min(8, len(desc))].astype(float), 3
            ).tolist()
        return OpResult(image=_draw_keypoints(img, kps), data=data)

    register(
        op_id, label, "feat",
        params=(extra_params or []) + [
            IntParam("max_points", min=10, max=5000, default=500),
            BoolParam("show_descriptors", default=False),
        ],
        summary=doc, formula_latex=formula, when_to_use=when, references=refs,
        syllabus=syllabus, cost=cost,
    )(fn)


_detector_op(
    "feat.sift", "SIFT", lambda kw: cv2.SIFT_create(
        nfeatures=int(kw.get("n_features", 0)),
        nOctaveLayers=int(kw.get("octave_layers", 3)),
        contrastThreshold=float(kw.get("contrast_threshold", 0.04)),
        edgeThreshold=float(kw.get("edge_threshold", 10.0)),
        sigma=float(kw.get("sigma", 1.6)),
    ),
    "Scale-Invariant Feature Transform: DoG scale-space extrema, refined "
    "sub-pixel and sub-scale, each given a dominant orientation and a 128-"
    "dimensional gradient-histogram descriptor.",
    r"D(x,y,\sigma) = \big(G_{k\sigma} - G_{\sigma}\big) * f",
    "The benchmark local feature. Invariant to scale, rotation and moderate "
    "illumination change. Its patent expired, so it is in mainline OpenCV.",
    ["Lowe, Distinctive Image Features from Scale-Invariant Keypoints, IJCV 2004"],
    syllabus=True,
    extra_params=[
        IntParam("n_features", min=0, max=5000, default=0,
                 help="0 keeps all detected features."),
        IntParam("octave_layers", min=1, max=6, default=3),
        FloatParam("contrast_threshold", min=0.001, max=0.2, default=0.04, step=0.001),
        FloatParam("edge_threshold", min=1.0, max=50.0, default=10.0, step=0.5),
        FloatParam("sigma", min=0.5, max=4.0, default=1.6, step=0.1),
    ],
)

_detector_op(
    "feat.akaze", "AKAZE", lambda kw: cv2.AKAZE_create(
        threshold=float(kw.get("threshold", 0.001)),
        nOctaves=int(kw.get("octaves", 4)),
        nOctaveLayers=int(kw.get("octave_layers", 4)),
    ),
    "Accelerated KAZE: builds a NONLINEAR scale space by anisotropic "
    "diffusion, so smoothing does not blur across object boundaries the way "
    "Gaussian scale space does.",
    r"\frac{\partial L}{\partial t} = \nabla\cdot(c(x,y,t)\,\nabla L)",
    "A free, wheel-available alternative to SURF with comparable or better "
    "repeatability. Offered alongside our SURF implementation for comparison.",
    ["Alcantarilla, Nuevo & Bartoli, Fast Explicit Diffusion for Accelerated "
     "Features in Nonlinear Scale Spaces, BMVC 2013"],
    extra_params=[
        FloatParam("threshold", min=1e-5, max=0.05, default=0.001, step=1e-5,
                   log_scale=True),
        IntParam("octaves", min=1, max=6, default=4),
        IntParam("octave_layers", min=1, max=6, default=4),
    ],
)

_detector_op(
    "feat.orb", "ORB", lambda kw: cv2.ORB_create(
        nfeatures=int(kw.get("n_features", 500)),
        scaleFactor=float(kw.get("scale_factor", 1.2)),
        nlevels=int(kw.get("levels", 8)),
    ),
    "Oriented FAST keypoints with rotation-aware BRIEF descriptors: a binary "
    "descriptor compared by Hamming distance.",
    r"d_H(a,b) = \sum_i a_i \oplus b_i",
    "Real-time matching. Orders of magnitude faster than SIFT, at some cost in "
    "repeatability.",
    ["Rublee et al., ORB: An efficient alternative to SIFT or SURF, ICCV 2011"],
    extra_params=[
        IntParam("n_features", min=10, max=5000, default=500),
        FloatParam("scale_factor", min=1.05, max=2.0, default=1.2, step=0.05),
        IntParam("levels", min=1, max=16, default=8),
    ],
    cost="fast",
)

_detector_op(
    "feat.brisk", "BRISK", lambda kw: cv2.BRISK_create(
        thresh=int(kw.get("threshold", 30)), octaves=int(kw.get("octaves", 3))
    ),
    "Binary Robust Invariant Scalable Keypoints: a concentric sampling pattern "
    "whose long pairs give orientation and short pairs give the descriptor.",
    r"\text{descriptor bit} = \mathbb{1}\big[I(p_i) > I(p_j)\big]",
    "Scale- and rotation-invariant binary features; a middle ground between "
    "ORB's speed and SIFT's robustness.",
    ["Leutenegger, Chli & Siegwart, BRISK, ICCV 2011"],
    extra_params=[
        IntParam("threshold", min=5, max=120, default=30),
        IntParam("octaves", min=0, max=6, default=3),
    ],
)


@register(
    "feat.fast",
    "FAST corners",
    "feat",
    params=[
        IntParam("threshold", min=1, max=100, default=25),
        BoolParam("nonmax_suppression", default=True),
        EnumParam("type", options=enum_opts(("9_16", "9 of 16"), ("7_12", "7 of 12"),
                                            ("5_8", "5 of 8")), default="9_16"),
    ],
    summary="Tests a circle of 16 pixels around each candidate and declares a "
    "corner when enough contiguous ones are all brighter or all darker.",
    formula_latex=r"\exists\ n \text{ contiguous } p \in C:\ "
    r"|I_p - I_c| > t",
    when_to_use="The fastest corner detector available; the detection stage of "
    "ORB. A decision tree makes the contiguity test nearly free.",
    references=["Rosten & Drummond, Machine learning for high-speed corner "
                "detection, ECCV 2006"],
    cost="fast",
)
def fast_corners(img, threshold=25, nonmax_suppression=True, type="9_16", ctx=None):
    types = {"9_16": cv2.FAST_FEATURE_DETECTOR_TYPE_9_16,
             "7_12": cv2.FAST_FEATURE_DETECTOR_TYPE_7_12,
             "5_8": cv2.FAST_FEATURE_DETECTOR_TYPE_5_8}
    det = cv2.FastFeatureDetector_create(
        threshold=int(threshold), nonmaxSuppression=bool(nonmax_suppression),
        type=types[type]
    )
    kps = det.detect(gray_u8(img), None)
    kps = sorted(kps, key=lambda k: -k.response)[:2000]
    return OpResult(image=_draw_keypoints(img, kps, (0.9, 0.9, 0.2), rich=False),
                    data={"keypoint_count": len(kps), "keypoints": _kp_list(kps)})


@register(
    "feat.hog",
    "HOG (Histogram of Oriented Gradients)",
    "feat",
    params=[
        IntParam("cell", min=4, max=32, default=8, scale_aware=True),
        IntParam("block", min=1, max=4, default=2),
        IntParam("bins", min=4, max=18, default=9),
        BoolParam("signed", default=False,
                  help="Signed uses 0-360 degrees; unsigned folds to 0-180."),
    ],
    summary="Divides the image into cells, builds a gradient-orientation "
    "histogram in each, then normalises over overlapping blocks for "
    "illumination invariance.",
    formula_latex=r"h_c(b) = \sum_{p \in c} \|\nabla f(p)\|\,"
    r"\mathbb{1}[\theta(p) \in b],\quad "
    r"\hat{h} = \frac{h}{\sqrt{\|h\|^2 + \epsilon}}",
    when_to_use="The classic shape descriptor for detection. The visualisation "
    "shows each cell as a rosette whose spoke lengths are the bin magnitudes.",
    syllabus=True,
    cost="slow",
    references=["Dalal & Triggs, Histograms of Oriented Gradients for Human "
                "Detection, CVPR 2005"],
)
def hog(img, cell=8, block=2, bins=9, signed=False, ctx=None):
    g = (luma(img) if img.ndim == 3 else img).astype(np.float32)
    feat, vis = sk_hog(
        g, orientations=int(bins), pixels_per_cell=(int(cell), int(cell)),
        cells_per_block=(int(block), int(block)), visualize=True,
        feature_vector=True, block_norm="L2-Hys",
    )
    return OpResult(
        image=normalize01(vis.astype(np.float32)),
        data={"descriptor_length": int(feat.size),
              "cells": [g.shape[0] // int(cell), g.shape[1] // int(cell)],
              "orientations": int(bins),
              "descriptor_mean": round(float(feat.mean()), 6),
              "descriptor_histogram": np.histogram(feat, bins=32, range=(0, 1))[0]
              .astype(int).tolist()},
    )


@register(
    "feat.lbp",
    "Local Binary Patterns",
    "feat",
    params=[
        IntParam("radius", min=1, max=10, default=1, scale_aware=True),
        IntParam("points", min=4, max=32, default=8),
        EnumParam("method", options=enum_opts(("uniform", "Uniform"),
                                              ("default", "Default"),
                                              ("ror", "Rotation-invariant"),
                                              ("nri_uniform", "Non-rotation-invariant uniform")),
                  default="uniform"),
    ],
    summary="Thresholds each neighbour against the centre pixel and reads the "
    "result as a binary number, giving a compact texture code per pixel.",
    formula_latex=r"LBP_{P,R} = \sum_{p=0}^{P-1} "
    r"\mathbb{1}\big[I_p \ge I_c\big]\,2^p",
    when_to_use="Texture classification and face description. Invariant to any "
    "monotonic intensity change, because only the ORDER of intensities matters.",
    tags=["beyond-syllabus"],
    cost="medium",
    references=["Ojala, Pietikainen & Maenpaa, Multiresolution gray-scale and "
                "rotation invariant texture classification with LBP, 2002"],
)
def lbp(img, radius=1, points=8, method="uniform", ctx=None):
    # LBP compares adjacent pixels for strict inequality, so float input makes
    # the result depend on numerical noise. Quantise to uint8 first.
    g = gray_u8(img)
    codes = local_binary_pattern(g, int(points), float(radius), method=method)
    nbins = int(codes.max()) + 1 if codes.max() > 0 else 2
    hist, _ = np.histogram(codes, bins=min(nbins, 256), range=(0, min(nbins, 256)))
    hist = hist.astype(float) / max(hist.sum(), 1)
    return OpResult(
        image=colorize(normalize01(codes), "viridis"),
        data={"histogram": np.round(hist, 6).tolist(),
              "unique_codes": int(len(np.unique(codes))),
              "uniformity": round(float(hist.max()), 5)},
    )


@register(
    "feat.gabor",
    "Gabor filter bank",
    "feat",
    params=[
        FloatParam("wavelength", min=2.0, max=40.0, default=8.0, step=0.5,
                   scale_aware=True),
        IntParam("orientations", min=1, max=12, default=4),
        FloatParam("sigma", min=1.0, max=20.0, default=4.0, step=0.5, scale_aware=True),
        FloatParam("gamma", min=0.1, max=2.0, default=0.5, step=0.05,
                   help="Spatial aspect ratio of the Gaussian envelope."),
        FloatParam("psi", min=0.0, max=6.28, default=0.0, step=0.1),
        EnumParam("combine", options=enum_opts(("max", "Maximum response"),
                                               ("sum", "Sum"),
                                               ("grid", "Show all orientations")),
                  default="max"),
    ],
    summary="A bank of oriented sinusoids under a Gaussian envelope -- jointly "
    "localised in space and frequency, and a standard model of early visual "
    "cortex receptive fields.",
    formula_latex=r"g(x,y) = \exp\!\left(-\frac{x'^2 + \gamma^2 y'^2}{2\sigma^2}"
    r"\right)\cos\!\left(\frac{2\pi x'}{\lambda} + \psi\right)",
    when_to_use="Texture analysis and orientation-selective filtering. The grid "
    "mode makes the orientation selectivity obvious.",
    tags=["beyond-syllabus"],
    cost="fast",
    references=["Gabor, Theory of Communication, 1946",
                "Daugman, Uncertainty relation for resolution in space, 1985"],
)
def gabor(img, wavelength=8.0, orientations=4, sigma=4.0, gamma=0.5, psi=0.0,
          combine="max", ctx=None):
    g = (luma(img) if img.ndim == 3 else img).astype(np.float32)
    n = int(orientations)
    ksize = int(max(7, sigma * 6)) | 1
    responses = []
    for i in range(n):
        theta = np.pi * i / n
        k = cv2.getGaborKernel((ksize, ksize), float(sigma), float(theta),
                               float(wavelength), float(gamma), float(psi),
                               ktype=cv2.CV_32F)
        k -= k.mean()  # zero-DC so flat regions give no response
        responses.append(np.abs(cv2.filter2D(g, cv2.CV_32F, k)))

    stack = np.stack(responses)
    if combine == "grid":
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        tiles = [normalize01(r) for r in responses]
        while len(tiles) < rows * cols:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[r * cols : (r + 1) * cols])
                          for r in range(rows)])
        out_img = normalize01(grid)
    elif combine == "sum":
        out_img = normalize01(stack.sum(axis=0))
    else:
        out_img = normalize01(stack.max(axis=0))

    energies = [round(float(r.mean()), 6) for r in responses]
    return OpResult(
        image=out_img,
        data={"orientation_energy": energies,
              "dominant_orientation_deg": round(180.0 * int(np.argmax(energies)) / n, 1),
              "orientations": n},
    )


@register(
    "feat.pyramid_gauss",
    "Gaussian pyramid",
    "feat",
    params=[
        IntParam("levels", min=1, max=7, default=4),
        IntParam("show_level", min=0, max=7, default=0,
                 help="0 is the full-resolution base."),
        BoolParam("show_grid", default=True),
    ],
    summary="Repeated smoothing and 2x downsampling, giving a multi-resolution "
    "representation.",
    formula_latex=r"G_{l+1} = \mathrm{down}\big(G_l * \hat{w}\big)",
    when_to_use="Scale-space analysis, coarse-to-fine search, and fast "
    "approximate blurring at large radii.",
    syllabus=True,
    references=["Burt & Adelson, The Laplacian Pyramid as a Compact Image Code, 1983"],
)
def pyramid_gauss(img, levels=4, show_level=0, show_grid=True, ctx=None):
    pyr = [img]
    cur = img
    for _ in range(int(levels)):
        if min(cur.shape[:2]) < 8:
            break
        cur = cv2.pyrDown(cur)
        pyr.append(cur)
    lvl = min(int(show_level), len(pyr) - 1)
    res = OpResult(
        image=clamp01(pyr[lvl]),
        data={"levels": len(pyr),
              "sizes": [[p.shape[1], p.shape[0]] for p in pyr],
              "total_pixels_ratio": round(
                  sum(p.shape[0] * p.shape[1] for p in pyr)
                  / max(img.shape[0] * img.shape[1], 1), 4)},
    )
    if show_grid and len(pyr) > 1:
        h = img.shape[0]
        canvas = np.zeros((h, img.shape[1] + pyr[1].shape[1], 3), np.float32)
        base = img if img.ndim == 3 else np.dstack([img] * 3)
        canvas[: base.shape[0], : base.shape[1]] = base
        y = 0
        for p in pyr[1:]:
            pp = p if p.ndim == 3 else np.dstack([p] * 3)
            if y + pp.shape[0] > h:
                break
            canvas[y : y + pp.shape[0], img.shape[1] : img.shape[1] + pp.shape[1]] = pp
            y += pp.shape[0]
        res.overlays["pyramid"] = clamp01(canvas)
    return res


@register(
    "feat.pyramid_laplacian",
    "Laplacian pyramid",
    "feat",
    params=[
        IntParam("levels", min=1, max=7, default=4),
        BoolParam("reconstruct", default=True,
                  help="Rebuild from the pyramid and report the error."),
        IntParam("show_level", min=0, max=7, default=0),
        FloatParam("gain", min=0.5, max=8.0, default=3.0, step=0.1,
                   help="Display gain; detail levels are near zero-mean."),
    ],
    summary="Each level stores what the Gaussian pyramid DISCARDED at that "
    "scale, making the representation complete and invertible.",
    formula_latex=r"L_l = G_l - \mathrm{up}(G_{l+1}),\qquad "
    r"G_l = L_l + \mathrm{up}(G_{l+1})",
    when_to_use="Band-pass decomposition, compression, and seamless blending. "
    "The reconstruction is exact, which the reported error confirms.",
    syllabus=True,
    references=["Burt & Adelson, 1983"],
)
def pyramid_laplacian(img, levels=4, reconstruct=True, show_level=0, gain=3.0,
                      ctx=None):
    gp = [img]
    cur = img
    for _ in range(int(levels)):
        if min(cur.shape[:2]) < 8:
            break
        cur = cv2.pyrDown(cur)
        gp.append(cur)

    lp = []
    for i in range(len(gp) - 1):
        up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        lp.append(gp[i] - up)
    lp.append(gp[-1])

    err = None
    if reconstruct:
        rec = lp[-1]
        for i in range(len(lp) - 2, -1, -1):
            rec = cv2.pyrUp(rec, dstsize=(lp[i].shape[1], lp[i].shape[0])) + lp[i]
        err = float(np.max(np.abs(rec - img)))

    lvl = min(int(show_level), len(lp) - 1)
    show = lp[lvl]
    out = clamp01(show * gain + 0.5) if lvl < len(lp) - 1 else clamp01(show)
    return OpResult(
        image=out,
        data={"levels": len(lp),
              "reconstruction_max_error": round(err, 9) if err is not None else None,
              "level_energy": [round(float(np.mean(np.abs(l))), 6) for l in lp]},
    )


@register(
    "feat.pyramid_blend",
    "Pyramid blending",
    "feat",
    params=[
        ImageRefParam("reference", help="Second image to blend with."),
        IntParam("levels", min=2, max=8, default=5),
        EnumParam("mask_kind", options=enum_opts(("vertical", "Vertical split"),
                                                 ("horizontal", "Horizontal split"),
                                                 ("radial", "Radial")),
                  default="vertical"),
        FloatParam("position", min=0.0, max=1.0, default=0.5, step=0.01),
    ],
    summary="Blends two images by combining their Laplacian pyramids under a "
    "Gaussian-blurred mask, so each frequency band is blended over a width "
    "appropriate to its own scale.",
    formula_latex=r"L^{blend}_l = M_l \cdot L^A_l + (1 - M_l)\cdot L^B_l",
    when_to_use="Seamless compositing. A single-scale alpha blend either shows "
    "a visible seam (sharp mask) or ghosting (soft mask); blending per band "
    "avoids both. This is the classic payoff of the pyramid representation.",
    tags=["beyond-syllabus"],
    cost="medium",
    requires={"second_image": True, "second_image_param": "reference"},
    references=["Burt & Adelson, A Multiresolution Spline With Application to "
                "Image Mosaics, 1983"],
)
def pyramid_blend(img, reference=None, levels=5, mask_kind="vertical",
                  position=0.5, ctx=None):
    ctx = ctx or {}
    other = resolve_ref_image(ctx, reference, img)
    a = img if img.ndim == 3 else np.dstack([img] * 3)
    b = other if other.ndim == 3 else np.dstack([other] * 3)
    a, b = a[:, :, :3], b[:, :, :3]

    h, w = a.shape[:2]
    if mask_kind == "radial":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
        m = (r < position * max(h, w) / 2).astype(np.float32)
    elif mask_kind == "horizontal":
        m = np.zeros((h, w), np.float32)
        m[: int(position * h), :] = 1.0
    else:
        m = np.zeros((h, w), np.float32)
        m[:, : int(position * w)] = 1.0

    n = int(levels)

    def build_lap(x):
        gp = [x]
        for _ in range(n):
            if min(gp[-1].shape[:2]) < 8:
                break
            gp.append(cv2.pyrDown(gp[-1]))
        lp = []
        for i in range(len(gp) - 1):
            up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
            lp.append(gp[i] - up)
        lp.append(gp[-1])
        return lp

    la, lb = build_lap(a), build_lap(b)
    gm = [m]
    for _ in range(len(la) - 1):
        gm.append(cv2.pyrDown(gm[-1]))

    blended = []
    for i in range(len(la)):
        mi = cv2.GaussianBlur(gm[i], (0, 0), 2.0)[:, :, None]
        blended.append(la[i] * mi + lb[i] * (1 - mi))

    rec = blended[-1]
    for i in range(len(blended) - 2, -1, -1):
        rec = cv2.pyrUp(rec, dstsize=(blended[i].shape[1], blended[i].shape[0]))
        rec = rec + blended[i]
    return OpResult(image=clamp01(rec), data={"levels": len(la)})


@register(
    "feat.match",
    "Keypoint matching",
    "feat",
    params=[
        ImageRefParam("reference"),
        EnumParam("detector", options=enum_opts(("sift", "SIFT"), ("orb", "ORB"),
                                                ("akaze", "AKAZE"), ("brisk", "BRISK")),
                  default="sift"),
        EnumParam("matcher", options=enum_opts(("bf", "Brute force"), ("flann", "FLANN")),
                  default="bf"),
        FloatParam("ratio", min=0.3, max=0.95, default=0.75, step=0.01,
                   help="Lowe's ratio test threshold."),
        BoolParam("ransac", default=True),
        FloatParam("ransac_thresh", min=1.0, max=20.0, default=4.0, step=0.5),
        IntParam("max_draw", min=10, max=400, default=60),
    ],
    summary="Detects and describes features in both images, matches them with "
    "Lowe's ratio test, then fits a homography with RANSAC to separate true "
    "correspondences from outliers.",
    formula_latex=r"\frac{d_1}{d_2} < \tau \quad\text{(ratio test)};\quad "
    r"x' \sim H x \quad\text{(RANSAC)}",
    when_to_use="Registration, stitching and verifying that two images show the "
    "same scene. The inlier count is the practical measure of that.",
    syllabus=True,
    cost="medium",
    requires={"second_image": True, "second_image_param": "reference"},
    references=["Lowe 2004; Fischler & Bolles, RANSAC, 1981"],
)
def match_features(img, reference=None, detector="sift", matcher="bf", ratio=0.75,
                   ransac=True, ransac_thresh=4.0, max_draw=60, ctx=None):
    ctx = ctx or {}
    other = resolve_ref_image(ctx, reference, img)
    g1, g2 = gray_u8(img), gray_u8(other)

    dets = {"sift": cv2.SIFT_create, "orb": cv2.ORB_create,
            "akaze": cv2.AKAZE_create, "brisk": cv2.BRISK_create}
    det = dets[detector]()
    k1, d1 = det.detectAndCompute(g1, None)
    k2, d2 = det.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
        return OpResult(image=img, data={"matches": 0, "inliers": 0},
                        notes=["Too few keypoints to match."])

    binary = detector in ("orb", "brisk", "akaze")
    if matcher == "flann" and not binary:
        index = dict(algorithm=1, trees=5)
        m = cv2.FlannBasedMatcher(index, dict(checks=50))
        knn = m.knnMatch(d1.astype(np.float32), d2.astype(np.float32), k=2)
    else:
        norm = cv2.NORM_HAMMING if binary else cv2.NORM_L2
        m = cv2.BFMatcher(norm)
        knn = m.knnMatch(d1, d2, k=2)

    good = [a for pair in knn if len(pair) == 2
            for a, b in [pair] if a.distance < ratio * b.distance]

    data = {"keypoints_a": len(k1), "keypoints_b": len(k2), "matches": len(good),
            "detector": detector}
    mask = None
    if ransac and len(good) >= 4:
        src = np.float32([k1[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, float(ransac_thresh))
        if H is not None:
            inliers = int(mask.sum())
            data["inliers"] = inliers
            data["inlier_ratio"] = round(inliers / max(len(good), 1), 4)
            data["homography"] = np.round(H, 6).tolist()
            # Reprojection RMS over the inliers -- the honest accuracy number.
            if inliers >= 4:
                sel = mask.ravel().astype(bool)
                proj = cv2.perspectiveTransform(src[sel], H)
                rms = float(np.sqrt(np.mean(np.sum((proj - dst[sel]) ** 2, axis=2))))
                data["reprojection_rms_px"] = round(rms, 3)

    draw = sorted(good, key=lambda g: g.distance)[: int(max_draw)]
    i1 = (np.clip(img if img.ndim == 3 else np.dstack([img] * 3), 0, 1) * 255
          ).astype(np.uint8)[:, :, ::-1]
    i2 = (np.clip(other if other.ndim == 3 else np.dstack([other] * 3), 0, 1) * 255
          ).astype(np.uint8)[:, :, ::-1]
    vis = cv2.drawMatches(i1, k1, i2, k2, draw, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return OpResult(image=None, data=data,
                    overlays={"matches": vis[:, :, ::-1].astype(np.float32) / 255.0})


@register(
    "feat.stitch",
    "Two-image stitch",
    "feat",
    params=[
        ImageRefParam("reference"),
        EnumParam("detector", options=enum_opts(("sift", "SIFT"), ("akaze", "AKAZE"),
                                                ("orb", "ORB")), default="sift"),
        FloatParam("ratio", min=0.3, max=0.95, default=0.75, step=0.01),
        EnumParam("blend", options=enum_opts(("feather", "Feathered"),
                                             ("average", "Average"),
                                             ("none", "Hard seam")),
                  default="feather"),
    ],
    summary="Matches features, fits a homography, warps the second image into "
    "the first's frame and blends the overlap.",
    formula_latex=r"I_{pano}(x) = \alpha(x) I_1(x) + (1-\alpha(x))\,I_2(H^{-1}x)",
    when_to_use="The payoff demo for detection + description + matching + "
    "RANSAC + blending: every stage is visible in one result.",
    tags=["beyond-syllabus"],
    cost="slow",
    requires={"second_image": True, "second_image_param": "reference"},
)
def stitch(img, reference=None, detector="sift", ratio=0.75, blend="feather",
           ctx=None):
    ctx = ctx or {}
    other = resolve_ref_image(ctx, reference, img)
    g1, g2 = gray_u8(img), gray_u8(other)
    dets = {"sift": cv2.SIFT_create, "orb": cv2.ORB_create, "akaze": cv2.AKAZE_create}
    det = dets[detector]()
    k1, d1 = det.detectAndCompute(g1, None)
    k2, d2 = det.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return OpResult(image=img, notes=["No descriptors found."])
    norm = cv2.NORM_HAMMING if detector in ("orb",) else cv2.NORM_L2
    knn = cv2.BFMatcher(norm).knnMatch(d1, d2, k=2)
    good = [a for pair in knn if len(pair) == 2
            for a, b in [pair] if a.distance < ratio * b.distance]
    if len(good) < 4:
        return OpResult(image=img, data={"matches": len(good)},
                        notes=["Too few matches for a homography."])

    src = np.float32([k2[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None:
        return OpResult(image=img, notes=["Homography estimation failed."])

    h1, w1 = img.shape[:2]
    h2, w2 = other.shape[:2]
    corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    allc = np.concatenate([warped_corners,
                           np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]
                                      ).reshape(-1, 1, 2)])
    xmin, ymin = np.int32(allc.min(axis=0).ravel() - 0.5)
    xmax, ymax = np.int32(allc.max(axis=0).ravel() + 0.5)
    # Cap the canvas so a bad homography cannot allocate gigabytes.
    if (xmax - xmin) * (ymax - ymin) > 40_000_000:
        return OpResult(image=img, notes=["Stitched canvas would be unreasonably "
                                          "large; the homography is probably wrong."])
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], np.float32)
    size = (xmax - xmin, ymax - ymin)

    a3 = img if img.ndim == 3 else np.dstack([img] * 3)
    b3 = other if other.ndim == 3 else np.dstack([other] * 3)
    pano = cv2.warpPerspective(b3[:, :, :3], T @ H, size)
    base = cv2.warpPerspective(a3[:, :, :3], T, size)

    mask_a = (cv2.warpPerspective(np.ones((h1, w1), np.float32), T, size) > 0.5)
    mask_b = (cv2.warpPerspective(np.ones((h2, w2), np.float32), T @ H, size) > 0.5)
    both = mask_a & mask_b

    out = np.where(mask_a[:, :, None], base, pano)
    if blend == "average" and both.any():
        out[both] = (base[both] + pano[both]) / 2.0
    elif blend == "feather" and both.any():
        # Weight by distance from each image's own border, so the seam fades.
        da = cv2.distanceTransform(mask_a.astype(np.uint8), cv2.DIST_L2, 5)
        db = cv2.distanceTransform(mask_b.astype(np.uint8), cv2.DIST_L2, 5)
        w = (da / np.maximum(da + db, EPS))[:, :, None]
        out = np.where(both[:, :, None], base * w + pano * (1 - w), out)

    return OpResult(
        image=clamp01(out),
        data={"matches": len(good), "inliers": int(mask.sum()) if mask is not None else 0,
              "panorama_size": [int(size[0]), int(size[1])],
              "homography": np.round(H, 6).tolist()},
    )


@register(
    "feat.template",
    "Template matching",
    "feat",
    params=[
        ImageRefParam("template"),
        EnumParam("method", options=enum_opts(
            ("TM_CCOEFF_NORMED", "Normalised correlation coefficient"),
            ("TM_CCORR_NORMED", "Normalised cross correlation"),
            ("TM_SQDIFF_NORMED", "Normalised squared difference"),
            ("TM_CCOEFF", "Correlation coefficient"),
            ("TM_CCORR", "Cross correlation"),
            ("TM_SQDIFF", "Squared difference")),
            default="TM_CCOEFF_NORMED"),
        FloatParam("threshold", min=0.0, max=1.0, default=0.8, step=0.01),
        IntParam("max_matches", min=1, max=200, default=10),
        BoolParam("show_surface", default=True),
    ],
    summary="Slides a template over the image and records a similarity score at "
    "every position.",
    formula_latex=r"R(x,y) = \frac{\sum_{x',y'} T'(x',y')\cdot I'(x+x',y+y')}"
    r"{\sqrt{\sum T'^2 \cdot \sum I'^2}}",
    when_to_use="Finding a known pattern at a known scale and orientation. The "
    "score surface shows why the normalised methods are preferred: the "
    "unnormalised ones peak wherever the image is simply bright.",
    cost="medium",
    requires={"second_image": True, "second_image_param": "template"},
)
def template_match(img, template=None, method="TM_CCOEFF_NORMED", threshold=0.8,
                   max_matches=10, show_surface=True, ctx=None):
    from ..core.errors import SecondImageRequired
    from ..core.store import store

    ctx = ctx or {}
    if not template:
        raise SecondImageRequired("Select a template image.")
    tpl_img = store.get(template).get(ctx.get("target", "proxy"))
    g = gray_u8(img)
    t = gray_u8(tpl_img)
    if t.shape[0] > g.shape[0] or t.shape[1] > g.shape[1]:
        f = min(g.shape[0] / t.shape[0], g.shape[1] / t.shape[1]) * 0.9
        t = cv2.resize(t, (max(1, int(t.shape[1] * f)), max(1, int(t.shape[0] * f))))

    m = getattr(cv2, method)
    res = cv2.matchTemplate(g, t, m)
    inverted = method.startswith("TM_SQDIFF")
    norm = normalize01(-res if inverted else res)

    th, tw = t.shape[:2]
    found = []
    work = norm.copy()
    for _ in range(int(max_matches)):
        _, mx, _, mloc = cv2.minMaxLoc(work)
        if mx < threshold:
            break
        found.append({"x": int(mloc[0]), "y": int(mloc[1]), "w": tw, "h": th,
                      "score": round(float(mx), 4)})
        x0, y0 = max(0, mloc[0] - tw // 2), max(0, mloc[1] - th // 2)
        work[y0 : y0 + th, x0 : x0 + tw] = 0

    vis = (np.clip(img if img.ndim == 3 else np.dstack([img] * 3), 0, 1) * 255
           ).astype(np.uint8)[:, :, ::-1].copy()
    for f in found:
        cv2.rectangle(vis, (f["x"], f["y"]), (f["x"] + tw, f["y"] + th),
                      (60, 60, 255), 2)
    out = OpResult(
        image=vis[:, :, ::-1].astype(np.float32) / 255.0,
        data={"matches": found, "match_count": len(found), "method": method,
              "template_size": [tw, th]},
    )
    if show_surface:
        out.overlays["score_surface"] = colorize(norm, "turbo")
    return out


@register(
    "feat.hough_lines",
    "Hough lines",
    "feat",
    params=[
        FloatParam("rho", min=0.5, max=10.0, default=1.0, step=0.5, scale_aware=True),
        FloatParam("theta_deg", min=0.1, max=5.0, default=1.0, step=0.1),
        IntParam("threshold", min=10, max=500, default=100),
        EnumParam("mode", options=enum_opts(("probabilistic", "Probabilistic"),
                                            ("standard", "Standard")),
                  default="probabilistic"),
        FloatParam("min_length", min=0.0, max=1.0, default=0.1, step=0.01,
                   help="As a fraction of the long edge."),
        FloatParam("max_gap", min=0.0, max=0.2, default=0.01, step=0.002),
        FloatParam("canny_low", min=0.0, max=1.0, default=0.2, step=0.01),
        FloatParam("canny_high", min=0.0, max=1.0, default=0.6, step=0.01),
        BoolParam("show_accumulator", default=True),
    ],
    summary="Each edge pixel votes for every line through it, in (rho, theta) "
    "parameter space; peaks in the accumulator are the lines.",
    formula_latex=r"\rho = x\cos\theta + y\sin\theta",
    when_to_use="Finding straight lines, measuring document skew, detecting "
    "architectural edges. The accumulator view makes parameter space concrete.",
    syllabus=True,
    cost="fast",
    references=["Duda & Hart, Use of the Hough transformation, 1972"],
)
def hough_lines(img, rho=1.0, theta_deg=1.0, threshold=100, mode="probabilistic",
                min_length=0.1, max_gap=0.01, canny_low=0.2, canny_high=0.6,
                show_accumulator=True, ctx=None):
    g = gray_u8(img)
    edges = cv2.Canny(g, canny_low * 255, canny_high * 255, apertureSize=3)
    long_edge = max(g.shape[:2])
    theta = np.deg2rad(theta_deg)

    vis = (np.clip(img if img.ndim == 3 else np.dstack([img] * 3), 0, 1) * 255
           ).astype(np.uint8)[:, :, ::-1].copy()
    lines_out = []
    if mode == "standard":
        lines = cv2.HoughLines(edges, float(rho), theta, int(threshold))
        if lines is not None:
            for r, t in lines[:200, 0]:
                a, b = np.cos(t), np.sin(t)
                x0, y0 = a * r, b * r
                p1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * a))
                p2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * a))
                cv2.line(vis, p1, p2, (60, 220, 60), 2)
                lines_out.append({"rho": round(float(r), 2),
                                  "theta_deg": round(float(np.degrees(t)), 2)})
    else:
        lines = cv2.HoughLinesP(edges, float(rho), theta, int(threshold),
                                minLineLength=int(min_length * long_edge),
                                maxLineGap=max(1, int(max_gap * long_edge)))
        if lines is not None:
            for x1, y1, x2, y2 in lines[:400, 0]:
                cv2.line(vis, (x1, y1), (x2, y2), (60, 220, 60), 2)
                ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                lines_out.append({"x1": int(x1), "y1": int(y1), "x2": int(x2),
                                  "y2": int(y2), "angle_deg": round(ang, 2),
                                  "length": round(float(np.hypot(x2 - x1, y2 - y1)), 1)})

    data = {"line_count": len(lines_out), "lines": lines_out[:200]}
    if lines_out and mode == "probabilistic":
        angs = np.array([l["angle_deg"] for l in lines_out])
        folded = ((angs + 45.0) % 90.0) - 45.0
        data["angle_histogram"] = np.histogram(folded, bins=45, range=(-45, 45))[0] \
            .astype(int).tolist()
        data["dominant_angle"] = round(float(np.median(folded)), 3)

    out = OpResult(image=vis[:, :, ::-1].astype(np.float32) / 255.0, data=data)
    if show_accumulator:
        # Build the (rho, theta) accumulator explicitly for display.
        ys, xs = np.nonzero(edges)
        if len(ys):
            nth = 180
            thetas = np.deg2rad(np.arange(nth))
            diag = int(np.ceil(np.hypot(*g.shape[:2])))
            acc = np.zeros((2 * diag, nth), np.float32)
            step = max(1, len(ys) // 20000)  # cap the cost on dense edge maps
            rr = (np.outer(xs[::step], np.cos(thetas))
                  + np.outer(ys[::step], np.sin(thetas)))
            ri = np.clip((rr + diag).astype(np.int32), 0, 2 * diag - 1)
            for j in range(nth):
                np.add.at(acc[:, j], ri[:, j], 1.0)
            out.overlays["accumulator"] = colorize(normalize01(acc), "turbo")
    return out


@register(
    "feat.hough_circles",
    "Hough circles",
    "feat",
    params=[
        FloatParam("dp", min=1.0, max=4.0, default=1.5, step=0.1),
        FloatParam("min_dist", min=0.01, max=0.5, default=0.05, step=0.005,
                   help="Fraction of the long edge."),
        IntParam("param1", min=10, max=400, default=120,
                 help="Upper Canny threshold."),
        IntParam("param2", min=5, max=300, default=40,
                 help="Accumulator threshold; lower finds more (and more false) circles."),
        FloatParam("min_radius", min=0.0, max=0.5, default=0.02, step=0.005),
        FloatParam("max_radius", min=0.0, max=1.0, default=0.3, step=0.01),
    ],
    summary="Gradient-based Hough transform for circles: votes are cast along "
    "the gradient direction, reducing a 3-D parameter space to something "
    "tractable.",
    formula_latex=r"(x - a)^2 + (y - b)^2 = r^2",
    when_to_use="Detecting coins, pupils, cells, dials and holes.",
    cost="fast",
)
def hough_circles(img, dp=1.5, min_dist=0.05, param1=120, param2=40,
                  min_radius=0.02, max_radius=0.3, ctx=None):
    g = gray_u8(img)
    g = cv2.medianBlur(g, 5)
    long_edge = max(g.shape[:2])
    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, float(dp), max(1.0, min_dist * long_edge),
        param1=float(param1), param2=float(param2),
        minRadius=int(min_radius * long_edge), maxRadius=int(max_radius * long_edge),
    )
    vis = (np.clip(img if img.ndim == 3 else np.dstack([img] * 3), 0, 1) * 255
           ).astype(np.uint8)[:, :, ::-1].copy()
    found = []
    if circles is not None:
        for x, y, r in np.uint16(np.around(circles))[0, :200]:
            cv2.circle(vis, (int(x), int(y)), int(r), (255, 120, 40), 2)
            cv2.circle(vis, (int(x), int(y)), 2, (40, 40, 255), 3)
            found.append({"x": int(x), "y": int(y), "r": int(r)})
    return OpResult(image=vis[:, :, ::-1].astype(np.float32) / 255.0,
                    data={"circle_count": len(found), "circles": found})


@register(
    "feat.contours",
    "Contours + shape descriptors",
    "feat",
    params=[
        FloatParam("threshold", min=0.0, max=1.0, default=0.5, step=0.01, auto=True),
        BoolParam("use_otsu", default=True),
        EnumParam("mode", options=enum_opts(("external", "External only"),
                                            ("list", "All contours"),
                                            ("tree", "Full hierarchy")),
                  default="external"),
        EnumParam("approx", options=enum_opts(("simple", "Simple"), ("none", "None"),
                                              ("tc89", "Teh-Chin")),
                  default="simple"),
        FloatParam("min_area", min=0.0, max=0.5, default=0.001, step=0.0005,
                   help="Fraction of total image area."),
        BoolParam("draw_labels", default=True),
    ],
    summary="Traces region boundaries and computes a shape descriptor table: "
    "area, perimeter, circularity, solidity, extent, aspect ratio and Hu moments.",
    formula_latex=r"\text{circularity} = \frac{4\pi A}{P^2},\quad "
    r"\text{solidity} = \frac{A}{A_{hull}},\quad "
    r"\phi_1 = \eta_{20} + \eta_{02}",
    when_to_use="Object measurement and shape-based classification. Hu moments "
    "are invariant to translation, scale and rotation, so the same shape scores "
    "alike however it is placed.",
    syllabus=True,
    references=["Hu, Visual Pattern Recognition by Moment Invariants, 1962",
                "Suzuki & Abe, Topological structural analysis, 1985"],
)
def contours(img, threshold=0.5, use_otsu=True, mode="external", approx="simple",
             min_area=0.001, draw_labels=True, ctx=None):
    g = gray_u8(img)
    if use_otsu:
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, b = cv2.threshold(g, int(threshold * 255), 255, cv2.THRESH_BINARY)

    modes = {"external": cv2.RETR_EXTERNAL, "list": cv2.RETR_LIST,
             "tree": cv2.RETR_TREE}
    approxs = {"simple": cv2.CHAIN_APPROX_SIMPLE, "none": cv2.CHAIN_APPROX_NONE,
               "tc89": cv2.CHAIN_APPROX_TC89_L1}
    cnts, hier = cv2.findContours(b, modes[mode], approxs[approx])

    total = g.shape[0] * g.shape[1]
    vis = (np.clip(img if img.ndim == 3 else np.dstack([img] * 3), 0, 1) * 255
           ).astype(np.uint8)[:, :, ::-1].copy()
    rows = []
    for i, c in enumerate(cnts):
        a = cv2.contourArea(c)
        if a < min_area * total:
            continue
        p = cv2.arcLength(c, True)
        x, y, w, h = cv2.boundingRect(c)
        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull)
        mom = cv2.moments(c)
        hu = cv2.HuMoments(mom).ravel()
        # Log-scale Hu moments: raw values span many orders of magnitude.
        hu_log = [round(float(-np.sign(v) * np.log10(abs(v) + 1e-30)), 4) for v in hu]
        cx = int(mom["m10"] / mom["m00"]) if mom["m00"] else x + w // 2
        cy = int(mom["m01"] / mom["m00"]) if mom["m00"] else y + h // 2
        rows.append({
            "id": len(rows),
            "area": round(a, 1),
            "perimeter": round(p, 1),
            "circularity": round(4 * np.pi * a / max(p * p, EPS), 4),
            "solidity": round(a / max(ha, EPS), 4),
            "extent": round(a / max(w * h, EPS), 4),
            "aspect_ratio": round(w / max(h, 1), 4),
            "bbox": [int(x), int(y), int(w), int(h)],
            "centroid": [cx, cy],
            "hu_log": hu_log,
        })
        cv2.drawContours(vis, [c], -1, (60, 255, 220), 2)
        if draw_labels:
            cv2.putText(vis, str(len(rows) - 1), (cx - 6, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 240), 1,
                        cv2.LINE_AA)

    return OpResult(
        image=vis[:, :, ::-1].astype(np.float32) / 255.0,
        data={"contour_count": len(rows), "contours": rows[:300],
              "total_area_fraction": round(
                  sum(r["area"] for r in rows) / max(total, 1), 5)},
    )


@register(
    "feat.fourier_desc",
    "Fourier descriptors",
    "feat",
    params=[
        IntParam("n_coeffs", min=2, max=120, default=12),
        FloatParam("threshold", min=0.0, max=1.0, default=0.5, step=0.01),
        BoolParam("use_otsu", default=True),
        BoolParam("reconstruct", default=True),
    ],
    summary="Represents a closed contour as a complex sequence z(t) = x + iy and "
    "takes its DFT; a handful of low-frequency coefficients reconstruct the "
    "overall shape.",
    formula_latex=r"a_k = \frac{1}{N}\sum_{t=0}^{N-1} z(t)\,e^{-j2\pi kt/N}",
    when_to_use="Compact, rotation- and scale-normalisable shape description. "
    "Reconstructing from 4 vs 40 coefficients shows exactly which coefficients "
    "carry gross shape and which carry fine boundary detail.",
    tags=["beyond-syllabus"],
    references=["Zahn & Roskies, Fourier Descriptors for Plane Closed Curves, 1972"],
)
def fourier_descriptors(img, n_coeffs=12, threshold=0.5, use_otsu=True,
                        reconstruct=True, ctx=None):
    g = gray_u8(img)
    if use_otsu:
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, b = cv2.threshold(g, int(threshold * 255), 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return OpResult(image=img, data={"contour_found": False},
                        notes=["No contour found to describe."])
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    z = c[:, 0].astype(np.float64) + 1j * c[:, 1].astype(np.float64)
    F = np.fft.fft(z)
    n = int(n_coeffs)

    # Invariant descriptor: drop a0 (translation), divide by |a1| (scale),
    # take magnitudes (rotation + starting point).
    mags = np.abs(F)
    inv = (mags[2 : 2 + n] / max(mags[1], EPS)).tolist()

    vis = (np.clip(img if img.ndim == 3 else np.dstack([img] * 3), 0, 1) * 255
           ).astype(np.uint8)[:, :, ::-1].copy()
    cv2.drawContours(vis, [c.reshape(-1, 1, 2)], -1, (120, 120, 120), 1)
    if reconstruct:
        Ft = np.zeros_like(F)
        keep = min(n, len(F) // 2)
        Ft[:keep] = F[:keep]
        Ft[-keep:] = F[-keep:]
        rec = np.fft.ifft(Ft)
        pts = np.stack([rec.real, rec.imag], axis=1).astype(np.int32)
        cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, (40, 80, 255), 2)

    return OpResult(
        image=vis[:, :, ::-1].astype(np.float32) / 255.0,
        data={"contour_points": int(len(c)), "n_coeffs": n,
              "invariant_descriptor": [round(v, 6) for v in inv]},
    )


@register(
    "feat.chaincode",
    "Chain code",
    "feat",
    params=[
        EnumParam("connectivity", options=enum_opts(("8", "8-connected"),
                                                    ("4", "4-connected")),
                  default="8"),
        FloatParam("threshold", min=0.0, max=1.0, default=0.5, step=0.01),
        BoolParam("use_otsu", default=True),
        IntParam("resample", min=0, max=200, default=0,
                 help="0 uses every boundary pixel."),
    ],
    summary="Encodes a boundary as a sequence of direction codes between "
    "successive pixels, plus the rotation-invariant first difference and the "
    "shape number.",
    formula_latex=r"c_i \in \{0..7\},\quad d_i = (c_{i+1} - c_i) \bmod 8",
    when_to_use="Compact lossless boundary representation. The first difference "
    "is rotation-invariant, and its minimum circular rotation is the canonical "
    "shape number.",
    syllabus=True,
    references=["Freeman, On the encoding of arbitrary geometric configurations, 1961",
                "Gonzalez & Woods, Ch. 11.1.1"],
)
def chaincode(img, connectivity="8", threshold=0.5, use_otsu=True, resample=0,
              ctx=None):
    g = gray_u8(img)
    if use_otsu:
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, b = cv2.threshold(g, int(threshold * 255), 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return OpResult(image=img, data={"found": False},
                        notes=["No boundary to encode."])
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    if resample and len(c) > resample:
        idx = np.linspace(0, len(c) - 1, int(resample)).astype(int)
        c = c[idx]

    # Freeman direction codes (8-connected): 0=E, counter-clockwise.
    dirs8 = {(1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
             (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7}
    dirs4 = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3}
    table = dirs8 if connectivity == "8" else dirs4
    n = 8 if connectivity == "8" else 4

    codes = []
    for i in range(len(c)):
        dx = int(np.sign(c[(i + 1) % len(c), 0] - c[i, 0]))
        dy = int(np.sign(c[(i + 1) % len(c), 1] - c[i, 1]))
        if (dx, dy) in table:
            codes.append(table[(dx, dy)])
    diff = [(codes[(i + 1) % len(codes)] - codes[i]) % n for i in range(len(codes))] \
        if codes else []
    # Shape number = the lexicographically smallest rotation of the difference.
    shape_number = None
    if diff and len(diff) <= 4000:
        rots = [tuple(diff[i:] + diff[:i]) for i in range(len(diff))]
        shape_number = list(min(rots))

    vis = (np.clip(img if img.ndim == 3 else np.dstack([img] * 3), 0, 1) * 255
           ).astype(np.uint8)[:, :, ::-1].copy()
    cv2.polylines(vis, [c.reshape(-1, 1, 2)], True, (60, 200, 255), 2)
    return OpResult(
        image=vis[:, :, ::-1].astype(np.float32) / 255.0,
        data={"length": len(codes), "connectivity": n,
              "chain_code": codes[:600],
              "first_difference": diff[:600],
              "shape_number_prefix": (shape_number or [])[:120],
              "direction_histogram": np.histogram(codes, bins=n, range=(0, n))[0]
              .astype(int).tolist() if codes else []},
    )
