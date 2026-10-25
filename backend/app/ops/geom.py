"""Geometric transformations. Catalogue Section 7 (syllabus block)."""
from __future__ import annotations

import cv2
import numpy as np

from ..core.colorspace import join_alpha, luma, split_alpha
from ..core.params import (
    BoolParam,
    ColorParam,
    EnumParam,
    FloatParam,
    ImageRefParam,
    IntParam,
    PointParam,
    PointsParam,
    RectParam,
    enum_opts,
)
from ..core.registry import OpResult, register
from ._util import EPS, border_flag, clamp01, interp_flag, normalize01, resolve_ref_image

INTERP_OPTS = enum_opts(
    ("nearest", "Nearest neighbour"),
    ("bilinear", "Bilinear"),
    ("bicubic", "Bicubic"),
    ("lanczos", "Lanczos 4"),
    ("area", "Area (downscale only)"),
    help_map={
        "nearest": "No interpolation; blocky but preserves exact label values. "
                   "The only correct choice for segmentation masks.",
        "bilinear": "Fast and smooth; slightly soft.",
        "bicubic": "Sharper than bilinear, with mild overshoot at edges.",
        "lanczos": "Sharpest resampling; visible ringing on hard edges.",
        "area": "Proper averaging for downscaling -- the only option that does "
                "not alias when shrinking.",
    },
)

BORDER_OPTS = enum_opts(
    ("reflect", "Reflect"), ("replicate", "Replicate"), ("wrap", "Wrap"),
    ("constant", "Constant"),
)


def _warp(img, M, size, interp, border, fill=(0, 0, 0)):
    flags = interp_flag(interp)
    bm = border_flag(border)
    val = tuple(float(v) for v in fill) if border == "constant" else 0
    if M.shape[0] == 2:
        return cv2.warpAffine(img, M, size, flags=flags, borderMode=bm, borderValue=val)
    return cv2.warpPerspective(img, M, size, flags=flags, borderMode=bm, borderValue=val)


@register(
    "geom.translate",
    "Translation",
    "geom",
    params=[
        FloatParam("tx", min=-1.0, max=1.0, default=0.0, step=0.005,
                   help="Fraction of image width."),
        FloatParam("ty", min=-1.0, max=1.0, default=0.0, step=0.005),
        EnumParam("interp", options=INTERP_OPTS, default="bilinear"),
        EnumParam("border", options=BORDER_OPTS, default="constant"),
        ColorParam("fill", default=(0.0, 0.0, 0.0)),
    ],
    summary="Shifts the image by a displacement vector.",
    formula_latex=r"\begin{bmatrix}x'\\y'\end{bmatrix} = "
    r"\begin{bmatrix}1&0&t_x\\0&1&t_y\end{bmatrix}"
    r"\begin{bmatrix}x\\y\\1\end{bmatrix}",
    when_to_use="Registration and alignment. Offsets are stored as fractions of "
    "the image size so they survive the proxy/full-resolution switch.",
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 2.6.5"],
)
def translate(img, tx=0.0, ty=0.0, interp="bilinear", border="constant",
              fill=(0, 0, 0), ctx=None):
    h, w = img.shape[:2]
    M = np.array([[1, 0, tx * w], [0, 1, ty * h]], np.float32)
    return clamp01(_warp(img, M, (w, h), interp, border, fill))


@register(
    "geom.rotate",
    "Rotation",
    "geom",
    params=[
        FloatParam("angle", min=-180.0, max=180.0, default=0.0, step=0.1, unit="deg"),
        PointParam("center", default=(0.5, 0.5)),
        FloatParam("scale", min=0.1, max=4.0, default=1.0, step=0.01),
        BoolParam("expand", default=False,
                  help="Grows the canvas so no content is cropped."),
        EnumParam("interp", options=INTERP_OPTS, default="bicubic"),
        EnumParam("border", options=BORDER_OPTS, default="constant"),
        ColorParam("fill", default=(0.0, 0.0, 0.0)),
    ],
    summary="Rotates about a point, optionally expanding the canvas to fit.",
    formula_latex=r"\begin{bmatrix}\cos\theta&-\sin\theta\\ \sin\theta&\cos\theta"
    r"\end{bmatrix}",
    when_to_use="Levelling a horizon, or demonstrating that rotation requires "
    "interpolation and is therefore lossy -- rotate by 1 degree forty times and "
    "compare with a single 40-degree rotation.",
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 2.6.5"],
)
def rotate(img, angle=0.0, center=(0.5, 0.5), scale=1.0, expand=False,
           interp="bicubic", border="constant", fill=(0, 0, 0), ctx=None):
    h, w = img.shape[:2]
    cx, cy = center[0] * w, center[1] * h
    M = cv2.getRotationMatrix2D((cx, cy), float(angle), float(scale))
    out_size = (w, h)
    if expand:
        rad = np.deg2rad(angle)
        c, s = abs(np.cos(rad)), abs(np.sin(rad))
        nw = int(np.ceil((w * c + h * s) * scale))
        nh = int(np.ceil((w * s + h * c) * scale))
        M[0, 2] += nw / 2.0 - cx
        M[1, 2] += nh / 2.0 - cy
        out_size = (nw, nh)
    return clamp01(_warp(img, M, out_size, interp, border, fill))


@register(
    "geom.scale",
    "Scaling / Resize",
    "geom",
    params=[
        EnumParam("mode", options=enum_opts(("factor", "Scale factor"),
                                            ("long_edge", "Target long edge"),
                                            ("megapixels", "Target megapixels"),
                                            ("exact", "Exact size")),
                  default="factor"),
        FloatParam("sx", min=0.05, max=8.0, default=1.0, step=0.01,
                   depends_on={"param": "mode", "equals": "factor"}),
        FloatParam("sy", min=0.05, max=8.0, default=1.0, step=0.01,
                   depends_on={"param": "mode", "equals": "factor"}),
        BoolParam("lock_aspect", default=True),
        IntParam("long_edge", min=16, max=8192, default=1024,
                 depends_on={"param": "mode", "equals": "long_edge"}),
        FloatParam("megapixels", min=0.01, max=64.0, default=2.0, step=0.01,
                   depends_on={"param": "mode", "equals": "megapixels"}),
        IntParam("width", min=1, max=8192, default=512,
                 depends_on={"param": "mode", "equals": "exact"}),
        IntParam("height", min=1, max=8192, default=512,
                 depends_on={"param": "mode", "equals": "exact"}),
        EnumParam("interp", options=INTERP_OPTS, default="area"),
    ],
    summary="Resamples the image to a new size.",
    formula_latex=r"\begin{bmatrix}s_x&0\\0&s_y\end{bmatrix}",
    when_to_use="Resizing, and demonstrating that downscaling without area "
    "averaging aliases: try 'nearest' on a fine checkerboard.",
    syllabus=True,
    domain="linear",
    references=["Gonzalez & Woods, Ch. 2.4.4"],
)
def scale_op(img, mode="factor", sx=1.0, sy=1.0, lock_aspect=True, long_edge=1024,
             megapixels=2.0, width=512, height=512, interp="area", ctx=None):
    h, w = img.shape[:2]
    if mode == "long_edge":
        f = long_edge / float(max(h, w))
        nw, nh = int(round(w * f)), int(round(h * f))
    elif mode == "megapixels":
        f = np.sqrt(megapixels * 1e6 / float(w * h))
        nw, nh = int(round(w * f)), int(round(h * f))
    elif mode == "exact":
        nw, nh = int(width), int(height)
    else:
        fy = sx if lock_aspect else sy
        nw, nh = int(round(w * sx)), int(round(h * fy))
    nw, nh = max(1, min(nw, 16384)), max(1, min(nh, 16384))
    flag = interp_flag(interp)
    if interp == "area" and (nw > w or nh > h):
        flag = cv2.INTER_CUBIC  # INTER_AREA is meaningless when upscaling
    out = cv2.resize(img, (nw, nh), interpolation=flag)
    return OpResult(image=clamp01(out),
                    data={"from": [w, h], "to": [nw, nh],
                          "factor": round(nw / max(w, 1), 4)})


@register(
    "geom.shear",
    "Shearing",
    "geom",
    params=[
        FloatParam("shx", min=-1.0, max=1.0, default=0.0, step=0.005),
        FloatParam("shy", min=-1.0, max=1.0, default=0.0, step=0.005),
        BoolParam("expand", default=True),
        EnumParam("interp", options=INTERP_OPTS, default="bilinear"),
        EnumParam("border", options=BORDER_OPTS, default="constant"),
    ],
    summary="Slants the image: displaces each row (or column) proportionally to "
    "its distance from the origin.",
    formula_latex=r"\begin{bmatrix}1&sh_x&0\\ sh_y&1&0\end{bmatrix}",
    when_to_use="Italicising, deskewing text, and demonstrating that affine "
    "transforms preserve parallel lines but not angles.",
    syllabus=True,
    references=["Gonzalez & Woods, Ch. 2.6.5"],
)
def shear(img, shx=0.0, shy=0.0, expand=True, interp="bilinear",
          border="constant", ctx=None):
    h, w = img.shape[:2]
    M = np.array([[1, shx, 0], [shy, 1, 0]], np.float32)
    size = (w, h)
    if expand:
        nw = int(np.ceil(w + abs(shx) * h))
        nh = int(np.ceil(h + abs(shy) * w))
        if shx < 0:
            M[0, 2] = -shx * h
        if shy < 0:
            M[1, 2] = -shy * w
        size = (nw, nh)
    return clamp01(_warp(img, M, size, interp, border))


@register(
    "geom.flip",
    "Flip / Transpose",
    "geom",
    params=[
        EnumParam("axis", options=enum_opts(("h", "Horizontal"), ("v", "Vertical"),
                                            ("both", "Both (180 deg)"),
                                            ("transpose", "Transpose")),
                  default="h"),
    ],
    summary="Mirrors or transposes the image. Exact, with no interpolation.",
    formula_latex=r"f'(x,y) = f(W-1-x,\ y)",
    when_to_use="Correcting mirrored scans, and the cheapest augmentation there "
    "is -- it is lossless, unlike every other geometric transform.",
)
def flip(img, axis="h", ctx=None):
    if axis == "h":
        return np.ascontiguousarray(img[:, ::-1])
    if axis == "v":
        return np.ascontiguousarray(img[::-1, :])
    if axis == "both":
        return np.ascontiguousarray(img[::-1, ::-1])
    return np.ascontiguousarray(np.swapaxes(img, 0, 1))


@register(
    "geom.affine3",
    "Affine (3-point)",
    "geom",
    params=[
        PointsParam("src", default=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
                    min_count=3, max_count=3),
        PointsParam("dst", default=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
                    min_count=3, max_count=3),
        EnumParam("interp", options=INTERP_OPTS, default="bilinear"),
        EnumParam("border", options=BORDER_OPTS, default="constant"),
    ],
    summary="The unique affine transform mapping three source points to three "
    "destination points.",
    formula_latex=r"\begin{bmatrix}a_{11}&a_{12}&b_1\\a_{21}&a_{22}&b_2\end{bmatrix}",
    when_to_use="Any combination of translation, rotation, scale and shear "
    "specified by example. Three point pairs are exactly enough to determine "
    "the six affine parameters.",
    syllabus=True,
)
def affine3(img, src=None, dst=None, interp="bilinear", border="constant", ctx=None):
    h, w = img.shape[:2]
    s = np.array([(p[0] * w, p[1] * h) for p in (src or [])], np.float32)
    d = np.array([(p[0] * w, p[1] * h) for p in (dst or [])], np.float32)
    if len(s) != 3 or len(d) != 3:
        return img
    M = cv2.getAffineTransform(s, d)
    return OpResult(image=clamp01(_warp(img, M, (w, h), interp, border)),
                    data={"matrix": np.round(M, 5).tolist()})


@register(
    "geom.perspective4",
    "Perspective (4-point)",
    "geom",
    params=[
        PointsParam("src", default=[(0.05, 0.05), (0.95, 0.05), (0.95, 0.95),
                                    (0.05, 0.95)], min_count=4, max_count=4),
        BoolParam("to_rect", default=True,
                  help="Maps the four points to the full output rectangle."),
        PointsParam("dst", default=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
                    min_count=4, max_count=4,
                    depends_on={"param": "to_rect", "equals": False}),
        EnumParam("interp", options=INTERP_OPTS, default="bicubic"),
        EnumParam("border", options=BORDER_OPTS, default="constant"),
    ],
    summary="Projective transform from four point correspondences, with eight "
    "degrees of freedom.",
    formula_latex=r"\begin{bmatrix}x'\\y'\\w'\end{bmatrix} = H "
    r"\begin{bmatrix}x\\y\\1\end{bmatrix},\quad H \in \mathbb{R}^{3\times3}",
    when_to_use="Keystone correction: photographing a document or building at an "
    "angle. Unlike affine, a homography does NOT preserve parallel lines, which "
    "is exactly what lets it undo perspective.",
    tags=["beyond-syllabus"],
)
def perspective4(img, src=None, to_rect=True, dst=None, interp="bicubic",
                 border="constant", ctx=None):
    h, w = img.shape[:2]
    s = np.array([(p[0] * w, p[1] * h) for p in (src or [])], np.float32)
    if len(s) != 4:
        return img
    if to_rect:
        d = np.array([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)], np.float32)
    else:
        d = np.array([(p[0] * w, p[1] * h) for p in (dst or [])], np.float32)
        if len(d) != 4:
            return img
    H = cv2.getPerspectiveTransform(s, d)
    return OpResult(image=clamp01(_warp(img, H, (w, h), interp, border)),
                    data={"homography": np.round(H, 6).tolist()})


@register(
    "geom.crop",
    "Crop",
    "geom",
    params=[
        RectParam("rect", default=(0.1, 0.1, 0.8, 0.8)),
        EnumParam("ratio", options=enum_opts(("free", "Free"), ("1:1", "1:1"),
                                             ("4:3", "4:3"), ("3:2", "3:2"),
                                             ("16:9", "16:9"), ("9:16", "9:16")),
                  default="free"),
    ],
    summary="Extracts a rectangular region. Stored in normalised coordinates so "
    "it is resolution-independent.",
    formula_latex=r"g(x,y) = f(x + x_0,\ y + y_0)",
    when_to_use="Composition, and restricting later operators to a region.",
)
def crop(img, rect=(0.1, 0.1, 0.8, 0.8), ratio="free", ctx=None):
    h, w = img.shape[:2]
    x, y, rw, rh = rect
    if ratio != "free":
        a, b = (float(v) for v in ratio.split(":"))
        target = a / b
        cur_w, cur_h = rw * w, rh * h
        if cur_w / max(cur_h, EPS) > target:
            cur_w = cur_h * target
        else:
            cur_h = cur_w / target
        rw, rh = cur_w / w, cur_h / h
    x0, y0 = int(round(x * w)), int(round(y * h))
    x1, y1 = int(round((x + rw) * w)), int(round((y + rh) * h))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, max(x0 + 1, x1)), min(h, max(y0 + 1, y1))
    return np.ascontiguousarray(img[y0:y1, x0:x1])


def _hough_tilt(gray: np.ndarray, max_angle: float = 20.0) -> tuple[float, float]:
    """Dominant deviation from axis-alignment, weighted by line length.

    Returns (angle_degrees, confidence). advisor-spec.md 2.7.
    """
    g = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    edges = cv2.Canny(g, 50, 150, apertureSize=3)
    h, w = g.shape
    min_len = max(20, int(0.12 * max(h, w)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360.0, threshold=60,
                            minLineLength=min_len, maxLineGap=12)
    if lines is None or len(lines) == 0:
        return 0.0, 0.0
    angles, weights = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = float(np.hypot(dx, dy))
        a = np.degrees(np.arctan2(dy, dx))
        # Fold into (-45, 45]: deviation from the nearest axis.
        a = ((a + 45.0) % 90.0) - 45.0
        if abs(a) <= max_angle:
            angles.append(a)
            weights.append(length)
    if not angles:
        return 0.0, 0.0
    angles = np.array(angles)
    weights = np.array(weights)
    # Weighted mode via a histogram, then a weighted mean inside the winning bin.
    hist, edges_b = np.histogram(angles, bins=81, range=(-max_angle, max_angle),
                                 weights=weights)
    b = int(np.argmax(hist))
    lo, hi = edges_b[b], edges_b[b + 1]
    sel = (angles >= lo - 0.5) & (angles <= hi + 0.5)
    if not sel.any():
        return 0.0, 0.0
    ang = float(np.average(angles[sel], weights=weights[sel]))
    agreement = float(weights[sel].sum() / max(weights.sum(), EPS))
    # Scale by how much line length was found at all, relative to the image
    # size. Agreement alone is 1.0 when a single short segment is the only
    # evidence, which is not a basis for rotating someone's photograph.
    long_edge = float(max(h, w))
    evidence = min(1.0, float(weights[sel].sum()) / (2.0 * long_edge))
    return ang, agreement * evidence


@register(
    "geom.deskew",
    "Auto deskew",
    "geom",
    params=[
        FloatParam("max_angle", min=1.0, max=45.0, default=20.0, step=0.5, unit="deg"),
        BoolParam("auto_crop", default=True,
                  help="Crops away the empty corners the rotation creates."),
        EnumParam("interp", options=INTERP_OPTS, default="bicubic"),
    ],
    summary="Detects the dominant deviation from axis-alignment using a "
    "length-weighted Hough angle histogram, then rotates to correct it.",
    formula_latex=r"\theta^* = \arg\max_\theta \sum_i \ell_i\,"
    r"\delta(\theta_i \approx \theta)",
    when_to_use="Scanned documents and photographs with a tilted horizon. "
    "Weighting by line length matters: one long building edge is better evidence "
    "than twenty short texture edges.",
    tags=["beyond-syllabus"],
    cost="fast",
)
def deskew(img, max_angle=20.0, auto_crop=True, interp="bicubic", ctx=None):
    g = luma(img) if img.ndim == 3 else img
    angle, conf = _hough_tilt(g, max_angle)
    if abs(angle) < 0.05 or conf < 0.05:
        return OpResult(image=img, data={"angle": round(angle, 3),
                                         "confidence": round(conf, 3),
                                         "applied": False},
                        notes=["No reliable tilt detected."])
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    out = cv2.warpAffine(img, M, (w, h), flags=interp_flag(interp),
                         borderMode=cv2.BORDER_REPLICATE)
    if auto_crop:
        rad = abs(np.deg2rad(angle))
        # Largest axis-aligned rectangle inside the rotated frame.
        c, s = np.cos(rad), np.sin(rad)
        denom = c * c - s * s
        if denom > 1e-6:
            nw = int((w * c - h * s) / denom)
            nh = int((h * c - w * s) / denom)
            nw, nh = max(8, min(nw, w)), max(8, min(nh, h))
            x0, y0 = (w - nw) // 2, (h - nh) // 2
            out = np.ascontiguousarray(out[y0 : y0 + nh, x0 : x0 + nw])
    return OpResult(image=clamp01(out),
                    data={"angle": round(angle, 3), "confidence": round(conf, 3),
                          "applied": True})


@register(
    "geom.interp_compare",
    "Interpolation comparison",
    "geom",
    params=[
        FloatParam("zoom", min=1.0, max=16.0, default=6.0, step=0.5),
        PointParam("center", default=(0.5, 0.5)),
    ],
    summary="Magnifies the same patch with four interpolation methods side by "
    "side, so their differences are visible rather than theoretical.",
    formula_latex=r"\text{nearest} \mid \text{bilinear} \mid \text{bicubic} "
    r"\mid \text{Lanczos}",
    when_to_use="Choosing a resampling method, and seeing exactly what each one "
    "does to edges: blocky, soft, slightly overshooting, or ringing.",
    is_view=True,
    tags=["teaching"],
)
def interp_compare(img, zoom=6.0, center=(0.5, 0.5), ctx=None):
    h, w = img.shape[:2]
    pw, ph = max(8, int(w / zoom)), max(8, int(h / zoom))
    cx, cy = int(center[0] * w), int(center[1] * h)
    x0 = max(0, min(w - pw, cx - pw // 2))
    y0 = max(0, min(h - ph, cy - ph // 2))
    patch = img[y0 : y0 + ph, x0 : x0 + pw]
    out_size = (pw * 2, ph * 2)
    tiles = {}
    for name in ("nearest", "bilinear", "bicubic", "lanczos"):
        tiles[name] = clamp01(
            cv2.resize(patch, out_size, interpolation=interp_flag(name))
        )
    top = np.hstack([tiles["nearest"], tiles["bilinear"]])
    bot = np.hstack([tiles["bicubic"], tiles["lanczos"]])
    grid = np.vstack([top, bot])
    return OpResult(
        image=None,
        data={"layout": [["nearest", "bilinear"], ["bicubic", "lanczos"]],
              "patch_rect": [x0 / w, y0 / h, pw / w, ph / h]},
        overlays={"comparison": grid},
    )


@register(
    "geom.lens_distort",
    "Lens distortion",
    "geom",
    params=[
        FloatParam("k1", min=-0.6, max=0.6, default=0.0, step=0.005,
                   help="Primary radial term. Negative = barrel, positive = pincushion."),
        FloatParam("k2", min=-0.3, max=0.3, default=0.0, step=0.005),
        FloatParam("p1", min=-0.05, max=0.05, default=0.0, step=0.001, advanced=True),
        FloatParam("p2", min=-0.05, max=0.05, default=0.0, step=0.001, advanced=True),
        BoolParam("invert", default=False,
                  help="Applies the inverse, i.e. REMOVES distortion."),
        EnumParam("interp", options=INTERP_OPTS, default="bicubic"),
    ],
    summary="Applies or removes the Brown-Conrady radial and tangential lens "
    "distortion model.",
    formula_latex=r"r' = r(1 + k_1 r^2 + k_2 r^4),\quad "
    r"\Delta x = 2p_1 xy + p_2(r^2 + 2x^2)",
    when_to_use="Straightening the bowed lines of a wide-angle lens, or "
    "synthesising distortion as an augmentation.",
    tags=["beyond-syllabus"],
)
def lens_distort(img, k1=0.0, k2=0.0, p1=0.0, p2=0.0, invert=False,
                 interp="bicubic", ctx=None):
    if abs(k1) < 1e-6 and abs(k2) < 1e-6 and abs(p1) < 1e-6 and abs(p2) < 1e-6:
        return img
    h, w = img.shape[:2]
    f = max(w, h) / 2.0
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], np.float64)
    D = np.array([k1, k2, p1, p2, 0.0], np.float64)
    if invert:
        mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
    else:
        # To *apply* distortion we need, for each output pixel, where to read
        # from: project the ideal grid through the distortion model.
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        x = (xx - w / 2.0) / f
        y = (yy - h / 2.0) / f
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        mapx = (xd * f + w / 2.0).astype(np.float32)
        mapy = (yd * f + h / 2.0).astype(np.float32)
    out = cv2.remap(img, mapx, mapy, interp_flag(interp),
                    borderMode=cv2.BORDER_REFLECT_101)
    return clamp01(out)


def fit_vignette(gray: np.ndarray) -> tuple[float, float, float]:
    """Fit L(r) = a + b r^2 + c r^4 to heavily low-passed luma.

    Used both by the vignette corrector and by the Advisor's uniformity metric
    (advisor-spec.md 2.8).
    """
    return fit_vignette_q(gray)[:3]


def fit_vignette_q(gray: np.ndarray) -> tuple[float, float, float, float]:
    """As fit_vignette, plus the R^2 of the radial fit.

    The R^2 matters: least squares always returns coefficients, so on an image
    with no radial structure at all it happily reports a large spurious
    falloff. Only a fit that actually explains the variance should be believed,
    so callers gate on this value.
    """
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 6.0)
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    cx = cy = 31.5
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / np.sqrt(2 * 31.5**2)
    A = np.stack([np.ones_like(r).ravel(), (r**2).ravel(), (r**4).ravel()], axis=1)
    y = small.ravel().astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return float(coef[0]), float(coef[1]), float(coef[2]), float(np.clip(r2, 0.0, 1.0))


@register(
    "geom.vignette",
    "Vignette add / remove",
    "geom",
    params=[
        FloatParam("amount", min=-1.0, max=1.0, default=-0.4, step=0.02,
                   help="Negative darkens the corners; positive brightens them."),
        FloatParam("radius", min=0.1, max=2.0, default=0.8, step=0.02),
        FloatParam("feather", min=0.05, max=3.0, default=1.0, step=0.05),
        BoolParam("auto_remove", default=False,
                  help="Fits a radial polynomial to the image and inverts it."),
    ],
    summary="Applies a radial brightness gradient, or measures and removes an "
    "existing one by fitting L(r) = a + b r^2 + c r^4.",
    formula_latex=r"g = f \cdot \big(1 + a\,(r/R)^{\,p}\big)",
    when_to_use="Adding a vignette for effect, or correcting real optical "
    "falloff. The auto mode measures the falloff rather than guessing it, and "
    "backs the Advisor's UNI_VIGNETTE fix.",
    tags=["beyond-syllabus"],
)
def vignette(img, amount=-0.4, radius=0.8, feather=1.0, auto_remove=False, ctx=None):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / np.sqrt(cx**2 + cy**2)

    if auto_remove:
        g = luma(img) if img.ndim == 3 else img
        a, b, c = fit_vignette(g)
        model = a + b * r**2 + c * r**4
        centre = max(a, 1e-4)
        gain = centre / np.maximum(model, 1e-4)
        gain = np.clip(gain, 0.25, 4.0)
        falloff = 1.0 - (a + b + c) / centre
        out = img * (gain[:, :, None] if img.ndim == 3 else gain)
        return OpResult(image=clamp01(out),
                        data={"fit": [round(a, 5), round(b, 5), round(c, 5)],
                              "measured_falloff": round(float(falloff), 4)})

    rr = np.clip(r / max(radius, EPS), 0, 4) ** max(feather, 0.05)
    gain = 1.0 + amount * rr
    gain = np.clip(gain, 0.0, 4.0)
    out = img * (gain[:, :, None] if img.ndim == 3 else gain)
    return clamp01(out)


@register(
    "geom.logpolar",
    "Log-polar transform",
    "geom",
    params=[
        PointParam("center", default=(0.5, 0.5)),
        BoolParam("inverse", default=False),
        EnumParam("interp", options=INTERP_OPTS, default="bilinear"),
    ],
    summary="Remaps to a coordinate system where radius is logarithmic, so "
    "rotation and scale about the centre become pure translations.",
    formula_latex=r"(\rho, \theta) = \left(\log\sqrt{x^2+y^2},\ "
    r"\arctan\frac{y}{x}\right)",
    when_to_use="Rotation- and scale-invariant matching: correlate two log-polar "
    "images and a peak offset gives you the rotation and scale directly.",
    tags=["beyond-syllabus"],
)
def logpolar(img, center=(0.5, 0.5), inverse=False, interp="bilinear", ctx=None):
    h, w = img.shape[:2]
    c = (center[0] * w, center[1] * h)
    maxr = np.hypot(max(c[0], w - c[0]), max(c[1], h - c[1]))
    m = w / np.log(max(maxr, 2.0))
    flags = interp_flag(interp) | cv2.WARP_FILL_OUTLIERS
    if inverse:
        flags |= cv2.WARP_INVERSE_MAP
    return clamp01(cv2.logPolar(img, c, m, flags))


@register(
    "geom.polar",
    "Polar unwrap",
    "geom",
    params=[
        PointParam("center", default=(0.5, 0.5)),
        BoolParam("inverse", default=False),
        EnumParam("interp", options=INTERP_OPTS, default="bilinear"),
    ],
    summary="Linear polar remap: concentric circles become horizontal lines.",
    formula_latex=r"(r, \theta) \mapsto (x, y)",
    when_to_use="Unwrapping circular content -- a clock face, an iris, a "
    "circular gauge -- into a rectangle where it can be processed row-wise.",
    tags=["beyond-syllabus"],
)
def polar(img, center=(0.5, 0.5), inverse=False, interp="bilinear", ctx=None):
    h, w = img.shape[:2]
    c = (center[0] * w, center[1] * h)
    maxr = np.hypot(max(c[0], w - c[0]), max(c[1], h - c[1]))
    flags = interp_flag(interp) | cv2.WARP_FILL_OUTLIERS
    if inverse:
        flags |= cv2.WARP_INVERSE_MAP
    return clamp01(cv2.linearPolar(img, c, maxr, flags))


@register(
    "geom.register_phase",
    "Register (phase correlation)",
    "geom",
    params=[
        ImageRefParam("reference"),
        BoolParam("apply_shift", default=True),
    ],
    summary="Estimates pure translation between two images from the phase of "
    "their cross-power spectrum.",
    formula_latex=r"R = \frac{F_1 \odot \overline{F_2}}{|F_1 \odot \overline{F_2}|},"
    r"\quad \Delta = \arg\max \mathcal{F}^{-1}\{R\}",
    when_to_use="Sub-pixel alignment of two shots of the same scene. Uses only "
    "phase, so it is immune to overall brightness differences.",
    tags=["beyond-syllabus"],
    requires={"second_image": True, "second_image_param": "reference"},
    references=["Kuglin & Hines, The phase correlation image alignment method, 1975"],
)
def register_phase(img, reference=None, apply_shift=True, ctx=None):
    ctx = ctx or {}
    ref = resolve_ref_image(ctx, reference, img)
    a = luma(img) if img.ndim == 3 else img
    b = luma(ref) if ref.ndim == 3 else ref
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(
        np.ascontiguousarray(b, np.float64), np.ascontiguousarray(a, np.float64), win
    )
    data = {"dx": round(float(dx), 3), "dy": round(float(dy), 3),
            "response": round(float(response), 5)}
    if not apply_shift:
        return OpResult(image=img, data=data)
    h, w = img.shape[:2]
    M = np.array([[1, 0, -dx], [0, 1, -dy]], np.float32)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REFLECT_101)
    return OpResult(image=clamp01(out), data=data)


@register(
    "geom.register_ecc",
    "Register (ECC)",
    "geom",
    params=[
        ImageRefParam("reference"),
        EnumParam("motion", options=enum_opts(("translation", "Translation"),
                                              ("euclidean", "Euclidean"),
                                              ("affine", "Affine"),
                                              ("homography", "Homography")),
                  default="affine"),
        IntParam("iterations", min=10, max=2000, default=200),
        FloatParam("eps", min=1e-8, max=1e-2, default=1e-5, log_scale=True,
                   advanced=True),
    ],
    summary="Iteratively maximises the enhanced correlation coefficient between "
    "the image and a reference, solving for a parametric warp.",
    formula_latex=r"\max_{p} \ \frac{\langle f, g(W(x;p))\rangle}"
    r"{\|f\|\,\|g\|}",
    when_to_use="Intensity-based alignment when there are too few features for "
    "keypoint matching -- smooth, low-texture images in particular.",
    tags=["beyond-syllabus"],
    cost="slow",
    requires={"second_image": True, "second_image_param": "reference"},
    references=["Evangelidis & Psarakis, Parametric Image Alignment Using ECC, 2008"],
)
def register_ecc(img, reference=None, motion="affine", iterations=200,
                 eps=1e-5, ctx=None):
    ctx = ctx or {}
    ref = resolve_ref_image(ctx, reference, img)
    a = (luma(img) if img.ndim == 3 else img).astype(np.float32)
    b = (luma(ref) if ref.ndim == 3 else ref).astype(np.float32)
    modes = {"translation": cv2.MOTION_TRANSLATION, "euclidean": cv2.MOTION_EUCLIDEAN,
             "affine": cv2.MOTION_AFFINE, "homography": cv2.MOTION_HOMOGRAPHY}
    mode = modes[motion]
    warp = (np.eye(3, dtype=np.float32) if mode == cv2.MOTION_HOMOGRAPHY
            else np.eye(2, 3, dtype=np.float32))
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iterations), float(eps))
    try:
        cc, warp = cv2.findTransformECC(b, a, warp, mode, crit, None, 5)
    except cv2.error as exc:
        return OpResult(image=img, data={"converged": False},
                        notes=[f"ECC did not converge: {exc}"])
    h, w = img.shape[:2]
    if mode == cv2.MOTION_HOMOGRAPHY:
        out = cv2.warpPerspective(img, np.linalg.inv(warp), (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REFLECT_101)
    else:
        out = cv2.warpAffine(img, warp, (w, h),
                             flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REFLECT_101)
    return OpResult(image=clamp01(out),
                    data={"converged": True, "correlation": round(float(cc), 5),
                          "warp": np.round(warp, 6).tolist()})


@register(
    "geom.seam_carve",
    "Content-aware resize (seam carving)",
    "geom",
    params=[
        IntParam("remove_cols", min=0, max=400, default=40, scale_aware=True),
        IntParam("remove_rows", min=0, max=400, default=0, scale_aware=True),
        EnumParam("energy", options=enum_opts(("sobel", "Gradient (Sobel)"),
                                              ("laplacian", "Laplacian"),
                                              ("entropy", "Local entropy")),
                  default="sobel"),
        BoolParam("show_seams", default=False),
    ],
    summary="Removes the lowest-energy connected seam of pixels repeatedly, "
    "found by dynamic programming over a cumulative energy map, so the image "
    "shrinks without squashing its salient content.",
    formula_latex=r"M(i,j) = e(i,j) + \min\big(M(i{-}1,j{-}1),\ M(i{-}1,j),\ "
    r"M(i{-}1,j{+}1)\big)",
    when_to_use="Changing aspect ratio without distorting subjects. Implemented "
    "from scratch -- the DP recurrence above is the whole algorithm.",
    tags=["beyond-syllabus"],
    cost="slow",
    references=["Avidan & Shamir, Seam Carving for Content-Aware Image Resizing, 2007"],
)
def seam_carve(img, remove_cols=40, remove_rows=0, energy="sobel",
               show_seams=False, ctx=None):
    def energy_map(a):
        g = luma(a) if a.ndim == 3 else a
        if energy == "laplacian":
            return np.abs(cv2.Laplacian(g, cv2.CV_32F))
        if energy == "entropy":
            from skimage.filters.rank import entropy as sk_entropy
            from skimage.morphology import disk
            u8 = (np.clip(g, 0, 1) * 255).astype(np.uint8)
            return sk_entropy(u8, disk(5)).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return np.abs(gx) + np.abs(gy)

    def carve_vertical(a, n, seam_log=None):
        for _ in range(int(n)):
            h, w = a.shape[:2]
            if w <= 2:
                break
            e = energy_map(a)
            M = e.copy()
            back = np.zeros_like(M, np.int32)
            for i in range(1, h):
                left = np.concatenate([[np.inf], M[i - 1, :-1]])
                mid = M[i - 1]
                right = np.concatenate([M[i - 1, 1:], [np.inf]])
                stack = np.vstack([left, mid, right])
                choice = np.argmin(stack, axis=0)
                back[i] = choice - 1
                M[i] = e[i] + stack[choice, np.arange(w)]
            # Backtrack the minimal seam.
            j = int(np.argmin(M[-1]))
            seam = np.empty(h, np.int32)
            for i in range(h - 1, -1, -1):
                seam[i] = j
                j = min(max(j + int(back[i, j]), 0), w - 1)
            if seam_log is not None:
                seam_log.append(seam.copy())
            mask = np.ones((h, w), bool)
            mask[np.arange(h), seam] = False
            if a.ndim == 3:
                a = a[mask].reshape(h, w - 1, a.shape[2])
            else:
                a = a[mask].reshape(h, w - 1)
        return a

    seams: list = [] if show_seams else None
    out = carve_vertical(img, remove_cols, seams)
    if remove_rows > 0:
        out = np.swapaxes(carve_vertical(np.swapaxes(out, 0, 1), remove_rows), 0, 1)

    res = OpResult(image=clamp01(np.ascontiguousarray(out)),
                   data={"removed_cols": int(remove_cols),
                         "removed_rows": int(remove_rows),
                         "final_size": [out.shape[1], out.shape[0]]})
    if show_seams and seams:
        vis = (luma(img) if img.ndim == 3 else img).copy()
        vis = np.dstack([vis] * 3)
        for s in seams[:60]:
            vis[np.arange(len(s)), np.clip(s, 0, vis.shape[1] - 1)] = (1.0, 0.1, 0.1)
        res.overlays["seams"] = vis
    return res


@register(
    "geom.pad",
    "Pad / Canvas",
    "geom",
    params=[
        FloatParam("top", min=0.0, max=1.0, default=0.05, step=0.005),
        FloatParam("right", min=0.0, max=1.0, default=0.05, step=0.005),
        FloatParam("bottom", min=0.0, max=1.0, default=0.05, step=0.005),
        FloatParam("left", min=0.0, max=1.0, default=0.05, step=0.005),
        EnumParam("border", options=BORDER_OPTS, default="constant"),
        ColorParam("fill", default=(0.0, 0.0, 0.0)),
    ],
    summary="Extends the canvas with a chosen border extension mode.",
    formula_latex=r"g(x,y) = f(x - l,\ y - t)",
    when_to_use="Adding a frame, or supplying margin for a filter whose kernel "
    "would otherwise read past the edge.",
)
def pad(img, top=0.05, right=0.05, bottom=0.05, left=0.05, border="constant",
        fill=(0, 0, 0), ctx=None):
    h, w = img.shape[:2]
    t, b = int(top * h), int(bottom * h)
    l, r = int(left * w), int(right * w)
    val = tuple(float(v) for v in fill) if border == "constant" else 0
    return clamp01(cv2.copyMakeBorder(img, t, b, l, r, border_flag(border),
                                      value=val))
