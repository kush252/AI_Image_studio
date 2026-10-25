"""Decode, encode, proxy generation and upload hardening. PLAN.md D3, D4, D10."""
from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import ExifTags, Image

from ..config import settings
from .colorspace import join_alpha, split_alpha
from .errors import DecodeFailed, ImageTooLarge, UnsupportedFormat

# D10: validate by magic bytes, never by extension.
MAGIC = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"GIF87a": "gif",
    b"GIF89a": "gif",
    b"BM": "bmp",
    b"II*\x00": "tiff",
    b"MM\x00*": "tiff",
}


def sniff_format(data: bytes) -> str:
    for magic, name in MAGIC.items():
        if data.startswith(magic):
            return name
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    raise UnsupportedFormat(
        "Unrecognised image format. Supported: JPEG, PNG, WebP, TIFF, BMP."
    )


@dataclass
class DecodedImage:
    """A decoded image plus everything we learned while decoding it."""

    data: np.ndarray  # float32 [0,1], HxW or HxWx3 or HxWx4
    width: int
    height: int
    channels: int
    source_format: str
    source_bit_depth: int
    exif: dict
    had_alpha: bool
    was_grayscale: bool


def _exif_summary(data: bytes) -> tuple[dict, int]:
    """Extract a readable EXIF subset and the orientation tag."""
    summary: dict = {}
    orientation = 1
    try:
        with Image.open(io.BytesIO(data)) as im:
            raw = im.getexif()
            if raw:
                tagmap = {ExifTags.TAGS.get(k, str(k)): v for k, v in raw.items()}
                orientation = int(tagmap.get("Orientation", 1) or 1)
                for key in (
                    "Make",
                    "Model",
                    "DateTime",
                    "ExposureTime",
                    "FNumber",
                    "ISOSpeedRatings",
                    "FocalLength",
                    "LensModel",
                    "Software",
                ):
                    if key in tagmap:
                        v = tagmap[key]
                        summary[key] = float(v) if isinstance(v, (int, float)) else str(v)
            summary["icc_profile"] = bool(im.info.get("icc_profile"))
    except Exception:  # EXIF is advisory; never fail a decode over it
        pass
    return summary, orientation


def _apply_orientation(img: np.ndarray, orientation: int) -> np.ndarray:
    """D10: honour the EXIF orientation tag so phone photos are not sideways."""
    if orientation == 2:
        return np.ascontiguousarray(img[:, ::-1])
    if orientation == 3:
        return np.ascontiguousarray(img[::-1, ::-1])
    if orientation == 4:
        return np.ascontiguousarray(img[::-1, :])
    if orientation == 5:
        return np.ascontiguousarray(np.swapaxes(img, 0, 1))
    if orientation == 6:
        return np.ascontiguousarray(np.swapaxes(img, 0, 1)[:, ::-1])
    if orientation == 7:
        return np.ascontiguousarray(np.swapaxes(img, 0, 1)[::-1, ::-1])
    if orientation == 8:
        return np.ascontiguousarray(np.swapaxes(img, 0, 1)[::-1, :])
    return img


def decode(data: bytes) -> DecodedImage:
    """Decode bytes to float32 [0,1]. Enforces the D10 guards before allocating."""
    if len(data) > settings.max_upload_bytes:
        raise ImageTooLarge(
            f"File is {len(data) / 1e6:.1f} MB; the limit is "
            f"{settings.max_upload_bytes / 1e6:.0f} MB."
        )
    fmt = sniff_format(data)

    # Check declared dimensions before decoding: cheap defence against
    # decompression bombs, which look tiny on disk.
    try:
        with Image.open(io.BytesIO(data)) as probe:
            w, h = probe.size
    except Exception as exc:
        raise DecodeFailed(f"Could not read image header: {exc}") from exc
    mp = (w * h) / 1e6
    if mp > settings.max_megapixels:
        raise ImageTooLarge(
            f"Image is {mp:.1f} MP ({w}x{h}); the limit is "
            f"{settings.max_megapixels:.0f} MP."
        )

    exif, orientation = _exif_summary(data)

    buf = np.frombuffer(data, dtype=np.uint8)
    raw = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise DecodeFailed("Decoder returned no data; the file may be corrupt.")

    bit_depth = 16 if raw.dtype == np.uint16 else 8
    if raw.dtype == np.uint8:
        arr = raw.astype(np.float32) / 255.0
    elif raw.dtype == np.uint16:
        arr = raw.astype(np.float32) / 65535.0
    else:
        arr = np.clip(raw.astype(np.float32), 0.0, 1.0)

    was_gray = arr.ndim == 2
    had_alpha = arr.ndim == 3 and arr.shape[2] == 4

    # OpenCV decodes BGR(A); the whole app works in RGB(A).
    if arr.ndim == 3:
        if arr.shape[2] == 3:
            arr = arr[:, :, ::-1]
        elif arr.shape[2] == 4:
            arr = np.dstack([arr[:, :, 2], arr[:, :, 1], arr[:, :, 0], arr[:, :, 3]])
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr = _apply_orientation(arr, orientation)

    return DecodedImage(
        data=arr,
        width=arr.shape[1],
        height=arr.shape[0],
        channels=1 if arr.ndim == 2 else arr.shape[2],
        source_format=fmt,
        source_bit_depth=bit_depth,
        exif=exif,
        had_alpha=had_alpha,
        was_grayscale=was_gray,
    )


def make_proxy(img: np.ndarray, max_edge: int) -> tuple[np.ndarray, float]:
    """Downscale so the long edge is at most max_edge. Returns (proxy, scale)."""
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return img, 1.0
    scale = max_edge / float(long_edge)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    # INTER_AREA is the correct choice for downscaling (proper averaging).
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(out, dtype=np.float32), scale


def build_pyramid(img: np.ndarray, levels: int = 4) -> list[np.ndarray]:
    """Gaussian pyramid, cached on upload for the zoom/scale-space views."""
    out = [img]
    cur = img
    for _ in range(levels):
        if min(cur.shape[:2]) < 16:
            break
        cur = cv2.pyrDown(cur)
        out.append(np.ascontiguousarray(cur, dtype=np.float32))
    return out


def encode(
    img: np.ndarray,
    fmt: str = "jpeg",
    quality: int = 88,
    bit_depth: int = 8,
) -> bytes:
    """Quantise once, at the very end (D4), and encode."""
    arr = np.clip(np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    rgb, alpha = split_alpha(arr)

    if rgb.ndim == 2:
        bgr = rgb
    else:
        bgr = rgb[:, :, ::-1]
    if alpha is not None and fmt in ("png", "webp", "tiff"):
        bgr = np.dstack([bgr, alpha]) if bgr.ndim == 3 else bgr

    fmt = fmt.lower()
    if bit_depth == 16 and fmt in ("png", "tiff"):
        out = (bgr * 65535.0 + 0.5).astype(np.uint16)
    else:
        out = (bgr * 255.0 + 0.5).astype(np.uint8)

    if fmt in ("jpeg", "jpg"):
        ext, params = ".jpg", [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
        if out.ndim == 3 and out.shape[2] == 4:
            out = out[:, :, :3]  # JPEG has no alpha
    elif fmt == "png":
        ext, params = ".png", [cv2.IMWRITE_PNG_COMPRESSION, 6]
    elif fmt == "webp":
        ext, params = ".webp", [cv2.IMWRITE_WEBP_QUALITY, int(quality)]
    elif fmt in ("tiff", "tif"):
        ext, params = ".tiff", []
    elif fmt == "bmp":
        ext, params = ".bmp", []
    else:
        raise UnsupportedFormat(f"Cannot encode to '{fmt}'.")

    ok, enc = cv2.imencode(ext, out, params)
    if not ok:
        raise DecodeFailed(f"Encoding to {fmt} failed.")
    return enc.tobytes()


def mime_for(fmt: str) -> str:
    return {
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "tiff": "image/tiff",
        "tif": "image/tiff",
        "bmp": "image/bmp",
    }.get(fmt.lower(), "application/octet-stream")


__all__ = [
    "DecodedImage",
    "build_pyramid",
    "decode",
    "encode",
    "join_alpha",
    "make_proxy",
    "mime_for",
    "sniff_format",
    "split_alpha",
]
