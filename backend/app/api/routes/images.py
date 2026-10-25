"""Image upload and render retrieval. PLAN.md Section 4."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, File, Query, Response, UploadFile

from ...config import settings
from ...core import imageio as iio
from ...core.cache import render_cache
from ...core.errors import ImageTooLarge, RenderNotFound
from ...core.store import store

router = APIRouter(prefix="/api", tags=["images"])


def publish_render(arr: np.ndarray, fmt: str | None = None,
                   quality: int | None = None, meta: dict | None = None) -> dict:
    """Encode an array and register it for content-addressed retrieval (D5)."""
    fmt = fmt or settings.preview_format
    q = quality or settings.preview_jpeg_quality
    payload = iio.encode(arr, fmt=fmt, quality=q)
    r = render_cache.put(payload, iio.mime_for(fmt), arr.shape[1], arr.shape[0], meta)
    return {
        "render_id": r.render_id,
        "url": f"/api/render/{r.render_id}",
        "width": r.width,
        "height": r.height,
        "bytes": len(payload),
    }


@router.post("/images")
async def upload_image(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise ImageTooLarge(
            f"File is {len(data) / 1e6:.1f} MB; the limit is "
            f"{settings.max_upload_bytes / 1e6:.0f} MB."
        )
    decoded = iio.decode(data)
    img = store.add(decoded, name=file.filename or "upload")
    preview = publish_render(img.proxy)
    thumb = publish_render(
        iio.make_proxy(img.proxy, settings.thumbnail_edge)[0], quality=80
    )
    return {
        **img.summary(),
        "preview": preview,
        "thumbnail": thumb,
        "upload_bytes": len(data),
    }


@router.get("/images")
async def list_images():
    return {"images": store.list(), "stats": store.stats()}


@router.get("/images/{image_id}")
async def get_image(image_id: str, target: str = Query("proxy")):
    img = store.get(image_id)
    return {**img.summary(), "render": publish_render(img.get(target))}


@router.delete("/images/{image_id}")
async def delete_image(image_id: str):
    store.get(image_id)  # 404s if absent
    store.remove(image_id)
    return {"deleted": image_id}


@router.get("/render/{render_id}")
async def get_render(render_id: str):
    r = render_cache.get(render_id)
    if r is None:
        raise RenderNotFound(
            f"Render '{render_id}' has expired from the cache; re-request the "
            f"preview."
        )
    # Immutable: the id is content-addressed, so the browser may cache forever.
    return Response(
        content=r.payload,
        media_type=r.mime,
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "Content-Length": str(len(r.payload)),
        },
    )


@router.get("/samples")
async def list_samples():
    """Built-in synthetic samples, so the app is explorable with no upload."""
    from ...core.samples import SAMPLES

    return {"samples": [{"id": k, **v["meta"]} for k, v in SAMPLES.items()]}


@router.post("/samples/{sample_id}")
async def load_sample(sample_id: str):
    from ...core.samples import build_sample

    decoded = build_sample(sample_id)
    img = store.add(decoded, name=f"sample:{sample_id}")
    return {
        **img.summary(),
        "preview": publish_render(img.proxy),
        "thumbnail": publish_render(
            iio.make_proxy(img.proxy, settings.thumbnail_edge)[0], quality=80
        ),
    }
