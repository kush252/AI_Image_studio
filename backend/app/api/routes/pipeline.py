"""Pipeline preview and probe. PLAN.md Section 4, D5, D8."""
from __future__ import annotations

import threading

import numpy as np
from anyio import to_thread
from fastapi import APIRouter
from pydantic import BaseModel, Field

from ...config import settings
from ...core import pipeline as pl
from ...core import registry
from ...core.cache import pipeline_cache, render_cache
from ...core.pipeline import CancelToken, OpInvocation, execute, post_stats
from ...core.store import store
from .images import publish_render

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

# D8: one in-flight render per (image, client session). A newer request cancels
# the older one rather than queueing behind it.
_inflight: dict[str, tuple[int, CancelToken]] = {}
_inflight_lock = threading.Lock()


class OpSpec(BaseModel):
    op_id: str
    params: dict = Field(default_factory=dict)
    enabled: bool = True
    opacity: float = 1.0
    blend: str = "normal"
    mask: dict | None = None
    uid: str = ""


class PreviewRequest(BaseModel):
    image_id: str
    ops: list[OpSpec] = Field(default_factory=list)
    target: str = "proxy"
    seq: int = 0
    session: str = "default"
    include_stats: bool = True
    include_overlays: bool = True
    format: str | None = None
    quality: int | None = None


def _claim(session: str, seq: int) -> CancelToken:
    """Register this request as the newest, cancelling any older one."""
    token = CancelToken()
    with _inflight_lock:
        prev = _inflight.get(session)
        if prev is not None and prev[0] < seq:
            prev[1].cancel()
        _inflight[session] = (seq, token)
    return token


def _release(session: str, seq: int) -> None:
    with _inflight_lock:
        cur = _inflight.get(session)
        if cur is not None and cur[0] == seq:
            _inflight.pop(session, None)


@router.post("/preview")
async def preview(req: PreviewRequest):
    img = store.get(req.image_id)
    token = _claim(req.session, req.seq)

    base = img.get(req.target)
    scale = 1.0
    if req.target == "full" and img.proxy_scale > 0:
        # D3: rescale scale_aware params by the proxy-to-full ratio.
        scale = img.long_edge("full") / max(img.long_edge("proxy"), 1)

    ops = [OpInvocation.from_dict(o.model_dump()) for o in req.ops]
    ctx = {"original": base, "image_id": req.image_id}

    def work():
        return execute(
            base, ops, image_id=req.image_id, target=req.target,
            scale_factor=scale, token=token, collect_views=req.include_overlays,
            ctx=ctx,
        )

    try:
        result = await to_thread.run_sync(work)
    finally:
        _release(req.session, req.seq)

    render = publish_render(result.image, req.format, req.quality,
                            meta={"image_id": req.image_id})

    overlays = {}
    if req.include_overlays:
        for name, arr in result.overlays.items():
            overlays[name] = publish_render(arr, quality=82)

    out = {
        "seq": req.seq,
        "render": render,
        "timings": [{"op_id": t.op_id, "ms": t.ms, "cached": t.cached}
                    for t in result.timings],
        "total_ms": result.total_ms,
        "cache_hits": result.cache_hits,
        "data": result.data,
        "overlays": overlays,
        "notes": result.notes,
        "scale_factor": round(scale, 4),
        "width": int(result.image.shape[1]),
        "height": int(result.image.shape[0]),
    }
    if req.include_stats:
        out["stats"] = post_stats(result.image)
    return out


class ProbeRequest(BaseModel):
    image_id: str
    ops: list[OpSpec] = Field(default_factory=list)
    target: str = "proxy"


@router.post("/probe")
async def probe(req: ProbeRequest):
    """Cheap readouts for the current stack top, with no new render (Section 4)."""
    img = store.get(req.image_id)
    base = img.get(req.target)
    ops = [OpInvocation.from_dict(o.model_dump()) for o in req.ops]

    def work():
        res = execute(base, ops, image_id=req.image_id, target=req.target,
                      collect_views=False, ctx={"original": base})
        return post_stats(res.image)

    stats = await to_thread.run_sync(work)
    return {"stats": stats}


@router.get("/cache")
async def cache_stats():
    return {
        "pipeline": pipeline_cache.stats(),
        "renders": render_cache.stats(),
        "store": store.stats(),
    }


@router.post("/cache/clear")
async def clear_cache():
    pipeline_cache.invalidate_prefix("")
    return {"cleared": True, "pipeline": pipeline_cache.stats()}
