"""Export, batch apply and job endpoints. PLAN.md Phase 9."""
from __future__ import annotations

import io
import json
import zipfile

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field

from ...core import imageio as iio
from ...core.cache import render_cache
from ...core.errors import AppError, RenderNotFound
from ...core.jobs import jobs
from ...core.pipeline import OpInvocation, execute
from ...core.store import store

router = APIRouter(prefix="/api", tags=["export"])


class ExportRequest(BaseModel):
    image_id: str
    ops: list[dict] = Field(default_factory=list)
    format: str = "png"
    quality: int = 95
    bit_depth: int = 8
    resize_long_edge: int | None = None
    strip_metadata: bool = True
    filename: str | None = None


@router.post("/export")
async def export(req: ExportRequest):
    """Full-resolution render as a background job (D3: scale_aware rescaling)."""
    img = store.get(req.image_id)
    job = jobs.create("export")

    def run(j):
        jobs.progress(j, 0.05, "Rendering at full resolution")
        base = img.get("full")
        scale = img.long_edge("full") / max(img.long_edge("proxy"), 1)
        res = execute(base, [OpInvocation.from_dict(o) for o in req.ops],
                      image_id=req.image_id, target="full", scale_factor=scale,
                      token=j.token, collect_views=False,
                      ctx={"original": base})
        arr = res.image
        jobs.progress(j, 0.75, "Encoding")

        if req.resize_long_edge:
            arr, _ = iio.make_proxy(arr, int(req.resize_long_edge))
        payload = iio.encode(arr, fmt=req.format, quality=req.quality,
                             bit_depth=req.bit_depth)
        r = render_cache.put(payload, iio.mime_for(req.format),
                             arr.shape[1], arr.shape[0], {"kind": "export"})
        ext = {"jpeg": "jpg"}.get(req.format, req.format)
        stem = (req.filename or (img.name or "export").rsplit(".", 1)[0])
        return {
            "download_id": r.render_id,
            "url": f"/api/export/download/{r.render_id}",
            "filename": f"{stem}_edited.{ext}",
            "bytes": len(payload),
            "width": int(arr.shape[1]),
            "height": int(arr.shape[0]),
            "scale_factor": round(scale, 4),
            "timings": [{"op_id": t.op_id, "ms": t.ms} for t in res.timings],
        }

    jobs.run(job, run)
    return {"job_id": job.job_id}


@router.get("/export/download/{render_id}")
async def download(render_id: str):
    r = render_cache.get(render_id)
    if r is None:
        raise RenderNotFound("That export has expired; render it again.")
    return Response(
        content=r.payload,
        media_type=r.mime,
        headers={"Content-Disposition": 'attachment; filename="export"'},
    )


class BatchRequest(BaseModel):
    image_ids: list[str]
    ops: list[dict] = Field(default_factory=list)
    format: str = "jpeg"
    quality: int = 92
    target: str = "full"
    resize_long_edge: int | None = None


@router.post("/batch/apply")
async def batch_apply(req: BatchRequest):
    """Apply one recipe to many images, each with its own scale handling."""
    if not req.image_ids:
        raise AppError("Select at least one image.")
    for iid in req.image_ids:
        store.get(iid)
    job = jobs.create("batch.apply")
    total = len(req.image_ids)

    def run(j):
        buf = io.BytesIO()
        manifest = {"ops": req.ops, "outputs": []}
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, iid in enumerate(req.image_ids):
                j.token.check()
                src = store.get(iid)
                base = src.get(req.target)
                scale = 1.0
                if req.target == "full":
                    scale = src.long_edge("full") / max(src.long_edge("proxy"), 1)
                res = execute(base, [OpInvocation.from_dict(o) for o in req.ops],
                              image_id=iid, target=req.target,
                              scale_factor=scale, token=j.token,
                              collect_views=False, ctx={"original": base})
                arr = res.image
                if req.resize_long_edge:
                    arr, _ = iio.make_proxy(arr, int(req.resize_long_edge))
                ext = {"jpeg": "jpg"}.get(req.format, req.format)
                stem = (src.name or iid).rsplit(".", 1)[0].replace("/", "_")
                name = f"{stem}_edited.{ext}"
                zf.writestr(name, iio.encode(arr, req.format, req.quality))
                manifest["outputs"].append({
                    "file": name, "image_id": iid,
                    "scale_factor": round(scale, 4),
                    "size": [int(arr.shape[1]), int(arr.shape[0])],
                })
                jobs.progress(j, (i + 1) / total, f"{i + 1}/{total} images")
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        payload = buf.getvalue()
        r = render_cache.put(payload, "application/zip", 0, 0, {"kind": "batch"})
        return {"download_id": r.render_id, "bytes": len(payload),
                "count": total, "url": f"/api/augment/download/{r.render_id}"}

    jobs.run(job, run)
    return {"job_id": job.job_id, "total": total}


@router.get("/jobs")
async def list_jobs():
    return {"jobs": jobs.list()}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    return jobs.get(job_id).summary()


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    jobs.cancel(job_id)
    return jobs.get(job_id).summary()
