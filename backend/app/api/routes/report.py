"""Report generation and websocket progress."""
from __future__ import annotations

import asyncio

from anyio import to_thread
from fastapi import APIRouter, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ...core.cache import pipeline_cache, render_cache
from ...core.jobs import jobs
from ...core.store import store
from ...reporting.report import build_report

router = APIRouter(prefix="/api", tags=["report"])


class ReportRequest(BaseModel):
    image_id: str
    ops: list[dict] = Field(default_factory=list)
    target: str = "proxy"
    title: str = "Image Processing Report"
    include_advisor: bool = True
    include_images: bool = True


@router.post("/report")
async def report(req: ReportRequest):
    img = store.get(req.image_id)

    def work():
        diagnosis = verification = None
        if req.include_advisor:
            from ...advisor.recipe import diagnose as _diag
            from ...advisor.verify import verify as _verify
            from ...core.pipeline import OpInvocation, execute

            base = img.get(req.target)
            diagnosis = _diag(base, long_edge=int(max(base.shape[:2])))
            if req.ops:
                res = execute(base, [OpInvocation.from_dict(o) for o in req.ops],
                              image_id=req.image_id, target=req.target,
                              collect_views=False, ctx={"original": base})
                verification = _verify(base, res.image, ops=req.ops)
        return build_report(img, req.ops, target=req.target,
                            diagnosis=diagnosis, verification=verification,
                            title=req.title, embed_images=req.include_images)

    return await to_thread.run_sync(work)


@router.post("/report/markdown")
async def report_markdown(req: ReportRequest):
    data = await report(req)
    return Response(
        content=data["markdown"],
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="report.md"'},
    )


@router.websocket("/ws/session/{sid}")
async def session_ws(ws: WebSocket, sid: str):
    """Job progress and cache stats. PLAN.md Section 4."""
    await ws.accept()
    watching: set[str] = set()
    try:
        await ws.send_json({"type": "hello", "session": sid})
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=1.0)
                if msg.get("type") == "watch":
                    watching.add(str(msg.get("job_id")))
                elif msg.get("type") == "unwatch":
                    watching.discard(str(msg.get("job_id")))
                elif msg.get("type") == "ping":
                    await ws.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                pass

            for jid in list(watching):
                try:
                    j = jobs.get(jid)
                except Exception:
                    watching.discard(jid)
                    continue
                await ws.send_json({"type": "job", **j.summary()})
                if j.status in ("done", "error", "cancelled"):
                    watching.discard(jid)

            await ws.send_json({
                "type": "stats",
                "pipeline_cache": pipeline_cache.stats(),
                "render_cache": render_cache.stats(),
                "store": store.stats(),
            })
            await asyncio.sleep(0.6)
    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass
