"""Analysis and Advisor endpoints. PLAN.md Section 4, advisor-spec.md."""
from __future__ import annotations

import numpy as np
from anyio import to_thread
from fastapi import APIRouter
from pydantic import BaseModel, Field

from ...advisor import recipe as rcp
from ...advisor import verify as vfy
from ...advisor.rules import RULES
from ...analysis import metrics as M
from ...analysis.score import OVERALL_WEIGHTS, SCORING
from ...core.pipeline import OpInvocation, execute
from ...core.store import store
from .images import publish_render

router = APIRouter(prefix="/api", tags=["analysis"])


class AnalysisRequest(BaseModel):
    image_id: str
    ops: list[dict] = Field(default_factory=list)
    target: str = "proxy"
    groups: list[str] | None = None
    include_scores: bool = True


def _resolve(req) -> np.ndarray:
    img = store.get(req.image_id)
    base = img.get(req.target)
    if not req.ops:
        return base
    res = execute(base, [OpInvocation.from_dict(o) for o in req.ops],
                  image_id=req.image_id, target=req.target,
                  collect_views=False, ctx={"original": base})
    return res.image


@router.post("/analysis")
async def analysis(req: AnalysisRequest):
    def work():
        arr = _resolve(req)
        metrics = M.measure_all(arr, req.groups)
        out = {"metrics": metrics}
        if req.include_scores:
            from ...analysis.score import score_all
            out["scores"] = score_all(metrics)
        return out

    return await to_thread.run_sync(work)


@router.get("/analysis/schema")
async def analysis_schema():
    """What the Advisor measures and how each metric is scored."""
    return {
        "groups": sorted(M.GROUPS),
        "scoring": {
            dim: [
                {"group": g, "metric": n, "weight": w,
                 "ideal": [ilo, ihi], "fail": [flo, fhi]}
                for g, n, w, ilo, ihi, flo, fhi in rows
            ]
            for dim, rows in SCORING.items()
        },
        "overall_weights": OVERALL_WEIGHTS,
        "rules": [
            {"id": r.id, "dimension": r.dimension, "title": r.title,
             "warning_only": r.warning_only, "opt_in": r.opt_in,
             "tags": r.tags}
            for r in RULES
        ],
    }


class DiagnoseRequest(BaseModel):
    image_id: str
    ops: list[dict] = Field(default_factory=list)
    target: str = "proxy"
    include_maps: bool = True
    strength: float = 1.0
    include_opt_in: bool = False


@router.post("/advisor/diagnose")
async def diagnose(req: DiagnoseRequest):
    img = store.get(req.image_id)

    def work():
        arr = _resolve(req)
        long_edge = int(max(arr.shape[0], arr.shape[1]))
        d = rcp.diagnose(arr, long_edge=long_edge)
        if abs(req.strength - 1.0) > 1e-6 or req.include_opt_in:
            d["recipe"] = rcp.build_recipe(
                d["findings"], include_opt_in=req.include_opt_in,
                strength=req.strength)
        maps = {}
        if req.include_maps:
            maps = _evidence_maps(arr)
        return d, maps

    d, maps = await to_thread.run_sync(work)

    overlays = {name: publish_render(arr, quality=82) for name, arr in maps.items()}
    return {
        "image_id": req.image_id,
        "scores": d["scores"],
        "findings": d["findings"],
        "recipe": d["recipe"],
        "metrics": d["metrics"],
        "document_mode": d["document_mode"],
        "monochrome": d["monochrome"],
        "notes": d["notes"],
        "maps": overlays,
        "strength": req.strength,
    }


def _evidence_maps(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Visual evidence the finding cards display alongside their numbers."""
    from ...core.colorspace import luma
    from ...ops._util import clamp01, colorize, normalize01

    out: dict[str, np.ndarray] = {}
    try:
        y = np.clip(luma(arr), 0, 1).astype(np.float32)

        # Clipping overlay
        vis = (arr if arr.ndim == 3 else np.dstack([arr] * 3))[:, :, :3] * 0.35
        vis = vis.copy()
        vis[y <= 0.008] = (0.1, 0.35, 1.0)
        vis[y >= 0.992] = (1.0, 0.1, 0.1)
        out["clipping"] = clamp01(vis)

        # Noise map
        import cv2
        med = cv2.medianBlur(y, 3)
        resid = np.abs(y - med)
        out["noise"] = colorize(normalize01(cv2.blur(resid, (9, 9))), "magma")

        # Saliency (ROI weighting evidence)
        from ...ops.inspect import spectral_saliency
        s = spectral_saliency(y)
        s = cv2.resize(s, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_CUBIC)
        out["saliency"] = colorize(normalize01(s), "inferno")

        # Spectrum with detected peaks circled
        F = np.fft.fftshift(np.fft.fft2(y - y.mean()))
        spec = colorize(normalize01(np.log1p(np.abs(F))), "turbo")
        from ...ops.freq import detect_spectral_peaks
        peaks = detect_spectral_peaks(y, 4.5, 6)
        sv = (spec * 255).astype(np.uint8)[:, :, ::-1].copy()
        for p in peaks:
            cx, cy = int(p["x"] * spec.shape[1]), int(p["y"] * spec.shape[0])
            cv2.circle(sv, (cx, cy), 14, (60, 60, 255), 2)
        out["spectrum"] = sv[:, :, ::-1].astype(np.float32) / 255.0

        # Dark channel (haze evidence)
        base = arr if arr.ndim == 3 else np.dstack([arr] * 3)
        mn = np.min(base[:, :, :3], axis=2)
        k = max(3, (int(0.02 * max(arr.shape[:2])) | 1))
        out["dark_channel"] = colorize(cv2.erode(mn, np.ones((k, k), np.uint8)), "bone")
    except Exception:
        pass
    return out


class VerifyRequest(BaseModel):
    image_id: str
    ops: list[dict] = Field(default_factory=list)
    target: str = "proxy"
    attribute_regressions: bool = True


@router.post("/advisor/verify")
async def verify(req: VerifyRequest):
    """Re-measure after a recipe and report the delta (advisor-spec.md 7)."""
    img = store.get(req.image_id)
    base = img.get(req.target)

    def work():
        res = execute(base, [OpInvocation.from_dict(o) for o in req.ops],
                      image_id=req.image_id, target=req.target,
                      collect_views=False, ctx={"original": base})
        v = vfy.verify(base, res.image, ops=req.ops)
        if req.attribute_regressions and v["regressions"]:
            for r in v["regressions"]:
                culprit = vfy.attribute_regression(
                    base, req.ops, r["dimension"], req.image_id, req.target)
                if culprit:
                    r["likely_cause"] = culprit
        return v, res.image

    v, after = await to_thread.run_sync(work)
    return {**v, "render": publish_render(after)}
