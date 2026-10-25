"""Augmentation Lab endpoints. PLAN.md Phase 7."""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
from anyio import to_thread
from fastapi import APIRouter, Response
from pydantic import BaseModel, Field

from ...config import settings
from ...core import imageio as iio
from ...core.errors import AppError
from ...core.jobs import jobs
from ...core.pipeline import OpInvocation, execute
from ...core.store import store
from .images import publish_render

router = APIRouter(prefix="/api/augment", tags=["augment"])


class Stage(BaseModel):
    op_id: str
    params: dict = Field(default_factory=dict)
    p: float = 1.0
    group: str | None = None  # "one_of" / "some_of" container id


class AugRequest(BaseModel):
    image_id: str
    stages: list[Stage] = Field(default_factory=list)
    count: int = 8
    seed: int = 0
    target: str = "proxy"
    thumbnail_edge: int = 256
    one_of_groups: dict[str, list[int]] = Field(default_factory=dict)


def _sample_recipe(stages: list[Stage], seed: int,
                   one_of: dict[str, list[int]]) -> list[dict]:
    """Realise one concrete recipe from the stage distribution."""
    rng = np.random.default_rng(seed)
    chosen_idx = set(range(len(stages)))
    # OneOf groups: keep exactly one member of each group.
    for _gid, members in (one_of or {}).items():
        valid = [i for i in members if 0 <= i < len(stages)]
        if len(valid) > 1:
            keep = int(rng.choice(valid))
            for i in valid:
                if i != keep:
                    chosen_idx.discard(i)

    ops = []
    for i, st in enumerate(stages):
        if i not in chosen_idx:
            continue
        params = dict(st.params)
        # Each stage gets its own deterministic sub-seed, so a stage's sampling
        # does not shift when an earlier stage is skipped.
        params.setdefault("seed", int(rng.integers(0, 2**31 - 1)))
        params["p"] = float(st.p)
        ops.append({"op_id": st.op_id, "params": params, "enabled": True,
                    "uid": f"aug-{i}-{st.op_id}"})
    return ops


@router.post("/preview")
async def preview(req: AugRequest):
    """Sample N variants at thumbnail size for the gallery."""
    img = store.get(req.image_id)
    base = img.get(req.target)
    small, _ = iio.make_proxy(base, req.thumbnail_edge)

    def work():
        out = []
        for n in range(int(req.count)):
            recipe = _sample_recipe(req.stages, req.seed + n * 7919,
                                    req.one_of_groups)
            res = execute(small, [OpInvocation.from_dict(o) for o in recipe],
                          image_id="", target="proxy", use_cache=False,
                          collect_views=True, ctx={"original": small})
            applied = {k: v for k, v in res.data.items()}
            out.append({"index": n, "image": res.image, "recipe": recipe,
                        "applied": applied, "notes": res.notes})
        return out

    variants = await to_thread.run_sync(work)
    return {
        "variants": [
            {"index": v["index"],
             "render": publish_render(v["image"], quality=80),
             "recipe": v["recipe"],
             "applied": v["applied"],
             "notes": v["notes"]}
            for v in variants
        ],
        "seed": req.seed,
    }


class GenerateRequest(BaseModel):
    image_ids: list[str]
    stages: list[Stage] = Field(default_factory=list)
    variants_per_image: int = 5
    seed: int = 0
    target: str = "proxy"
    format: str = "jpeg"
    quality: int = 92
    one_of_groups: dict[str, list[int]] = Field(default_factory=dict)


@router.post("/generate")
async def generate(req: GenerateRequest):
    """Batch job: K variants per image, zipped with a reproducibility manifest."""
    if not req.image_ids:
        raise AppError("Select at least one image.")
    for iid in req.image_ids:
        store.get(iid)  # validate up front

    job = jobs.create("augment.generate")
    total = len(req.image_ids) * max(1, req.variants_per_image)

    def run(j):
        buf = io.BytesIO()
        manifest = {
            "seed": req.seed,
            "variants_per_image": req.variants_per_image,
            "format": req.format,
            "quality": req.quality,
            "stages": [s.model_dump() for s in req.stages],
            "outputs": [],
        }
        done = 0
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for iid in req.image_ids:
                src = store.get(iid)
                base = src.get(req.target)
                stem = (src.name or iid).rsplit(".", 1)[0].replace("/", "_")
                for n in range(int(req.variants_per_image)):
                    j.token.check()
                    sub = req.seed + n * 7919 + abs(hash(iid)) % 100003
                    recipe = _sample_recipe(req.stages, sub, req.one_of_groups)
                    res = execute(base,
                                  [OpInvocation.from_dict(o) for o in recipe],
                                  image_id="", target=req.target,
                                  use_cache=False, collect_views=True,
                                  ctx={"original": base})
                    name = f"{stem}_aug{n:03d}.{'jpg' if req.format == 'jpeg' else req.format}"
                    zf.writestr(name, iio.encode(res.image, req.format, req.quality))
                    manifest["outputs"].append({
                        "file": name,
                        "source_image_id": iid,
                        "source_name": src.name,
                        "variant": n,
                        "sub_seed": sub,
                        # The exact realised parameters, so the output can be
                        # reproduced by replaying this recipe.
                        "recipe": recipe,
                        "applied": res.data,
                    })
                    done += 1
                    jobs.progress(j, done / total, f"{done}/{total} variants")
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        payload = buf.getvalue()
        from ...core.cache import render_cache
        r = render_cache.put(payload, "application/zip", 0, 0,
                             {"kind": "augment_zip"})
        return {"download_id": r.render_id, "bytes": len(payload),
                "count": done, "url": f"/api/augment/download/{r.render_id}"}

    jobs.run(job, run)
    return {"job_id": job.job_id, "total": total}


@router.get("/download/{render_id}")
async def download(render_id: str):
    from ...core.cache import render_cache
    from ...core.errors import RenderNotFound

    r = render_cache.get(render_id)
    if r is None:
        raise RenderNotFound("That archive has expired; regenerate it.")
    return Response(
        content=r.payload,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="augmented.zip"'},
    )


# Shipped preset recipes (PLAN.md Phase 7).
PRESETS: dict[str, dict] = {
    "classification_light": {
        "name": "Classification (light)",
        "blurb": "Safe geometric and photometric jitter for general training.",
        "stages": [
            {"op_id": "aug.flip", "params": {"p_h": 0.5}, "p": 1.0},
            {"op_id": "aug.random_crop", "params": {"scale_range": [0.85, 1.0]}, "p": 0.8},
            {"op_id": "aug.brightness_jitter", "params": {"amount_range": [-0.12, 0.12]}, "p": 0.7},
            {"op_id": "aug.contrast_jitter", "params": {"amount_range": [-0.15, 0.15]}, "p": 0.7},
            {"op_id": "aug.saturation_jitter", "params": {"amount_range": [-0.15, 0.15]}, "p": 0.5},
        ],
    },
    "classification_heavy": {
        "name": "Classification (heavy)",
        "blurb": "Aggressive policy for small datasets; includes occlusion.",
        "stages": [
            {"op_id": "aug.flip", "params": {"p_h": 0.5}, "p": 1.0},
            {"op_id": "aug.random_crop", "params": {"scale_range": [0.6, 1.0]}, "p": 0.9},
            {"op_id": "aug.rotate", "params": {"angle_range": [-20, 20]}, "p": 0.6},
            {"op_id": "aug.brightness_jitter", "params": {"amount_range": [-0.25, 0.25]}, "p": 0.8},
            {"op_id": "aug.contrast_jitter", "params": {"amount_range": [-0.3, 0.3]}, "p": 0.8},
            {"op_id": "aug.hue_jitter", "params": {"degrees_range": [-20, 20]}, "p": 0.5},
            {"op_id": "aug.gaussian_noise", "params": {"sigma_range": [0.0, 0.05]}, "p": 0.4},
            {"op_id": "aug.random_erase", "params": {}, "p": 0.4},
        ],
    },
    "detection_safe": {
        "name": "Detection (geometry-safe)",
        "blurb": "Photometric only plus horizontal flip, so boxes stay valid.",
        "stages": [
            {"op_id": "aug.flip", "params": {"p_h": 0.5, "p_v": 0.0}, "p": 1.0},
            {"op_id": "aug.brightness_jitter", "params": {"amount_range": [-0.2, 0.2]}, "p": 0.8},
            {"op_id": "aug.contrast_jitter", "params": {"amount_range": [-0.2, 0.2]}, "p": 0.8},
            {"op_id": "aug.hue_jitter", "params": {"degrees_range": [-10, 10]}, "p": 0.5},
            {"op_id": "aug.jpeg_artifacts", "params": {"quality_range": [45, 90]}, "p": 0.3},
        ],
    },
    "ocr_document": {
        "name": "OCR / document",
        "blurb": "Small rotations, warps and print-and-scan degradation.",
        "stages": [
            {"op_id": "aug.rotate", "params": {"angle_range": [-4, 4]}, "p": 0.8},
            {"op_id": "aug.perspective_jitter", "params": {"magnitude": 0.03}, "p": 0.5},
            {"op_id": "aug.elastic", "params": {"alpha": 12.0, "sigma": 5.0}, "p": 0.3},
            {"op_id": "aug.gaussian_noise", "params": {"sigma_range": [0.005, 0.03]}, "p": 0.6},
            {"op_id": "aug.jpeg_artifacts", "params": {"quality_range": [30, 80]}, "p": 0.6},
            {"op_id": "aug.downscale_upscale", "params": {"factor_range": [0.5, 0.9]}, "p": 0.4},
        ],
    },
    "lowlight_robustness": {
        "name": "Low-light robustness",
        "blurb": "Physically-modelled sensor noise at high ISO plus darkening.",
        "stages": [
            {"op_id": "aug.gamma_jitter", "params": {"amount_range": [0.2, 0.9]}, "p": 0.9},
            {"op_id": "aug.iso_simulation", "params": {"iso_range": [1600, 12800]}, "p": 1.0},
            {"op_id": "aug.motion_blur", "params": {"length_range": [3, 12]}, "p": 0.4},
        ],
    },
    "medical_conservative": {
        "name": "Medical (conservative)",
        "blurb": "Intensity-preserving: no hue change, no occlusion.",
        "stages": [
            {"op_id": "aug.flip", "params": {"p_h": 0.5}, "p": 1.0},
            {"op_id": "aug.rotate", "params": {"angle_range": [-8, 8]}, "p": 0.6},
            {"op_id": "aug.elastic", "params": {"alpha": 18.0, "sigma": 6.0}, "p": 0.4},
            {"op_id": "aug.brightness_jitter", "params": {"amount_range": [-0.08, 0.08]}, "p": 0.5},
            {"op_id": "aug.gaussian_noise", "params": {"sigma_range": [0.002, 0.02]}, "p": 0.5},
        ],
    },
    "sensor_simulation": {
        "name": "Sensor simulation",
        "blurb": "A full capture chain: optics, sensor noise, then the codec.",
        "stages": [
            {"op_id": "aug.defocus_blur", "params": {"radius_range": [1.0, 3.0]}, "p": 0.5},
            {"op_id": "aug.chromatic_aberration", "params": {}, "p": 0.4},
            {"op_id": "aug.vignette", "params": {"amount_range": [-0.35, -0.1]}, "p": 0.6},
            {"op_id": "aug.iso_simulation", "params": {"iso_range": [200, 6400]}, "p": 1.0},
            {"op_id": "aug.jpeg_artifacts", "params": {"quality_range": [60, 95]}, "p": 0.8},
        ],
    },
}


@router.get("/presets")
async def presets():
    return {"presets": [{"id": k, **v} for k, v in PRESETS.items()]}
