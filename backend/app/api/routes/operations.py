"""Operation catalogue and auto-parameter endpoints. PLAN.md D1, Section 5."""
from __future__ import annotations

from anyio import to_thread
from fastapi import APIRouter
from pydantic import BaseModel, Field

from ...core import registry
from ...core.blend import MODE_LABELS
from ...core.errors import AppError
from ...core.pipeline import OpInvocation, execute
from ...core.store import store

router = APIRouter(prefix="/api/operations", tags=["operations"])


@router.get("")
async def catalogue():
    cat = registry.catalogue()
    cat["blend_modes"] = MODE_LABELS
    return cat


@router.get("/{op_id}")
async def one(op_id: str):
    spec = registry.get(op_id)
    return {**spec.schema(), "defaults": spec.defaults()}


class AutoRequest(BaseModel):
    image_id: str
    ops: list[dict] = Field(default_factory=list)
    target: str = "proxy"
    params: dict = Field(default_factory=dict)


@router.post("/{op_id}/auto")
async def auto_params(op_id: str, req: AutoRequest):
    """Compute `auto`-capable parameters from the current stack output."""
    spec = registry.get(op_id)
    if spec.auto_fn is None:
        raise AppError(f"'{spec.name}' has no automatic parameter estimation.")
    img = store.get(req.image_id)
    base = img.get(req.target)

    def work():
        if req.ops:
            res = execute(base, [OpInvocation.from_dict(o) for o in req.ops],
                          image_id=req.image_id, target=req.target,
                          collect_views=False, ctx={"original": base})
            arr = res.image
        else:
            arr = base
        return spec.auto_fn(arr, **(req.params or {}))

    values = await to_thread.run_sync(work)
    return {"op_id": op_id, "params": values}
