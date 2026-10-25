"""Pipeline executor. PLAN.md D2 (prefix cache), D3 (scale), D4 (domain), D8 (cancel)."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import registry
from .blend import apply_blend
from .cache import pipeline_cache, prefix_key
from .colorspace import join_alpha, linear_to_srgb, split_alpha, srgb_to_linear
from .errors import (
    AppError,
    Cancelled,
    OpRequiresColor,
    OpRequiresGrayscale,
    SecondImageRequired,
)
from .masks import resolve_mask
from .registry import OpResult


class CancelToken:
    """D8: the executor checks this between stages."""

    def __init__(self) -> None:
        self._flag = threading.Event()

    def cancel(self) -> None:
        self._flag.set()

    @property
    def cancelled(self) -> bool:
        return self._flag.is_set()

    def check(self) -> None:
        if self._flag.is_set():
            raise Cancelled("Superseded by a newer request.")


@dataclass
class OpInvocation:
    """One entry in the client's op stack."""

    op_id: str
    params: dict = field(default_factory=dict)
    enabled: bool = True
    opacity: float = 1.0
    blend: str = "normal"
    mask: dict | None = None
    uid: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> OpInvocation:
        return cls(
            op_id=str(d.get("op_id") or d.get("id") or ""),
            params=dict(d.get("params") or {}),
            enabled=bool(d.get("enabled", True)),
            opacity=float(d.get("opacity", 1.0)),
            blend=str(d.get("blend", "normal")),
            mask=d.get("mask"),
            uid=str(d.get("uid", "")),
        )

    def cache_dict(self, resolved_params: dict) -> dict:
        """What the prefix hash sees. Must be fully deterministic (D2)."""
        return {
            "op": self.op_id,
            "p": resolved_params,
            "o": round(float(self.opacity), 4),
            "b": self.blend,
            "m": self.mask,
        }


@dataclass
class StageTiming:
    op_id: str
    ms: float
    cached: bool


@dataclass
class PipelineResult:
    image: np.ndarray
    timings: list[StageTiming] = field(default_factory=list)
    data: dict = field(default_factory=dict)  # op uid/id -> view data
    overlays: dict = field(default_factory=dict)  # name -> ndarray
    notes: list[str] = field(default_factory=list)
    cache_hits: int = 0

    @property
    def total_ms(self) -> float:
        return round(sum(t.ms for t in self.timings), 3)


def _check_requires(spec: registry.OperationSpec, img: np.ndarray, params: dict) -> None:
    req = spec.requires or {}
    ch = req.get("channels")
    is_color = img.ndim == 3 and img.shape[2] >= 3
    if ch == 3 and not is_color:
        raise OpRequiresColor(f"'{spec.name}' needs a colour image.")
    if ch == 1 and is_color:
        # Not an error: we silently reduce, which is what a user expects from
        # an operator like Canny. Only *hard* grayscale requirements raise.
        if req.get("strict"):
            raise OpRequiresGrayscale(f"'{spec.name}' needs a single-channel image.")
    if req.get("second_image"):
        key = req.get("second_image_param", "reference")
        if not params.get(key):
            raise SecondImageRequired(
                f"'{spec.name}' needs a second image; select one in the "
                f"'{key}' slot."
            )


def _run_single(
    spec: registry.OperationSpec,
    img: np.ndarray,
    params: dict,
    ctx: dict,
) -> OpResult:
    """Invoke one operator, normalising its return into an OpResult."""
    _check_requires(spec, img, params)

    # D4: operators declaring a domain get converted in and out.
    work = img
    alpha = None
    if spec.domain == "linear":
        work, alpha = split_alpha(work)
        work = srgb_to_linear(work)

    out = spec.fn(work, ctx=ctx, **params)

    if isinstance(out, OpResult):
        result = out
    elif isinstance(out, tuple) and len(out) == 2:
        result = OpResult(image=out[0], data=out[1] or {})
    elif isinstance(out, dict):
        result = OpResult(image=None, data=out)
    elif out is None:
        result = OpResult(image=None)
    else:
        result = OpResult(image=out)

    if result.image is not None and spec.domain == "linear":
        result.image = linear_to_srgb(result.image)
        if alpha is not None:
            result.image = join_alpha(result.image, alpha)

    if result.image is not None:
        result.image = np.ascontiguousarray(
            np.nan_to_num(result.image, nan=0.0, posinf=1.0, neginf=0.0),
            dtype=np.float32,
        )
    return result


def execute(
    base: np.ndarray,
    ops: list[OpInvocation],
    *,
    image_id: str = "",
    target: str = "proxy",
    scale_factor: float = 1.0,
    token: CancelToken | None = None,
    use_cache: bool = True,
    collect_views: bool = True,
    ctx: dict | None = None,
) -> PipelineResult:
    """Run the stack, reusing cached prefixes where possible.

    `scale_factor` is full_edge / proxy_edge; it rescales scale_aware params so
    a full-resolution export matches the preview (D3).
    """
    ctx = dict(ctx or {})
    ctx.setdefault("target", target)
    ctx.setdefault("scale_factor", scale_factor)

    result = PipelineResult(image=base)
    chain: list[dict] = []
    current = base

    # Find the longest cached prefix, so a slider move recomputes one operator.
    resolved: list[tuple[OpInvocation, registry.OperationSpec, dict, dict]] = []
    for inv in ops:
        if not inv.op_id:
            continue
        spec = registry.get(inv.op_id)
        params = spec.validate(inv.params)
        if abs(scale_factor - 1.0) > 1e-6:
            params = spec.rescale(params, scale_factor)
        resolved.append((inv, spec, params, inv.cache_dict(params)))

    start_index = 0
    if use_cache and image_id and resolved:
        probe: list[dict] = []
        for i, (_inv, _spec, _params, cd) in enumerate(resolved):
            probe.append(cd)
            hit = pipeline_cache.get(prefix_key(image_id, target, probe))
            if hit is None:
                break
            current = hit
            start_index = i + 1
            result.cache_hits += 1
        chain = probe[:start_index]
        for i in range(start_index):
            result.timings.append(StageTiming(resolved[i][0].op_id, 0.0, True))

    for i in range(start_index, len(resolved)):
        inv, spec, params, cd = resolved[i]
        if token:
            token.check()

        if not inv.enabled:
            # A disabled op still participates in the cache key, so toggling it
            # back on reuses the prefix built while it was off.
            chain.append(cd)
            result.timings.append(StageTiming(inv.op_id, 0.0, False))
            continue

        t0 = time.perf_counter()
        try:
            out = _run_single(spec, current, params, ctx)
        except AppError:
            raise
        except Exception as exc:  # an operator failing must not kill the stack
            result.notes.append(f"{spec.name}: {type(exc).__name__}: {exc}")
            chain.append(cd)
            result.timings.append(
                StageTiming(inv.op_id, (time.perf_counter() - t0) * 1000.0, False)
            )
            continue

        if out.image is not None and not spec.is_view:
            new_img = out.image
            if new_img.shape == current.shape and (
                inv.blend != "normal" or inv.opacity < 1.0 or inv.mask
            ):
                mask = resolve_mask(inv.mask, current) if inv.mask else None
                new_img = apply_blend(current, new_img, inv.blend, inv.opacity, mask)
            current = new_img

        if collect_views:
            if out.data:
                result.data[inv.uid or f"{inv.op_id}#{i}"] = out.data
            if out.overlays:
                for k, v in out.overlays.items():
                    result.overlays[f"{inv.uid or inv.op_id}:{k}"] = v
        result.notes.extend(out.notes)

        ms = (time.perf_counter() - t0) * 1000.0
        result.timings.append(StageTiming(inv.op_id, round(ms, 3), False))

        chain.append(cd)
        if use_cache and image_id and not spec.is_view:
            pipeline_cache.put(prefix_key(image_id, target, list(chain)), current)

    result.image = current
    return result


def post_stats(img: np.ndarray) -> dict:
    """Cheap readouts returned with every preview so the UI never re-requests."""
    from .colorspace import luma

    y = luma(img)
    hist, _ = np.histogram(y, bins=256, range=(0.0, 1.0))
    out: dict[str, Any] = {
        "mean": round(float(y.mean()), 5),
        "std": round(float(y.std()), 5),
        "min": round(float(y.min()), 5),
        "max": round(float(y.max()), 5),
        "clip_low": round(float((y <= 2 / 255).mean()), 6),
        "clip_high": round(float((y >= 253 / 255).mean()), 6),
        "histogram": hist.astype(int).tolist(),
    }
    if img.ndim == 3 and img.shape[2] >= 3:
        for i, name in enumerate(("r", "g", "b")):
            h, _ = np.histogram(img[:, :, i], bins=256, range=(0.0, 1.0))
            out[f"histogram_{name}"] = h.astype(int).tolist()
    return out
