"""Operation registry. PLAN.md D1.

One @register per operator. The catalogue this produces is the *only* source the
frontend uses to build controls, so an operator that registers correctly needs
no frontend work at all.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import UnknownOperation
from .params import ParamSpec, rescale_params, validate_params

# Category metadata drives the left-rail / accordion grouping in the UI.
CATEGORIES: dict[str, dict] = {
    "inspect": {"label": "Inspect", "order": 10, "icon": "search",
                "blurb": "Measure and visualise what is in the image."},
    "tone": {"label": "Tone", "order": 20, "icon": "sun",
             "blurb": "Point operations on intensity."},
    "color": {"label": "Color", "order": 30, "icon": "palette",
              "blurb": "White balance, saturation and colour mapping."},
    "smooth": {"label": "Smooth / Denoise", "order": 40, "icon": "droplet",
               "blurb": "Noise reduction and neighbourhood averaging."},
    "sharpen": {"label": "Sharpen / Detail", "order": 50, "icon": "zap",
                "blurb": "High-pass enhancement and restoration."},
    "freq": {"label": "Frequency", "order": 60, "icon": "radio",
             "blurb": "Transforms and filtering in the frequency domain."},
    "geom": {"label": "Geometric", "order": 70, "icon": "move",
             "blurb": "Spatial transformations and warps."},
    "morph": {"label": "Morphology", "order": 80, "icon": "box",
              "blurb": "Set-theoretic shape operations."},
    "edge": {"label": "Edges", "order": 90, "icon": "git-branch",
             "blurb": "Gradient and zero-crossing edge detection."},
    "feat": {"label": "Features", "order": 100, "icon": "crosshair",
             "blurb": "Interest points, descriptors and matching."},
    "seg": {"label": "Segmentation", "order": 110, "icon": "layers",
            "blurb": "Partition the image into regions."},
    "aug": {"label": "Augment", "order": 120, "icon": "shuffle",
            "blurb": "Randomised transformations for dataset generation."},
    "compose": {"label": "Compose", "order": 130, "icon": "copy",
                "blurb": "Masks, blending and pipeline composition."},
}


@dataclass
class OpResult:
    """Returned when an operator produces more than a new image.

    `image` is the new pipeline image (None = pass the input through unchanged,
    which is how pure *views* behave). `data` carries numbers for the UI;
    `overlays` carries extra renders (heatmaps, keypoint plots).
    """

    image: np.ndarray | None = None
    data: dict = field(default_factory=dict)
    overlays: dict[str, np.ndarray] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class OperationSpec:
    id: str
    name: str
    category: str
    fn: Callable[..., Any]
    params: list[ParamSpec] = field(default_factory=list)

    # Documentation (required -- the schema-completeness test enforces it)
    summary: str = ""
    formula_latex: str = ""
    when_to_use: str = ""
    references: list[str] = field(default_factory=list)

    # Behaviour flags
    domain: str = "srgb"  # "srgb" | "linear" | "either"  (D4)
    is_view: bool = False  # produces data/overlays, does not alter the image
    syllabus: bool = False  # [S] in the catalogue
    requires: dict = field(default_factory=dict)  # channels / binary / second_image
    tags: list[str] = field(default_factory=list)
    cost: str = "fast"  # fast | medium | slow -- UI warns before slow ops at full res
    auto_fn: Callable[..., dict] | None = None  # backs the "Auto" buttons

    def schema(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "params": [p.schema() for p in self.params],
            "summary": self.summary,
            "formula_latex": self.formula_latex,
            "when_to_use": self.when_to_use,
            "references": self.references,
            "domain": self.domain,
            "is_view": self.is_view,
            "syllabus": self.syllabus,
            "requires": self.requires,
            "tags": self.tags,
            "cost": self.cost,
            "has_auto": self.auto_fn is not None,
        }

    def validate(self, supplied: dict | None) -> dict:
        return validate_params(self.params, supplied)

    def rescale(self, values: dict, factor: float) -> dict:
        return rescale_params(self.params, values, factor)

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}


_REGISTRY: dict[str, OperationSpec] = {}


def register(
    id: str,
    name: str,
    category: str,
    *,
    params: list[ParamSpec] | None = None,
    summary: str = "",
    formula_latex: str = "",
    when_to_use: str = "",
    references: list[str] | None = None,
    domain: str = "srgb",
    is_view: bool = False,
    syllabus: bool = False,
    requires: dict | None = None,
    tags: list[str] | None = None,
    cost: str = "fast",
    auto_fn: Callable[..., dict] | None = None,
) -> Callable:
    """Decorator registering one operator."""

    def deco(fn: Callable) -> Callable:
        if id in _REGISTRY:
            raise ValueError(f"Duplicate operation id '{id}'")
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category '{category}' for operation '{id}'")
        _REGISTRY[id] = OperationSpec(
            id=id,
            name=name,
            category=category,
            fn=fn,
            params=params or [],
            summary=summary.strip(),
            formula_latex=formula_latex.strip(),
            when_to_use=when_to_use.strip(),
            references=references or [],
            domain=domain,
            is_view=is_view,
            syllabus=syllabus,
            requires=requires or {},
            tags=tags or [],
            cost=cost,
            auto_fn=auto_fn,
        )
        fn._op_id = id  # type: ignore[attr-defined]
        return fn

    return deco


def get(op_id: str) -> OperationSpec:
    spec = _REGISTRY.get(op_id)
    if spec is None:
        raise UnknownOperation(f"No operation with id '{op_id}'.")
    return spec


def exists(op_id: str) -> bool:
    return op_id in _REGISTRY


def all_ops() -> list[OperationSpec]:
    return sorted(
        _REGISTRY.values(),
        key=lambda s: (CATEGORIES[s.category]["order"], s.name),
    )


def catalogue() -> dict:
    ops = all_ops()
    used = {o.category for o in ops}
    return {
        "categories": [
            {"id": cid, **meta, "count": sum(1 for o in ops if o.category == cid)}
            for cid, meta in sorted(CATEGORIES.items(), key=lambda kv: kv[1]["order"])
            if cid in used
        ],
        "operations": [o.schema() for o in ops],
        "total": len(ops),
        "syllabus_count": sum(1 for o in ops if o.syllabus),
    }


def load_all() -> int:
    """Import every operator module so the decorators run. Called once at startup."""
    from .. import ops  # noqa: F401  (ops/__init__.py imports each family)

    return len(_REGISTRY)
