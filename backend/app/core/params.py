"""Parameter type system. PLAN.md Section 5.

One ParamSpec subclass per UI control. This module is the single place where
validation, coercion and JSON-schema emission live, shared by the HTTP layer,
the Advisor, the augmentation sampler and the tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .errors import ParamInvalid, ParamOutOfRange


@dataclass
class ParamSpec:
    """Base parameter. `scale_aware` is load-bearing -- see PLAN.md D3."""

    name: str
    label: str = ""
    default: Any = None
    help: str = ""
    group: str = ""
    advanced: bool = False
    scale_aware: bool = False
    # Normalised-frequency parameters scale INVERSELY with resolution: a
    # cutoff expressed as a fraction of Nyquist selects a different physical
    # frequency on a proxy than on the full image, because the proxy's Nyquist
    # sits lower in scene terms. Such params set both flags.
    scale_inverse: bool = False
    depends_on: dict | None = None  # {"param": "mode", "equals": "manual"}
    auto: bool = False  # offers an "Auto" button backed by the op's auto_params
    unit: str = ""

    kind: str = "any"

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.name.replace("_", " ").strip().capitalize()

    def schema(self) -> dict:
        d = {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
            "help": self.help,
            "group": self.group,
            "advanced": self.advanced,
            "scale_aware": self.scale_aware,
            "scale_inverse": self.scale_inverse,
            "auto": self.auto,
            "unit": self.unit,
        }
        if self.depends_on:
            d["depends_on"] = self.depends_on
        return d

    def coerce(self, value: Any) -> Any:
        return value

    def rescale(self, value: Any, factor: float) -> Any:
        """Rescale for full-resolution export. Only meaningful if scale_aware."""
        return value


@dataclass
class FloatParam(ParamSpec):
    min: float = 0.0
    max: float = 1.0
    step: float = 0.01
    log_scale: bool = False
    kind: str = "float"

    def schema(self) -> dict:
        d = super().schema()
        d.update(
            {"min": self.min, "max": self.max, "step": self.step, "log_scale": self.log_scale}
        )
        return d

    def coerce(self, value: Any) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ParamInvalid(f"'{self.name}' must be a number.", field=self.name)
        if not math.isfinite(v):
            raise ParamInvalid(f"'{self.name}' must be finite.", field=self.name)
        if v < self.min - 1e-9 or v > self.max + 1e-9:
            raise ParamOutOfRange(
                f"'{self.name}' = {v} is outside [{self.min}, {self.max}].",
                field=self.name,
            )
        return min(max(v, self.min), self.max)

    def rescale(self, value: Any, factor: float) -> float:
        if not self.scale_aware:
            return value
        f = (1.0 / factor) if (self.scale_inverse and factor != 0) else factor
        return min(max(float(value) * f, self.min), self.max)


@dataclass
class IntParam(ParamSpec):
    min: int = 0
    max: int = 100
    step: int = 1
    kind: str = "int"

    def schema(self) -> dict:
        d = super().schema()
        d.update({"min": self.min, "max": self.max, "step": self.step})
        return d

    def coerce(self, value: Any) -> int:
        try:
            v = int(round(float(value)))
        except (TypeError, ValueError):
            raise ParamInvalid(f"'{self.name}' must be an integer.", field=self.name)
        if v < self.min or v > self.max:
            raise ParamOutOfRange(
                f"'{self.name}' = {v} is outside [{self.min}, {self.max}].",
                field=self.name,
            )
        return v

    def rescale(self, value: Any, factor: float) -> int:
        if not self.scale_aware:
            return value
        return int(min(max(round(int(value) * factor), self.min), self.max))


@dataclass
class OddIntParam(IntParam):
    """Kernel sizes. Snaps to odd so OpenCV never raises on an even kernel."""

    kind: str = "odd_int"

    def coerce(self, value: Any) -> int:
        v = super().coerce(value)
        if v % 2 == 0:
            v += 1
        return min(max(v, self.min | 1), self.max)

    def rescale(self, value: Any, factor: float) -> int:
        if not self.scale_aware:
            return value
        v = int(round(int(value) * factor))
        if v % 2 == 0:
            v += 1
        return min(max(v, self.min | 1), self.max)


@dataclass
class BoolParam(ParamSpec):
    kind: str = "bool"
    default: Any = False

    def coerce(self, value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)


@dataclass
class EnumParam(ParamSpec):
    options: list[dict] = field(default_factory=list)
    kind: str = "enum"

    def schema(self) -> dict:
        d = super().schema()
        d["options"] = self.options
        return d

    @property
    def values(self) -> list[str]:
        return [str(o["value"]) for o in self.options]

    def coerce(self, value: Any) -> str:
        v = str(value)
        if v not in self.values:
            raise ParamInvalid(
                f"'{self.name}' must be one of {self.values}; got '{v}'.", field=self.name
            )
        return v


@dataclass
class ColorParam(ParamSpec):
    kind: str = "color"
    default: Any = (1.0, 1.0, 1.0)

    def coerce(self, value: Any) -> tuple[float, float, float]:
        if isinstance(value, str) and value.startswith("#"):
            h = value.lstrip("#")
            if len(h) == 3:
                h = "".join(c * 2 for c in h)
            if len(h) != 6:
                raise ParamInvalid(f"'{self.name}' is not a hex colour.", field=self.name)
            return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore
        try:
            seq = [float(x) for x in value]
        except (TypeError, ValueError):
            raise ParamInvalid(f"'{self.name}' must be a colour.", field=self.name)
        if len(seq) < 3:
            raise ParamInvalid(f"'{self.name}' needs 3 components.", field=self.name)
        return tuple(min(max(x, 0.0), 1.0) for x in seq[:3])  # type: ignore


@dataclass
class PointParam(ParamSpec):
    """Normalised (x, y) in [0,1]; resolution-independent by construction."""

    kind: str = "point"
    default: Any = (0.5, 0.5)

    def coerce(self, value: Any) -> tuple[float, float]:
        try:
            x, y = float(value[0]), float(value[1])
        except Exception:
            raise ParamInvalid(f"'{self.name}' must be [x, y].", field=self.name)
        return (min(max(x, -2.0), 3.0), min(max(y, -2.0), 3.0))


@dataclass
class PointsParam(ParamSpec):
    min_count: int = 2
    max_count: int = 64
    kind: str = "points"
    default: Any = field(default_factory=list)

    def schema(self) -> dict:
        d = super().schema()
        d.update({"min_count": self.min_count, "max_count": self.max_count})
        return d

    def coerce(self, value: Any) -> list[tuple[float, float]]:
        if value is None:
            return []
        try:
            pts = [(float(p[0]), float(p[1])) for p in value]
        except Exception:
            raise ParamInvalid(f"'{self.name}' must be a list of [x, y].", field=self.name)
        if len(pts) > self.max_count:
            raise ParamOutOfRange(
                f"'{self.name}' accepts at most {self.max_count} points.", field=self.name
            )
        return pts


@dataclass
class RectParam(ParamSpec):
    """Normalised (x, y, w, h)."""

    kind: str = "rect"
    default: Any = (0.0, 0.0, 1.0, 1.0)

    def coerce(self, value: Any) -> tuple[float, float, float, float]:
        try:
            x, y, w, h = (float(v) for v in value)
        except Exception:
            raise ParamInvalid(f"'{self.name}' must be [x, y, w, h].", field=self.name)
        x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
        w, h = min(max(w, 1e-4), 1.0 - x), min(max(h, 1e-4), 1.0 - y)
        return (x, y, w, h)


@dataclass
class LineParam(ParamSpec):
    kind: str = "line"
    default: Any = ((0.1, 0.5), (0.9, 0.5))

    def coerce(self, value: Any) -> tuple[tuple[float, float], tuple[float, float]]:
        try:
            (x1, y1), (x2, y2) = value
            return ((float(x1), float(y1)), (float(x2), float(y2)))
        except Exception:
            raise ParamInvalid(
                f"'{self.name}' must be [[x1,y1],[x2,y2]].", field=self.name
            )


@dataclass
class RangeParam(ParamSpec):
    """Dual-thumb slider: hue windows, luminosity masks."""

    min: float = 0.0
    max: float = 1.0
    step: float = 0.01
    kind: str = "range"
    default: Any = (0.0, 1.0)

    def schema(self) -> dict:
        d = super().schema()
        d.update({"min": self.min, "max": self.max, "step": self.step})
        return d

    def coerce(self, value: Any) -> tuple[float, float]:
        try:
            lo, hi = float(value[0]), float(value[1])
        except Exception:
            raise ParamInvalid(f"'{self.name}' must be [low, high].", field=self.name)
        lo = min(max(lo, self.min), self.max)
        hi = min(max(hi, self.min), self.max)
        return (min(lo, hi), max(lo, hi))


@dataclass
class MatrixParam(ParamSpec):
    """Convolution kernel editor."""

    rows: int = 3
    cols: int = 3
    min: float = -10.0
    max: float = 10.0
    presets: list[dict] = field(default_factory=list)
    kind: str = "matrix"

    def schema(self) -> dict:
        d = super().schema()
        d.update(
            {
                "rows": self.rows,
                "cols": self.cols,
                "min": self.min,
                "max": self.max,
                "presets": self.presets,
            }
        )
        return d

    def coerce(self, value: Any) -> list[list[float]]:
        try:
            m = [[float(x) for x in row] for row in value]
        except Exception:
            raise ParamInvalid(f"'{self.name}' must be a 2-D numeric array.", field=self.name)
        if not m or not m[0]:
            raise ParamInvalid(f"'{self.name}' cannot be empty.", field=self.name)
        w = len(m[0])
        if any(len(r) != w for r in m):
            raise ParamInvalid(f"'{self.name}' rows must be equal length.", field=self.name)
        if len(m) % 2 == 0 or w % 2 == 0:
            raise ParamInvalid(
                f"'{self.name}' must have odd dimensions so it has a centre tap.",
                field=self.name,
            )
        return m


@dataclass
class CurveParam(ParamSpec):
    """Spline control points, each normalised (x, y) in [0,1]."""

    kind: str = "curve"
    default: Any = ((0.0, 0.0), (1.0, 1.0))
    channels: list[str] = field(default_factory=lambda: ["rgb", "r", "g", "b"])

    def schema(self) -> dict:
        d = super().schema()
        d["channels"] = self.channels
        return d

    def coerce(self, value: Any) -> list[tuple[float, float]]:
        try:
            pts = sorted(
                (min(max(float(p[0]), 0.0), 1.0), min(max(float(p[1]), 0.0), 1.0))
                for p in value
            )
        except Exception:
            raise ParamInvalid(
                f"'{self.name}' must be a list of [x, y] control points.", field=self.name
            )
        if len(pts) < 2:
            raise ParamInvalid(f"'{self.name}' needs at least 2 points.", field=self.name)
        return pts


@dataclass
class ImageRefParam(ParamSpec):
    """A second image: histogram matching, colour transfer, mixup, registration."""

    kind: str = "image_ref"
    default: Any = None

    def coerce(self, value: Any) -> str | None:
        if value in (None, ""):
            return None
        return str(value)


@dataclass
class MaskRefParam(ParamSpec):
    kind: str = "mask_ref"
    default: Any = None

    def coerce(self, value: Any) -> Any:
        return value


@dataclass
class FreqMaskParam(ParamSpec):
    """Paintable frequency-domain mask: a run-length-encoded binary bitmap."""

    kind: str = "freq_mask"
    default: Any = None

    def coerce(self, value: Any) -> Any:
        return value


@dataclass
class LutFileParam(ParamSpec):
    kind: str = "lut_file"
    default: Any = None

    def coerce(self, value: Any) -> Any:
        return value


@dataclass
class SeedParam(IntParam):
    kind: str = "seed"
    min: int = 0
    max: int = 2**31 - 1
    default: Any = 0


@dataclass
class StringParam(ParamSpec):
    kind: str = "string"
    default: Any = ""
    max_length: int = 256

    def coerce(self, value: Any) -> str:
        s = str(value)
        if len(s) > self.max_length:
            raise ParamOutOfRange(
                f"'{self.name}' exceeds {self.max_length} characters.", field=self.name
            )
        return s


def enum_opts(*pairs: tuple[str, str] | str, help_map: dict | None = None) -> list[dict]:
    """Shorthand for enum options: enum_opts('a', ('b', 'Label B'))."""
    out = []
    for p in pairs:
        if isinstance(p, tuple):
            value, label = p
        else:
            value, label = p, p.replace("_", " ").capitalize()
        o = {"value": value, "label": label}
        if help_map and value in help_map:
            o["help"] = help_map[value]
        out.append(o)
    return out


def validate_params(specs: list[ParamSpec], supplied: dict | None) -> dict:
    """Coerce supplied values against the specs, filling in defaults."""
    supplied = supplied or {}
    by_name = {s.name: s for s in specs}
    unknown = set(supplied) - set(by_name)
    if unknown:
        raise ParamInvalid(
            f"Unknown parameter(s): {sorted(unknown)}. "
            f"Expected any of {sorted(by_name)}."
        )
    out: dict = {}
    for spec in specs:
        if spec.name in supplied and supplied[spec.name] is not None:
            out[spec.name] = spec.coerce(supplied[spec.name])
        else:
            out[spec.name] = spec.default
    return out


def rescale_params(specs: list[ParamSpec], values: dict, factor: float) -> dict:
    """D3: rescale scale_aware params when rendering at a different resolution."""
    if abs(factor - 1.0) < 1e-6:
        return values
    by_name = {s.name: s for s in specs}
    out = dict(values)
    for name, val in values.items():
        spec = by_name.get(name)
        if spec is not None and spec.scale_aware and val is not None:
            out[name] = spec.rescale(val, factor)
    return out
