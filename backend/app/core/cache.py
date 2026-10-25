"""Byte-bounded LRU caches. PLAN.md D2 (pipeline prefix cache) and D5 (renders)."""
from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import settings


def canonical_json(obj: Any) -> str:
    """Stable serialisation so equal recipes hash equal. D2 depends on this."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_fallback)


def _fallback(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return round(float(o), 6)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def prefix_key(image_id: str, target: str, ops: list[dict]) -> str:
    """key_i = sha1(image_id | target | canonical_json(ops[0..i]))  -- PLAN.md D2."""
    payload = f"{image_id}|{target}|{canonical_json(ops)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class ArrayCache:
    """LRU of numpy arrays bounded by total bytes, spilling to .npy on eviction.

    Spilling (rather than dropping) matters because recomputing a 10-operator
    prefix is far more expensive than reading it back off disk.
    """

    def __init__(self, max_bytes: int, spill_dir: Path | None = None):
        self.max_bytes = max_bytes
        self._mem: OrderedDict[str, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()
        self._spill_dir = spill_dir or (settings.temp_dir / "spill")
        self._spill_dir.mkdir(parents=True, exist_ok=True)
        self._spilled: set[str] = set()
        self.hits = 0
        self.misses = 0
        self.spill_hits = 0

    def _path(self, key: str) -> Path:
        return self._spill_dir / f"{key}.npy"

    def get(self, key: str) -> np.ndarray | None:
        with self._lock:
            if key in self._mem:
                self._mem.move_to_end(key)
                self.hits += 1
                return self._mem[key]
            if key in self._spilled:
                p = self._path(key)
                if p.exists():
                    try:
                        arr = np.load(p, allow_pickle=False)
                    except Exception:
                        self._spilled.discard(key)
                        self.misses += 1
                        return None
                    self.spill_hits += 1
                    self.hits += 1
                    self._insert(key, arr)
                    return arr
                self._spilled.discard(key)
            self.misses += 1
            return None

    def put(self, key: str, arr: np.ndarray) -> None:
        with self._lock:
            self._insert(key, arr)

    def _insert(self, key: str, arr: np.ndarray) -> None:
        if key in self._mem:
            self._bytes -= self._mem[key].nbytes
            del self._mem[key]
        if arr.nbytes > self.max_bytes:
            return  # a single array larger than the whole budget is not cacheable
        self._mem[key] = arr
        self._bytes += arr.nbytes
        self._mem.move_to_end(key)
        self._evict()

    def _evict(self) -> None:
        while self._bytes > self.max_bytes and self._mem:
            k, v = self._mem.popitem(last=False)  # oldest
            self._bytes -= v.nbytes
            try:
                np.save(self._path(k), v, allow_pickle=False)
                self._spilled.add(k)
            except Exception:
                pass  # a failed spill is just a cache miss later

    def invalidate_prefix(self, image_id: str) -> None:
        """Drop everything belonging to one image (on delete or re-upload)."""
        with self._lock:
            self._mem.clear()
            self._bytes = 0
            for k in list(self._spilled):
                try:
                    self._path(k).unlink(missing_ok=True)
                except Exception:
                    pass
            self._spilled.clear()

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "entries": len(self._mem),
                "bytes": self._bytes,
                "max_bytes": self.max_bytes,
                "spilled": len(self._spilled),
                "hits": self.hits,
                "misses": self.misses,
                "spill_hits": self.spill_hits,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
            }


@dataclass
class Render:
    """An encoded image held for content-addressed retrieval (D5)."""

    render_id: str
    payload: bytes
    mime: str
    width: int
    height: int
    meta: dict = field(default_factory=dict)


class RenderCache:
    """LRU of encoded bytes. Keys are opaque ids handed to the client."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._items: OrderedDict[str, Render] = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()

    def put(
        self, payload: bytes, mime: str, width: int, height: int, meta: dict | None = None
    ) -> Render:
        rid = uuid.uuid4().hex
        r = Render(rid, payload, mime, width, height, meta or {})
        with self._lock:
            self._items[rid] = r
            self._bytes += len(payload)
            while self._bytes > self.max_bytes and len(self._items) > 1:
                _, old = self._items.popitem(last=False)
                self._bytes -= len(old.payload)
        return r

    def get(self, render_id: str) -> Render | None:
        with self._lock:
            r = self._items.get(render_id)
            if r is not None:
                self._items.move_to_end(render_id)
            return r

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._items),
                "bytes": self._bytes,
                "max_bytes": self.max_bytes,
            }


pipeline_cache = ArrayCache(settings.cache_max_bytes)
render_cache = RenderCache(settings.render_cache_max_bytes)
analysis_cache = ArrayCache(256_000_000, spill_dir=settings.temp_dir / "analysis_spill")
