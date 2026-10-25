"""Session image store: in-memory, TTL'd, byte-capped. PLAN.md D9."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from ..config import settings
from .errors import ImageNotFound
from .imageio import DecodedImage, build_pyramid, make_proxy


@dataclass
class StoredImage:
    image_id: str
    original: np.ndarray  # float32, native resolution
    proxy: np.ndarray  # float32, long edge <= preview_max_edge
    proxy_scale: float
    pyramid: list[np.ndarray] = field(default_factory=list)
    name: str = ""
    source_format: str = "jpeg"
    source_bit_depth: int = 8
    exif: dict = field(default_factory=dict)
    had_alpha: bool = False
    was_grayscale: bool = False
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    @property
    def width(self) -> int:
        return self.original.shape[1]

    @property
    def height(self) -> int:
        return self.original.shape[0]

    @property
    def channels(self) -> int:
        return 1 if self.original.ndim == 2 else self.original.shape[2]

    @property
    def nbytes(self) -> int:
        return (
            self.original.nbytes
            + self.proxy.nbytes
            + sum(p.nbytes for p in self.pyramid)
        )

    def get(self, target: str) -> np.ndarray:
        """'proxy' for interactive work, 'full' for export (D3)."""
        return self.original if target == "full" else self.proxy

    def long_edge(self, target: str) -> int:
        a = self.get(target)
        return max(a.shape[0], a.shape[1])

    def summary(self) -> dict:
        return {
            "image_id": self.image_id,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "megapixels": round(self.width * self.height / 1e6, 2),
            "proxy_width": self.proxy.shape[1],
            "proxy_height": self.proxy.shape[0],
            "proxy_scale": round(self.proxy_scale, 4),
            "source_format": self.source_format,
            "source_bit_depth": self.source_bit_depth,
            "had_alpha": self.had_alpha,
            "was_grayscale": self.was_grayscale,
            "exif": self.exif,
            "created_at": self.created_at,
        }


class ImageStore:
    def __init__(self, ttl: int, max_bytes: int):
        self.ttl = ttl
        self.max_bytes = max_bytes
        self._items: dict[str, StoredImage] = {}
        self._lock = threading.RLock()

    def add(self, decoded: DecodedImage, name: str = "") -> StoredImage:
        proxy, scale = make_proxy(decoded.data, settings.preview_max_edge)
        img = StoredImage(
            image_id=uuid.uuid4().hex,
            original=decoded.data,
            proxy=proxy,
            proxy_scale=scale,
            pyramid=build_pyramid(proxy, 4),
            name=name,
            source_format=decoded.source_format,
            source_bit_depth=decoded.source_bit_depth,
            exif=decoded.exif,
            had_alpha=decoded.had_alpha,
            was_grayscale=decoded.was_grayscale,
        )
        with self._lock:
            self._items[img.image_id] = img
            self._sweep()
        return img

    def get(self, image_id: str) -> StoredImage:
        with self._lock:
            img = self._items.get(image_id)
            if img is None:
                raise ImageNotFound(
                    f"Image '{image_id}' is not in the store; it may have expired. "
                    "Please re-upload."
                )
            img.last_used = time.time()
            return img

    def has(self, image_id: str) -> bool:
        with self._lock:
            return image_id in self._items

    def remove(self, image_id: str) -> None:
        with self._lock:
            self._items.pop(image_id, None)

    def list(self) -> list[dict]:
        with self._lock:
            return [i.summary() for i in sorted(
                self._items.values(), key=lambda x: -x.created_at
            )]

    def _sweep(self) -> None:
        """TTL eviction first, then byte-cap eviction of the least recently used."""
        now = time.time()
        for k, v in list(self._items.items()):
            if now - v.last_used > self.ttl:
                del self._items[k]
        total = sum(v.nbytes for v in self._items.values())
        if total <= self.max_bytes:
            return
        for k, v in sorted(self._items.items(), key=lambda kv: kv[1].last_used):
            if total <= self.max_bytes or len(self._items) <= 1:
                break
            total -= v.nbytes
            del self._items[k]

    def stats(self) -> dict:
        with self._lock:
            return {
                "images": len(self._items),
                "bytes": sum(v.nbytes for v in self._items.values()),
                "max_bytes": self.max_bytes,
                "ttl_seconds": self.ttl,
            }


store = ImageStore(settings.image_ttl_seconds, settings.store_max_bytes)
