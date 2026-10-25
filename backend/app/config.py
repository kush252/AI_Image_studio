"""Application settings. See PLAN.md D3, D9, D10."""
from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Image Processing Platform"
    debug: bool = True

    # D3 - proxy resolution
    preview_max_edge: int = 1600
    preview_allowed_edges: tuple[int, ...] = (1024, 1600, 2048)
    thumbnail_edge: int = 256

    # D10 - upload hardening
    max_upload_bytes: int = 25 * 1024 * 1024
    max_megapixels: float = 40.0

    # D2 - pipeline cache
    cache_max_bytes: int = 1_500_000_000
    render_cache_max_bytes: int = 400_000_000

    # D9 - session store
    image_ttl_seconds: int = 7200
    store_max_bytes: int = 2_000_000_000

    # D6 - worker pool
    worker_threads: int = 4
    cv2_threads: int = 2

    # encoding
    preview_jpeg_quality: int = 88
    preview_format: str = "jpeg"

    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )

    @property
    def temp_dir(self) -> Path:
        d = Path(tempfile.gettempdir()) / "ipcv_platform"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def static_dir(self) -> Path:
        # This file lives in backend/app/, and the Vite build writes to
        # backend/app/static, so the directory sits beside this module.
        return Path(__file__).resolve().parent / "static"


settings = Settings()
