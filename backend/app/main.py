"""Application factory. PLAN.md Section 3, D6, D10."""
from __future__ import annotations

import logging
import time

import cv2
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .core import registry
from .core.errors import AppError

log = logging.getLogger("ipcv")


def create_app() -> FastAPI:
    # D6: cap OpenCV's internal threading so one request cannot starve others.
    cv2.setNumThreads(settings.cv2_threads)
    cv2.ocl.setUseOpenCL(False)  # deterministic results across machines

    count = registry.load_all()
    log.info("registered %d operations", count)

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description=(
            "Classical image-processing platform: a manual Studio of ~240 "
            "operators, and an Advisor that measures image quality and derives "
            "corrective recipes. No learned models are used anywhere."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-ms"] = (
            f"{(time.perf_counter() - t0) * 1000:.1f}"
        )
        return response

    # Uniform error envelope (PLAN.md Section 4)
    @app.exception_handler(AppError)
    async def app_error(_request: Request, exc: AppError):
        return JSONResponse(status_code=exc.status, content=exc.envelope())

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "INVALID_REQUEST",
                               "message": "Request body failed validation.",
                               "detail": exc.errors()[:8]}},
        )

    @app.exception_handler(Exception)
    async def unhandled(_request: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "INTERNAL_ERROR",
                               "message": f"{type(exc).__name__}: {exc}"}},
        )

    from .api.routes import (
        analysis,
        augment,
        export,
        images,
        operations,
        pipeline,
        report,
    )

    for mod in (images, operations, pipeline, analysis, augment, export, report):
        app.include_router(mod.router)

    @app.get("/api/health")
    async def health():
        cat = registry.catalogue()
        return {
            "status": "ok",
            "operations": cat["total"],
            "syllabus_operations": cat["syllabus_count"],
            "categories": len(cat["categories"]),
            "preview_max_edge": settings.preview_max_edge,
            "max_megapixels": settings.max_megapixels,
            "uses_learned_models": False,
        }

    # Single-container mode: serve the built frontend if it is present.
    static = settings.static_dir
    if static.exists() and (static / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(static), html=True), name="web")

    return app


app = create_app()
