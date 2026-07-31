"""Application factory and wiring.

Layering, enforced by tests/test_api.py: `api/` holds routers with no logic, `services/` holds the
domain code and never imports FastAPI, `repositories/` and `serving/` own everything that touches
the filesystem or torch.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api import content, health
from app.config import get_settings
from app.repositories.content import ContentRepository

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build the application: settings, middleware, routers."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Benchmark and trajectory payloads are JSON in the tens of kilobytes and compress well.
    # Server-sent events are excluded by media type, so this does not buffer progress streams.
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    app.state.settings = settings
    # Committed JSON, read once. Nothing here needs weights, which is why the methodology half of
    # the app works on a clean clone.
    app.state.content = ContentRepository.load(settings.data_dir)

    app.include_router(health.router)
    app.include_router(content.router)

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        """Nothing reaches a client as a raw traceback; every failure is a logged 500."""
        logger.exception("unhandled error", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    return app


app = create_app()
