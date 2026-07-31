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

from app.api import content, health, models
from app.config import Settings, get_settings
from app.repositories.content import ContentRepository
from app.repositories.examples import ExampleRepository
from app.repositories.model_registry import ModelRegistry
from app.services.inference import InferenceService

logger = logging.getLogger(__name__)


def _build_registry(app: FastAPI, settings: Settings) -> None:
    """Load the checkpoints, or explain in the log why there are none.

    Eagerly, at startup, rather than on first use. The application publishes measured latency, and
    a model loaded on demand would fold seconds of loading into the first measurement a visitor
    ever saw. The cost is a slower boot; the alternative is a wrong number.

    Weights are gitignored, so an empty registry is the ordinary state of a fresh clone. Whatever
    fails to load is kept as a skipped entry with its reason, and the app comes up regardless -
    the methodology half of it needs no weights at all.
    """
    if settings.model_tier == "none":
        logger.info("serving is switched off (model_tier=none)")
        return

    from app.serving.runtime import configure_torch_threads

    configure_torch_threads(settings.torch_threads)
    registry = ModelRegistry(
        models_dir=settings.models_dir,
        data_dir=settings.data_dir,
        baseline=settings.baseline_model,
        device=settings.serve_device,
    )
    registry.load()
    # Warm-up here is the polarity check: it is the last thing standing between a checkpoint that
    # was replaced upstream and a confidently inverted prediction.
    registry.warmup()
    logger.info(
        "registry ready: %d loaded, %d skipped", len(registry.list_infos()), len(registry.skipped)
    )

    app.state.registry = registry
    app.state.inference = InferenceService(registry, settings)


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
    # Committed data too, and deliberately loaded whether or not serving is on: the reviews are
    # readable before anyone has downloaded a checkpoint.
    app.state.examples = ExampleRepository.load(settings.data_dir)
    _build_registry(app, settings)

    app.include_router(health.router)
    app.include_router(content.router)
    app.include_router(models.router)

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        """Nothing reaches a client as a raw traceback; every failure is a logged 500."""
        logger.exception("unhandled error", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    return app


# No module-level `app = create_app()`. Building the application now loads a gigabyte of weights,
# and at module level that would happen on any import of this module - including pytest collecting
# a test that never asks for a model. Uvicorn is pointed at the factory instead:
#     uvicorn app.main:create_app --factory
