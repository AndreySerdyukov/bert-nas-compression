"""Liveness endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas.health import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Report that the process is up, and how much of the model registry came with it.

    The two counts are the useful part: a clean clone has no weights, so `models_loaded` is 0 and
    the app still answers. That is the state CI's docker job asserts.
    """
    registry = getattr(request.app.state, "registry", None)
    return HealthResponse(
        status="ok",
        app=request.app.state.settings.app_name,
        models_loaded=len(registry.list_infos()) if registry is not None else 0,
        models_skipped=len(registry.skipped) if registry is not None else 0,
    )
