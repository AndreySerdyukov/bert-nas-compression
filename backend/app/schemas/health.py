"""Health endpoint schema."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Liveness plus how much of the registry is actually available."""

    status: str
    app: str
    # A clean clone has no weights: both counts are meaningful, and `models_skipped` carries the
    # difference between "nothing is configured" and "everything failed to load".
    models_loaded: int = 0
    models_skipped: int = 0
