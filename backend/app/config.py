"""Application settings, read from the environment with an `APP_` prefix."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parents[1]

# Which models to build at startup. "core" is the five the product is about; "all" adds the two
# extra full-BERT runs used only to show the spread between three fine-tunes of one architecture;
# "none" starts with an empty registry, which is what CI's docker job exercises.
ModelTier = Literal["core", "all", "none"]


class Settings(BaseSettings):
    """Runtime configuration. Every field is overridable as `APP_<FIELD>`."""

    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    app_name: str = "BERT NAS Compression"

    models_dir: Path = _BACKEND_DIR / "models"
    data_dir: Path = _BACKEND_DIR / "data"

    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    model_tier: ModelTier = "core"
    # Off by default and deliberately so: the app publishes measured latency, and a lazily loaded
    # model would fold seconds of loading into the first measurement a visitor ever sees.
    lazy_models: bool = False

    # Pinned to 1 so latency is comparable between models and between runs. Reported alongside
    # every measurement - a latency figure without its thread count is not a figure.
    torch_threads: int = 1

    # Warm-up passes discarded before timing, then this many timed repeats. The offline benchmark
    # uses larger values; these are what an interactive request can afford.
    latency_warmup: int = 3
    latency_repeats: int = 5

    # Upper bound on an interactive evaluation job, so a stray request cannot pin the single
    # worker thread for an hour.
    max_eval_sample: int = 2000


@lru_cache
def get_settings() -> Settings:
    """Settings singleton (parsed once per process)."""
    return Settings()
