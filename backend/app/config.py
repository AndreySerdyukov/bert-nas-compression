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

    # The model every comparison is measured against. One baseline, on purpose: the point of
    # comparison is the uncompressed reference, not three teammates' fine-tunes of it.
    baseline_model: str = "bert-imdb"

    # Serving runs on the CPU even where MPS is available. Latency is the number this application
    # publishes, and a CPU figure is the one a reader can reproduce and compare; accuracy runs
    # offline in training/benchmark.py, where mps is worth the extra device to reason about.
    serve_device: str = "cpu"

    # Pinned to 1 so latency is comparable between models and between runs. Reported alongside
    # every measurement - a latency figure without its thread count is not a figure.
    torch_threads: int = 1

    # Warm-up passes discarded before timing, then this many timed repeats. The offline benchmark
    # uses larger values; these are what an interactive request can afford.
    latency_warmup: int = 3
    latency_repeats: int = 5
    # Batch size for the throughput measurement, which is a different quantity from latency and
    # is reported under a different name.
    latency_batch: int = 16

    # How long a request waits for the models before it is told the service is busy. The five
    # checkpoints share one process and a pinned thread budget, so overlapping requests would
    # measure each other rather than themselves.
    busy_timeout_s: float = 20.0

    # Upper bound on an interactive evaluation job, so a stray request cannot pin the single
    # worker thread for an hour.
    max_eval_sample: int = 2000


@lru_cache
def get_settings() -> Settings:
    """Settings singleton (parsed once per process)."""
    return Settings()
