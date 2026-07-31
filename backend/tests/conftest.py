"""Fixtures shared by the API tests.

Two decisions shape this file.

**Serving is off unless a test asks for it.** Building the application now loads up to a gigabyte of
weights, and most of this suite is about documents, schemas and status codes. The autouse fixture
sets `APP_MODEL_TIER=none`, and the tests that want models turn it back on.

**The no-weights state is constructed, not assumed.** Whether the checkpoints are on the machine
running the tests is exactly what the suite cannot know - they are there on a laptop that has run
the fetch script and absent in CI - so `no_weights_dir` builds a models directory holding the
manifests and no weights. That is the same everywhere, and it is the state the 503 answers describe.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.config import get_settings

BACKEND_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BACKEND_DIR / "models"

# A checkpoint is only present once scripts/fetch_models.py has run. Weight-backed tests skip
# rather than fail, and `pytest -rs` in CI makes the skip count visible.
WEIGHTS_PRESENT = (MODELS_DIR / "bert-imdb" / "config.json").exists()
needs_weights = pytest.mark.skipif(
    not WEIGHTS_PRESENT, reason="run scripts/fetch_models.py to download the checkpoints"
)


@pytest.fixture(autouse=True)
def _serving_off_by_default(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No registry, and settings re-read per test since they are cached per process."""
    monkeypatch.setenv("APP_MODEL_TIER", "none")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def serving(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn the registry back on, against the real models directory."""
    monkeypatch.setenv("APP_MODEL_TIER", "core")


@pytest.fixture
def no_weights_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every manifest, none of the weights: a clean clone, reproduced on any machine."""
    models = tmp_path / "models"
    models.mkdir()
    for manifest in sorted(MODELS_DIR.glob("*.meta.json")):
        shutil.copy(manifest, models / manifest.name)
    monkeypatch.setenv("APP_MODEL_TIER", "core")
    monkeypatch.setenv("APP_MODELS_DIR", str(models))
    return models


@pytest.fixture
def broken_manifest_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """One manifest with an unusable `kind`, to prove a bad file does not take the app down."""
    models = tmp_path / "models"
    models.mkdir()
    manifest = json.loads((MODELS_DIR / "bert-imdb.meta.json").read_text(encoding="utf-8"))
    manifest["kind"] = "typo"
    (models / "bert-imdb.meta.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("APP_MODEL_TIER", "core")
    monkeypatch.setenv("APP_MODELS_DIR", str(models))
    return models
