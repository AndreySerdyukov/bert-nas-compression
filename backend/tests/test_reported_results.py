"""Consistency of the transcribed source tables.

These do not check that the original numbers are right - they are not, which is the point. They
check that the transcription is internally coherent, so chapter 9 is rendering a real disagreement
rather than a typo introduced while copying one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services import architecture as arch

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def reported() -> dict:
    return json.loads((DATA_DIR / "reported_results.json").read_text(encoding="utf-8"))


def test_exactly_one_source_is_canonical(reported: dict) -> None:
    canonical = [k for k, v in reported["sources"].items() if v.get("canonical")]
    assert canonical == ["nas_results"]


def test_every_model_has_the_canonical_figure(reported: dict) -> None:
    """A model missing from the canonical table has nothing to be measured against."""
    for model in reported["models"]:
        assert "nas_results" in model["accuracy"], model["key"]


def test_every_cited_source_exists(reported: dict) -> None:
    known = set(reported["sources"])
    for model in reported["models"]:
        assert set(model["accuracy"]) <= known, model["key"]
        assert set(model.get("ms_per_example", {})) <= known, model["key"]
        assert set(model.get("notes", {})) <= known, model["key"]


def test_parameter_counts_follow_the_formula(reported: dict) -> None:
    """Except AdaBERT, which is not a layer mask and carries its own arithmetic."""
    for model in reported["models"]:
        if model["n_layers"] is None:
            assert model["params"] == 30522 * 256 + 256 * 2 + 2
            continue
        assert model["params"] == arch.params_for(model["n_layers"]), model["key"]


def test_the_shipped_models_line_up_with_the_architectures_file(reported: dict) -> None:
    architectures = json.loads((DATA_DIR / "architectures.json").read_text(encoding="utf-8"))
    by_repo = {m["hf_repo"]: m for m in reported["models"]}
    for name, method in architectures["methods"].items():
        model = by_repo[method["hf_repo"]]
        assert model["params"] == method["shipped"]["params"], name


def test_the_sources_actually_disagree(reported: dict) -> None:
    """If they ever all agreed, chapter 9 would have nothing to show and should be deleted."""
    disagreeing = [
        model["key"]
        for model in reported["models"]
        if len({round(v, 4) for v in model["accuracy"].values()}) > 1
    ]
    # Six of the seven; BANANAS is the one whose notebook and benchmark match, and even it
    # disagrees with its tech report.
    assert len(disagreeing) >= 5


def test_every_referenced_file_is_in_the_repository(reported: dict) -> None:
    repo_root = DATA_DIR.parents[1]
    for source in reported["sources"].values():
        if "path" in source:
            assert (repo_root / source["path"]).exists(), source["path"]
