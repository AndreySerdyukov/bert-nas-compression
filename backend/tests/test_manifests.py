"""The model manifests, and the committed data the app scores against.

These run without weights: the manifests and the evaluation sample are tracked, the snapshots are
not. What cannot be checked here is that a manifest still describes the checkpoint on the Hub - the
lock file records digests for that, and `scripts/fetch_models.py` is what re-verifies it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services import architecture as arch

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

EXPECTED_MODELS = {"bert-imdb", "random-search", "alphanas", "bananas", "adabert"}


def manifests() -> dict[str, dict]:
    return {
        path.name.removesuffix(".meta.json"): json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(MODELS_DIR.glob("*.meta.json"))
    }


def test_every_model_has_a_manifest() -> None:
    assert set(manifests()) == EXPECTED_MODELS


@pytest.mark.parametrize("name", sorted(EXPECTED_MODELS))
def test_the_revision_is_pinned(name: str) -> None:
    """`main` moves. A measurement has to name the commit it was taken on."""
    manifest = manifests()[name]
    assert len(manifest["hf_revision"]) == 40, name
    assert manifest["hf_repo"].count("/") == 1


@pytest.mark.parametrize("name", sorted(EXPECTED_MODELS))
def test_the_label_order_was_measured_not_assumed(name: str) -> None:
    """The single most dangerous thing this project could get wrong.

    No checkpoint carries `id2label`, so the mapping is established by running unambiguous reviews
    through the model at download time. A model that only mostly agrees with the probe is a model
    whose polarity is not actually known.
    """
    order = manifests()[name]["label_order"]
    assert order["source"].startswith("measured")
    assert order["positive_index"] in (0, 1)
    assert order["probe_agreement"] == 1.0, name
    assert order["id2label"][str(order["positive_index"])] == "positive"


@pytest.mark.parametrize("name", sorted(EXPECTED_MODELS - {"adabert"}))
def test_the_parameter_count_matches_the_formula(name: str) -> None:
    """Measured off the loaded model at download time, checked against the closed form here."""
    info = manifests()[name]["model_info"]
    assert info["params"] == arch.params_for(info["n_layers"]), name


def test_the_manifests_agree_with_the_extracted_architectures() -> None:
    """Two independent sources: the Hub config, and the eval notebooks' layers_flag."""
    architectures = json.loads((DATA_DIR / "architectures.json").read_text(encoding="utf-8"))
    by_name = manifests()

    for method, entry in architectures["methods"].items():
        if entry["shipped"]["n_layers"] is None:
            continue
        assert by_name[method]["model_info"]["n_layers"] == entry["shipped"]["n_layers"], method
        assert by_name[method]["model_info"]["params"] == entry["shipped"]["params"], method
        assert by_name[method]["hf_repo"] == entry["hf_repo"], method


def test_adabert_is_a_state_dict_not_a_snapshot() -> None:
    """The checkpoint carries no config, so the manifest has to carry the architecture instead."""
    manifest = manifests()["adabert"]
    arch = manifest["arch"]
    assert manifest["kind"] == "adabert"
    assert arch["weights"] == "frozen_adabert.pt"
    assert arch["vocab_size"] * arch["hidden_size"] + 256 * 2 + 2 == 7_814_146
    assert manifest["model_info"]["params"] == 7_814_146
    # Transcribed from the notebook that saved the file, and named there, because `strict=True`
    # cannot choose between the two divergent class definitions: the selected operation carries no
    # parameters, so both produce identical state-dict keys.
    assert arch["selected_ops"] == ["avg_pool"] * 5
    assert arch["selected_layers"] == [1, 2, 5, 6, 7]
    assert "notebooks/dnas" in arch["source"]


def test_adabert_is_served_at_the_length_it_was_trained_at() -> None:
    """It has no attention mask and means over its padding, so 128 is part of the model."""
    assert manifests()["adabert"]["arch"]["max_length"] == 128
    for name in EXPECTED_MODELS - {"adabert"}:
        assert manifests()[name]["arch"]["max_length"] == 512


@pytest.mark.parametrize("name", sorted(EXPECTED_MODELS))
def test_a_lock_file_records_what_was_downloaded(name: str) -> None:
    """So a checkpoint being replaced upstream is detectable rather than merely unlikely."""
    lock = json.loads((MODELS_DIR / f"{name}.lock.json").read_text(encoding="utf-8"))
    assert lock["files"], name
    for entry in lock["files"].values():
        assert len(entry["sha256"]) == 64
        assert entry["bytes"] > 0


# --- the committed evaluation sample --------------------------------------------------------


def test_the_evaluation_sample_is_balanced_and_traceable() -> None:
    text = (DATA_DIR / "imdb_eval_sample.jsonl").read_text(encoding="utf-8")
    # Not splitlines(): reviews contain characters such as U+2028 that Python treats as line breaks
    # and JSON does not escape.
    lines = text.strip("\n").split("\n")
    provenance = json.loads(lines[0])
    rows = [json.loads(line) for line in lines[1:]]

    assert provenance["n"] == len(rows) == 2000
    assert len(provenance["test_index_sha256"]) == 64
    assert sum(row["label"] for row in rows) == 1000
    assert len({row["id"] for row in rows}) == 2000
    assert all(row["text"].strip() for row in rows)


def test_the_label_probe_is_unambiguous_and_balanced() -> None:
    probe = json.loads((DATA_DIR / "label_probe.json").read_text(encoding="utf-8"))
    labels = [review["label"] for review in probe["reviews"]]
    assert labels.count("positive") == labels.count("negative")
    assert probe["min_agreement"] >= 0.95
