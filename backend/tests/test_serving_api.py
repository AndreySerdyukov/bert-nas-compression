"""The scoring endpoints, mostly exercised without any weights.

The no-weights path is the one worth testing hardest, because it is the state of a fresh clone and
the state CI's docker job runs in. What it has to prove is that "not downloaded yet" is answered
differently from "no such model", that the answer says what to run, and that the application comes
up either way.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

from .conftest import needs_weights

REVIEW = "A complete waste of time. Terrible acting and a boring, predictable plot."


# --- without a registry at all -----------------------------------------------------------------


def test_serving_switched_off_is_a_503_that_says_so() -> None:
    """`model_tier=none` is a supported configuration, not a broken one."""
    with TestClient(create_app()) as client:
        response = client.get("/api/models")
    assert response.status_code == 503
    assert "APP_MODEL_TIER" in response.json()["detail"]


# --- manifests present, weights absent ---------------------------------------------------------


def test_the_app_starts_with_no_weights_and_says_what_is_missing(no_weights_dir: object) -> None:
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["models_loaded"] == 0
        body = client.get("/api/models").json()

    assert body["models"] == []
    assert {entry["name"] for entry in body["skipped"]} == {
        "bert-imdb",
        "random-search",
        "alphanas",
        "bananas",
        "adabert",
    }
    for entry in body["skipped"]:
        assert "not downloaded" in entry["reason"]
        # The reason without the remedy would leave a reader guessing at a script name.
        assert entry["remedy"] == f"python scripts/fetch_models.py --only {entry['name']}"


def test_the_runtime_travels_with_the_catalog(no_weights_dir: object) -> None:
    """Every timing this app publishes is comparable only because these fields go with it."""
    with TestClient(create_app()) as client:
        runtime = client.get("/api/models").json()["runtime"]
    assert runtime["threads"] >= 1
    assert runtime["device"] == "cpu"
    assert runtime["machine"]
    assert runtime["torch_version"]


def test_a_known_model_without_weights_is_503_naming_the_fetch_script(
    no_weights_dir: object,
) -> None:
    with TestClient(create_app()) as client:
        response = client.post("/api/models/bert-imdb/predict", json={"text": REVIEW})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "scripts/fetch_models.py" in detail
    assert "bert-imdb" in detail


def test_an_unknown_model_is_404_not_503(no_weights_dir: object) -> None:
    """The two are different failures: one is a typo, the other is a missing download."""
    with TestClient(create_app()) as client:
        response = client.post("/api/models/bert-large/predict", json={"text": REVIEW})
    assert response.status_code == 404


def test_an_empty_review_is_422_before_any_model_is_consulted(no_weights_dir: object) -> None:
    with TestClient(create_app()) as client:
        response = client.post("/api/models/bert-imdb/predict", json={"text": "   "})
    assert response.status_code == 422


def test_compare_with_nothing_loaded_is_503(no_weights_dir: object) -> None:
    with TestClient(create_app()) as client:
        response = client.post("/api/compare", json={"text": REVIEW})
    assert response.status_code == 503


def test_a_manifest_with_an_unusable_kind_does_not_take_the_app_down(
    broken_manifest_dir: object,
) -> None:
    """The raise for an unknown kind sits inside the try, so it skips a model rather than startup."""
    with TestClient(create_app()) as client:
        body = client.get("/api/models").json()
    assert body["models"] == []
    assert "typo" in body["skipped"][0]["reason"]


def test_a_manifest_whose_label_order_contradicts_itself_is_refused(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Half the manifest saying class 1 and the other half saying class 0 makes neither usable."""
    from .conftest import MODELS_DIR

    models = tmp_path / "models"  # type: ignore[operator]
    models.mkdir()
    manifest = json.loads((MODELS_DIR / "bert-imdb.meta.json").read_text(encoding="utf-8"))
    manifest["label_order"]["id2label"] = {"0": "positive", "1": "negative"}
    (models / "bert-imdb.meta.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("APP_MODEL_TIER", "core")
    monkeypatch.setenv("APP_MODELS_DIR", str(models))

    with TestClient(create_app()) as client:
        body = client.get("/api/models").json()
    assert body["models"] == []
    assert "label_order" in body["skipped"][0]["reason"]


# --- with the checkpoints on disk ---------------------------------------------------------------


@needs_weights
@pytest.mark.needs_weights
def test_every_checkpoint_loads_and_agrees_with_its_pinned_polarity(serving: object) -> None:
    """Warm-up is the polarity probe, so a model that survives startup has passed it."""
    with TestClient(create_app()) as client:
        body = client.get("/api/models").json()
    assert body["skipped"] == []
    assert {entry["name"] for entry in body["models"]} == {
        "bert-imdb",
        "random-search",
        "alphanas",
        "bananas",
        "adabert",
    }
    # The baseline sorts first: it is what every comparison is measured against.
    assert body["models"][0]["name"] == body["baseline"] == "bert-imdb"


@needs_weights
@pytest.mark.needs_weights
def test_a_prediction_carries_the_machine_that_produced_its_timing(serving: object) -> None:
    with TestClient(create_app()) as client:
        body = client.post(
            "/api/models/bert-imdb/predict", json={"text": REVIEW, "measure_latency": True}
        ).json()

    assert body["verdict"] == "negative"
    assert body["probabilities"]["negative"] > body["probabilities"]["positive"]
    assert body["n_tokens"] > 0
    assert body["params"] == 109_483_778

    assert body["latency"]["batch_size"] == 1
    assert body["latency"]["repeats"] >= 1
    assert body["latency"]["p95_ms"] >= body["latency"]["median_ms"]
    # The token count is part of the figure: these models truncate at different lengths.
    assert body["latency"]["n_tokens"] == body["n_tokens"]
    assert body["runtime"]["threads"] >= 1
    # Throughput is a different quantity and is not produced unless it was asked for.
    assert body["throughput"] is None


@needs_weights
@pytest.mark.needs_weights
def test_latency_is_not_measured_unless_it_was_asked_for(serving: object) -> None:
    """The interactive path should not pay for eight extra forward passes by default."""
    with TestClient(create_app()) as client:
        body = client.post("/api/models/adabert/predict", json={"text": REVIEW}).json()
    assert body["latency"] is None
    # One sample, warm-up included, and named so.
    assert body["elapsed_ms"] > 0


@needs_weights
@pytest.mark.needs_weights
def test_adabert_is_served_at_the_length_it_was_trained_at(serving: object) -> None:
    """128, not 512. The network means over its padding, so the length is part of the model."""
    with TestClient(create_app()) as client:
        catalog = client.get("/api/models").json()
        body = client.post("/api/models/adabert/predict", json={"text": REVIEW}).json()

    adabert = next(m for m in catalog["models"] if m["name"] == "adabert")
    assert adabert["max_length"] == 128
    assert adabert["n_layers"] is None
    assert body["n_tokens"] == 128


@needs_weights
@pytest.mark.needs_weights
def test_compare_scores_every_model_and_marks_the_disagreements(serving: object) -> None:
    with TestClient(create_app()) as client:
        body = client.post(
            "/api/compare",
            json={"text": REVIEW, "models": ["bert-imdb", "adabert"], "measure_latency": False},
        ).json()

    assert body["baseline"] == "bert-imdb"
    results = {entry["name"]: entry for entry in body["results"]}
    assert set(results) == {"bert-imdb", "adabert"}
    # The baseline is not compared against itself.
    assert results["bert-imdb"]["agrees_with_baseline"] is None
    assert results["adabert"]["agrees_with_baseline"] is not None
    assert set(body["disagree"]) <= {"adabert"}


@needs_weights
@pytest.mark.needs_weights
def test_throughput_is_reported_separately_from_latency(serving: object) -> None:
    """Two quantities, two names. Dividing a batch time by 16 is not a latency.

    Measured on the baseline rather than on AdaBERT, and that is the fix for a flake rather than a
    preference. AdaBERT answers in about 0.7 ms, where the per-example batch time and the
    single-example median sit within a few percent of each other and swap places run to run - the
    assertion below then failed about one time in three for reasons that had nothing to do with
    the code under test. On the twelve-layer baseline the batch is several times cheaper per
    example, which is the effect this test means to pin.
    """
    with TestClient(create_app()) as client:
        body = client.post(
            "/api/compare",
            json={
                "text": REVIEW,
                "models": ["bert-imdb"],
                "measure_latency": True,
                "measure_throughput": True,
            },
        ).json()

    entry = body["results"][0]
    assert entry["latency"]["batch_size"] == 1
    assert entry["throughput"]["batch_size"] == 16
    assert entry["throughput"]["per_example_ms"] < entry["latency"]["median_ms"]
    assert entry["throughput"]["examples_per_second"] > 0
