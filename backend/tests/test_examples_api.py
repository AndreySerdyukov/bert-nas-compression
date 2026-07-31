"""The reviews the playground offers, and the shape of what serving returns.

The second half of this file is unusual and deliberate. The browser tests run against a fully
stubbed network, so nothing over there notices if a field is renamed on this side - the stubs would
keep serving the old shape and keep passing. Pinning the key sets here is the cheap half of that
problem: a rename fails a backend test, which is a reminder that `frontend/src/api.ts` has to move
with it.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app

from .conftest import needs_weights

REVIEW = "A complete waste of time. Terrible acting and a boring, predictable plot."


def test_examples_need_no_models() -> None:
    """A visitor can read the reviews the project is scored on before downloading a gigabyte."""
    with TestClient(create_app()) as client:
        response = client.get("/api/examples")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2000
    assert body["source"] == "data/imdb_eval_sample.jsonl"
    assert body["examples"]


def test_the_offered_reviews_are_balanced_and_traceable() -> None:
    """A rule, not a hand-picked list: equal polarities, real row ids, and one long pair."""
    with TestClient(create_app()) as client:
        examples = client.get("/api/examples").json()["examples"]

    labels = [entry["label"] for entry in examples]
    assert labels.count(1) == labels.count(0)
    assert len({entry["id"] for entry in examples}) == len(examples)
    assert all(entry["n_chars"] == len(entry["text"]) for entry in examples)
    # At least one review per polarity runs past the 512-token window, which is how a reader sees
    # for themselves that the models are being handed a truncated review.
    truncated = [entry for entry in examples if entry["truncated"]]
    assert {entry["label"] for entry in truncated} == {0, 1}


def test_the_examples_come_from_the_evaluation_sample() -> None:
    """Not written for the occasion: every offered review is a row every model is scored on."""
    from .conftest import BACKEND_DIR

    text = (BACKEND_DIR / "data" / "imdb_eval_sample.jsonl").read_text(encoding="utf-8")
    # Not splitlines(): the reviews contain U+2028.
    import json

    rows = {
        json.loads(line)["id"]: json.loads(line)["text"]
        for line in text.strip("\n").split("\n")[1:]
    }

    with TestClient(create_app()) as client:
        examples = client.get("/api/examples").json()["examples"]
    for entry in examples:
        assert rows[entry["id"]] == entry["text"]


# --- the shape the frontend is written against ---------------------------------------------------

MODEL_KEYS = {
    "name",
    "label",
    "description",
    "method",
    "params",
    "n_layers",
    "max_length",
    "hf_repo",
    "hf_revision",
    "positive_index",
    "is_baseline",
}
PREDICTION_KEYS = {
    "name",
    "label",
    "verdict",
    "positive_probability",
    "probabilities",
    "n_tokens",
    "params",
    "n_layers",
    "elapsed_ms",
    "latency",
    "throughput",
    "agrees_with_baseline",
}
RUNTIME_KEYS = {"device", "threads", "interop_threads", "machine", "torch_version"}


def test_the_catalog_shape_is_pinned(no_weights_dir: object) -> None:
    with TestClient(create_app()) as client:
        body = client.get("/api/models").json()
    assert set(body) == {"models", "skipped", "baseline", "runtime"}
    assert set(body["runtime"]) == RUNTIME_KEYS
    assert set(body["skipped"][0]) == {"name", "label", "reason", "remedy"}


@needs_weights
def test_the_comparison_shape_is_pinned(serving: object) -> None:
    with TestClient(create_app()) as client:
        catalog = client.get("/api/models").json()
        body = client.post(
            "/api/compare",
            json={"text": REVIEW, "models": ["adabert"], "measure_latency": True},
        ).json()

    assert set(catalog["models"][0]) == MODEL_KEYS
    assert set(body) == {"baseline", "results", "disagree", "runtime"}
    assert set(body["results"][0]) == PREDICTION_KEYS
    assert set(body["results"][0]["latency"]) == {
        "batch_size",
        "warmup",
        "repeats",
        "median_ms",
        "p95_ms",
        "n_tokens",
    }
