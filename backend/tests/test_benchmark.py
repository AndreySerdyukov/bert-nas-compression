"""The measurement harness, and the file it produces.

Two different jobs here. The first half checks the arithmetic with stubs, so it runs in
milliseconds and needs no checkpoint. The second half validates the committed
`data/benchmark.json` itself - it is the file the README is generated from and the one number
anybody will quote, so it is worth asserting that it is internally consistent rather than trusting
that it was produced correctly on the day.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.services import architecture as arch
from app.serving.base import Prediction
from scripts.render_readme_table import MISSING, render
from training.benchmark import score_model

BACKEND_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_FILE = BACKEND_DIR / "data" / "benchmark.json"


class Counter:
    def update(self, n: int = 1) -> None:
        pass


class StubPredictor:
    def __init__(self, verdicts: list[str]) -> None:
        self._verdicts = verdicts

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        taken, self._verdicts = self._verdicts[: len(texts)], self._verdicts[len(texts) :]
        return [
            Prediction(
                label=verdict,
                positive_probability=1.0 if verdict == "positive" else 0.0,
                probabilities={"negative": 0.0, "positive": 1.0},
                n_tokens=8,
            )
            for verdict in taken
        ]


# --- the arithmetic ------------------------------------------------------------------------------


def test_the_confusion_matrix_names_its_corners() -> None:
    """Which corner is which is exactly what a reader should not have to work out."""
    labels = [1, 1, 0, 0]
    predicted = ["positive", "negative", "negative", "positive"]
    result = score_model(StubPredictor(predicted), ["a", "b", "c", "d"], labels, Counter())

    assert result["accuracy"] == 0.5
    assert result["correct"] == 2
    assert result["confusion"] == {
        "true_positive": 1,
        "false_negative": 1,
        "true_negative": 1,
        "false_positive": 1,
    }


def test_a_model_that_says_one_thing_scores_half_and_says_so() -> None:
    """The failure mode a randomly initialised head produces, and what it should look like here."""
    labels = [1, 0] * 50
    result = score_model(StubPredictor(["positive"] * 100), ["x"] * 100, labels, Counter())

    assert result["accuracy"] == 0.5
    assert result["positive_recall"] == 1.0
    # Macro F1 is what catches it: accuracy alone would look like a coin flip rather than a stuck
    # classifier.
    assert result["macro_f1"] < 0.4
    assert result["confusion"]["true_negative"] == 0


# --- the README table ----------------------------------------------------------------------------


def _benchmark_fixture() -> dict[str, Any]:
    return {
        "protocol": {
            "test_rows": 15_000,
            "test_index_sha256": "33d7fee1070469ac628d568dfa95fc960d8d76e8a548acdb553fbb15ce1796c9",
            "accuracy_device": "mps",
            "latency_device": "cpu",
            "latency_threads": 1,
            "latency_warmup": 5,
            "latency_repeats": 25,
            "throughput_batch": 16,
            "machine": "Darwin arm64",
            "torch": "2.13.0",
            "transformers": "5.14.1",
        },
        "models": [
            {
                "name": "bert-imdb",
                "label": "BERT-base",
                "method": None,
                "params": 109_483_778,
                "accuracy": 0.9331,
                "macro_f1": 0.9330,
                "latency": {"median_ms": 24.7, "p95_ms": 26.0, "n_tokens": 32},
                "throughput": {"per_example_ms": 6.2, "examples_per_second": 161.0},
            }
        ],
    }


def test_the_table_keeps_latency_and_throughput_in_separate_columns() -> None:
    """The one column this generator must never be able to produce is a single "ms/example"."""
    rendered = render(_benchmark_fixture())
    header = rendered.splitlines()[0]
    assert "Latency, median" in header
    assert "Throughput" in header
    assert "ms/example" not in rendered


def test_the_table_carries_the_protocol_that_makes_it_comparable() -> None:
    rendered = render(_benchmark_fixture())
    assert "15,000 row test split" in rendered
    assert "33d7fee10704" in rendered
    assert "`cpu`, 1 thread" in rendered
    assert "5 warm-ups discarded" in rendered
    assert "round-robin" in rendered
    assert "Darwin arm64" in rendered


def test_a_latency_column_over_unequal_token_counts_says_so() -> None:
    """AdaBERT's 0.9 ms is over 128 padded tokens; everything else's is over 33 of real review.

    Printed side by side with no note, that column reads as "AdaBERT is 23x faster at the same
    job", which is not what was measured. The sentence only appears when the counts actually
    differ, so it cannot become boilerplate nobody reads.
    """
    fixture = _benchmark_fixture()
    fixture["models"].append(
        {
            **fixture["models"][0],
            "name": "adabert",
            "label": "AdaBERT",
            "method": "adabert",
            "latency": {"median_ms": 0.9, "p95_ms": 0.9, "n_tokens": 128},
        }
    )
    rendered = render(fixture)
    assert "AdaBERT 128" in rendered and "BERT-base 32" in rendered

    # One model, or several that agree, and the note has nothing to warn about.
    assert "not on the same amount of work" not in render(_benchmark_fixture())


def test_the_memory_column_names_the_floor_it_excludes() -> None:
    """A bare "88 MB" next to a 500 MB process is a number a reader cannot act on."""
    fixture = _benchmark_fixture()
    fixture["models"][0]["resident_bytes"] = 797_900_000
    fixture["models"][0]["runtime_baseline_bytes"] = 411_600_000
    rendered = render(fixture)

    assert "| 386 MB |" in rendered
    assert "412 MB floor torch and transformers occupy" in rendered
    # Without a reading the column is empty rather than zero, and the paragraph stays away.
    assert "| n/a |" in render(_benchmark_fixture())
    assert "resident set" not in render(_benchmark_fixture())


def test_a_clone_with_no_measurements_says_so_rather_than_showing_a_stale_table() -> None:
    assert "No measurements yet" in MISSING
    assert "training.benchmark" in MISSING


# --- the committed measurement file ---------------------------------------------------------------


@pytest.mark.skipif(not BENCHMARK_FILE.exists(), reason="run python -m training.benchmark")
def test_the_committed_benchmark_is_internally_consistent() -> None:
    """Every number in the README comes from this file, so the file itself is worth checking."""
    payload = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
    protocol = payload["protocol"]
    rows = protocol["test_rows"]

    assert protocol["latency_device"] == "cpu"
    assert protocol["latency_threads"] == 1, (
        "a latency measured on more than one thread is not ours"
    )
    assert len(protocol["test_index_sha256"]) == 64

    for entry in payload["models"]:
        name = entry["name"]
        assert entry["correct"] / rows == pytest.approx(entry["accuracy"], abs=5e-7), name
        assert sum(entry["confusion"].values()) == rows, name
        # Measured off the loaded module and computed from the closed form; the run asserts they
        # agree, and this asserts the run really did.
        if entry["params_formula"] is not None:
            assert entry["params_agree"] is True, name
            assert entry["params_formula"] == arch.params_for(entry["n_layers"]), name
        assert entry["bytes_fp32"] == entry["params"] * 4, name

        latency = entry.get("latency")
        if latency is not None:
            assert latency["batch_size"] == 1, name
            assert latency["p95_ms"] >= latency["median_ms"], name
        throughput = entry.get("throughput")
        if throughput is not None:
            # The whole point of keeping them apart: batching wins, so per-example time under a
            # batch is below the single-example latency. Equal would mean one was copied.
            assert throughput["per_example_ms"] < latency["median_ms"], name


@pytest.mark.skipif(not BENCHMARK_FILE.exists(), reason="run python -m training.benchmark")
def test_every_model_was_weighed_against_the_same_floor() -> None:
    """The check that caught the first memory pass, which measured the allocator instead.

    Weighing all five in one interpreter gave the 438 MB baseline a 1.8 MB cost and the 32 MB
    AdaBERT 74 MB, because a freed model's pages go back to Python's pool rather than the OS and
    the next load is served out of them. Two byte-identical architectures came out 2.6x apart,
    which is the tell: they cannot differ, so any spread between them is the instrument. Each
    model is now weighed in its own process, and what makes those numbers comparable is that the
    floor is the same in all of them.
    """
    payload = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
    weighed = [entry for entry in payload["models"] if entry.get("resident_bytes")]
    if not weighed:
        pytest.skip("no memory readings in this benchmark")

    floors = [entry["runtime_baseline_bytes"] for entry in weighed]
    assert min(floors) > 0
    assert max(floors) - min(floors) < 0.05 * min(floors), (
        "the runtime floor moved between processes, so the readings are not comparable"
    )

    for entry in weighed:
        name = entry["name"]
        # A total below the floor would mean the model was never actually loaded.
        assert entry["resident_bytes"] > entry["runtime_baseline_bytes"], name
        # Weights are mapped, not copied, so a model may be less resident than its fp32 size -
        # but never more than the whole process, which is what a swapped field would produce.
        assert entry["resident_bytes"] - entry["runtime_baseline_bytes"] < entry["resident_bytes"]

    # The two four-layer BERTs are the same architecture down to the byte count on disk. They are
    # the control for this measurement: if they disagree by much, it is the ruler, not the models.
    twins = {
        entry["name"]: entry["resident_bytes"] - entry["runtime_baseline_bytes"]
        for entry in weighed
        if entry["name"] in {"alphanas", "bananas"}
    }
    if len(twins) == 2:
        smaller, larger = sorted(twins.values())
        assert larger < 1.15 * smaller, f"identical architectures weighed differently: {twins}"


@pytest.mark.skipif(not BENCHMARK_FILE.exists(), reason="run python -m training.benchmark")
def test_the_benchmark_covers_every_model_that_has_a_manifest() -> None:
    payload = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
    measured = {entry["name"] for entry in payload["models"]}
    manifests = {
        path.name.removesuffix(".meta.json")
        for path in (BACKEND_DIR / "models").glob("*.meta.json")
    }
    assert measured == manifests
