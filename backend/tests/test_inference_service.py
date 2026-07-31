"""The inference service, driven with stub predictors instead of checkpoints.

What is worth testing here is the behaviour around the models rather than the models: that
overlapping requests are refused instead of quietly measuring each other, that a comparison marks
disagreement against the baseline and not against whichever model came first, and that the two
kinds of "not available" stay distinguishable.

The stubs implement the same `ModelPredictor` contract the real wrappers do, which is what lets the
whole file run in milliseconds with neither torch nor a gigabyte of weights.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.config import Settings
from app.repositories.model_registry import LoadedModel, ModelRegistry
from app.schemas.models import ModelInfo, SkippedModel
from app.services.inference import (
    InferenceService,
    InvalidInputError,
    ModelNotFoundError,
    ModelUnavailableError,
    PredictionError,
    ServiceBusyError,
)
from app.serving.base import Prediction

REVIEW = "Painfully bad. The dialogue is embarrassing and nothing about it works."


class StubPredictor:
    """Answers instantly and always the same way, optionally after a delay."""

    def __init__(self, info: ModelInfo, verdict: str = "positive", delay: float = 0.0) -> None:
        self.info = info
        self._verdict = verdict
        self._delay = delay
        self.calls = 0

    def predict(self, text: str) -> Prediction:
        self.calls += 1
        if self._delay:
            time.sleep(self._delay)
        positive = 0.9 if self._verdict == "positive" else 0.1
        return Prediction(
            label=self._verdict,
            positive_probability=positive,
            probabilities={"negative": 1 - positive, "positive": positive},
            n_tokens=len(text.split()),
        )

    def predict_many(self, texts: Sequence[str]) -> list[Prediction]:
        return [self.predict(text) for text in texts]

    def warmup(self) -> None:
        pass


def _info(name: str, *, baseline: bool = False) -> ModelInfo:
    return ModelInfo(
        name=name,
        label=name.title(),
        max_length=512,
        hf_repo=f"someone/{name}",
        hf_revision="0" * 40,
        positive_index=1,
        is_baseline=baseline,
        params=1_000,
        n_layers=4,
    )


def _service(
    predictors: dict[str, StubPredictor],
    skipped: dict[str, SkippedModel] | None = None,
    **overrides: object,
) -> InferenceService:
    settings = Settings(
        latency_warmup=1,
        latency_repeats=2,
        latency_batch=4,
        **overrides,  # type: ignore[arg-type]
    )
    registry = ModelRegistry(
        models_dir=Path("models"), data_dir=Path("data"), baseline=settings.baseline_model
    )
    # Reaching into the registry's own storage rather than loading manifests: the point of this
    # file is the service, and going through `load()` would drag in torch and the checkpoints.
    for name, predictor in predictors.items():
        registry._models[name] = LoadedModel(predictor=predictor, info=predictor.info)
    for name, entry in (skipped or {}).items():
        registry._skipped[name] = entry
    return InferenceService(registry, settings)


def test_an_empty_review_never_reaches_a_model() -> None:
    predictor = StubPredictor(_info("bert-imdb", baseline=True))
    service = _service({"bert-imdb": predictor})
    with pytest.raises(InvalidInputError):
        service.predict("bert-imdb", "   ")
    assert predictor.calls == 0


def test_an_unknown_name_and_a_skipped_model_are_different_failures() -> None:
    skipped = {
        "adabert": SkippedModel(
            name="adabert", label="AdaBERT", reason="the weights are not downloaded", remedy="fetch"
        )
    }
    service = _service({}, skipped)

    with pytest.raises(ModelNotFoundError):
        service.predict("bert-huge", REVIEW)
    with pytest.raises(ModelUnavailableError) as caught:
        service.predict("adabert", REVIEW)
    assert caught.value.remedy == "fetch"


def test_a_failing_predictor_is_a_prediction_error_not_a_traceback() -> None:
    class Broken(StubPredictor):
        def predict(self, text: str) -> Prediction:
            raise RuntimeError("the tensor is on the wrong device")

    service = _service({"bananas": Broken(_info("bananas"))})
    with pytest.raises(PredictionError, match="bananas"):
        service.predict("bananas", REVIEW)


def test_measuring_latency_is_opt_in() -> None:
    predictor = StubPredictor(_info("bert-imdb", baseline=True))
    service = _service({"bert-imdb": predictor})

    plain = service.predict("bert-imdb", REVIEW)
    assert plain.latency is None
    assert predictor.calls == 1

    measured = service.predict("bert-imdb", REVIEW, measure_latency=True)
    assert measured.latency is not None
    assert measured.latency.repeats == 2
    # One scoring, then one warm-up and two timed repeats on top of it.
    assert predictor.calls == 5


def test_every_response_carries_the_runtime_behind_its_timings() -> None:
    service = _service({"bert-imdb": StubPredictor(_info("bert-imdb", baseline=True))})
    response = service.predict("bert-imdb", REVIEW)
    assert response.runtime.machine
    assert response.runtime.threads >= 0


def test_compare_marks_disagreement_against_the_baseline_only() -> None:
    service = _service(
        {
            "bert-imdb": StubPredictor(_info("bert-imdb", baseline=True), verdict="negative"),
            "alphanas": StubPredictor(_info("alphanas"), verdict="negative"),
            "bananas": StubPredictor(_info("bananas"), verdict="positive"),
        }
    )
    response = service.compare(REVIEW, measure_latency=False)
    results = {entry.name: entry for entry in response.results}

    assert response.baseline == "bert-imdb"
    assert results["bert-imdb"].agrees_with_baseline is None
    assert results["alphanas"].agrees_with_baseline is True
    assert results["bananas"].agrees_with_baseline is False
    assert response.disagree == ["bananas"]


def test_compare_without_the_baseline_loaded_claims_no_disagreement() -> None:
    """Two compressed models differing from each other says nothing about either being wrong."""
    service = _service(
        {
            "alphanas": StubPredictor(_info("alphanas"), verdict="negative"),
            "bananas": StubPredictor(_info("bananas"), verdict="positive"),
        }
    )
    response = service.compare(REVIEW, measure_latency=False)
    assert response.disagree == []
    assert all(entry.agrees_with_baseline is None for entry in response.results)


def test_compare_measures_throughput_under_its_own_name() -> None:
    service = _service({"alphanas": StubPredictor(_info("alphanas"))})
    response = service.compare(REVIEW, ["alphanas"], measure_latency=True, measure_throughput=True)
    entry = response.results[0]
    assert entry.latency is not None and entry.latency.batch_size == 1
    assert entry.throughput is not None and entry.throughput.batch_size == 4
    assert entry.throughput.per_example_ms <= entry.throughput.median_batch_ms


def test_an_overlapping_request_is_refused_rather_than_measured_alongside() -> None:
    """The models share one process and a pinned thread budget: two at once measure each other."""
    predictor = StubPredictor(_info("bert-imdb", baseline=True), delay=0.3)
    service = _service({"bert-imdb": predictor}, busy_timeout_s=0.05)

    started = threading.Event()

    def hold() -> None:
        started.set()
        service.predict("bert-imdb", REVIEW)

    holder = threading.Thread(target=hold)
    holder.start()
    started.wait(timeout=1)
    time.sleep(0.05)
    try:
        with pytest.raises(ServiceBusyError):
            service.predict("bert-imdb", REVIEW)
    finally:
        holder.join()

    # And the lock is released afterwards, so the next request is served normally.
    assert service.predict("bert-imdb", REVIEW).verdict == "positive"
