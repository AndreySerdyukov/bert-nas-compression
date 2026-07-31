"""Scoring reviews: the domain layer. No FastAPI here, and `tests/test_api.py` parses this
directory to prove it.

Everything that touches a model goes through one lock. The five checkpoints share a process and a
pinned thread budget, so two concurrent requests would not merely be slower - they would each
measure the other's work and report it as their own latency. A caller that cannot get the lock in
time is told the service is busy rather than handed a number that means nothing.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

from app.config import Settings
from app.repositories.model_registry import LoadedModel, ModelRegistry
from app.schemas.models import ModelsResponse
from app.schemas.predict import (
    CompareResponse,
    ModelPrediction,
    PredictResponse,
)
from app.services.latency import latency_stats, throughput_stats, time_call, time_round_robin
from app.serving.base import Prediction
from app.serving.runtime import describe_runtime


class ModelNotFoundError(Exception):
    """No manifest carries this name."""


class ModelUnavailableError(Exception):
    """The manifest is there, the model is not. Usually because the weights are not downloaded."""

    def __init__(self, name: str, reason: str, remedy: str | None) -> None:
        super().__init__(reason)
        self.name = name
        self.reason = reason
        self.remedy = remedy


class InvalidInputError(Exception):
    """There is nothing here to score."""


class PredictionError(Exception):
    """The model could not score this input."""


class ServiceBusyError(Exception):
    """Another scoring request holds the models, and timings taken alongside it would be junk."""


class InferenceService:
    """Everything the API can ask a model to do."""

    def __init__(self, registry: ModelRegistry, settings: Settings) -> None:
        self._registry = registry
        self._settings = settings
        self._lock = threading.Lock()

    # --- catalog -------------------------------------------------------------------------------

    def catalog(self) -> ModelsResponse:
        """What is loaded, what was skipped and why, and the machine behind every timing."""
        return ModelsResponse(
            models=self._registry.list_infos(),
            skipped=sorted(self._registry.skipped.values(), key=lambda entry: entry.name),
            baseline=self._settings.baseline_model,
            runtime=describe_runtime(self._settings.serve_device),
        )

    # --- scoring -------------------------------------------------------------------------------

    def predict(self, name: str, text: str, *, measure_latency: bool = False) -> PredictResponse:
        """Score one review with one model."""
        review = self._validated(text)
        loaded = self._loaded(name)

        with self._exclusive():
            prediction, elapsed_ms = self._score(loaded, review)
            entry = self._entry(loaded, prediction, elapsed_ms)
            if measure_latency:
                timing = time_call(
                    lambda: loaded.predictor.predict(review),
                    warmup=self._settings.latency_warmup,
                    repeats=self._settings.latency_repeats,
                )
                entry.latency = latency_stats(
                    timing, warmup=self._settings.latency_warmup, n_tokens=prediction.n_tokens
                )

        return PredictResponse(
            **entry.model_dump(), runtime=describe_runtime(self._settings.serve_device)
        )

    def compare(
        self,
        text: str,
        names: Sequence[str] | None = None,
        *,
        measure_latency: bool = True,
        measure_throughput: bool = False,
    ) -> CompareResponse:
        """Score one review with several models, measured round-robin, against the baseline."""
        review = self._validated(text)
        selected = list(names) if names else [info.name for info in self._registry.list_infos()]
        if not selected:
            raise ModelUnavailableError(
                "any", "no models are loaded", f"{self._settings.models_dir} has no weights yet"
            )
        loaded = {name: self._loaded(name) for name in dict.fromkeys(selected)}

        with self._exclusive():
            entries: dict[str, ModelPrediction] = {}
            predictions: dict[str, Prediction] = {}
            for name, model in loaded.items():
                prediction, elapsed_ms = self._score(model, review)
                predictions[name] = prediction
                entries[name] = self._entry(model, prediction, elapsed_ms)

            if measure_latency:
                calls: dict[str, Callable[[], object]] = {
                    name: (lambda m=model: m.predictor.predict(review))  # type: ignore[misc]
                    for name, model in loaded.items()
                }
                for name, timing in time_round_robin(
                    calls,
                    warmup=self._settings.latency_warmup,
                    repeats=self._settings.latency_repeats,
                ).items():
                    entries[name].latency = latency_stats(
                        timing,
                        warmup=self._settings.latency_warmup,
                        n_tokens=predictions[name].n_tokens,
                    )

            if measure_throughput:
                size = self._settings.latency_batch
                # The same review repeated: per-example work is then constant across the batch, so
                # what the number isolates is the effect of batching and nothing else.
                batch = [review] * size
                calls = {
                    name: (lambda m=model: m.predictor.predict_many(batch))  # type: ignore[misc]
                    for name, model in loaded.items()
                }
                for name, timing in time_round_robin(
                    calls,
                    warmup=self._settings.latency_warmup,
                    repeats=self._settings.latency_repeats,
                ).items():
                    entries[name].throughput = throughput_stats(
                        timing,
                        warmup=self._settings.latency_warmup,
                        batch_size=size,
                        n_tokens=predictions[name].n_tokens,
                    )

        baseline = self._settings.baseline_model
        disagree: list[str] = []
        if baseline in entries:
            verdict = entries[baseline].verdict
            for name, entry in entries.items():
                if name == baseline:
                    continue
                entry.agrees_with_baseline = entry.verdict == verdict
                if not entry.agrees_with_baseline:
                    disagree.append(name)

        return CompareResponse(
            baseline=baseline,
            results=[entries[name] for name in loaded],
            disagree=disagree,
            runtime=describe_runtime(self._settings.serve_device),
        )

    # --- internals -----------------------------------------------------------------------------

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        """Hold the models for the duration of a request, or refuse the request."""
        if not self._lock.acquire(timeout=self._settings.busy_timeout_s):
            raise ServiceBusyError(
                "another request is being scored. The models share one process and a pinned "
                "thread budget, so overlapping requests would measure each other."
            )
        try:
            yield
        finally:
            self._lock.release()

    def _validated(self, text: str) -> str:
        if not text or not text.strip():
            raise InvalidInputError("there is no review to score")
        return text

    def _loaded(self, name: str) -> LoadedModel:
        loaded = self._registry.get(name)
        if loaded is not None:
            return loaded
        if self._registry.is_known(name):
            skipped = self._registry.skipped[name]
            raise ModelUnavailableError(name, skipped.reason, skipped.remedy)
        raise ModelNotFoundError(name)

    @staticmethod
    def _score(loaded: LoadedModel, review: str) -> tuple[Prediction, float]:
        """One scoring, timed as one sample. Not a measurement, and never labelled as one."""
        started = time.perf_counter()
        try:
            prediction = loaded.predictor.predict(review)
        except Exception as exc:
            raise PredictionError(f"{loaded.info.name}: {exc}") from exc
        return prediction, (time.perf_counter() - started) * 1000

    @staticmethod
    def _entry(loaded: LoadedModel, prediction: Prediction, elapsed_ms: float) -> ModelPrediction:
        info = loaded.info
        return ModelPrediction(
            name=info.name,
            label=info.label,
            verdict=prediction.label,
            positive_probability=prediction.positive_probability,
            probabilities=prediction.probabilities,
            n_tokens=prediction.n_tokens,
            params=info.params,
            n_layers=info.n_layers,
            elapsed_ms=round(elapsed_ms, 3),
        )
