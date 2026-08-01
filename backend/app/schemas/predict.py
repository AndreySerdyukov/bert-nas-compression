"""Request and response shapes for scoring reviews.

The naming here is deliberate and is the reason this file is longer than the endpoint needs.

`elapsed_ms` is how long this one scoring took, warm-up included, measured once. It is what the
interactive card shows, and it is honest about being a single sample.

`LatencyStats` is a measurement: warm-up passes discarded, several timed repeats, median and p95.

`ThroughputStats` is a different quantity with a different name. Dividing a batch time by the batch
size gives a per-example number that is always smaller than latency, and reporting that number as
latency is the specific defect this rebuild exists to correct. The two never share a field.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.models import RuntimeInfo


class LatencyStats(BaseModel):
    """Single-example latency: what one user waits for one review."""

    batch_size: int = 1
    warmup: int
    repeats: int
    median_ms: float
    # With the handful of repeats an interactive request can afford, p95 sits at or near the
    # slowest sample. It is reported anyway because the spread is the interesting part.
    p95_ms: float
    n_tokens: int


class ThroughputStats(BaseModel):
    """Batched throughput. `per_example_ms` here is not latency and is never labelled as such."""

    batch_size: int
    warmup: int
    repeats: int
    median_batch_ms: float
    per_example_ms: float
    examples_per_second: float
    n_tokens: int


class ModelPrediction(BaseModel):
    """One model's verdict on one review, with whatever timing was asked for."""

    name: str
    label: str
    verdict: str
    positive_probability: float
    probabilities: dict[str, float]
    n_tokens: int
    params: int | None = None
    n_layers: int | None = None

    # One sample, warm-up included. Always present, because it costs nothing to record.
    elapsed_ms: float
    latency: LatencyStats | None = None
    throughput: ThroughputStats | None = None

    # Only set in a comparison. None on the baseline itself.
    agrees_with_baseline: bool | None = None


class PredictRequest(BaseModel):
    """Score one review with one model."""

    text: str
    # Off by default: a proper measurement is several more forward passes, and the interactive
    # path should not pay for one unless the reader asked for it.
    measure_latency: bool = False


class PredictResponse(ModelPrediction):
    """One prediction plus the machine behind its timings."""

    runtime: RuntimeInfo


class CompareRequest(BaseModel):
    """Score one review with several models, measured round-robin."""

    text: str
    # None means every loaded model.
    models: list[str] | None = None
    measure_latency: bool = True
    # Off by default: a batch of 16 full-length reviews through five models is a long wait.
    measure_throughput: bool = False


class CompareResponse(BaseModel):
    """Every model's verdict on the same review, and where they part company."""

    baseline: str
    # Whether the baseline was among the models scored. False whenever it is not being served -
    # a clean clone that fetched one checkpoint, or a baseline the polarity probe dropped - and
    # then `disagree` is empty because nothing was compared, not because everything agreed. Those
    # two are indistinguishable without this field, and the page said the second.
    baseline_scored: bool = True
    results: list[ModelPrediction]
    # Names of the models whose verdict differs from the baseline's. This is the empirical face of
    # the headline number: a 2.5 pp accuracy gap is fifty reviews out of two thousand.
    disagree: list[str] = Field(default_factory=list)
    runtime: RuntimeInfo
