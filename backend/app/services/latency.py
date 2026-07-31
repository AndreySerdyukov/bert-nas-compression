"""Timing, done the way this project claims to do it.

Three rules, each of which exists because the notebooks this rebuild replaces broke it.

**Warm-up is discarded.** The first forward pass through a freshly loaded model pays for lazy
allocations and cache misses that no subsequent request will. Including it inflates the smallest
model most, because it has the least real work to hide the overhead behind.

**Median and p95, never a bare mean.** One descheduled run drags a mean somewhere no request ever
was. The median says what usually happens and the p95 says how bad it gets.

**Latency and throughput are different quantities.** Time one review to get latency; time a batch
and divide to get a per-example number that is always smaller. This module produces them through two
functions with two names, and the schemas keep them in two fields, so the two cannot be conflated by
accident. That conflation is the specific defect the rebuild exists to correct.

Comparing several models adds a fourth rule: **measure them round-robin**. Measuring model A to
completion and then model B hands any drift over the measurement window - a background process, a
thermal ramp - entirely to whichever model went last.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

from app.schemas.predict import LatencyStats, ThroughputStats

T = TypeVar("T")


@dataclass(frozen=True)
class Timing:
    """The timed samples for one thing, and the two statistics published from them."""

    samples: tuple[float, ...]

    @property
    def median_ms(self) -> float:
        return percentile(self.samples, 50)

    @property
    def p95_ms(self) -> float:
        return percentile(self.samples, 95)


def percentile(samples: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile over unsorted samples.

    With the five repeats an interactive request can afford, p95 lands on or beside the slowest
    sample. That is not a flaw to be smoothed over - it is what a five-sample p95 means, and the
    repeat count travels in the response so the reader can see it.
    """
    if not samples:
        raise ValueError("no samples to take a percentile of")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    position = (q / 100) * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def time_call(call: Callable[[], object], *, warmup: int, repeats: int) -> Timing:
    """Run `call` warm-up times, then time it `repeats` times."""
    if repeats < 1:
        raise ValueError("timing needs at least one repeat")
    for _ in range(max(warmup, 0)):
        call()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1000)
    return Timing(samples=tuple(samples))


def time_round_robin(
    calls: Mapping[str, Callable[[], object]], *, warmup: int, repeats: int
) -> dict[str, Timing]:
    """Time several things interleaved, rotating the order every round.

    Interleaving spreads any drift across all of them; rotating stops the model that happens to sit
    first in each round from systematically paying for whatever the round boundary costs.
    """
    if repeats < 1:
        raise ValueError("timing needs at least one repeat")
    names = list(calls)
    for _ in range(max(warmup, 0)):
        for name in names:
            calls[name]()

    samples: dict[str, list[float]] = {name: [] for name in names}
    for round_index in range(repeats):
        offset = round_index % len(names)
        for name in names[offset:] + names[:offset]:
            started = time.perf_counter()
            calls[name]()
            samples[name].append((time.perf_counter() - started) * 1000)
    return {name: Timing(samples=tuple(values)) for name, values in samples.items()}


def latency_stats(timing: Timing, *, warmup: int, n_tokens: int) -> LatencyStats:
    """A single-example timing, rendered as the API publishes it."""
    return LatencyStats(
        batch_size=1,
        warmup=warmup,
        repeats=len(timing.samples),
        median_ms=round(timing.median_ms, 3),
        p95_ms=round(timing.p95_ms, 3),
        n_tokens=n_tokens,
    )


def throughput_stats(
    timing: Timing, *, warmup: int, batch_size: int, n_tokens: int
) -> ThroughputStats:
    """A batch timing, rendered with `per_example_ms` explicitly not called latency."""
    if batch_size < 1:
        raise ValueError("batch size must be at least 1")
    median = timing.median_ms
    per_example = median / batch_size
    return ThroughputStats(
        batch_size=batch_size,
        warmup=warmup,
        repeats=len(timing.samples),
        median_batch_ms=round(median, 3),
        per_example_ms=round(per_example, 3),
        examples_per_second=round(1000 / per_example, 2) if per_example > 0 else 0.0,
        n_tokens=n_tokens,
    )
