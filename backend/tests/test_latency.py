"""The timing primitives, tested without a model in sight.

These are twenty lines of arithmetic that every published number passes through, so they are worth
pinning directly rather than only through an endpoint that also has to load half a gigabyte first.
"""

from __future__ import annotations

import pytest

from app.services.latency import (
    Timing,
    latency_stats,
    percentile,
    throughput_stats,
    time_call,
    time_round_robin,
)


def test_percentile_interpolates_between_samples() -> None:
    samples = [10.0, 20.0, 30.0, 40.0]
    assert percentile(samples, 50) == 25.0
    assert percentile(samples, 0) == 10.0
    assert percentile(samples, 100) == 40.0


def test_percentile_does_not_care_about_input_order() -> None:
    assert percentile([30.0, 10.0, 20.0], 50) == percentile([10.0, 20.0, 30.0], 50) == 20.0


def test_a_five_sample_p95_sits_beside_the_slowest_sample() -> None:
    """Stated as a test because it is what the published p95 means at this repeat count."""
    timing = Timing(samples=(1.0, 1.0, 1.0, 1.0, 9.0))
    assert timing.median_ms == 1.0
    assert timing.p95_ms == pytest.approx(7.4)


def test_warmup_calls_are_run_and_not_timed() -> None:
    calls = []
    time_call(lambda: calls.append(1), warmup=3, repeats=5)
    # Eight calls happened; only five of them produced a sample.
    assert len(calls) == 8


def test_time_call_produces_one_sample_per_repeat() -> None:
    timing = time_call(lambda: None, warmup=0, repeats=4)
    assert len(timing.samples) == 4
    assert all(sample >= 0 for sample in timing.samples)


def test_round_robin_interleaves_and_rotates_the_order() -> None:
    """Measuring one model to completion hands any drift entirely to whichever went last."""
    order: list[str] = []
    calls = {name: (lambda n=name: order.append(n)) for name in ("a", "b", "c")}
    timings = time_round_robin(calls, warmup=1, repeats=3)

    assert {name: len(t.samples) for name, t in timings.items()} == {"a": 3, "b": 3, "c": 3}
    rounds = [order[3:][i : i + 3] for i in range(0, 9, 3)]
    assert rounds == [["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"]]


def test_throughput_divides_the_batch_and_never_calls_it_latency() -> None:
    timing = Timing(samples=(160.0,))
    stats = throughput_stats(timing, warmup=1, batch_size=16, n_tokens=128)
    assert stats.median_batch_ms == 160.0
    assert stats.per_example_ms == 10.0
    assert stats.examples_per_second == 100.0
    # The fields are disjoint by construction: there is no `median_ms` to mistake for a latency.
    assert not hasattr(stats, "median_ms")


def test_latency_stats_report_the_repeat_count_they_came_from() -> None:
    stats = latency_stats(Timing(samples=(1.0, 2.0, 3.0)), warmup=2, n_tokens=17)
    assert (stats.batch_size, stats.warmup, stats.repeats) == (1, 2, 3)
    assert stats.median_ms == 2.0
    assert stats.n_tokens == 17


def test_timing_needs_at_least_one_repeat() -> None:
    with pytest.raises(ValueError):
        time_call(lambda: None, warmup=1, repeats=0)
    with pytest.raises(ValueError):
        percentile([], 50)
