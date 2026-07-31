"""The disagreement scan: where two models actually part company, over the whole sample.

This is the empirical face of the project's headline number. "Four encoder layers cost two and a
half points of accuracy" is a sentence nobody can picture; "fifty of these two thousand reviews get
a different answer, and here are eight of them" is one anybody can. So the scan does not report a
percentage and stop - it keeps the rows.

It is not a timing measurement, and that shapes two decisions. It runs on every core rather than
the pinned single thread, and it scores in batches rather than one review at a time. Both would be
wrong for a latency figure and are simply faster here.

Accuracy against the corpus label comes out of the same pass for free, so it is reported - clearly
labelled as being over this 2000-row sample. It is not the benchmark. The benchmark is the full
15 000-row test split and lives in `training/benchmark.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.repositories.examples import SampleRow
from app.repositories.model_registry import LoadedModel
from app.services.progress import ProgressReporter
from app.serving.base import POSITIVE

# One forward pass per chunk. Sixteen full-length reviews is a comfortable tensor on a laptop and
# most of the batching win is already in by then.
CHUNK = 16

# How many disagreeing reviews travel back to the page. Both this and the number found are
# reported: a cap that is not stated reads as "that was all of them".
MAX_ROWS_RETURNED = 40
# Reviews run to thousands of characters and the page shows an excerpt.
EXCERPT_CHARS = 600


def _verdict_label(row_label: int) -> str:
    return POSITIVE if row_label == 1 else "negative"


def scan_disagreements(
    models: dict[str, LoadedModel],
    rows: Sequence[SampleRow],
    *,
    baseline: str,
    progress: ProgressReporter,
) -> dict[str, Any]:
    """Score every row with every model and collect the rows where they do not agree.

    One stage per model, named, because a run of this length is unreadable as a bare counter.
    """
    if not models:
        raise ValueError("no models to scan with")
    if not rows:
        raise ValueError("no rows to scan")

    # The baseline goes first: everything after it is compared against its verdicts, and a reader
    # watching the stages should see the reference established before the comparisons start.
    names = sorted(models, key=lambda name: (name != baseline, name))
    texts = [row.text for row in rows]

    verdicts: dict[str, list[str]] = {}
    for name in names:
        info = models[name].info
        with progress.stage(f"scoring with {info.label}"):
            scored: list[str] = []
            with progress.sub(len(rows), info.label) as counter:
                for start in range(0, len(texts), CHUNK):
                    batch = texts[start : start + CHUNK]
                    scored.extend(
                        prediction.label
                        for prediction in models[name].predictor.predict_many(batch)
                    )
                    counter.update(len(batch))
            verdicts[name] = scored
        progress.write(f"{info.label}: {len(scored)} reviews scored")

    with progress.stage("comparing"):
        report = _compare(models, rows, verdicts, names, baseline)
    progress.write(
        f"{report['disagreements_found']} of {len(rows)} reviews split the models"
        if report["disagreements_found"]
        else f"every model agreed on all {len(rows)} reviews"
    )
    return report


def _compare(
    models: dict[str, LoadedModel],
    rows: Sequence[SampleRow],
    verdicts: dict[str, list[str]],
    names: Sequence[str],
    baseline: str,
) -> dict[str, Any]:
    """Turn per-model verdict lists into per-model counts and the rows that split them."""
    truths = [_verdict_label(row.label) for row in rows]
    has_baseline = baseline in verdicts

    summaries = []
    for name in names:
        scored = verdicts[name]
        correct = sum(int(v == t) for v, t in zip(scored, truths, strict=True))
        agree = (
            sum(int(v == b) for v, b in zip(scored, verdicts[baseline], strict=True))
            if has_baseline
            else 0
        )
        summaries.append(
            {
                "name": name,
                "label": models[name].info.label,
                "params": models[name].info.params,
                "n_layers": models[name].info.n_layers,
                # Against the corpus label, on this sample. Not the benchmark.
                "correct": correct,
                "accuracy": round(correct / len(rows), 4),
                # Against the baseline's verdicts, whatever those were worth.
                "agree_with_baseline": agree if has_baseline and name != baseline else None,
                "agreement": (
                    round(agree / len(rows), 4) if has_baseline and name != baseline else None
                ),
            }
        )

    split: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row_verdicts = {name: verdicts[name][index] for name in names}
        if len(set(row_verdicts.values())) == 1:
            continue
        split.append(
            {
                "id": row.id,
                "label": _verdict_label(row.label),
                "n_chars": len(row.text),
                "excerpt": row.text[:EXCERPT_CHARS],
                "truncated": len(row.text) > EXCERPT_CHARS,
                "verdicts": row_verdicts,
            }
        )

    return {
        "baseline": baseline if has_baseline else None,
        "rows_scanned": len(rows),
        "models": summaries,
        "disagreements_found": len(split),
        "disagreements_shown": min(len(split), MAX_ROWS_RETURNED),
        "disagreements": split[:MAX_ROWS_RETURNED],
    }
