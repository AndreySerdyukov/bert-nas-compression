"""What a mask is worth without retraining, measured against the same model at full depth.

The explorer's whole claim rests on the difference between two operations that look alike written
down. NAS does `finetune(mask(pretrained))`. This does `mask(finetuned)`, and the accuracy collapse
that follows is the measurement: it is what training every candidate buys, and it is why a search
cannot skip that cost.

Two decisions here exist to keep the comparison honest.

**The full model is re-scored in the same pass, on the same rows.** Not read off the benchmark. The
benchmark is the 15 000-row split and the explorer runs over the 2 000-row sample, so quoting one
next to the other would be comparing an ablation on one set of reviews with a baseline on a
different one. Scoring both here costs a second forward pass and removes the question.

**Every accuracy carries a Wilson interval.** On 2 000 rows the interval is a little over a point
wide, which is the same order as the gaps people will want to read off this page. Printing a bare
percentage would invite conclusions the sample size does not support.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from app.repositories.examples import SampleRow
from app.repositories.model_registry import LoadedModel
from app.services.architecture import (
    layers_from_mask,
    mask_from_layers,
    matches_known_architecture,
    params_for,
)
from app.services.progress import ProgressReporter
from app.serving.base import POSITIVE
from app.serving.masked_bert import amputated

# One forward pass per chunk, same size the disagreement scan uses.
CHUNK = 16

# 95%, the interval this project publishes everywhere.
Z = 1.96


def wilson(correct: int, total: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Wilson rather than the normal approximation because the normal one misbehaves exactly where an
    ablation lands: near 0.5 it is fine, but an amputated encoder can score in the low fifties or
    the high nineties, and at the ends the naive interval runs past 0 or 1 and stops meaning
    anything.
    """
    if total == 0:
        return (0.0, 0.0)
    proportion = correct / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
    spread /= denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _scored(verdicts: Sequence[str], truths: Sequence[str]) -> dict[str, Any]:
    correct = sum(int(verdict == truth) for verdict, truth in zip(verdicts, truths, strict=True))
    low, high = wilson(correct, len(truths))
    return {
        "correct": correct,
        "accuracy": round(correct / len(truths), 4) if truths else 0.0,
        "wilson_low": round(low, 4),
        "wilson_high": round(high, 4),
    }


def _predict_all(
    loaded: LoadedModel, texts: Sequence[str], progress: ProgressReporter, label: str
) -> list[str]:
    verdicts: list[str] = []
    with progress.sub(len(texts), label) as counter:
        for start in range(0, len(texts), CHUNK):
            batch = texts[start : start + CHUNK]
            verdicts.extend(prediction.label for prediction in loaded.predictor.predict_many(batch))
            counter.update(len(batch))
    return verdicts


def score_ablation(
    loaded: LoadedModel,
    module: Any,
    rows: Sequence[SampleRow],
    layers: Sequence[int],
    *,
    progress: ProgressReporter,
) -> dict[str, Any]:
    """Score `layers` of a fine-tuned model with no retraining, and the same model at full depth.

    The caller holds the inference lock: this mutates the shared module for the duration of the
    first pass, and an overlapping request would otherwise be served by an amputated encoder.
    """
    if not rows:
        raise ValueError("no rows to score")

    texts = [row.text for row in rows]
    truths = [POSITIVE if row.label == 1 else "negative" for row in rows]

    with progress.stage("scoring the amputated model"), amputated(module, layers) as kept:
        ablated_verdicts = _predict_all(loaded, texts, progress, f"{len(kept)} layers")
    ablated = _scored(ablated_verdicts, truths)
    progress.write(
        f"{len(kept)} of {loaded.info.n_layers or '?'} layers, no retraining: "
        f"accuracy {ablated['accuracy']:.4f}"
    )

    with progress.stage("scoring the full model"):
        full_verdicts = _predict_all(loaded, texts, progress, "12 layers")
    full = _scored(full_verdicts, truths)
    progress.write(f"the same model at full depth: accuracy {full['accuracy']:.4f}")

    mask = mask_from_layers(list(kept))
    agree = sum(int(a == b) for a, b in zip(ablated_verdicts, full_verdicts, strict=True))

    return {
        "mask": list(mask),
        "layers": list(kept),
        "n_layers": len(kept),
        "params": params_for(len(kept)),
        "matches_known_architecture": matches_known_architecture(mask),
        "baseline": loaded.info.name,
        "baseline_label": loaded.info.label,
        "rows_scanned": len(rows),
        "ablated": ablated,
        "full": full,
        # How often the amputated model still says what the full one said. An accuracy that barely
        # moved while agreement collapsed would mean it is right about different reviews.
        "agreement_with_full": round(agree / len(rows), 4),
        "protocol": (
            f"Both figures are the same {len(rows)} reviews from the evaluation sample, scored in "
            "one pass. This is not the benchmark, which is the full 15 000-row test split."
        ),
        "note": (
            "The layers were removed from an already fine-tuned model and nothing was retrained. "
            "The searched architectures were trained after their layers were chosen, which is the "
            "expensive half of NAS and the half this measurement leaves out."
        ),
    }


def ablation_stages() -> list[str]:
    """Named up front so the whole bar can be drawn before the first forward pass."""
    return ["scoring the amputated model", "scoring the full model"]


def layers_of(mask: Sequence[int]) -> tuple[int, ...]:
    """Convenience for callers holding a 0/1 mask rather than indices."""
    return layers_from_mask(tuple(mask))
