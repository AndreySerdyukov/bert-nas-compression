"""The label probe: a handful of unambiguous reviews, used to check polarity at every load.

No checkpoint in this project carries `id2label`, so which class index means "positive" is genuinely
unknown until someone asks the model. `scripts/fetch_models.py` asks at download time and pins the
answer in the manifest; this is the same question asked again at startup, so that a checkpoint
replaced upstream is caught before it serves an inverted prediction rather than after.

The reviews live in `data/label_probe.json` and are committed, so both askings use the same eight
sentences.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

PROBE_FILE = "label_probe.json"
POSITIVE_LABEL = "positive"


@dataclass(frozen=True)
class LabelProbe:
    """Reviews whose sentiment is not in question, and how much agreement is enough."""

    texts: list[str]
    # True where the review is positive. Same order as `texts`.
    truths: list[bool]
    min_agreement: float

    @classmethod
    def load(cls, data_dir: Path) -> LabelProbe:
        """Read the committed probe. Missing or malformed is fatal, not a warning.

        There is no sensible fallback: skipping the check would let a model with unknown polarity
        serve predictions, which is exactly the outcome the probe exists to prevent.
        """
        path = data_dir / PROBE_FILE
        payload = json.loads(path.read_text(encoding="utf-8"))
        reviews = payload["reviews"]
        if not reviews:
            raise ValueError(f"{path} contains no probe reviews")
        return cls(
            texts=[str(review["text"]) for review in reviews],
            truths=[review["label"] == POSITIVE_LABEL for review in reviews],
            min_agreement=float(payload["min_agreement"]),
        )

    def agreement(self, predictions: Sequence[bool]) -> float:
        """Fraction of probe reviews the model got right, given `True` means it said positive."""
        if len(predictions) != len(self.truths):
            raise ValueError(
                f"probe has {len(self.truths)} reviews but {len(predictions)} predictions arrived"
            )
        hits = sum(int(p == t) for p, t in zip(predictions, self.truths, strict=True))
        return hits / len(self.truths)

    def __len__(self) -> int:
        return len(self.texts)
