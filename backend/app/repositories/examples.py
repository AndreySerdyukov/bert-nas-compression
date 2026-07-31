"""Reviews the playground offers as a starting point, drawn from the committed evaluation sample.

Typing a film review from memory to try a classifier is a worse experience than it sounds, so the
page offers a handful. They come from `data/imdb_eval_sample.jsonl` - the same 2000 rows every
model is scored on - rather than being written for the occasion, which means each one carries its
row id and its true label, and a visitor can trace any of them back to the corpus.

The selection is a **rule, not a hand-picked list**: the first few short reviews of each polarity in
file order, plus the first long one of each. The long pair earns its place by being longer than the
512-token window - it is how a visitor sees for themselves that the models are reading a truncated
review, which is otherwise an invisible detail of the protocol.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SAMPLE_FILE = "imdb_eval_sample.jsonl"

# A short review fits on the page without a scrollbar; a long one runs past the token window.
SHORT_CHARS = 700
LONG_CHARS = 3_000
SHORT_PER_LABEL = 4
LONG_PER_LABEL = 1


@dataclass(frozen=True)
class Example:
    """One review, with its provenance intact."""

    id: int
    text: str
    label: int
    n_chars: int
    # True when the review runs past the 512-token window the BERT descendants read.
    truncated: bool


@dataclass(frozen=True)
class ExampleRepository:
    """A dozen reviews chosen by rule from the evaluation sample, read once at startup."""

    examples: tuple[Example, ...]
    total: int

    @classmethod
    def load(cls, data_dir: Path) -> ExampleRepository:
        """Read the sample and apply the selection rule. A missing file is not fatal."""
        path = data_dir / SAMPLE_FILE
        if not path.exists():
            logger.warning(
                "no evaluation sample at %s; the playground will offer no examples", path
            )
            return cls(examples=(), total=0)

        # Not splitlines(): IMDB reviews contain U+2028, which Python treats as a line break and
        # JSON does not escape. The naive version counted 3% more rows than the file holds.
        lines = path.read_text(encoding="utf-8").strip("\n").split("\n")
        rows = [json.loads(line) for line in lines[1:]]

        short: dict[int, list[Example]] = {0: [], 1: []}
        long: dict[int, list[Example]] = {0: [], 1: []}
        for row in rows:
            text = str(row["text"])
            label = int(row["label"])
            entry = Example(
                id=int(row["id"]),
                text=text,
                label=label,
                n_chars=len(text),
                truncated=len(text) >= LONG_CHARS,
            )
            if len(text) <= SHORT_CHARS and len(short[label]) < SHORT_PER_LABEL:
                short[label].append(entry)
            elif len(text) >= LONG_CHARS and len(long[label]) < LONG_PER_LABEL:
                long[label].append(entry)

        # Alternating polarity, so the list does not read as "the positives, then the negatives".
        ordered: list[Example] = []
        for bucket in (short, long):
            for pair in zip(bucket[1], bucket[0], strict=False):
                ordered.extend(pair)
        return cls(examples=tuple(ordered), total=len(rows))
