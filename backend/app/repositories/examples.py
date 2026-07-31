"""The evaluation sample: all two thousand rows, plus the dozen the playground offers.

One repository owns the file because two features read it. The disagreement scan needs every row -
the whole point of it is that the headline number is a count of rows, not a percentage - and the
playground needs a handful to start from. Loading it twice would be two parses of the same 2.7 MB
and two chances for the two to disagree about what the sample is.

The offered dozen is a **rule, not a hand-picked list**: the first few short reviews of each
polarity in file order, plus the first long one of each. The long pair earns its place by running
past the 512-token window - it is how a visitor sees for themselves that the models are reading a
truncated review, which is otherwise an invisible detail of the protocol.
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
class SampleRow:
    """One row of the evaluation sample: what the models are scored on."""

    id: int
    text: str
    # 1 positive, 0 negative, straight from the corpus.
    label: int


@dataclass(frozen=True)
class Example:
    """A row offered by the playground, with the shape the page wants."""

    id: int
    text: str
    label: int
    n_chars: int
    # True when the review runs past the 512-token window the BERT descendants read.
    truncated: bool


@dataclass(frozen=True)
class ExampleRepository:
    """The sample, read once at startup."""

    rows: tuple[SampleRow, ...]
    examples: tuple[Example, ...]

    @property
    def total(self) -> int:
        return len(self.rows)

    @classmethod
    def load(cls, data_dir: Path) -> ExampleRepository:
        """Read the sample and apply the selection rule. A missing file is not fatal."""
        path = data_dir / SAMPLE_FILE
        if not path.exists():
            logger.warning("no evaluation sample at %s; the playground will offer none", path)
            return cls(rows=(), examples=())

        # Not splitlines(): IMDB reviews contain U+2028, which Python treats as a line break and
        # JSON does not escape. The naive version counted 3% more rows than the file holds.
        lines = path.read_text(encoding="utf-8").strip("\n").split("\n")
        rows = tuple(
            SampleRow(id=int(row["id"]), text=str(row["text"]), label=int(row["label"]))
            for row in (json.loads(line) for line in lines[1:])
        )
        return cls(rows=rows, examples=cls._offer(rows))

    @staticmethod
    def _offer(rows: tuple[SampleRow, ...]) -> tuple[Example, ...]:
        """Apply the selection rule to the loaded sample."""
        short: dict[int, list[Example]] = {0: [], 1: []}
        long: dict[int, list[Example]] = {0: [], 1: []}
        for row in rows:
            entry = Example(
                id=row.id,
                text=row.text,
                label=row.label,
                n_chars=len(row.text),
                truncated=len(row.text) >= LONG_CHARS,
            )
            if len(row.text) <= SHORT_CHARS and len(short[row.label]) < SHORT_PER_LABEL:
                short[row.label].append(entry)
            elif len(row.text) >= LONG_CHARS and len(long[row.label]) < LONG_PER_LABEL:
                long[row.label].append(entry)

        # Alternating polarity, so the list does not read as "the positives, then the negatives".
        ordered: list[Example] = []
        for bucket in (short, long):
            for pair in zip(bucket[1], bucket[0], strict=False):
                ordered.extend(pair)
        return tuple(ordered)
