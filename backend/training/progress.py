"""Progress output for the offline scripts: named stages plus nested bars.

Copied from the sibling `dl-playground` on purpose, method for method, because the application's
`app/services/progress.py` mirrors this same shape. A benchmark run and a browser job execute the
same evaluation code; one of them ends up here drawing tqdm bars, the other streams to a page, and
the code between them knows about neither.

Why it exists at all: a benchmark runs for minutes, and from bare output there is no telling which
stage the process is in or how much is left. The top bar tracks whole stages ("3/8 scoring with
BANANAS"), the nested one tracks rows within a stage. Both live at once, so overall and local
progress are visible together.

One detail that matters: a plain `print` in the middle of a live tqdm bar shreds the terminal, so
every message goes through `tracker.write()` - that is `tqdm.write`, which redraws the bars.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Self

from tqdm.auto import tqdm


class StageTracker:
    """A top-level bar over stages plus a factory for nested bars.

    The stage list is fixed up front and never changes: even a skipped stage still counts towards
    overall progress, because otherwise the time-remaining estimate would lie.
    """

    def __init__(self, stages: list[str], *, desc: str, enabled: bool = True) -> None:
        self.stages = stages
        # Outside a terminal (piped to a file, or CI) the bars turn into miles of ANSI garbage, so
        # they disable themselves there and only `write` messages remain.
        self.enabled = enabled and sys.stderr.isatty()
        self._prefix = desc
        self._bar = tqdm(
            total=len(stages),
            desc=desc,
            position=0,
            leave=True,
            disable=not self.enabled,
            bar_format="{desc}: {n_fmt}/{total_fmt} stages |{bar}| [{elapsed}<{remaining}]",
        )

    def _set_stage(self, label: str) -> None:
        self._bar.set_description_str(f"{self._prefix} · {label}")

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Context for one stage: shows its name, advances the top bar on exit."""
        self._set_stage(name)
        try:
            yield
        finally:
            self._bar.update(1)

    def skip(self, name: str) -> None:
        """Mark a stage as skipped without losing the count."""
        self._set_stage(f"{name} (skipped)")
        self._bar.update(1)

    def sub(self, total: int, desc: str) -> tqdm[Any]:
        """A nested bar (rows, batches, epochs). Closes itself once the `with` block exits."""
        return tqdm(total=total, desc=desc, position=1, leave=False, disable=not self.enabled)

    def write(self, message: str) -> None:
        """A message printed above the bars, replacing `print` in the scripts."""
        tqdm.write(message)

    def close(self) -> None:
        self._set_stage("done")
        self._bar.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
