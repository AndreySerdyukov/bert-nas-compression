"""The progress protocol, and the do-nothing implementation of it.

The same evaluation code runs in two places: inside a request, streaming to a browser, and inside
`training/benchmark.py`, drawing a tqdm bar in a terminal. It should not know which. So the shape
here is copied from `training/progress.py`'s `StageTracker` - same method names, same semantics -
and both the terminal tracker and the job reporter satisfy it without either being aware of the
other.

Naming the stages rather than only counting them is deliberate: a bar that says "1400/2000" tells a
reader how much is left, and one that says "scoring with BANANAS, 1400/2000" tells them what the
machine is doing. On a run that takes minutes, the second is the difference between waiting and
wondering whether it has hung.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from types import TracebackType
from typing import Protocol, Self, runtime_checkable


@runtime_checkable
class SubProgress(Protocol):
    """A nested counter within one stage: rows, batches, epochs."""

    def update(self, n: int = 1) -> None: ...

    def close(self) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


@runtime_checkable
class ProgressReporter(Protocol):
    """What the evaluation code is allowed to assume about being watched."""

    def stage(self, name: str) -> AbstractContextManager[None]:
        """Context manager for one named stage; advances the outer count on exit."""
        ...

    def skip(self, name: str) -> None:
        """Mark a stage as skipped without losing it from the count."""
        ...

    def sub(self, total: int, desc: str) -> SubProgress:
        """A nested counter for the work inside the current stage."""
        ...

    def write(self, message: str) -> None:
        """A line for the reader, above the bars or into the stream."""
        ...

    def close(self) -> None:
        """No more progress is coming."""
        ...


class NullSub:
    """A nested counter nobody is watching."""

    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class NullProgress:
    """Progress reporting turned off, for tests and for callers that do not want it.

    A real implementation rather than `None` threaded through the evaluation code: an optional
    reporter means every call site grows an `if progress is not None`, and one of them eventually
    will not.
    """

    def __init__(self) -> None:
        self.messages: list[str] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        yield

    def skip(self, name: str) -> None:
        pass

    def sub(self, total: int, desc: str) -> NullSub:
        return NullSub()

    def write(self, message: str) -> None:
        # Kept rather than dropped: tests assert on what a run said it was doing.
        self.messages.append(message)

    def close(self) -> None:
        pass
