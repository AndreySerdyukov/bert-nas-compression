"""Work that outlives a request: one job at a time, watched over a stream.

Scanning two thousand reviews with five models takes minutes. That is too long for a request to
hold open and far too long for a page to show nothing, so the work moves to a worker thread and the
page watches it.

**One at a time, and a refusal rather than a queue.** The executor has a single worker, and a second
submission is turned away with 429 instead of being queued. Queueing would be worse than it sounds:
the models are one shared object, so a queued job would be invisible for however long the first one
takes, and the reader would be looking at a page that says nothing is happening.

**The stream is the transport, not the truth.** Every frame is a complete snapshot rather than a
delta, and the same snapshot is what `GET /api/jobs/{id}` returns. A dropped connection therefore
costs nothing: the client asks once and carries on. Deltas would have made a lost frame permanent.

Snapshots are throttled. A scan updates its counter ten thousand times; a page needs a few frames a
second, and stage changes and terminal states are never throttled away.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self

from app.schemas.jobs import JobSnapshot

logger = logging.getLogger(__name__)

# A page redraws happily at four frames a second, and the scan would otherwise emit thousands.
THROTTLE_S = 0.25
# How long the stream waits before sending a comment frame, so proxies and browsers keep the
# connection rather than reaping it as idle.
HEARTBEAT_S = 15.0
# Finished jobs kept for their results to be fetched after the stream has closed.
KEEP_FINISHED = 8


class JobBusyError(Exception):
    """Another job holds the worker."""


class JobNotFoundError(Exception):
    """No job by that id, or it has aged out of the finished list."""


@dataclass
class Job:
    """One unit of long work, and everything visible about it while it runs."""

    id: str
    kind: str
    stages: list[str]

    status: str = "running"
    stage: str | None = None
    stage_index: int = 0
    processed: int = 0
    total: int = 0
    messages: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None

    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _watchers: list[queue.Queue[JobSnapshot]] = field(default_factory=list, repr=False)

    @property
    def finished(self) -> bool:
        return self.status in ("done", "failed")

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            end = self.finished_at if self.finished_at is not None else time.monotonic()
            return JobSnapshot(
                id=self.id,
                kind=self.kind,
                status=self.status,  # type: ignore[arg-type]
                stages=list(self.stages),
                stage=self.stage,
                stage_index=self.stage_index,
                processed=self.processed,
                total=self.total,
                messages=list(self.messages),
                elapsed_s=round(end - self.started_at, 2),
                result=self.result,
                error=self.error,
            )

    def publish(self) -> None:
        """Hand the current snapshot to every watcher."""
        snapshot = self.snapshot()
        with self._lock:
            watchers = list(self._watchers)
        for watcher in watchers:
            watcher.put(snapshot)

    def watch(self) -> queue.Queue[JobSnapshot]:
        channel: queue.Queue[JobSnapshot] = queue.Queue()
        with self._lock:
            self._watchers.append(channel)
        return channel

    def unwatch(self, channel: queue.Queue[JobSnapshot]) -> None:
        with self._lock:
            if channel in self._watchers:
                self._watchers.remove(channel)


class JobSub:
    """A nested counter that publishes as it advances, at most a few times a second."""

    def __init__(self, progress: JobProgress, total: int, desc: str) -> None:
        self._progress = progress
        # `desc` is taken and dropped: the terminal tracker draws it on the nested bar, and this
        # side has no second line to draw it on - the stage name is already the label a reader
        # sees. Kept in the signature because `ProgressReporter.sub` is one shape for both.
        progress.set_total(total)

    def update(self, n: int = 1) -> None:
        self._progress.advance(n)

    def close(self) -> None:
        self._progress.publish_now()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class JobProgress:
    """`ProgressReporter` over a job. Same method names as the terminal tracker, by design.

    The evaluation code is written once against this shape and runs unchanged under tqdm in
    `training/benchmark.py` and under a browser here.
    """

    def __init__(self, job: Job) -> None:
        self._job = job
        self._last_publish = 0.0

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self._job.stage = name
        self._job.processed = 0
        # Stage changes are never throttled: they are the part a reader is actually reading.
        self.publish_now()
        try:
            yield
        finally:
            self._job.stage_index += 1
            self.publish_now()

    def skip(self, name: str) -> None:
        self._job.stage = f"{name} (skipped)"
        self._job.stage_index += 1
        self.publish_now()

    def sub(self, total: int, desc: str) -> JobSub:
        return JobSub(self, total, desc)

    def write(self, message: str) -> None:
        self._job.messages.append(message)
        self.publish_now()

    def close(self) -> None:
        self.publish_now()

    # --- internals ------------------------------------------------------------------------------

    def set_total(self, total: int) -> None:
        self._job.total = total
        self._job.processed = 0
        self.publish_now()

    def advance(self, n: int) -> None:
        self._job.processed += n
        now = time.monotonic()
        if now - self._last_publish >= THROTTLE_S:
            self._last_publish = now
            self._job.publish()

    def publish_now(self) -> None:
        self._last_publish = time.monotonic()
        self._job.publish()


class JobRunner:
    """One worker, one job, and everything anyone needs to watch it."""

    def __init__(self, keep: int = KEEP_FINISHED) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="job")
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._keep = keep
        self._lock = threading.Lock()

    def submit(
        self,
        kind: str,
        work: Callable[[JobProgress], dict[str, Any]],
        *,
        stages: list[str],
    ) -> Job:
        """Start a job, or refuse because one is already running.

        The check is explicit rather than left to the single-worker executor: a queued submission
        would sit invisible behind a job that takes minutes, and the caller would have no way to
        tell "waiting" from "running".
        """
        with self._lock:
            for job_id in self._order:
                if not self._jobs[job_id].finished:
                    raise JobBusyError(
                        f"a {self._jobs[job_id].kind} job is already running "
                        f"({self._jobs[job_id].id}). The models are one shared object, so there is "
                        "one worker and it is busy."
                    )
            job = Job(id=uuid.uuid4().hex[:12], kind=kind, stages=stages)
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._forget_old()

        self._executor.submit(self._run, job, work)
        return job

    def _run(self, job: Job, work: Callable[[JobProgress], dict[str, Any]]) -> None:
        progress = JobProgress(job)
        try:
            job.result = work(progress)
            job.status = "done"
        except Exception as exc:
            logger.exception("job %s (%s) failed", job.id, job.kind)
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = "failed"
        finally:
            job.finished_at = time.monotonic()
            # The terminal snapshot is what tells every watcher to stop reading.
            job.publish()

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job

    def stream(self, job_id: str) -> Iterator[str]:
        """Server-sent events for one job: a snapshot now, then one per change, then the end."""
        job = self.get(job_id)
        channel = job.watch()
        try:
            # The current state first, so a client that connected late is never left waiting for a
            # change that already happened - including a job that finished before it asked.
            snapshot = job.snapshot()
            yield _frame(snapshot)
            if snapshot.status != "running":
                return

            while True:
                try:
                    snapshot = channel.get(timeout=HEARTBEAT_S)
                except queue.Empty:
                    # A comment frame. Keeps proxies and browsers from reaping an idle connection
                    # during a long stage that has not ticked yet.
                    yield ": keep-alive\n\n"
                    continue
                yield _frame(snapshot)
                if snapshot.status != "running":
                    return
        finally:
            job.unwatch(channel)

    def _forget_old(self) -> None:
        """Drop the oldest finished jobs, keeping recent results fetchable after the stream ends."""
        finished = [job_id for job_id in self._order if self._jobs[job_id].finished]
        for job_id in finished[: max(0, len(finished) - self._keep)]:
            self._jobs.pop(job_id, None)
            self._order.remove(job_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _frame(snapshot: JobSnapshot) -> str:
    """One server-sent event carrying a whole snapshot."""
    return f"event: update\ndata: {json.dumps(snapshot.model_dump())}\n\n"


SELFTEST_STEPS = 10
SELFTEST_STAGES = ["waking up", "counting", "winding down"]


def selftest_work(
    progress: JobProgress, *, steps: int = SELFTEST_STEPS, delay: float = 0.4
) -> dict[str, Any]:
    """A job that only sleeps, and the reason the whole pipeline is testable without weights.

    It exercises everything the real scan does - submit, stages, a counter, the stream, a terminal
    snapshot - and needs no checkpoint, so CI can drive it through nginx on an image built with no
    models in it. That is where a missing `proxy_buffering off` shows up: with buffering on, every
    frame below arrives at once at the end, and nothing in development reproduces it because Vite's
    proxy streams.
    """
    with progress.stage(SELFTEST_STAGES[0]):
        time.sleep(delay)
    with progress.stage(SELFTEST_STAGES[1]), progress.sub(steps, "steps") as counter:
        for step in range(steps):
            time.sleep(delay)
            counter.update(1)
            if step == steps // 2:
                progress.write(f"halfway: {step + 1} of {steps}")
    with progress.stage(SELFTEST_STAGES[2]):
        time.sleep(delay)
    return {"steps": steps, "note": "no model was involved in this job"}
