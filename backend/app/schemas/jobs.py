"""Schemas for long-running work: what a job looks like from outside while it runs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatus = Literal["running", "done", "failed"]


class JobSnapshot(BaseModel):
    """Everything a page needs to draw a progress bar and then a result.

    One shape for both transports. The stream sends this and the polling endpoint returns this, so
    a client that loses the stream can fall back to asking, and nothing about its rendering changes.
    """

    id: str
    kind: str
    status: JobStatus

    # Named, not merely counted: "scoring with BANANAS, 1400/2000" tells a reader what the machine
    # is doing, where "1400/2000" only tells them how long they have left to wait.
    stages: list[str]
    stage: str | None = None
    stage_index: int = 0
    processed: int = 0
    total: int = 0

    messages: list[str] = Field(default_factory=list)
    elapsed_s: float = 0.0

    # Shape depends on `kind`; null until the job finishes. A failed job carries `error` instead.
    result: dict[str, Any] | None = None
    error: str | None = None


class StartJobRequest(BaseModel):
    """Start a job. `kind` picks the work, and the other fields belong to one kind each."""

    kind: Literal["disagreement", "selftest", "ablation"]
    # Rows of the evaluation sample to scan. None means all of them.
    limit: int | None = None
    # `disagreement` only. None means every loaded model.
    models: list[str] | None = None
    # `ablation` only: which encoder layers survive. Indices rather than a 0/1 mask, because that
    # is what the reader is choosing and what every error message names.
    layers: list[int] | None = None
