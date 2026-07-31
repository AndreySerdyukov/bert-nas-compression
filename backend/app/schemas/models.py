"""Descriptions of the served models, and of the machine that serves them.

`RuntimeInfo` is attached to every response that carries a timing. That is not defensive detail: a
millisecond figure without its thread count, device and machine is not comparable to anything, and
the whole point of this application is that its numbers can be compared.
"""

from __future__ import annotations

from pydantic import BaseModel


class RuntimeInfo(BaseModel):
    """What produced a timing. Rendered next to it, never dropped."""

    device: str
    # Pinned in settings so two models measured minutes apart are measured the same way.
    threads: int
    interop_threads: int
    machine: str
    torch_version: str


class ModelInfo(BaseModel):
    """Public description of a loaded model."""

    name: str
    label: str
    description: str = ""
    # None for the uncompressed reference: it is what the others are compressions of.
    method: str | None = None

    # Measured off the loaded module, then checked against the manifest and, for the BERT
    # descendants, against the closed-form parameter count.
    params: int | None = None
    n_layers: int | None = None

    # Truncation length. Not cosmetic: it is part of the model, because it is the length the
    # checkpoint was trained and evaluated at, and it is part of every latency figure.
    max_length: int

    hf_repo: str
    hf_revision: str

    # Which class index means "positive", measured at download time and re-checked at warmup.
    positive_index: int
    # The one model the others are compared against.
    is_baseline: bool = False


class SkippedModel(BaseModel):
    """A model with a manifest that did not load, and why.

    Exposed rather than logged. A clean clone has no weights at all, and an app that answered as if
    those models simply did not exist would be indistinguishable from an app whose checkpoints had
    silently broken.
    """

    name: str
    label: str
    reason: str
    # What the reader can do about it, when there is something. Usually the fetch command.
    remedy: str | None = None


class ModelsResponse(BaseModel):
    """The catalog: what loaded, what did not, and on what machine."""

    models: list[ModelInfo]
    skipped: list[SkippedModel]
    baseline: str
    runtime: RuntimeInfo
