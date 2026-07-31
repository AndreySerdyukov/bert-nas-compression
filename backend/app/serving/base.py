"""The serve-side contract every model in this project satisfies.

There is exactly one modality and one task here - a review goes in, a sentiment verdict comes out -
so this contract is much narrower than the one in the sibling playgrounds, and it can afford to be
specific instead of generic. What it buys is that `services/` never learns the difference between a
twelve-layer transformer and an embedding table with a linear head on top: both answer `predict`.

Two fields exist because of measurement rather than prediction.

`n_tokens` is on every prediction because a latency figure without the number of tokens behind it
describes nothing. The models here truncate at different lengths (512 for the BERT descendants, 128
for AdaBERT, which is the length it was trained at), so a comparison that does not carry the token
count is comparing two different amounts of work.

`predict_many` exists separately from a loop over `predict` because throughput and latency are two
quantities. Measuring one and reporting the other is the specific defect this rebuild exists to
correct, so the two paths are two methods with two names.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.schemas.models import ModelInfo

# Class index order is fixed project-wide: index 0 is negative, index 1 is positive. Which index a
# given checkpoint actually uses is measured, not assumed - see `label_order` in the manifests.
NEGATIVE = "negative"
POSITIVE = "positive"
CLASS_NAMES = (NEGATIVE, POSITIVE)


@dataclass(frozen=True)
class Prediction:
    """One scored review."""

    label: str
    # Broken out because the whole app is binary sentiment: the UI draws one bar, not a histogram,
    # and the disagreement search compares this number between models.
    positive_probability: float
    probabilities: dict[str, float]
    # Tokens actually fed to the model, after truncation. Part of every latency figure.
    n_tokens: int


@runtime_checkable
class ModelPredictor(Protocol):
    """What the inference service is allowed to assume about a loaded model."""

    info: ModelInfo

    def predict(self, text: str) -> Prediction:
        """Score one review. This is the batch=1 path, and the one latency is measured on."""
        ...

    def predict_many(self, texts: Sequence[str]) -> list[Prediction]:
        """Score a batch. This is the throughput path; it is never reported as latency."""
        ...

    def warmup(self) -> None:
        """Score the label probe and confirm the polarity pinned in the manifest still holds."""
        ...
