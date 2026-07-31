"""Serve wrapper for the four transformer checkpoints: a review in, a sentiment verdict out.

The uncompressed reference and the three searched descendants are all `BertForSequenceClassification`
snapshots that differ only in how many encoder layers survived, so one wrapper serves all four.

Two things here are not boilerplate.

**Loading is strictly offline** (`local_files_only=True`). Without it an incomplete snapshot makes
transformers reach for the network and download half a gigabyte inside `create_app()`, before the
application has bound its port.

**The label order comes from the manifest, never from the model.** Not one of these checkpoints
carries `id2label` - `scripts/fetch_models.py` established the polarity by running unambiguous
reviews through each model at download time, and pinned the answer. This wrapper re-runs that same
probe in `warmup()`, because a checkpoint replaced upstream, or a manifest edited by hand, would
otherwise serve inverted sentiment while every other signal stayed green. Inverted sentiment is the
worst outcome available to this project: it looks perfectly healthy from the outside.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from app.schemas.models import ModelInfo
from app.serving.base import CLASS_NAMES, POSITIVE, Prediction
from app.serving.probe import LabelProbe

logger = logging.getLogger(__name__)

SUPPORTED_ARCHITECTURE = "bert_sequence_classification"

# A character cap applied before tokenization. `truncation=True` only takes effect after the
# tokenizer has chewed through the whole string, so a megabyte of pasted text is a real stall on a
# synchronous worker. The longest review in the IMDB corpus is under 14 000 characters.
MAX_INPUT_CHARS = 20_000


class PolarityError(ValueError):
    """The model disagrees with the label order pinned in its manifest."""


class BertClassifier:
    """Tokenizer -> BERT -> softmax -> {negative, positive}."""

    def __init__(
        self,
        info: ModelInfo,
        model: Any,
        tokenizer: Any,
        probe: LabelProbe,
        *,
        device: str = "cpu",
    ) -> None:
        self.info = info
        self._model = model
        self._tokenizer = tokenizer
        self._probe = probe
        self._device = device
        # CLASS_NAMES is index-ordered project-wide; the manifest says which index this particular
        # checkpoint uses for positive, so the names are permuted rather than assumed.
        self._labels = (
            list(CLASS_NAMES) if info.positive_index == 1 else list(reversed(CLASS_NAMES))
        )

    def predict(self, text: str) -> Prediction:
        """Score one review. This is the path latency is measured on."""
        return self.predict_many([text])[0]

    def predict_many(self, texts: Sequence[str]) -> list[Prediction]:
        """Score a batch in a single forward pass.

        Deliberately not chunked: the caller decides the batch size, because the throughput
        measurement means to time one batch of exactly the size it asked for. Callers scoring
        thousands of reviews chunk on their side.
        """
        if not texts:
            return []
        probabilities, token_counts = self._forward(texts)
        return [
            self._to_prediction(row, n_tokens)
            for row, n_tokens in zip(probabilities, token_counts, strict=True)
        ]

    def warmup(self) -> None:
        """Score the label probe and confirm the manifest's polarity still describes this model.

        This is the second half of a check whose first half ran at download time. It runs on every
        boot because the weights are gitignored and fetched separately: nothing else stands between
        an upstream replacement and a served prediction.
        """
        predictions = self.predict_many(self._probe.texts)
        agreement = self._probe.agreement([p.label == POSITIVE for p in predictions])
        if agreement < self._probe.min_agreement:
            raise PolarityError(
                f"{self.info.name}: the label probe agrees with the manifest only "
                f"{agreement:.0%} of the time (need {self._probe.min_agreement:.0%}). "
                f"The manifest pins class {self.info.positive_index} as positive; either the "
                "checkpoint on the Hub was replaced or the manifest was edited by hand. "
                "Re-run scripts/fetch_models.py."
            )
        logger.debug("%s: polarity confirmed on %d probe reviews", self.info.name, len(predictions))

    def _forward(self, texts: Sequence[str]) -> tuple[torch.Tensor, list[int]]:
        """Tokenize, run the network, softmax. Returns per-row probabilities and token counts."""
        encoded = self._tokenizer(
            [text[:MAX_INPUT_CHARS] for text in texts],
            truncation=True,
            max_length=self.info.max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self._model(**encoded).logits

        probabilities = torch.softmax(logits.float(), dim=-1).cpu()
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError(f"{self.info.name} produced a non-finite prediction")
        # Real tokens only. These models mask their padding, so padding is not work the model did
        # and must not be counted in the number that accompanies a latency figure.
        token_counts = [int(row.sum()) for row in encoded["attention_mask"].cpu()]
        return probabilities, token_counts

    def _to_prediction(self, row: torch.Tensor, n_tokens: int) -> Prediction:
        probabilities = {name: float(row[i]) for i, name in enumerate(self._labels)}
        return Prediction(
            label=self._labels[int(torch.argmax(row))],
            positive_probability=probabilities[POSITIVE],
            probabilities=probabilities,
            n_tokens=n_tokens,
        )


def resolve_positive_index(manifest: dict[str, Any], name: str) -> int:
    """Read the measured label order out of a manifest, refusing anything ambiguous."""
    raw = manifest.get("label_order")
    order: dict[str, Any] = raw if isinstance(raw, dict) else {}

    # A manifest with no usable label_order and one with a nonsense index fail the same way: the
    # polarity of the checkpoint is unknown, and guessing it is the one thing never to do here.
    index = order.get("positive_index")
    if index not in (0, 1):
        raise ValueError(
            f"'{name}': label_order.positive_index is {index!r}, expected 0 or 1. The polarity of "
            "this checkpoint is not recorded; re-run scripts/fetch_models.py to measure it."
        )

    declared = order.get("id2label")
    if isinstance(declared, dict) and declared.get(str(index)) != POSITIVE:
        # The two halves of the manifest describe different mappings, so neither can be trusted.
        raise ValueError(
            f"'{name}': label_order says class {index} is positive but id2label says "
            f"{declared.get(str(index))!r}"
        )
    return int(index)


def build_bert_classifier(
    manifest: dict[str, Any],
    info: ModelInfo,
    models_dir: Path,
    probe: LabelProbe,
    *,
    device: str = "cpu",
) -> BertClassifier:
    """Build the classifier from a manifest and a local snapshot.

    Raises FileNotFoundError when the snapshot has not been downloaded - the registry turns that
    into a skipped model with the fetch command as its remedy, so a clean clone still starts.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    arch = manifest["arch"]
    if str(arch.get("name")) != SUPPORTED_ARCHITECTURE:
        raise ValueError(f"'{info.name}': unsupported architecture {arch.get('name')!r}")

    model_dir = models_dir / manifest["dir"]
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"no snapshot for '{info.name}' at {model_dir}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), local_files_only=True, output_loading_info=True
        )
    except (OSError, RuntimeError, KeyError) as exc:
        # Broken or half-downloaded snapshot. Normalised to ValueError so behaviour does not depend
        # on which exception type the next version of transformers happens to pick.
        raise ValueError(f"'{info.name}': cannot load the snapshot at {model_dir}: {exc}") from exc

    # In transformers 5 these come back as sets, not lists.
    missing = {key for key in loading_info["missing_keys"] if key.startswith("classifier.")}
    if missing:
        raise ValueError(
            f"'{info.name}': the classifier head is missing from the checkpoint "
            f"({sorted(missing)}). transformers has re-initialised it at random, which scores "
            "about 50% and looks entirely healthy from outside."
        )

    model.eval()
    model.to(device)

    measured = sum(p.numel() for p in model.parameters())
    if info.params is not None and measured != info.params:
        raise ValueError(
            f"'{info.name}': the checkpoint has {measured:,} parameters, the manifest says "
            f"{info.params:,}. The snapshot is not the one that was measured."
        )
    measured_layers = int(model.config.num_hidden_layers)
    if info.n_layers is not None and measured_layers != info.n_layers:
        raise ValueError(
            f"'{info.name}': the checkpoint has {measured_layers} encoder layers, the manifest "
            f"says {info.n_layers}"
        )

    return BertClassifier(info, model, tokenizer, probe, device=device)
