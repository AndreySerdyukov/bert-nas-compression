"""AdaBERT: the one checkpoint that is not a transformers snapshot.

Differentiable NAS with distillation searched over a small operation set per cell, and every
operation it selected came out parameter-free (`avg_pool`, five times). What shipped is therefore an
embedding table, some pooling along the token axis, a mean, and a linear layer - 7 814 146
parameters, no attention anywhere. The class has to be re-declared here because the artifact on the
Hub is a bare `state_dict`, not a model with a config beside it.

**Two divergent definitions exist, and `strict=True` cannot choose between them.** The notebook that
trained and saved the checkpoint (`notebooks/dnas/`) declares five `FrozenSearchCell`s over the
token axis; the notebook that later evaluated it (`notebooks/results/`) declares the same embedding
and classifier with no cells at all. Because the selected operation carries no parameters, both
definitions produce the identical set of state-dict keys, and both load strictly without a
complaint. Measured on the 2000-review evaluation sample they are not identical: 89.45% with the
cells against 89.25% without, differing on 3 reviews in 400.

So the tie is broken by provenance, not by the loader: the definition here is the one from the
notebook that wrote the file. `strict=True` is still used, and still earns its place - it is what
would catch a checkpoint whose search had selected a convolution, which is why the full operation
table is reproduced below rather than just the pooling.

**Truncation length is part of the model.** This network has no attention mask; it means over every
position it is given, padding included. It was trained at `max_length=128`, and evaluating the same
weights at 512 costs 2.5 points of accuracy (89.45% -> 86.95%) purely through the change in what the
mean is taken over. The length is pinned in the manifest and is not a serving preference.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from app.schemas.models import ModelInfo
from app.serving.base import CLASS_NAMES, POSITIVE, Prediction
from app.serving.bert_classifier import MAX_INPUT_CHARS, PolarityError
from app.serving.probe import LabelProbe

SUPPORTED_ARCHITECTURE = "frozen_adabert"


def _search_ops(hidden_size: int) -> nn.ModuleDict:
    """The search space, reproduced from `notebooks/dnas/Differentiable_NAS.ipynb`.

    All seven are here even though the search selected only `avg_pool`, and the module nesting
    matches the notebook exactly (`nn.Sequential`, same order), because that is what makes
    `strict=True` a real check: a checkpoint holding convolution weights would have keys like
    `cells.0.op.1.weight`, and they have to land somewhere or fail loudly.
    """
    return nn.ModuleDict(
        {
            "conv1": nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
                nn.BatchNorm1d(hidden_size),
            ),
            "conv3": nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_size),
            ),
            "conv5": nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_size),
            ),
            "dil_conv3": nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=2, padding=2),
                nn.BatchNorm1d(hidden_size),
            ),
            "max_pool": nn.Sequential(
                nn.ConstantPad1d((1, 1), 0), nn.MaxPool1d(kernel_size=3, stride=1)
            ),
            "avg_pool": nn.Sequential(
                nn.ConstantPad1d((1, 1), 0), nn.AvgPool1d(kernel_size=3, stride=1)
            ),
            "skip": nn.Identity(),
        }
    )


class FrozenSearchCell(nn.Module):
    """One cell with its search collapsed to the single operation that won."""

    def __init__(self, op: str, hidden_size: int) -> None:
        super().__init__()
        ops = _search_ops(hidden_size)
        if op not in ops:
            raise ValueError(f"unknown operation {op!r}; the search space is {sorted(ops)}")
        # Named `op` because the checkpoint's keys are `cells.<i>.op.*`.
        self.op = ops[op]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.op(x)
        return out


class FrozenAdaBERT(nn.Module):
    """Embeddings -> the selected cells along the token axis -> mean over tokens -> linear."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        selected_ops: Sequence[str] = (),
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.cells = nn.ModuleList([FrozenSearchCell(op, hidden_size) for op in selected_ops])
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # (batch, tokens, hidden) -> (batch, hidden, tokens): the cells are 1-D operations along
        # the token axis, so the channel axis has to come second.
        x = self.embedding(input_ids).transpose(1, 2)
        for cell in self.cells:
            x = cell(x)
        logits: torch.Tensor = self.classifier(x.mean(dim=-1))
        return logits


class AdaBertClassifier:
    """The same serve contract as the transformer wrapper, over a very different network."""

    def __init__(
        self,
        info: ModelInfo,
        model: FrozenAdaBERT,
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
        self._labels = (
            list(CLASS_NAMES) if info.positive_index == 1 else list(reversed(CLASS_NAMES))
        )

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]

    def predict_many(self, texts: Sequence[str]) -> list[Prediction]:
        """Score a batch in one forward pass.

        Padding is `max_length` rather than the batch maximum, and that is not a performance
        choice: with no attention mask, the number of padding positions is part of the arithmetic.
        Padding to the batch maximum would make a review's prediction depend on which other
        reviews happened to be scored alongside it.
        """
        if not texts:
            return []
        encoded = self._tokenizer(
            [text[:MAX_INPUT_CHARS] for text in texts],
            truncation=True,
            max_length=self.info.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self._device)
        with torch.inference_mode():
            logits = self._model(input_ids)

        probabilities = torch.softmax(logits.float(), dim=-1).cpu()
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError(f"{self.info.name} produced a non-finite prediction")

        # Every position counts, padding included, because every position entered the mean. This
        # is the honest token count for this model and it differs in kind from the BERT wrapper's.
        n_tokens = int(input_ids.shape[1])
        return [
            Prediction(
                label=self._labels[int(torch.argmax(row))],
                positive_probability=float(row[self._labels.index(POSITIVE)]),
                probabilities={name: float(row[i]) for i, name in enumerate(self._labels)},
                n_tokens=n_tokens,
            )
            for row in probabilities
        ]

    def warmup(self) -> None:
        """Confirm the polarity the manifest pins is the polarity this state dict actually has."""
        predictions = self.predict_many(self._probe.texts)
        agreement = self._probe.agreement([p.label == POSITIVE for p in predictions])
        if agreement < self._probe.min_agreement:
            raise PolarityError(
                f"{self.info.name}: the label probe agrees with the manifest only "
                f"{agreement:.0%} of the time (need {self._probe.min_agreement:.0%}). "
                "For this model the probe is also the only check that the class definition here "
                "is the one that produced the checkpoint, since the divergent definition loads "
                "just as strictly. Re-run scripts/fetch_models.py."
            )


def build_adabert(
    manifest: dict[str, Any],
    info: ModelInfo,
    models_dir: Path,
    probe: LabelProbe,
    *,
    device: str = "cpu",
) -> AdaBertClassifier:
    """Build AdaBERT from its manifest and the bare state dict beside it."""
    from transformers import AutoTokenizer

    arch = manifest["arch"]
    if str(arch.get("name")) != SUPPORTED_ARCHITECTURE:
        raise ValueError(f"'{info.name}': unsupported architecture {arch.get('name')!r}")

    model_dir = models_dir / manifest["dir"]
    weights = model_dir / str(arch["weights"])
    if not weights.exists():
        raise FileNotFoundError(f"no checkpoint for '{info.name}' at {weights}")

    model = FrozenAdaBERT(
        vocab_size=int(arch["vocab_size"]),
        hidden_size=int(arch["hidden_size"]),
        selected_ops=[str(op) for op in arch["selected_ops"]],
        num_labels=int(arch["num_labels"]),
    )
    # weights_only is torch's default from 2.6 on, and is stated here because this file comes from
    # a third-party repository: a state dict is data, and unpickling it must not execute code.
    state = torch.load(weights, map_location=device, weights_only=True)
    # strict=True: an artifact whose search selected something other than pooling would arrive with
    # keys this definition has nowhere to put, and that has to be loud.
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)

    measured = sum(p.numel() for p in model.parameters())
    if info.params is not None and measured != info.params:
        raise ValueError(
            f"'{info.name}': the checkpoint has {measured:,} parameters, the manifest says "
            f"{info.params:,}"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    return AdaBertClassifier(info, model, tokenizer, probe, device=device)
