"""The arithmetic of a layer mask: parameters, FLOPs, and which known architecture it is.

This module is the reason the architecture explorer feels instant. Everything here is closed-form -
the cost of an architecture depends only on which encoder layers it keeps, never on its weights - so
the whole cost side of the compression trade can be answered without loading a model, or indeed
without any model existing. Quality is the other half, and that one needs a GPU and a training run;
the asymmetry is the point the methodology section is built around.

`frontend/src/lib/arch.ts` is the same arithmetic in TypeScript, so the UI needs no round trip. The
two are kept honest by pinning the identical table of values in both test suites.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# BERT-base geometry. None of it was ever searched - the four methods vary only which layers to
# keep - so these are constants rather than configuration.
N_LAYERS = 12
HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 3072
VOCAB_SIZE = 30522
MAX_POSITION_EMBEDDINGS = 512

# Derived once from a real checkpoint and pinned: a full 12-layer model reports 109 483 778
# parameters, and each encoder layer accounts for 7 087 872 of them. What is left - token, position
# and type embeddings, their LayerNorm, the pooler and the 2-way classifier - is the same whatever
# the mask does.
PARAMS_PER_LAYER = 7_087_872
NON_LAYER_PARAMS = 24_429_314

BYTES_PER_FP32 = 4

MIN_LAYERS_SEARCHED = 4
MAX_LAYERS_SEARCHED = 12

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class InvalidMaskError(ValueError):
    """The mask is not twelve entries of 0 or 1."""


@dataclass(frozen=True)
class ArchitectureCost:
    """What a mask costs. Everything here is arithmetic, not measurement."""

    mask: tuple[int, ...]
    layers: tuple[int, ...]
    n_layers: int
    params: int
    params_pct_of_base: float
    flops_per_example: int
    bytes_fp32: int
    matches_known: str | None


def validate_mask(mask: object) -> tuple[int, ...]:
    """Coerce input into a 12-entry 0/1 tuple, or raise InvalidMaskError.

    An all-zero mask is allowed: embeddings plus a classifier is a real (bad) architecture, and the
    explorer should be able to price it rather than refuse it.
    """
    if not isinstance(mask, (list, tuple)):
        raise InvalidMaskError("mask must be a list of 12 entries of 0 or 1")
    if len(mask) != N_LAYERS:
        raise InvalidMaskError(f"mask must have {N_LAYERS} entries, got {len(mask)}")
    coerced: list[int] = []
    for entry in mask:
        # bool is an int subclass, so True/False are accepted deliberately - JSON clients send both.
        if isinstance(entry, bool):
            coerced.append(int(entry))
        elif isinstance(entry, int) and entry in (0, 1):
            coerced.append(entry)
        else:
            raise InvalidMaskError(f"mask entries must be 0 or 1, got {entry!r}")
    return tuple(coerced)


def layers_from_mask(mask: tuple[int, ...]) -> tuple[int, ...]:
    """Indices of the kept encoder layers, ascending."""
    return tuple(i for i, bit in enumerate(mask) if bit)


def mask_from_layers(layers: object) -> tuple[int, ...]:
    """Inverse of `layers_from_mask`, for callers that think in layer indices."""
    if not isinstance(layers, (list, tuple)):
        raise InvalidMaskError("layers must be a list of indices")
    mask = [0] * N_LAYERS
    for index in layers:
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < N_LAYERS:
            raise InvalidMaskError(f"layer index out of range: {index!r}")
        mask[index] = 1
    return tuple(mask)


def params_for(n_layers: int) -> int:
    """Parameter count of a classifier keeping `n_layers` encoder layers.

    The whole compression story is this one line: layers are the only thing the searches varied,
    and each one costs exactly the same.
    """
    if not 0 <= n_layers <= N_LAYERS:
        raise InvalidMaskError(f"n_layers must be between 0 and {N_LAYERS}, got {n_layers}")
    return NON_LAYER_PARAMS + PARAMS_PER_LAYER * n_layers


def flops_per_layer(seq_len: int) -> int:
    """Multiply-accumulate FLOPs for one encoder layer at `seq_len` tokens.

    Counts the matrix multiplications only - the four attention projections, the two attention
    batched matmuls, and the two feed-forward projections - at two FLOPs per multiply-accumulate.
    Softmax, LayerNorm, GELU and the residual adds are elementwise and contribute a percent or so;
    leaving them out is the usual convention and keeps the number comparable to published ones.

    The quadratic term is why the sequence-length slider in the playground changes the shape of the
    trade and not just its scale.
    """
    attention_projections = 4 * seq_len * HIDDEN_SIZE * HIDDEN_SIZE
    attention_scores = 2 * seq_len * seq_len * HIDDEN_SIZE
    feed_forward = 2 * seq_len * HIDDEN_SIZE * INTERMEDIATE_SIZE
    return 2 * (attention_projections + attention_scores + feed_forward)


def flops_for(n_layers: int, seq_len: int) -> int:
    """Encoder FLOPs for a whole forward pass, plus the classifier head."""
    if seq_len < 1 or seq_len > MAX_POSITION_EMBEDDINGS:
        raise InvalidMaskError(f"seq_len must be between 1 and {MAX_POSITION_EMBEDDINGS}")
    head = 2 * (HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * 2)  # pooler + classifier
    return n_layers * flops_per_layer(seq_len) + head


@lru_cache
def _known_architectures() -> dict[tuple[int, ...], str]:
    """Shipped masks, keyed by mask, from the committed architectures.json.

    Read from the file rather than hardcoded so this cannot disagree with what
    training/extract_notebook_data.py read out of the eval notebooks.
    """
    path = _DATA_DIR / "architectures.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    known: dict[tuple[int, ...], str] = {}
    for name, method in data["methods"].items():
        mask = method["shipped"]["mask"]
        if mask is not None:
            known[tuple(mask)] = name
    return known


def matches_known_architecture(mask: tuple[int, ...]) -> str | None:
    """The method that shipped this exact mask, or None.

    Note what this does *not* match: the mask Random Search's write-up names, `[0,1,6,8,10]`. The
    model on the Hub is `[0,1,5,7,9]`, and this function answers for the model, because that is
    what a user comparing their own ablation against a published number needs.
    """
    return _known_architectures().get(mask)


def describe(mask: object, seq_len: int = 256) -> ArchitectureCost:
    """Full cost breakdown for a mask. Pure arithmetic - no torch, no weights, no I/O."""
    validated = validate_mask(mask)
    layers = layers_from_mask(validated)
    params = params_for(len(layers))
    return ArchitectureCost(
        mask=validated,
        layers=layers,
        n_layers=len(layers),
        params=params,
        params_pct_of_base=params / params_for(N_LAYERS),
        flops_per_example=flops_for(len(layers), seq_len),
        bytes_fp32=params * BYTES_PER_FP32,
        matches_known=matches_known_architecture(validated),
    )


def search_space_size(
    min_layers: int = MIN_LAYERS_SEARCHED, max_layers: int = MAX_LAYERS_SEARCHED
) -> int:
    """How many masks keep between `min_layers` and `max_layers` layers.

    For the 4-12 range the searches used: 3 797. Random Search evaluated three of them.
    """
    from math import comb

    if not 0 <= min_layers <= max_layers <= N_LAYERS:
        raise InvalidMaskError("layer bounds must satisfy 0 <= min <= max <= 12")
    return sum(comb(N_LAYERS, k) for k in range(min_layers, max_layers + 1))
