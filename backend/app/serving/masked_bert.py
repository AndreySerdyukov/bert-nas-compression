"""Amputating a fine-tuned encoder in place, and putting it back.

This is the machinery behind the explorer, and the operation it performs is deliberately **not**
the one NAS performed. A search does `finetune(mask(pretrained))`: it removes layers from a
pre-trained encoder and then trains what is left, so the surviving layers get to adapt. The explorer
does `mask(finetuned)` and stops - no retraining, because retraining a candidate is the expensive
half and skipping it is the whole point of what this measures. Accuracy falls off a cliff, and that
cliff is the evidence that training each candidate is the irreducible cost of architecture search.

**Nothing is copied.** The surviving layers are the same `nn.Module` objects, rebound into a new
`ModuleList`, and the original list is restored in `finally`. A copy would be half a gigabyte per
request and would make the explorer the most expensive page in the app; a mutation without the
`finally` would leave the model that serves `/api/compare` permanently four layers deep, and every
number after it silently wrong.

The caller must hold the inference lock. This mutates the shared model that every other endpoint
scores with, so an overlapping request would be served by an amputated encoder without knowing it.
`InferenceService` owns that lock and is the only caller.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch

# The architecture every mask in this project is defined over.
ENCODER_LAYERS = 12


class MaskError(ValueError):
    """The requested layers are not ones this particular module has."""


def resolve_layers(layers: Sequence[int], *, total: int = ENCODER_LAYERS) -> tuple[int, ...]:
    """Normalise requested layer indices against the module that will be cut.

    Distinct from `services.architecture.validate_mask`, which checks a mask against this project's
    fixed twelve-layer search space. This one checks it against the encoder actually in memory, so
    a model with a different depth cannot be silently indexed past its end.

    Returned sorted and deduplicated: `[3, 1, 1]` and `[1, 3]` are the same network, and letting
    both through would produce two descriptions of one architecture.
    """
    if not layers:
        raise MaskError("a mask needs at least one layer; an encoder with none is not a model")
    outside = sorted({index for index in layers if not 0 <= index < total})
    if outside:
        raise MaskError(f"layers {outside} are outside the {total} this model has (0-{total - 1})")
    return tuple(sorted(set(layers)))


@contextmanager
def amputated(module: Any, layers: Sequence[int]) -> Iterator[tuple[int, ...]]:
    """Keep only `layers` for the duration of the block, then restore the encoder exactly.

    Restoration happens in `finally` and rebinds the original `ModuleList` object rather than
    rebuilding one, so the model is not merely equivalent afterwards - it is the same object graph
    the process started with.
    """
    kept = resolve_layers(layers, total=len(module.bert.encoder.layer))

    encoder = module.bert.encoder
    original_layers = encoder.layer
    original_depth = module.config.num_hidden_layers
    try:
        encoder.layer = torch.nn.ModuleList([original_layers[index] for index in kept])
        module.config.num_hidden_layers = len(kept)
        yield kept
    finally:
        encoder.layer = original_layers
        module.config.num_hidden_layers = original_depth
