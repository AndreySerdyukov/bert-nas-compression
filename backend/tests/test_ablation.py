"""Amputating the shared encoder, and putting it back.

The restore is the part worth testing hardest. This mutates the one model object every other
endpoint scores with, so a failure here does not surface as an error - it surfaces as every
subsequent number on the site being quietly taken from a four-layer model.

No weights are needed: the context manager is about object graphs, so a stand-in module with the
same shape exercises it exactly, and these run in CI where the checkpoints do not exist.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from app.services.ablation import wilson
from app.serving.masked_bert import MaskError, amputated, resolve_layers


class _Stub(torch.nn.Module):
    """The shape `amputated` reaches through: `.bert.encoder.layer` and `.config`."""

    def __init__(self, depth: int = 12) -> None:
        super().__init__()
        layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(depth)])
        self.bert = SimpleNamespace(encoder=SimpleNamespace(layer=layers))
        self.config = SimpleNamespace(num_hidden_layers=depth)


def test_the_encoder_is_restored_to_the_same_object_afterwards() -> None:
    """Not merely equivalent: the same ModuleList, so nothing downstream holds a stale reference."""
    model = _Stub()
    original = model.bert.encoder.layer
    identities = [id(layer) for layer in original]

    with amputated(model, [0, 4, 7, 11]) as kept:
        assert kept == (0, 4, 7, 11)
        assert len(model.bert.encoder.layer) == 4
        assert model.config.num_hidden_layers == 4
        # The survivors are the originals, not copies - half a gigabyte per request otherwise.
        assert id(model.bert.encoder.layer[0]) == identities[0]
        assert id(model.bert.encoder.layer[3]) == identities[11]

    assert model.bert.encoder.layer is original
    assert [id(layer) for layer in model.bert.encoder.layer] == identities
    assert model.config.num_hidden_layers == 12


def test_the_encoder_is_restored_when_scoring_raises() -> None:
    """The failure mode this exists to prevent: an exception leaving the model amputated."""
    model = _Stub()
    original = model.bert.encoder.layer

    with pytest.raises(RuntimeError, match="boom"), amputated(model, [0, 1]):
        assert len(model.bert.encoder.layer) == 2
        raise RuntimeError("boom")

    assert model.bert.encoder.layer is original
    assert model.config.num_hidden_layers == 12


def test_a_mask_is_normalised_so_one_architecture_has_one_description() -> None:
    assert resolve_layers([3, 1, 1]) == (1, 3)
    assert resolve_layers([11, 0]) == (0, 11)


@pytest.mark.parametrize(
    ("layers", "message"),
    [([], "at least one layer"), ([12], "outside"), ([-1], "outside"), ([0, 99], "outside")],
)
def test_a_mask_this_model_cannot_have_is_refused_by_name(layers: list[int], message: str) -> None:
    with pytest.raises(MaskError, match=message):
        resolve_layers(layers)


def test_the_stub_depth_is_what_bounds_the_mask() -> None:
    """Checked against the encoder in memory, not against the project's fixed twelve."""
    model = _Stub(depth=6)
    with pytest.raises(MaskError, match="outside the 6"), amputated(model, [7]):
        pass  # pragma: no cover - the context manager raises before the body


def test_wilson_brackets_the_proportion_and_stays_inside_zero_and_one() -> None:
    """The normal approximation runs past the ends exactly where an ablation lands."""
    low, high = wilson(50, 100)
    assert (low, high) == (pytest.approx(0.4038, abs=0.001), pytest.approx(0.5962, abs=0.001))

    # A model that got everything wrong, and one that got everything right. Approx at the top end
    # because the arithmetic lands a float's width under 1 rather than on it.
    assert wilson(0, 100)[0] == 0.0
    assert wilson(100, 100)[1] == pytest.approx(1.0)
    assert wilson(100, 100)[1] <= 1.0
    # Wider on less evidence, which is the whole reason the page prints it.
    assert (wilson(45, 50)[1] - wilson(45, 50)[0]) > (wilson(900, 1000)[1] - wilson(900, 1000)[0])
    assert wilson(0, 0) == (0.0, 0.0)
