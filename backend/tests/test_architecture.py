"""The mask arithmetic. No torch, no weights, no network - these run everywhere in milliseconds."""

from __future__ import annotations

import itertools
import json
from math import comb
from pathlib import Path

import pytest

from app.services import architecture as arch

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# The parameter ladder, pinned exactly. Every one of these appears in the notebooks' own output,
# so this table is simultaneously a unit test and a transcription check on the source material.
PARAMS_BY_LAYER_COUNT = {
    12: 109_483_778,
    10: 95_308_034,
    9: 88_220_162,
    7: 74_044_418,
    6: 66_956_546,
    5: 59_868_674,
    4: 52_780_802,
}


@pytest.mark.parametrize("n_layers,expected", sorted(PARAMS_BY_LAYER_COUNT.items()))
def test_the_parameter_ladder_is_exact(n_layers: int, expected: int) -> None:
    assert arch.params_for(n_layers) == expected


def test_the_formula_holds_for_every_mask() -> None:
    """All 4 096 masks, not a sample: the formula is the product's central claim."""
    for bits in itertools.product((0, 1), repeat=arch.N_LAYERS):
        cost = arch.describe(bits)
        assert cost.params == arch.NON_LAYER_PARAMS + arch.PARAMS_PER_LAYER * sum(bits)
        assert cost.n_layers == sum(bits)


def test_an_all_zero_mask_is_priced_rather_than_refused() -> None:
    """Embeddings plus a classifier is a real architecture, and a useful floor to show."""
    cost = arch.describe([0] * 12)
    assert cost.n_layers == 0
    assert cost.params == arch.NON_LAYER_PARAMS
    assert cost.layers == ()


@pytest.mark.parametrize(
    "bad",
    [
        [1] * 11,
        [1] * 13,
        "111111111111",
        [2] + [0] * 11,
        [-1] + [0] * 11,
        [None] + [0] * 11,
    ],
)
def test_malformed_masks_are_rejected(bad: object) -> None:
    with pytest.raises(arch.InvalidMaskError):
        arch.describe(bad)


def test_booleans_are_accepted_because_json_clients_send_them() -> None:
    assert arch.describe([True, True] + [False] * 10).layers == (0, 1)


def test_layers_and_masks_round_trip() -> None:
    for layers in [(0,), (0, 1, 5, 7, 9), (1, 2, 6, 11), tuple(range(12))]:
        assert arch.layers_from_mask(arch.mask_from_layers(list(layers))) == layers


@pytest.mark.parametrize("bad", [[12], [-1], [1.5], [True]])
def test_out_of_range_layer_indices_are_rejected(bad: object) -> None:
    with pytest.raises(arch.InvalidMaskError):
        arch.mask_from_layers(bad)


# --- what the searches shipped -------------------------------------------------------------


def test_the_shipped_masks_map_to_their_documented_layers() -> None:
    expected = {
        "random-search": (0, 1, 5, 7, 9),
        "alphanas": (1, 2, 6, 11),
        "bananas": (0, 1, 6, 9),
    }
    data = json.loads((DATA_DIR / "architectures.json").read_text(encoding="utf-8"))
    for name, layers in expected.items():
        assert tuple(data["methods"][name]["shipped"]["layers"]) == layers
        assert arch.matches_known_architecture(arch.mask_from_layers(list(layers))) == name


def test_random_search_ships_a_different_mask_from_the_one_it_reported() -> None:
    """The disagreement is pinned as a test so it cannot be silently "corrected" later.

    The selection notebook and its tech report both name [0, 1, 6, 8, 10]. The notebook that
    trained and uploaded the checkpoint set [0, 1, 5, 7, 9]. Both keep five layers, so no table of
    parameter counts ever noticed. `matches_known_architecture` answers for the model that exists.
    """
    shipped = arch.mask_from_layers([0, 1, 5, 7, 9])
    reported = arch.mask_from_layers([0, 1, 6, 8, 10])

    assert arch.matches_known_architecture(shipped) == "random-search"
    assert arch.matches_known_architecture(reported) is None
    assert arch.params_for(5) == arch.describe(reported).params  # which is why it went unnoticed

    data = json.loads((DATA_DIR / "architectures.json").read_text(encoding="utf-8"))
    assert data["methods"]["random-search"]["shipped_matches_search"] is False
    assert data["methods"]["alphanas"]["shipped_matches_search"] is True
    assert data["methods"]["bananas"]["shipped_matches_search"] is True


def test_adabert_carries_no_mask_and_says_why() -> None:
    data = json.loads((DATA_DIR / "architectures.json").read_text(encoding="utf-8"))
    adabert = data["methods"]["adabert"]
    assert adabert["shipped"]["mask"] is None
    # 30522*256 + 256*2 + 2 - every searched operation was parameter-free, so this is the model.
    assert adabert["shipped"]["params"] == 7_814_146
    assert "avg_pool" in adabert["note"]


def test_every_extracted_candidate_agrees_with_the_formula() -> None:
    """The searches printed a parameter count next to every candidate; all 26 must match."""
    data = json.loads((DATA_DIR / "search_trajectories.json").read_text(encoding="utf-8"))
    seen = 0
    for method in data["methods"].values():
        for candidate in method["candidates"]:
            assert candidate["params"] == arch.params_for(candidate["n_layers"])
            assert arch.layers_from_mask(tuple(candidate["mask"])) == tuple(candidate["layers"])
            seen += 1
    assert seen == 26


# --- FLOPs and the search space -------------------------------------------------------------


def test_flops_grow_with_layers_and_with_sequence_length() -> None:
    assert arch.flops_for(4, 256) < arch.flops_for(5, 256) < arch.flops_for(12, 256)
    assert arch.flops_for(4, 128) < arch.flops_for(4, 256) < arch.flops_for(4, 512)


def test_attention_is_quadratic_so_doubling_the_length_more_than_doubles_the_cost() -> None:
    """If this ever became linear, the sequence-length slider would be telling a different story."""
    assert arch.flops_for(12, 512) > 2 * arch.flops_for(12, 256)


def test_zero_layers_still_costs_the_classifier_head() -> None:
    assert arch.flops_for(0, 256) > 0


@pytest.mark.parametrize("seq_len", [0, -1, 513])
def test_out_of_range_sequence_lengths_are_rejected(seq_len: int) -> None:
    with pytest.raises(arch.InvalidMaskError):
        arch.flops_for(4, seq_len)


def test_the_searched_space_holds_3797_architectures() -> None:
    """Random Search evaluated three of them, which is the whole point of chapter 2."""
    assert arch.search_space_size(4, 12) == 3797
    assert arch.search_space_size(0, 12) == 2**12
    assert arch.search_space_size(4, 4) == comb(12, 4) == 495
