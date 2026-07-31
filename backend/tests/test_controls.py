"""The control plan: which runs exist, in what order, and what a night of them costs.

None of this loads a model. What is worth pinning here is the part that decides whether the
comparison is fair - the masks are naive by rule rather than by choice, the order puts the most
informative run first, and the protocol is the notebooks' own rather than one tuned here.
"""

from __future__ import annotations

import pytest

from training.train_reference import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MAX_LENGTH,
    RANDOM_SEEDS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    controls,
    evenly_spaced,
    project,
    random_mask,
)


def test_the_protocol_is_the_notebooks_own() -> None:
    """Read out of the eval notebooks, which agree to the digit. Tuning any of these is cheating.

    A control trained for fewer epochs, on fewer rows or at a shorter sequence than the models it
    is compared against would make every conclusion on this project's front page flattering and
    wrong. Pinned here so a well-meaning speed-up fails a test.
    """
    assert (EPOCHS, MAX_LENGTH, BATCH_SIZE) == (1, 512, 16)
    assert (LEARNING_RATE, WEIGHT_DECAY, WARMUP_RATIO) == (3e-5, 0.01, 0.1)


def test_the_evenly_spaced_mask_is_naive_by_rule() -> None:
    """The whole question is whether search beat this, so it must not be quietly clever."""
    assert evenly_spaced(4) == (0, 4, 7, 11)
    assert evenly_spaced(5) == (0, 3, 6, 8, 11)
    # It always reaches both ends: a "naive" baseline that skipped the last layer would be a
    # different, and weaker, thing to beat.
    for k in (4, 5, 6):
        mask = evenly_spaced(k)
        assert len(mask) == len(set(mask)) == k
        assert mask[0] == 0 and mask[-1] == 11


def test_a_random_mask_is_reproducible_from_its_name() -> None:
    """The seed is in the key, so any row of the results table can be re-run from what it says."""
    assert random_mask(4, 0) == random_mask(4, 0)
    assert random_mask(4, 0) != random_mask(4, 1)
    for seed in RANDOM_SEEDS:
        mask = random_mask(5, seed)
        assert len(set(mask)) == 5
        assert list(mask) == sorted(mask)
        assert all(0 <= index < 12 for index in mask)


def test_the_cheapest_and_most_surprising_control_goes_first() -> None:
    """A run cut short by a laptop lid should still have produced the most informative subset."""
    plan = controls()
    assert plan[0].key == "tfidf"
    assert [c.key for c in plan if c.priority == 2] == ["uniform-4", "uniform-5"]
    # Five seeds per size, because one random mask says nothing about a distribution.
    assert len([c for c in plan if c.key.startswith("random-4")]) == len(RANDOM_SEEDS)
    assert len([c for c in plan if c.key.startswith("random-5")]) == len(RANDOM_SEEDS)
    assert [c.priority for c in plan] == sorted(c.priority for c in plan)


def test_every_control_carries_the_question_it_exists_to_answer() -> None:
    """A control without a question is a number nobody can use."""
    for control in controls():
        assert control.question.endswith("?"), control.key
        assert control.label
        if control.kind == "mask":
            assert control.layers is not None and control.n_layers in (4, 5)


def test_the_projection_scales_with_depth_and_names_the_total() -> None:
    """Two measured depths pin the line; measuring all eighteen would cost as much as running them."""
    projected = project({4: 0.4, 12: 1.2}, steps_per_epoch=1750)
    by_key = {control.key: seconds for control, seconds in projected}

    # 4 layers at 0.4 s a step over 1750 steps.
    assert by_key["uniform-4"] == pytest.approx(0.4 * 1750)
    assert by_key["uniform-5"] == pytest.approx(0.5 * 1750)
    # DistilBERT is six layers of the same width, so it lands on the same line.
    assert by_key["distilbert"] == pytest.approx(0.6 * 1750)
    # The one control that is not a transformer does not get a transformer's projection.
    assert by_key["tfidf"] < 60 * 10
    assert len(projected) == len(controls())
