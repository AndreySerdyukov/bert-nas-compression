"""The control plan: which runs exist, in what order, and what a night of them costs.

None of this loads a model. What is worth pinning here is the part that decides whether the
comparison is fair - the masks are naive by rule rather than by choice, the order puts the most
informative run first, and the protocol is the notebooks' own rather than one tuned here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from training.train_reference import (
    BATCH_SIZE,
    DISTILBERT_LAYERS,
    EPOCHS,
    LEARNING_RATE,
    MAX_LENGTH,
    PROBE_DEPTHS,
    RANDOM_SEEDS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    DepthCost,
    _write,
    controls,
    evenly_spaced,
    print_schedule,
    project,
    random_mask,
    run_seed,
    seed_everything,
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
    """The seed is in the key, so the *mask* of any row can be re-drawn from what the row says.

    Only the mask. The number beside it needs the training seed as well, which is what `run_seed`
    is for - and which nothing pinned until 2026-08-01, so the committed results were produced
    under an unseeded head initialisation and an unseeded shuffle.
    """
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


def test_every_depth_in_the_plan_is_a_depth_the_probe_measures() -> None:
    """The projection reads a measurement per depth, so the probe has to cover the plan.

    A control at a depth nobody probed would be priced off the nearest one, which is how the
    first version of this - a line drawn from 4 layers to 12 - overstated a night by half.
    """
    needed = {
        control.n_layers if control.n_layers is not None else DISTILBERT_LAYERS
        for control in controls()
        if control.kind != "tfidf"
    }
    assert needed <= set(PROBE_DEPTHS)


def test_each_control_is_projected_from_its_own_measured_depth() -> None:
    """No interpolation: 5 layers costs what 5 layers was measured to cost, not what a line says."""
    measured = {
        4: DepthCost(train_seconds_per_step=0.4, eval_seconds_per_row=0.0),
        5: DepthCost(train_seconds_per_step=0.55, eval_seconds_per_row=0.0),
        6: DepthCost(train_seconds_per_step=0.9, eval_seconds_per_row=0.0),
    }
    projected = project(measured, steps_per_epoch=1750, test_rows=15_000)
    by_key = {control.key: seconds for control, seconds in projected}

    assert by_key["uniform-4"] == pytest.approx(0.4 * 1750)
    # A fitted line through 4 and 6 would say 0.65 here. The measurement says 0.55, and the
    # measurement wins - that gap over seven five-layer runs is an hour of a night.
    assert by_key["uniform-5"] == pytest.approx(0.55 * 1750)
    assert by_key["distilbert"] == pytest.approx(0.9 * 1750)
    # The one control that is not a transformer does not get a transformer's projection.
    assert by_key["tfidf"] < 60 * 10
    assert len(projected) == len(controls())


def test_the_schedule_counts_the_scoring_pass_every_control_pays() -> None:
    """Omitted at first, and it is an hour over seventeen controls - the tail of the priority list.

    A schedule that counts only training reports that everything fits in a night, and the runs it
    then drops without saying so are the last ones planned, which is the whole point of the order.
    """
    training_only = {4: DepthCost(train_seconds_per_step=0.4, eval_seconds_per_row=0.0)}
    with_scoring = {4: DepthCost(train_seconds_per_step=0.4, eval_seconds_per_row=0.01)}

    bare = {
        control.key: seconds
        for control, seconds in project(training_only, steps_per_epoch=1000, test_rows=15_000)
    }
    full = {
        control.key: seconds
        for control, seconds in project(with_scoring, steps_per_epoch=1000, test_rows=15_000)
    }
    assert full["uniform-4"] - bare["uniform-4"] == pytest.approx(150.0)
    # The tf-idf control is not scored on a GPU and does not pick up the transformer's eval cost.
    assert full["tfidf"] == bare["tfidf"]


def test_the_cooling_pause_goes_between_runs_and_into_the_total(capsys: Any) -> None:
    """Idle time is still time, and a schedule that hid it would promise a night it cannot deliver.

    Seventeen gaps for eighteen controls, not eighteen: nobody waits after the last one. At the
    900 s a cooling pause intuitively wants to be, those gaps are 4.25 h - which is why the flag
    documents 120 s instead.
    """
    measured = {4: DepthCost(train_seconds_per_step=1.0, eval_seconds_per_row=0.0)}
    projected = project(measured, steps_per_epoch=60, test_rows=0)

    print_schedule(projected, budget_hours=None, cooldown=0)
    without = capsys.readouterr().out
    print_schedule(projected, budget_hours=None, cooldown=900)
    with_cooling = capsys.readouterr().out

    gaps = len(projected) - 1
    assert f"of which {gaps * 900 / 3600:.1f} h is cooling" in with_cooling
    assert "cooling" not in without
    # The work itself is untouched: the pause moves the finish line, not any control's cost.
    assert without.splitlines()[2] == with_cooling.splitlines()[2]


def test_a_missing_depth_falls_back_to_the_nearest_measurement() -> None:
    """An interrupted probe should still produce a schedule, and an approximate one it can name."""
    measured = {4: DepthCost(train_seconds_per_step=0.4, eval_seconds_per_row=0.0)}
    projected = project(measured, steps_per_epoch=1000, test_rows=15_000)
    by_key = {control.key: seconds for control, seconds in projected}
    assert by_key["uniform-5"] == pytest.approx(0.4 * 1000)


def test_each_control_trains_from_its_own_pinned_seed() -> None:
    """Reproducibility of the number, not just of the mask.

    Two controls sharing a seed would share a classifier initialisation, which is a correlation
    between rows that are supposed to be independent observations. Derived from the key rather
    than from `hash()`, which is salted per process and would give a different run every time.
    """
    seeds = {control.key: run_seed(control.key) for control in controls()}
    assert len(set(seeds.values())) == len(seeds)
    assert run_seed("uniform-4") == run_seed("uniform-4")
    assert all(0 <= seed < 2**32 for seed in seeds.values())


def test_seeding_pins_the_head_initialisation_and_the_shuffle() -> None:
    """The two things that were loose. Torch only, since sklearn's control takes no seed here."""
    torch = pytest.importorskip("torch")

    seed_everything(run_seed("uniform-4"))
    first = torch.randn(4)
    seed_everything(run_seed("uniform-4"))
    assert torch.equal(first, torch.randn(4))

    seed_everything(run_seed("uniform-5"))
    assert not torch.equal(first, torch.randn(4))


# --- writing the results file --------------------------------------------------------------------


class _Split:
    """Enough of a Split for `_write`: it only reads the sizes and the test digest."""

    def __init__(self, digest: str = "a" * 64) -> None:
        self.train = [0] * 28_000
        self.test = [0] * 15_000
        self.test_index_sha256 = digest


def _result(key: str, accuracy: float) -> dict[str, Any]:
    return {"key": key, "label": key, "accuracy": accuracy, "train": {"seconds": 1.0}}


def test_rerunning_one_control_keeps_the_other_seventeen(tmp_path: Path) -> None:
    """The defect this merge exists for: `--only` used to rewrite the file from the current run.

    Nine hours of GPU time lived in that file and one documented command discarded it silently.
    So a partial run has to fold into what is already there, and `planned` has to stay the whole
    plan rather than shrinking to whatever this invocation happened to select.
    """
    out = tmp_path / "controls.json"
    plan = controls()
    _write(out, plan, [_result(c.key, 0.9) for c in plan], _Split(), "mps")
    assert len(json.loads(out.read_text())["controls"]) == len(plan)

    # The `--only tfidf` shape: one control selected, one result, run against the same protocol.
    _write(out, [plan[0]], [_result("tfidf", 0.9141)], _Split(), "mps")

    written = json.loads(out.read_text())
    assert len(written["controls"]) == len(plan)
    assert written["planned"] == [control.key for control in plan]
    assert written["not_run"] == []
    rerun = next(entry for entry in written["controls"] if entry["key"] == "tfidf")
    assert rerun["accuracy"] == 0.9141
    # Plan order, not arrival order: the file reads as the schedule it came from.
    assert [entry["key"] for entry in written["controls"]] == [control.key for control in plan]


def test_a_partial_file_names_every_control_that_has_not_run(tmp_path: Path) -> None:
    """A file with two results out of eighteen has to say so, or it reads as "the controls"."""
    out = tmp_path / "controls.json"
    plan = controls()
    _write(out, plan, [_result("tfidf", 0.91), _result("uniform-4", 0.89)], _Split(), "mps")

    written = json.loads(out.read_text())
    assert len(written["controls"]) == 2
    assert len(written["not_run"]) == len(plan) - 2
    assert "distilbert" in written["not_run"]


def test_results_from_another_split_are_refused_rather_than_mixed_in(tmp_path: Path) -> None:
    """Two protocols in one file look exactly like one protocol, which is the whole problem."""
    out = tmp_path / "controls.json"
    _write(out, controls(), [_result("tfidf", 0.91)], _Split(digest="a" * 64), "mps")

    with pytest.raises(SystemExit, match="different protocol"):
        _write(out, controls(), [_result("tfidf", 0.92)], _Split(digest="b" * 64), "mps")

    # And the file it refused to touch is untouched.
    assert json.loads(out.read_text())["controls"][0]["accuracy"] == 0.91


def test_replace_is_the_explicit_way_to_start_over(tmp_path: Path) -> None:
    """Merging is the default; discarding the file has to be asked for by name."""
    out = tmp_path / "controls.json"
    plan = controls()
    _write(out, plan, [_result(c.key, 0.9) for c in plan], _Split(), "mps")
    _write(out, [plan[0]], [_result("tfidf", 0.91)], _Split(), "mps", replace=True)

    written = json.loads(out.read_text())
    assert [entry["key"] for entry in written["controls"]] == ["tfidf"]
    assert written["planned"] == ["tfidf"]
