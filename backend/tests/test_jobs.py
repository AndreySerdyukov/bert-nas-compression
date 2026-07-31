"""Long-running work: the runner, the stream, and the scan that rides on them.

All of this runs without a checkpoint. That is the point of the `selftest` kind - it exercises
submit, stages, a counter, the stream and a terminal snapshot with no model involved - and it is
why CI can drive the whole progress pipeline through nginx on an image built with no weights.

The one thing these tests cannot show is that the frames arrive as they are produced: `TestClient`
collects the whole response before handing it over, so every frame here has the same arrival time
whatever the server did. Only the Docker stack can show a missing `proxy_buffering off`, which is
where the CI step for it lives.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.repositories.examples import SampleRow
from app.repositories.model_registry import LoadedModel
from app.schemas.models import ModelInfo
from app.services.evaluation import MAX_ROWS_RETURNED, scan_disagreements
from app.services.jobs import (
    JobBusyError,
    JobNotFoundError,
    JobProgress,
    JobRunner,
    selftest_work,
)
from app.services.progress import NullProgress, ProgressReporter
from app.serving.base import Prediction


def frames(client: TestClient, job_id: str) -> list[dict[str, Any]]:
    """Read a job's stream to its terminal frame."""
    collected: list[dict[str, Any]] = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            snapshot = json.loads(line[len("data: ") :])
            collected.append(snapshot)
            if snapshot["status"] != "running":
                break
    return collected


# --- the runner ---------------------------------------------------------------------------------


def test_the_terminal_and_the_browser_reporters_are_the_same_shape() -> None:
    """The evaluation code is written once and runs under both, so this is the whole contract.

    `runtime_checkable` only compares method names, and method names are exactly what would drift:
    `training/progress.py` is a copy that lives in another directory and is edited by hand.
    """
    from training.progress import StageTracker

    job_progress = JobProgress(JobRunner().submit("selftest", lambda _: {}, stages=[]))
    assert isinstance(job_progress, ProgressReporter)
    assert isinstance(NullProgress(), ProgressReporter)
    assert isinstance(StageTracker(["a"], desc="test", enabled=False), ProgressReporter)


def test_a_second_job_is_refused_rather_than_queued() -> None:
    runner = JobRunner()
    runner.submit(
        "selftest", lambda progress: selftest_work(progress, steps=2, delay=0.05), stages=[]
    )
    with pytest.raises(JobBusyError, match="already running"):
        runner.submit("selftest", lambda _: {}, stages=[])


def test_an_unknown_job_is_not_found() -> None:
    with pytest.raises(JobNotFoundError):
        JobRunner().get("nope")


def test_a_failing_job_is_a_result_not_a_crash() -> None:
    runner = JobRunner()

    def explode(_: JobProgress) -> dict[str, Any]:
        raise RuntimeError("the tensor is on the wrong device")

    job = runner.submit("selftest", explode, stages=[])
    # The stream ends on the terminal snapshot whatever the outcome was.
    for _ in runner.stream(job.id):
        pass
    snapshot = job.snapshot()
    assert snapshot.status == "failed"
    assert "wrong device" in (snapshot.error or "")
    assert snapshot.result is None


# --- through the API ----------------------------------------------------------------------------


def test_the_selftest_job_runs_end_to_end_without_a_single_model() -> None:
    """The state a clean clone and CI's docker job are in."""
    with TestClient(create_app()) as client:
        started = client.post("/api/jobs", json={"kind": "selftest"})
        assert started.status_code == 202
        job_id = started.json()["id"]
        assert started.json()["stages"]

        collected = frames(client, job_id)

    assert collected[-1]["status"] == "done"
    assert collected[-1]["result"]["steps"] == 10
    # The stages were named as they were walked, not merely counted.
    assert {snapshot["stage"] for snapshot in collected} >= {"waking up", "counting"}
    # And the nested counter advanced.
    assert max(snapshot["processed"] for snapshot in collected) == 10


def test_a_second_job_over_the_api_is_a_429() -> None:
    with TestClient(create_app()) as client:
        first = client.post("/api/jobs", json={"kind": "selftest"})
        second = client.post("/api/jobs", json={"kind": "selftest"})
        assert second.status_code == 429
        frames(client, first.json()["id"])


def test_a_finished_job_still_answers_and_streams() -> None:
    """A client that connects late gets the terminal frame rather than waiting forever."""
    with TestClient(create_app()) as client:
        job_id = client.post("/api/jobs", json={"kind": "selftest"}).json()["id"]
        frames(client, job_id)

        polled = client.get(f"/api/jobs/{job_id}")
        assert polled.status_code == 200
        assert polled.json()["status"] == "done"
        # The same shape from both transports, so a dropped stream costs a client nothing.
        assert set(polled.json()) == set(frames(client, job_id)[-1])


def test_an_unknown_job_is_404_on_both_transports() -> None:
    with TestClient(create_app()) as client:
        assert client.get("/api/jobs/nope").status_code == 404
        assert client.get("/api/jobs/nope/events").status_code == 404


def test_a_scan_without_models_is_503_naming_the_fetch_script(no_weights_dir: object) -> None:
    with TestClient(create_app()) as client:
        response = client.post("/api/jobs", json={"kind": "disagreement"})
    assert response.status_code == 503
    assert "fetch_models.py" in response.json()["detail"]


# --- the scan itself, with stub predictors -------------------------------------------------------


class StubPredictor:
    """Says whatever it was told to say, row by row."""

    def __init__(self, info: ModelInfo, verdicts: list[str]) -> None:
        self.info = info
        self._verdicts = verdicts

    def predict(self, text: str) -> Prediction:
        raise AssertionError("the scan must batch, not score one review at a time")

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        taken = self._verdicts[: len(texts)]
        self._verdicts = self._verdicts[len(texts) :]
        return [
            Prediction(
                label=verdict,
                positive_probability=1.0 if verdict == "positive" else 0.0,
                probabilities={"negative": 0.0, "positive": 1.0},
                n_tokens=8,
            )
            for verdict in taken
        ]

    def warmup(self) -> None:
        pass


def _model(name: str, verdicts: list[str]) -> LoadedModel:
    info = ModelInfo(
        name=name,
        label=name.title(),
        max_length=512,
        hf_repo=f"someone/{name}",
        hf_revision="0" * 40,
        positive_index=1,
    )
    return LoadedModel(predictor=StubPredictor(info, verdicts), info=info)


def test_the_scan_keeps_the_rows_that_split_the_models() -> None:
    """A percentage is not the deliverable here. The rows are."""
    rows = [SampleRow(id=i, text=f"review {i}", label=i % 2) for i in range(10)]
    truth = ["negative" if i % 2 == 0 else "positive" for i in range(10)]
    # The baseline is right about everything; the challenger is wrong about rows 3 and 7.
    challenger = list(truth)
    challenger[3] = "negative"
    challenger[7] = "negative"

    progress = NullProgress()
    report = scan_disagreements(
        {"bert-imdb": _model("bert-imdb", truth), "bananas": _model("bananas", challenger)},
        rows,
        baseline="bert-imdb",
        progress=progress,
    )

    assert report["rows_scanned"] == 10
    assert report["disagreements_found"] == 2
    assert [entry["id"] for entry in report["disagreements"]] == [3, 7]
    assert report["disagreements"][0]["verdicts"] == {
        "bert-imdb": "positive",
        "bananas": "negative",
    }

    summaries = {entry["name"]: entry for entry in report["models"]}
    assert summaries["bert-imdb"]["accuracy"] == 1.0
    assert summaries["bananas"]["accuracy"] == 0.8
    # The baseline is not compared against itself.
    assert summaries["bert-imdb"]["agreement"] is None
    assert summaries["bananas"]["agreement"] == 0.8
    # The stages said what they were doing, by name.
    assert any("scored" in message for message in progress.messages)


def test_the_scan_reports_what_it_left_out() -> None:
    """A cap that is not stated reads as "that was all of them"."""
    n = MAX_ROWS_RETURNED + 5
    rows = [SampleRow(id=i, text=f"review {i}", label=1) for i in range(n)]
    report = scan_disagreements(
        {
            "bert-imdb": _model("bert-imdb", ["positive"] * n),
            "bananas": _model("bananas", ["negative"] * n),
        },
        rows,
        baseline="bert-imdb",
        progress=NullProgress(),
    )

    assert report["disagreements_found"] == n
    assert report["disagreements_shown"] == MAX_ROWS_RETURNED
    assert len(report["disagreements"]) == MAX_ROWS_RETURNED


def test_the_scan_refuses_to_run_on_nothing() -> None:
    with pytest.raises(ValueError, match="no rows"):
        scan_disagreements(
            {"bert-imdb": _model("bert-imdb", [])},
            [],
            baseline="bert-imdb",
            progress=NullProgress(),
        )
    with pytest.raises(ValueError, match="no models"):
        scan_disagreements(
            {}, [SampleRow(id=1, text="a", label=1)], baseline="bert-imdb", progress=NullProgress()
        )
