"""The committed-document endpoints, including the one that is legitimately absent."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"

# Parsed out of the notebooks and committed, so they are present in every clone.
SPINE = ("architectures.json", "search_trajectories.json", "reported_results.json")


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(create_app()) as test_client:
        yield test_client


def _client_over(data: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("APP_DATA_DIR", str(data))
    get_settings.cache_clear()
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def without_measurements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """The spine, and neither measured document.

    Constructed rather than assumed, for the same reason `no_weights_dir` is: whether this machine
    has run the benchmark is exactly what the suite cannot know. The earlier version of this test
    read the real `data/` and skipped when it found a file - which meant that once both documents
    were committed, it skipped in every environment including CI, and a test that always skips is
    a test that does not exist.
    """
    data = tmp_path / "data"
    data.mkdir()
    for name in SPINE:
        shutil.copy(DATA_DIR / name, data / name)
    yield from _client_over(data, monkeypatch)


@pytest.fixture
def with_measurements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """The other side of the same switch: both documents present, whatever this machine has run."""
    data = tmp_path / "data"
    data.mkdir()
    for name in SPINE:
        shutil.copy(DATA_DIR / name, data / name)
    for name in ("benchmark.json", "controls.json"):
        (data / name).write_text(json.dumps({"schema": 1, "models": []}), encoding="utf-8")
    yield from _client_over(data, monkeypatch)


@pytest.mark.parametrize(
    "path", ["/api/architectures", "/api/trajectories", "/api/reported-results"]
)
def test_the_committed_documents_are_served(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "public, max-age=3600"
    assert isinstance(response.json(), dict)


@pytest.mark.parametrize(
    ("document", "script"),
    [("benchmark", "training.benchmark"), ("controls", "training.train_reference")],
)
def test_a_measured_document_says_what_to_run_when_it_is_missing(
    without_measurements: TestClient, document: str, script: str
) -> None:
    """Nine of the ten chapters work without either, so absence is a 503, not a broken app.

    Each names its own producer. One shared message that named the benchmark script would send a
    reader to run the wrong thing for hours.
    """
    response = without_measurements.get(f"/api/{document}")
    assert response.status_code == 503
    assert script in response.json()["detail"]


@pytest.mark.parametrize("document", ["benchmark", "controls"])
def test_a_measured_document_is_served_once_it_exists(
    with_measurements: TestClient, document: str
) -> None:
    """And the same endpoint answers 200 when the file is there, on any machine."""
    response = with_measurements.get(f"/api/{document}")
    assert response.status_code == 200
    assert response.json()["schema"] == 1


def test_architectures_carries_the_random_search_disagreement(client: TestClient) -> None:
    """The finding has to survive the trip through the API, not just exist in the file."""
    methods = client.get("/api/architectures").json()["methods"]
    assert methods["random-search"]["shipped"]["layers"] == [0, 1, 5, 7, 9]
    assert methods["random-search"]["shipped_matches_search"] is False


def test_trajectories_carry_the_surrogate_proposals(client: TestClient) -> None:
    """Chapter 6 plots predicted fitness against what the candidate actually scored."""
    bananas = client.get("/api/trajectories").json()["methods"]["bananas"]
    assert len(bananas["proposals"]) == 50
    assert {"round", "mask", "predicted_fitness"} <= set(bananas["proposals"][0])


def test_the_documents_are_byte_identical_to_what_is_committed(client: TestClient) -> None:
    """Served as bytes on purpose; this pins that nothing re-serialises them on the way out."""
    from app.config import get_settings

    committed = (get_settings().data_dir / "architectures.json").read_bytes()
    assert client.get("/api/architectures").content == committed
    assert json.loads(committed)["schema"] == 1
