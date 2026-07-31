"""The committed-document endpoints, including the one that is legitimately absent."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(create_app()) as test_client:
        yield test_client


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
    client: TestClient, document: str, script: str
) -> None:
    """Nine of the ten chapters work without either, so absence is a 503, not a broken app.

    Each names its own producer. One shared message that named the benchmark script would send a
    reader to run the wrong thing for hours.
    """
    response = client.get(f"/api/{document}")
    if response.status_code == 200:
        pytest.skip(f"{document}.json has been generated on this machine")
    assert response.status_code == 503
    assert script in response.json()["detail"]


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
