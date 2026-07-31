"""API-level tests: status codes, response shapes, and the layering invariant."""

from __future__ import annotations

import ast
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app

SERVICES_DIR = Path(__file__).resolve().parents[1] / "app" / "services"


def test_health_reports_ok_and_the_app_name() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"]


def test_health_reports_no_models_without_a_registry() -> None:
    """A clean clone has no weights, and the app still answers rather than failing to start."""
    with TestClient(create_app()) as client:
        body = client.get("/health").json()
    assert body["models_loaded"] == 0
    assert body["models_skipped"] == 0


def test_services_layer_does_not_import_fastapi() -> None:
    """The domain layer stays framework-free, so it is testable without a web server.

    Parsed rather than imported: an import-based check would pass simply because the module under
    test happened not to be imported yet.
    """
    offenders: list[str] = []
    for path in SERVICES_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.split(".")[0] in {"fastapi", "starlette"} for name in names):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"services/ must not import FastAPI: {offenders}"
