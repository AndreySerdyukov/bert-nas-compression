"""The committed JSON documents the methodology section reads.

This router is the documented exception to "routers hold no logic": the documents are served
straight from memory as bytes, so there is no service layer to route through. The same choice
billboard-planner made for its district polygons, and for the same reason - re-parsing a static
document on every request buys nothing.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response

from app.repositories.content import DOCUMENTS

router = APIRouter(prefix="/api")

# These never change without a redeploy, and the methodology pages fetch several of them.
CACHE_CONTROL = "public, max-age=3600"


def _document(request: Request, name: str) -> Response:
    repository = request.app.state.content
    payload = repository.get(name)
    if payload is None:
        # Only benchmark.json can realistically be missing, and when it is, the answer says what
        # produces it rather than pretending the document is unknown.
        raise HTTPException(
            status_code=503,
            detail=(
                f"{DOCUMENTS[name]} has not been generated yet. "
                "Run training/benchmark.py to produce it."
            ),
        )
    return Response(
        content=payload, media_type="application/json", headers={"Cache-Control": CACHE_CONTROL}
    )


@router.get("/architectures")
def architectures(request: Request) -> Response:
    """What each method shipped, read from the eval notebooks, with its search's own conclusion."""
    return _document(request, "architectures")


@router.get("/trajectories")
def trajectories(request: Request) -> Response:
    """Every candidate the three searches evaluated, parsed from their selection notebooks."""
    return _document(request, "trajectories")


@router.get("/reported-results")
def reported_results(request: Request) -> Response:
    """The four disagreeing source tables, with citations. Chapter 9 renders this as a matrix."""
    return _document(request, "reported-results")


@router.get("/benchmark")
def benchmark(request: Request) -> Response:
    """This project's own measurements. 503 until training/benchmark.py has been run."""
    return _document(request, "benchmark")
