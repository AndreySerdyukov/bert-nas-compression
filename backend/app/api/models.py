"""The scoring router: the catalog, one model on one review, and all of them side by side.

Thin by rule. It pulls the service off the application state, calls it, and translates the domain's
exceptions into status codes. The codes carry information, so they are worth stating:

    404  no manifest carries that name - the caller made it up
    422  there is nothing to score, or the model could not score it
    503  the manifest is there and the weights are not; the body names the script that fetches them
    429  another request holds the models, and a timing taken alongside it would be meaningless

There is no 500 for any of these. A clean clone with no weights is the normal state of this
repository, not an error, and the answers say so.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas.models import ModelsResponse
from app.schemas.predict import (
    CompareRequest,
    CompareResponse,
    PredictRequest,
    PredictResponse,
)
from app.services.inference import (
    InferenceService,
    InvalidInputError,
    ModelNotFoundError,
    ModelUnavailableError,
    PredictionError,
    ServiceBusyError,
)

router = APIRouter(prefix="/api", tags=["inference"])


def _service(request: Request) -> InferenceService:
    """The singleton built in `create_app`. 503 when the application came up without one."""
    service: InferenceService | None = getattr(request.app.state, "inference", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "serving is switched off (APP_MODEL_TIER=none). The methodology section does not "
                "need it; the playground does."
            ),
        )
    return service


def _unavailable(exc: ModelUnavailableError) -> HTTPException:
    detail = f"'{exc.name}' is not being served: {exc.reason}"
    if exc.remedy:
        detail = f"{detail}. Run: {exc.remedy}"
    return HTTPException(status_code=503, detail=detail)


@router.get("/models", response_model=ModelsResponse)
def list_models(request: Request) -> ModelsResponse:
    """Every model with a manifest: those being served, and those that are not, with the reason."""
    return _service(request).catalog()


@router.post("/models/{name}/predict", response_model=PredictResponse)
def predict(name: str, payload: PredictRequest, request: Request) -> PredictResponse:
    """Score one review with one model."""
    try:
        return _service(request).predict(
            name, payload.text, measure_latency=payload.measure_latency
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"no such model: {exc}") from exc
    except ModelUnavailableError as exc:
        raise _unavailable(exc) from exc
    except InvalidInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PredictionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ServiceBusyError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


@router.post("/compare", response_model=CompareResponse)
def compare(payload: CompareRequest, request: Request) -> CompareResponse:
    """Score one review with several models at once and say where they part from the baseline."""
    try:
        return _service(request).compare(
            payload.text,
            payload.models,
            measure_latency=payload.measure_latency,
            measure_throughput=payload.measure_throughput,
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"no such model: {exc}") from exc
    except ModelUnavailableError as exc:
        raise _unavailable(exc) from exc
    except InvalidInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PredictionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ServiceBusyError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
