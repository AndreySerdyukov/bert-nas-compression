"""Long-running work: start it, watch it, ask about it.

    POST /api/jobs             start one, or 429 because the single worker is busy
    GET  /api/jobs/{id}        the current snapshot
    GET  /api/jobs/{id}/events the same snapshots as server-sent events

The stream and the poll return the same shape on purpose, so a client that loses the connection
falls back by asking instead of by re-rendering. `X-Accel-Buffering: no` is set here as well as in
`nginx.conf`: the header travels with the response, so a proxy this project does not own still gets
told not to sit on the frames.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.schemas.jobs import JobSnapshot, StartJobRequest
from app.services.ablation import ablation_stages
from app.services.architecture import InvalidMaskError, mask_from_layers
from app.services.inference import (
    AblationUnsupportedError,
    InferenceService,
    ModelNotFoundError,
    ModelUnavailableError,
    ServiceBusyError,
)
from app.services.jobs import (
    SELFTEST_STAGES,
    JobBusyError,
    JobNotFoundError,
    JobProgress,
    JobRunner,
    selftest_work,
)

router = APIRouter(prefix="/api", tags=["jobs"])

STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # nginx buffers proxied responses by default, which turns a live progress stream into one
    # silent wait followed by every frame at once.
    "X-Accel-Buffering": "no",
}


def _runner(request: Request) -> JobRunner:
    runner: JobRunner | None = getattr(request.app.state, "jobs", None)
    if runner is None:  # pragma: no cover - the runner is built unconditionally
        raise HTTPException(status_code=503, detail="the job runner is not available")
    return runner


def _service(request: Request) -> InferenceService:
    service: InferenceService | None = getattr(request.app.state, "inference", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="serving is switched off (APP_MODEL_TIER=none), so there is nothing to scan",
        )
    return service


@router.post("/jobs", response_model=JobSnapshot, status_code=202)
def start_job(payload: StartJobRequest, request: Request) -> JobSnapshot:
    """Start a job. One at a time: the models are a single shared object."""
    runner = _runner(request)

    if payload.kind == "selftest":
        try:
            return runner.submit("selftest", selftest_work, stages=list(SELFTEST_STAGES)).snapshot()
        except JobBusyError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc

    service = _service(request)
    repository = request.app.state.examples
    limit = payload.limit or request.app.state.settings.max_eval_sample
    rows = repository.rows[:limit]
    if not rows:
        raise HTTPException(
            status_code=503,
            detail="the evaluation sample is not present; run training/build_eval_sample.py",
        )

    if payload.kind == "ablation":
        if not payload.layers:
            raise HTTPException(
                status_code=422,
                detail="an ablation needs `layers`: which encoder layers survive, e.g. [0, 1, 2, 3]",
            )
        layers = payload.layers
        # Validated before the job starts, so a typo comes back as 422 on the request that made it
        # rather than as a job that runs for a second and then reports itself failed.
        try:
            mask_from_layers(layers)
            service.check_ablation_supported()
        except InvalidMaskError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except AblationUnsupportedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        def ablation_work(progress: JobProgress) -> dict[str, object]:
            return service.ablate(rows, layers, progress)

        try:
            return runner.submit("ablation", ablation_work, stages=ablation_stages()).snapshot()
        except JobBusyError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc

    names = payload.models or [info.name for info in service.catalog().models]
    if not names:
        raise HTTPException(
            status_code=503,
            detail="no models are loaded. Run: python scripts/fetch_models.py",
        )

    try:
        stages = service.scan_stages(names)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"no such model: {exc}") from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"'{exc.name}' is not being served") from exc

    def work(progress: JobProgress) -> dict[str, object]:
        return service.scan_disagreements(rows, names, progress)

    try:
        return runner.submit("disagreement", work, stages=stages).snapshot()
    except JobBusyError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ServiceBusyError as exc:  # pragma: no cover - the runner refuses first
        raise HTTPException(status_code=429, detail=str(exc)) from exc


@router.get("/jobs/{job_id}", response_model=JobSnapshot)
def get_job(job_id: str, request: Request) -> JobSnapshot:
    """The current snapshot. The fallback for a client whose stream dropped."""
    try:
        return _runner(request).get(job_id).snapshot()
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}") from exc


@router.get("/jobs/{job_id}/events")
def stream_job(job_id: str, request: Request) -> StreamingResponse:
    """Snapshots as they change, then the terminal one, then the connection closes."""
    runner = _runner(request)
    try:
        runner.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}") from exc
    return StreamingResponse(
        runner.stream(job_id), media_type="text/event-stream", headers=STREAM_HEADERS
    )
