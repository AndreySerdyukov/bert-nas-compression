"""Torch runtime configuration, and the description of it that ships with every measurement.

A separate module so `main.py` never imports torch: the entry point is about wiring, not about
compute parameters.

The description is the load-bearing half. This project publishes latency, and a latency figure is
only a figure if it says how many threads and what machine produced it - so `describe_runtime()` is
attached to every prediction and every measurement rather than being logged once at startup.
"""

from __future__ import annotations

import logging
import platform

from app.schemas.models import RuntimeInfo

logger = logging.getLogger(__name__)


def configure_torch_threads(threads: int) -> None:
    """Pin torch to `threads` intra-op and one inter-op thread (0 = leave torch's defaults alone).

    Serving handles one request at a time, so a thread per core only makes the cores fight over one
    forward pass, and it makes latency depend on what else the machine was doing. Pinning it is
    what allows two models measured minutes apart to be compared at all.

    The inter-op cap matters as much as the intra-op one here: five models are measured round-robin
    in a single process, and torch's inter-op pool is shared between them. Inside the container
    OMP_NUM_THREADS and MKL_NUM_THREADS back this up one level down.
    """
    if threads <= 0:
        return
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is present in every working environment
        logger.warning("torch is not installed, skipping thread configuration")
        return

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Torch allows this exactly once, and only before any parallel work has started. Under
        # `uvicorn --reload` and under pytest the module is re-imported into a process that has
        # already run a forward pass, and the second call raises. The first call is the one that
        # counted, so there is nothing to fix.
        logger.debug("inter-op thread count was already fixed for this process")


def describe_runtime(device: str) -> RuntimeInfo:
    """The machine and thread configuration behind a measurement, as it will be rendered."""
    try:
        import torch

        threads = int(torch.get_num_threads())
        interop = int(torch.get_num_interop_threads())
        torch_version = str(torch.__version__)
    except ImportError:  # pragma: no cover - torch is present in every working environment
        threads, interop, torch_version = 0, 0, "not installed"

    return RuntimeInfo(
        device=device,
        threads=threads,
        interop_threads=interop,
        machine=f"{platform.system()} {platform.machine()}",
        torch_version=torch_version,
    )
