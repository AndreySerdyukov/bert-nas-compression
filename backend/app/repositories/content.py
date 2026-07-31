"""Committed JSON documents the methodology section reads.

These four files are the project's data spine. Three of them are parsed out of the notebooks by
`training/extract_notebook_data.py`, one is written by `training/benchmark.py`, and all of them are
committed - so the whole methodology half of the app works on a clean clone with no weights, no
corpus and no network.

They are served as bytes rather than parsed and re-serialised through pydantic. Documents this
shape (the trajectories file alone is a few hundred candidates deep) would need schemas that
duplicate the writers' structure and would have to be edited in lockstep with them, buying type
safety over data that a `--check` run in CI already verifies byte-for-byte against its source. The
schema that matters here is enforced where the file is written, not where it is read.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Document name -> filename. The name is what the URL and the frontend use.
DOCUMENTS = {
    "architectures": "architectures.json",
    "trajectories": "search_trajectories.json",
    "reported-results": "reported_results.json",
    "benchmark": "benchmark.json",
}


@dataclass(frozen=True)
class ContentRepository:
    """Committed JSON, read once at startup and held as bytes."""

    documents: dict[str, bytes]

    @classmethod
    def load(cls, data_dir: Path) -> ContentRepository:
        """Read every document that exists.

        A missing file is not an error. `benchmark.json` is absent until someone has run the
        benchmark, and the endpoint answers 503 in that case rather than the app refusing to start:
        the nine chapters that do not need it still work.
        """
        documents: dict[str, bytes] = {}
        for name, filename in DOCUMENTS.items():
            path = data_dir / filename
            if not path.exists():
                logger.warning("content document %r is missing at %s", name, path)
                continue
            documents[name] = path.read_bytes()
        return cls(documents=documents)

    def get(self, name: str) -> bytes | None:
        """Raw JSON for a document, or None if it was not present at startup."""
        return self.documents.get(name)

    @property
    def available(self) -> list[str]:
        """Names of the documents that loaded, for /health and the catalog."""
        return sorted(self.documents)
