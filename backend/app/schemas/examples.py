"""Schema for the reviews the playground offers to start from."""

from __future__ import annotations

from pydantic import BaseModel


class Example(BaseModel):
    """One review from the evaluation sample, with the provenance that makes it checkable."""

    id: int
    text: str
    # The corpus label, so the page can say whether a model got it right rather than only what it
    # said. These are not cherry-picked to flatter anyone: the selection is a rule.
    label: int
    n_chars: int
    truncated: bool


class ExamplesResponse(BaseModel):
    """The offered reviews, and what they were drawn from."""

    examples: list[Example]
    # Rows in the evaluation sample the examples came from, so the dozen is visibly a sample.
    total: int
    source: str = "data/imdb_eval_sample.jsonl"
