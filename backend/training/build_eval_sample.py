"""Build the two committed data files the app scores against.

  data/imdb_eval_sample.jsonl  2 000 rows drawn from the 15 000-row test split, stratified and
                               seeded. Everything that measures anything interactively uses these
                               rows, so the laptop, CI and the Docker image all score the same text.

  data/label_probe.json        A handful of unambiguous reviews used to work out which class index
                               means "positive". None of the checkpoints carry `id2label`, so the
                               mapping is measured rather than assumed - and re-checked every time a
                               model loads, because serving inverted sentiment looks perfectly
                               healthy from the outside.

Run (from the backend directory, inside .venv; needs the corpus):
    python -m training.build_eval_sample
    python -m training.build_eval_sample --check      # verify what is committed, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from training.dataset import ID_TO_LABEL, load_split

_BACKEND_DIR = Path(__file__).resolve().parents[1]
SAMPLE_OUT = _BACKEND_DIR / "data" / "imdb_eval_sample.jsonl"
PROBE_OUT = _BACKEND_DIR / "data" / "label_probe.json"

SAMPLE_SIZE = 2000
SEED = 7

# Short, blunt reviews written for this file rather than drawn from the corpus. Two reasons: they
# have to be unambiguous enough that a disagreement means a broken label map and not a hard example,
# and using held-out rows for a startup check would leak test data into the serving path.
PROBE_REVIEWS: list[dict[str, Any]] = [
    {"text": "Absolutely wonderful film. I loved every minute of it.", "label": "positive"},
    {
        "text": "One of the best movies I have ever seen - brilliant, moving, beautifully acted.",
        "label": "positive",
    },
    {
        "text": "A delight from start to finish. Warm, funny and genuinely surprising.",
        "label": "positive",
    },
    {
        "text": "Superb performances and a script that never wastes a scene. Highly recommended.",
        "label": "positive",
    },
    {
        "text": "A complete waste of time. Terrible acting and a boring, predictable plot.",
        "label": "negative",
    },
    {
        "text": "Awful. I walked out halfway through. Dreadful in every possible way.",
        "label": "negative",
    },
    {
        "text": "Painfully bad. The dialogue is embarrassing and nothing about it works.",
        "label": "negative",
    },
    {"text": "I regret watching this. Dull, ugly and far too long.", "label": "negative"},
]


def _index_digest(rows: list[dict[str, Any]]) -> str:
    """Fingerprint of which corpus rows the sample holds, in the order it holds them.

    One function, called where the file is written and where it is checked. Two copies of this
    line would be two chances for the check to agree with itself and with nothing else.
    """
    return hashlib.sha256(",".join(str(row["id"]) for row in rows).encode("ascii")).hexdigest()


def build_sample() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Draw the stratified sample, and describe where it came from."""
    split = load_split()
    test = split.test

    # Stratified so a 2 000-row accuracy is not skewed by an unbalanced draw; seeded so it is the
    # same 2 000 rows everywhere.
    per_class = SAMPLE_SIZE // 2
    frames = [
        test[test["label"] == label].sample(n=per_class, random_state=SEED)
        for label in sorted(ID_TO_LABEL)
    ]
    drawn = pd.concat(frames).sample(frac=1.0, random_state=SEED)

    rows = [
        {"id": int(index), "text": str(row["review"]), "label": int(row["label"])}
        for index, row in drawn.iterrows()
    ]
    provenance = {
        "n": len(rows),
        "seed": SEED,
        "drawn_from": "the 15 000-row test split",
        "test_index_sha256": split.test_index_sha256,
        "source_index_sha256": _index_digest(rows),
        "class_balance": {ID_TO_LABEL[0]: per_class, ID_TO_LABEL[1]: per_class},
    }
    return rows, provenance


def build_probe() -> dict[str, Any]:
    return {
        "schema": 1,
        "note": (
            "No checkpoint in this project carries id2label. scripts/fetch_models.py runs these "
            "reviews through each downloaded model, works out which class index means positive, "
            "and pins it in the manifest. Every load re-checks it and drops the model on a "
            "disagreement."
        ),
        "min_agreement": 0.95,
        "reviews": PROBE_REVIEWS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the committed files only")
    args = parser.parse_args()

    probe = json.dumps(build_probe(), indent=2) + "\n"

    if args.check:
        if not SAMPLE_OUT.exists() or not PROBE_OUT.exists():
            print("the committed sample or probe is missing", file=sys.stderr)
            return 1
        if PROBE_OUT.read_text(encoding="utf-8") != probe:
            print(f"{PROBE_OUT.name} does not match this script", file=sys.stderr)
            return 1
        # Split on "\n" rather than splitlines(): reviews contain characters such as U+2028 that
        # Python treats as line breaks and JSON does not escape, which inflates the count by ~3%.
        lines = SAMPLE_OUT.read_text(encoding="utf-8").strip("\n").split("\n")
        # The corpus is not available in CI, so the sample is checked structurally rather than
        # regenerated: the rows are all that matter and they are committed.
        if len(lines) != SAMPLE_SIZE + 1:
            print(
                f"{SAMPLE_OUT.name} has {len(lines)} lines, expected {SAMPLE_SIZE + 1}",
                file=sys.stderr,
            )
            return 1
        header = json.loads(lines[0])
        rows = [json.loads(line) for line in lines[1:]]
        labels = [row["label"] for row in rows]
        if sorted(set(labels)) != [0, 1] or labels.count(1) != SAMPLE_SIZE // 2:
            print(f"{SAMPLE_OUT.name} is not balanced", file=sys.stderr)
            return 1

        # The digest the file writes about itself, checked rather than merely carried. Without
        # this the check passed on row count and class balance alone, so a row swapped for another
        # from the same split - a rebase gone wrong, a hand edit - went through unnoticed, and the
        # provenance line went on describing a sample the file no longer held.
        digest = _index_digest(rows)
        if digest != header.get("source_index_sha256"):
            print(
                f"{SAMPLE_OUT.name} holds different rows than its own provenance line records.\n"
                f"  recorded sha256 {header.get('source_index_sha256')}\n"
                f"  rows here       {digest}\n"
                "Regenerate it with `python -m training.build_eval_sample` (needs the corpus).",
                file=sys.stderr,
            )
            return 1

        print(
            f"eval sample: {len(labels)} rows, balanced, row index {digest[:12]} as recorded, "
            f"from test index {header['test_index_sha256'][:12]}; probe: "
            f"{len(build_probe()['reviews'])} reviews"
        )
        return 0

    rows, provenance = build_sample()
    with SAMPLE_OUT.open("w", encoding="utf-8") as handle:
        # The first line is provenance, not a row: this file has to be able to say which test set it
        # came out of, and a sidecar would get separated from it.
        handle.write(json.dumps(provenance, ensure_ascii=False) + "\n")
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    PROBE_OUT.write_text(probe, encoding="utf-8")

    size_mb = SAMPLE_OUT.stat().st_size / 1_048_576
    print(f"  eval sample: {len(rows)} rows ({size_mb:.1f} MB) -> {SAMPLE_OUT.name}")
    print(f"  test index sha256: {provenance['test_index_sha256']}")
    print(f"  label probe: {len(PROBE_REVIEWS)} reviews -> {PROBE_OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
