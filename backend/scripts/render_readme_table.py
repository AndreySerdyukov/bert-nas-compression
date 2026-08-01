"""Generate the README's results table from `data/benchmark.json`, and check it has not drifted.

The README is where a reader meets this project's numbers, and prose drifts. Every figure between
the markers is written by this script from the measurement file, and CI runs it with `--check`, so a
number can only change in the README by being re-measured first.

The table deliberately carries two timing columns with two headings. Collapsing them into one
"ms/example" is the exact defect the rebuild exists to correct, and a generator that could produce
that column is a generator that eventually will.

Run (from `backend/`):
    python scripts/render_readme_table.py            # rewrite the block
    python scripts/render_readme_table.py --check    # fail if it would change
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_DIR.parent

BENCHMARK = _BACKEND_DIR / "data" / "benchmark.json"
CONTROLS = _BACKEND_DIR / "data" / "controls.json"
README = _REPO_ROOT / "README.md"

START = "<!-- benchmark:start -->"
END = "<!-- benchmark:end -->"

MISSING = (
    "_No measurements yet. Run `python -m training.benchmark` from `backend/` to produce\n"
    "`backend/data/benchmark.json`; this table is generated from it._"
)


def thousands(value: int | None) -> str:
    """Non-breaking thin spaces, so a number does not wrap mid-number.

    Used for every grouped figure this file emits rather than only for the parameter counts. The
    prose around the generated block writes "15 000" and "28 000", and a generated paragraph that
    said "15,000" two lines below made the same quantity look like two conventions.
    """
    return "n/a" if value is None else f"{value:,}".replace(",", " ")


def megabytes(value: int | None) -> str:
    return "n/a" if not value else f"{value / 1e6:,.0f} MB"


def own_memory(entry: dict[str, Any]) -> int | None:
    """What the model itself occupies: the process, less the runtime every model pays for."""
    total, floor = entry.get("resident_bytes"), entry.get("runtime_baseline_bytes")
    if not total or floor is None:
        return None
    return int(total) - int(floor)


def render(benchmark: dict[str, Any]) -> str:
    """The block between the markers: the table, then the protocol behind it."""
    protocol = benchmark["protocol"]
    lines = [
        "| Model | Method | Accuracy | Macro F1 | Params | Latency, median | Throughput | Memory |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in benchmark["models"]:
        latency = entry.get("latency")
        throughput = entry.get("throughput")
        lines.append(
            "| {label} | {method} | {accuracy:.4f} | {f1:.4f} | {params} | {latency} "
            "| {throughput} | {memory} |".format(
                label=entry["label"],
                method=entry["method"] or "fine-tune (baseline)",
                accuracy=entry["accuracy"],
                f1=entry["macro_f1"],
                params=thousands(entry["params"]),
                latency=f"{latency['median_ms']:.1f} ms" if latency else "not measured",
                throughput=(
                    f"{throughput['examples_per_second']:.0f}/s" if throughput else "not measured"
                ),
                memory=megabytes(own_memory(entry)),
            )
        )

    # Every column above is only comparable because of this paragraph, so it is generated with the
    # table rather than written beside it by hand.
    lines += [
        "",
        (
            f"Measured on the full {thousands(protocol['test_rows'])} row test split "
            f"(index sha256 `{protocol['test_index_sha256'][:12]}`) by "
            f"[`training/benchmark.py`](backend/training/benchmark.py)."
        ),
        "",
        (
            f"**Accuracy** on `{protocol['accuracy_device']}`. **Latency** is single-example on "
            f"`{protocol['latency_device']}`, {protocol['latency_threads']} thread"
            f"{'s' if protocol['latency_threads'] != 1 else ''}, "
            f"{protocol['latency_warmup']} warm-ups discarded, median of "
            f"{protocol['latency_repeats']}, round-robin. **Throughput** is a batch of "
            f"{protocol['throughput_batch']} and a different quantity: dividing it by the batch "
            f"size does not give the latency column. {protocol['machine']}, torch "
            f"{protocol['torch']}, transformers {protocol['transformers']}."
        ),
    ]

    # Every model timed the same review, but they do not all see the same number of tokens, and a
    # latency column read without that is a comparison of two different amounts of work.
    tokens = {
        entry["label"]: entry["latency"]["n_tokens"]
        for entry in benchmark["models"]
        if entry.get("latency")
    }
    if len(set(tokens.values())) > 1:
        counts = ", ".join(f"{label} {n}" for label, n in tokens.items())
        lines += [
            "",
            (
                f"All five timed the same review over different amounts of work: {counts} tokens. "
                "AdaBERT has no attention mask and is served at a fixed 128, so its column is a "
                "shorter time over *more* tokens."
            ),
        ]

    floors = [
        entry["runtime_baseline_bytes"]
        for entry in benchmark["models"]
        if entry.get("runtime_baseline_bytes")
    ]
    if floors:
        lines += [
            "",
            (
                f"**Memory** is one model per process, less the {megabytes(min(floors))} floor "
                "torch and transformers occupy before any model loads. Weighed one after another "
                "in a single process instead, a freed model's pages return to Python's allocator "
                "rather than to the OS and every model after the first looks nearly free."
            ),
        ]
    return "\n".join(lines)


def render_controls(controls: dict[str, Any]) -> str:
    """The controls table: what was actually run, and what was not.

    Sorted best first, because the ordering is the finding - a reader who sees TF-IDF and DistilBERT
    above every searched architecture has the point of the table before reading a word of it.

    The question each control answers is printed once per group rather than once per row. Ten
    consecutive rows repeating "is the found mask in the tail of the distribution, or the middle?"
    is the same sentence ten times, and it pushed the accuracy columns off the side of the page.

    `not_run` is printed rather than omitted. A list of six results where eighteen were planned
    reads as "these are the controls" unless the plan is written down beside it, and a control
    that quietly went missing is how a comparison ends up flattering whoever ran it.
    """
    protocol = controls["protocol"]
    ranked = sorted(controls["controls"], key=lambda entry: -entry["accuracy"])

    lines = ["| Control | Layers | Accuracy | Macro F1 |", "|---|---|---:|---:|"]
    for entry in ranked:
        layers = "-" if not entry["layers"] else ",".join(str(index) for index in entry["layers"])
        lines.append(
            f"| {entry['label']} | {layers} | {entry['accuracy']:.4f} | {entry['macro_f1']:.4f} |"
        )

    # Deduplicated in the order the controls were planned, so the list reads as the sequence of
    # questions the run was designed to answer.
    questions: list[str] = []
    for entry in controls["controls"]:
        if entry["question"] not in questions:
            questions.append(entry["question"])

    lines += ["", f"They answer {len(questions)} questions:", ""]
    lines += [f"{index}. {question}" for index, question in enumerate(questions, start=1)]

    lines += [
        "",
        (
            f"Trained under the shipped checkpoints' own protocol, read out of the eval notebooks: "
            f"{protocol['epochs']} epoch over all {thousands(protocol['train_rows'])} training "
            f"rows, "
            f"{protocol['max_length']} tokens, batch {protocol['batch_size']}, AdamW at "
            f"{protocol['learning_rate']} with weight decay {protocol['weight_decay']} and a "
            f"{protocol['warmup_ratio']:.0%} warm-up. Nothing was tuned - shortening the training "
            "would bias every comparison on this page in this project's favour."
        ),
    ]
    if controls["not_run"]:
        lines += [
            "",
            (
                f"**Not run: {', '.join(controls['not_run'])}.** "
                f"{len(controls['controls'])} of {len(controls['planned'])} planned controls "
                "completed within the time budget."
            ),
        ]
    return "\n".join(lines)


def block(readme: str) -> tuple[int, int]:
    start = readme.find(START)
    end = readme.find(END)
    if start == -1 or end == -1 or end < start:
        raise SystemExit(f"{README} has no {START} … {END} block")
    return start + len(START), end


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="exit non-zero if the README would change"
    )
    args = parser.parse_args()

    readme = README.read_text(encoding="utf-8")
    start, end = block(readme)

    if BENCHMARK.exists():
        parts = [render(json.loads(BENCHMARK.read_text(encoding="utf-8")))]
        if CONTROLS.exists():
            parts += [
                "",
                "### The controls",
                "",
                render_controls(json.loads(CONTROLS.read_text(encoding="utf-8"))),
            ]
        body = "\n".join(parts)
    else:
        # A clean clone has no measurements, and the README says so rather than showing a stale
        # table that nothing can be checked against.
        body = MISSING

    updated = f"{readme[:start]}\n\n{body}\n\n{readme[end:]}"
    if updated == readme:
        print("README table is current")
        return 0

    if args.check:
        print(
            "The README results table does not match backend/data/benchmark.json.\n"
            "Run: python scripts/render_readme_table.py",
            file=sys.stderr,
        )
        return 1

    README.write_text(updated, encoding="utf-8")
    print(f"rewrote the results table in {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
