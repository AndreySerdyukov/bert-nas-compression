"""Measure every model the same way, on the same rows, and write down how.

This file is the reason the project was rebuilt. The original head-to-head reported one accuracy per
model and a millisecond figure that was a batch time divided by sixteen, taken with no warm-up, and
called a latency. Everything here is arranged so that cannot happen again:

- **One harness, one test set.** The full 15 000-row split, identified by a digest of its sorted
  index, so "the test set" is a checkable claim rather than a phrase.
- **The same loader the application serves from.** Accuracy and latency come out of
  `ModelRegistry`, not out of a copy of the loading code that might differ from it by one
  tokenizer argument.
- **Three quantities, three names.** Single-example latency, batched throughput, and the wall clock
  of the accuracy pass are separate fields and never collapse into one.
- **Each number measured where it means something.** Accuracy on the GPU because it is the same
  arithmetic and three times faster; latency on the CPU with one pinned thread because that is a
  figure a reader can reproduce; memory in a child process that has loaded no other model, because
  a delta taken after four other checkpoints have been through the same interpreter measures the
  allocator's free pool rather than the model.

Run (from `backend/`, inside .venv):
    python -m training.benchmark                    # the full test split, ~20 minutes
    python -m training.benchmark --limit 500        # a quick check
    python -m training.benchmark --only adabert
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND_DIR))

DEFAULT_OUT = _BACKEND_DIR / "data" / "benchmark.json"

# Batch for the accuracy pass. It changes nothing about the answers - the transformers mask their
# padding and AdaBERT pads to a fixed length - so it is purely how fast the pass runs.
SCORE_BATCH = 32

# The accuracy pass is the same arithmetic on either device and three times faster on the GPU.
# Latency is not: it is the number a reader is invited to reproduce, so it stays on one CPU thread.
ACCURACY_DEVICE = "mps"
LATENCY_DEVICE = "cpu"

# More repeats than an interactive request can afford, because this run is not in a hurry.
LATENCY_WARMUP = 5
LATENCY_REPEATS = 25
THROUGHPUT_BATCH = 16
THROUGHPUT_REPEATS = 5

# The review latency is measured on. A fixed one, so the token count behind every model's figure is
# the same and the comparison is between models rather than between inputs.
LATENCY_REVIEW = (
    "The premise had promise and the first half hour genuinely works, but the film loses its nerve "
    "completely after that, and the ending is unforgivable."
)


@dataclass(frozen=True)
class Protocol:
    """Everything about how the numbers were taken, recorded next to them."""

    test_rows: int
    test_index_sha256: str
    corpus_sha256: str
    accuracy_device: str
    latency_device: str
    latency_threads: int
    latency_warmup: int
    latency_repeats: int
    throughput_batch: int
    score_batch: int
    memory_measured: str
    machine: str
    python: str
    torch: str
    transformers: str

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


def rss_bytes() -> int:
    """Resident set size of this process, right now.

    `resource.getrusage` reports a high-water mark, which only ever grows and so cannot say what
    the model just loaded costs. `ps` reports the current value, in kilobytes on both macOS and
    Linux. Zero when it cannot be read, which the caller reports rather than papers over.
    """
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(subprocess.os.getpid())],  # type: ignore[attr-defined]
            capture_output=True,
            text=True,
            check=True,
        )
        return int(out.stdout.strip()) * 1024
    except (subprocess.SubprocessError, ValueError, OSError):
        return 0


def weigh_here(name: str) -> dict[str, int]:
    """Load one model into a process that has loaded no other, and report what it cost.

    This runs in a child process, and that is the whole point. Measured in-process after the
    accuracy pass, the deltas were nonsense: the 438 MB baseline appeared to cost 1.8 MB and the
    32 MB AdaBERT 74 MB, because freeing the previous model returns its pages to Python's and
    torch's allocators rather than to the OS, so the next load is served out of memory that is
    already resident. Two byte-identical checkpoints came out at 3.8 MB and 9.9 MB, which is how
    the reading was caught. A fresh interpreter has no such pool to be served from.

    The baseline is taken with the whole runtime already imported, `AutoTokenizer` included, and
    that import is not a formality: in transformers 5 it is a lazy module and naming the symbol
    costs 187 MB. Measured without it, AdaBERT appeared to cost 268 MB for a 32 MB checkpoint,
    because it is the one model small enough that the fixed cost is not hidden inside memory the
    allocator had already taken for a 438 MB state dict and freed. Every model has to be charged
    the same floor or the smallest one pays for all of them.

    The reading is taken after `warmup()`: a safetensors checkpoint is mapped rather than copied,
    so pages nobody reads never become resident, and the honest question is what a process that has
    answered a request occupies. Both readings are absolute and both are published - what a reader
    sizes a container with is the total, and the baseline says how much of it is not the model.
    """
    import torch  # noqa: F401  - paid for before the baseline, so it stays out of the delta
    import transformers  # noqa: F401
    from transformers import AutoTokenizer  # noqa: F401

    from app.config import get_settings
    from app.repositories.model_registry import ModelRegistry
    from app.serving.runtime import configure_torch_threads

    settings = get_settings()
    configure_torch_threads(settings.torch_threads)
    registry = ModelRegistry(
        models_dir=settings.models_dir,
        data_dir=settings.data_dir,
        baseline=settings.baseline_model,
        device=LATENCY_DEVICE,
    )
    gc.collect()
    baseline = rss_bytes()
    registry.load(only=name)
    if registry.get(name) is None:
        return {"baseline_bytes": baseline, "resident_bytes": 0}
    registry.warmup()
    return {"baseline_bytes": baseline, "resident_bytes": rss_bytes()}


def weigh(name: str) -> dict[str, int] | None:
    """Ask a child process what a process serving `name` occupies. None if it could not be read."""
    completed = subprocess.run(
        [sys.executable, "-m", "training.benchmark", "--weigh", name],
        capture_output=True,
        text=True,
        cwd=_BACKEND_DIR,
        # A model that cannot be weighed is reported as a missing reading, not as a failed run:
        # the accuracy and latency it already has are worth writing down.
        check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        measured = json.loads(completed.stdout.strip().splitlines()[-1])
        resident = int(measured["resident_bytes"])
        baseline = int(measured["baseline_bytes"])
    except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None
    return {"resident_bytes": resident, "runtime_baseline_bytes": baseline} if resident else None


def _describe_environment(split_sha: str, corpus_sha: str, rows: int, threads: int) -> Protocol:
    import torch
    import transformers

    return Protocol(
        test_rows=rows,
        test_index_sha256=split_sha,
        corpus_sha256=corpus_sha,
        accuracy_device=ACCURACY_DEVICE,
        latency_device=LATENCY_DEVICE,
        latency_threads=threads,
        latency_warmup=LATENCY_WARMUP,
        latency_repeats=LATENCY_REPEATS,
        throughput_batch=THROUGHPUT_BATCH,
        score_batch=SCORE_BATCH,
        memory_measured=(
            "resident set size of a process holding exactly one loaded and warmed model; "
            "runtime_baseline_bytes is the same process with torch and transformers imported "
            "and no model, and is a fixed cost none of the models owns"
        ),
        machine=f"{platform.system()} {platform.machine()}",
        python=platform.python_version(),
        torch=str(torch.__version__),
        transformers=str(transformers.__version__),
    )


def score_predictions(predictions: list[int], labels: list[int]) -> dict[str, Any]:
    """Every metric this project publishes, from raw predictions.

    Shared with `training/train_reference.py` deliberately: a control scored by a second copy of
    this arithmetic would be comparable to the models only by coincidence.
    """
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    macro = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
    positive = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=1, zero_division=0
    )
    correct = sum(int(p == t) for p, t in zip(predictions, labels, strict=True))

    return {
        "correct": correct,
        "accuracy": round(correct / len(labels), 6),
        "macro_precision": round(float(macro[0]), 6),
        "macro_recall": round(float(macro[1]), 6),
        "macro_f1": round(float(macro[2]), 6),
        "positive_precision": round(float(positive[0]), 6),
        "positive_recall": round(float(positive[1]), 6),
        "positive_f1": round(float(positive[2]), 6),
        # Written out rather than left as a nested array: which corner is which is exactly the
        # thing a reader should not have to work out.
        "confusion": {
            "true_negative": int(matrix[0][0]),
            "false_positive": int(matrix[0][1]),
            "false_negative": int(matrix[1][0]),
            "true_positive": int(matrix[1][1]),
        },
    }


def score_model(
    predictor: Any, texts: list[str], labels: list[int], counter: Any
) -> dict[str, Any]:
    """Run every row through a loaded model and score what comes out."""
    predictions: list[int] = []
    for start in range(0, len(texts), SCORE_BATCH):
        batch = texts[start : start + SCORE_BATCH]
        predictions.extend(
            1 if prediction.label == "positive" else 0
            for prediction in predictor.predict_many(batch)
        )
        counter.update(len(batch))
    return score_predictions(predictions, labels)


def cost_of(info: Any, models_dir: Path) -> dict[str, Any]:
    """What the architecture costs, from three independent directions.

    The parameter count is taken from the loaded module, from the manifest, and from the closed
    form, and all three are reported. Two of them agreeing proves less than it looks; three of them
    disagreeing is what catches a checkpoint that is not what its manifest says.
    """
    from app.services import architecture as arch

    lock_path = models_dir / f"{info.name}.lock.json"
    on_disk = 0
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        on_disk = sum(int(entry["bytes"]) for entry in lock["files"].values())

    formula: int | None = None
    flops: int | None = None
    if info.n_layers is not None:
        formula = arch.params_for(info.n_layers)
        flops = arch.flops_for(info.n_layers, info.max_length)

    return {
        "params": info.params,
        # None for AdaBERT: it has no encoder layers, so the twelve-layer arithmetic does not
        # describe it and a number here would be a fabrication.
        "params_formula": formula,
        "params_agree": None if formula is None else formula == info.params,
        "flops_per_example": flops,
        "bytes_on_disk": on_disk,
        "bytes_fp32": (info.params or 0) * 4,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N test rows")
    parser.add_argument("--only", action="append", help="benchmark just this model (repeatable)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--skip-latency", action="store_true", help="accuracy only, for a quick check"
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="skip the corpus digest check (know what it costs)"
    )
    parser.add_argument(
        "--weigh",
        metavar="MODEL",
        help="internal: load one model in this process and print what it cost resident",
    )
    args = parser.parse_args()

    if args.weigh:
        # The child half of the memory pass. It has to be a whole process, so it is this one
        # re-invoked rather than a second script that could drift from the loader used here.
        print(json.dumps(weigh_here(args.weigh)))
        return 0

    from app.config import get_settings
    from app.repositories.model_registry import ModelRegistry
    from app.serving.runtime import configure_torch_threads
    from training.dataset import DEFAULT_CSV, file_sha256, load_split
    from training.progress import StageTracker

    settings = get_settings()
    configure_torch_threads(settings.torch_threads)

    # Which models exist is read from the manifests, so a model added to the registry appears here
    # without this file being edited.
    names = sorted(
        path.name.removesuffix(".meta.json") for path in settings.models_dir.glob("*.meta.json")
    )
    if args.only:
        names = [name for name in names if name in args.only]
    if not names:
        print("no models to benchmark", file=sys.stderr)
        return 1

    split = load_split(verify=not args.no_verify)
    test = split.test if args.limit is None else split.test.head(args.limit)
    texts = test["review"].tolist()
    labels = [int(label) for label in test["label"].tolist()]

    stages = (
        [f"scoring {name}" for name in names]
        + [f"weighing {name}" for name in names]
        + ([] if args.skip_latency else ["timing, round-robin"])
        + ["writing"]
    )

    results: dict[str, dict[str, Any]] = {}
    infos: dict[str, Any] = {}

    with StageTracker(stages, desc="benchmark") as tracker:
        tracker.write(
            f"{len(texts)} test rows, index sha256 {split.test_index_sha256[:12]}, "
            f"accuracy on {ACCURACY_DEVICE}, latency on {LATENCY_DEVICE}"
        )

        # --- accuracy, one model at a time on the fast device ------------------------------------
        for name in names:
            with tracker.stage(f"scoring {name}"):
                registry = ModelRegistry(
                    models_dir=settings.models_dir,
                    data_dir=settings.data_dir,
                    baseline=settings.baseline_model,
                    device=ACCURACY_DEVICE,
                )
                registry.load(only=name)
                loaded = registry.get(name)
                if loaded is None:
                    tracker.write(f"{name}: skipped ({registry.skipped[name].reason})")
                    continue
                loaded.predictor.warmup()
                infos[name] = loaded.info
                started = time.perf_counter()
                with tracker.sub(len(texts), name) as counter:
                    results[name] = score_model(loaded.predictor, texts, labels, counter)
                results[name]["score_seconds"] = round(time.perf_counter() - started, 2)
                del registry, loaded
                gc.collect()
            entry = results.get(name)
            if entry:
                tracker.write(
                    f"{name}: accuracy {entry['accuracy']:.4f} in {entry['score_seconds']}s"
                )

        # --- memory, one model in a process that has loaded no other ------------------------------
        for name in names:
            with tracker.stage(f"weighing {name}"):
                if name not in results:
                    continue
                measured = weigh(name)
                unread: dict[str, Any] = {"resident_bytes": None, "runtime_baseline_bytes": None}
                results[name].update(measured or unread)

        # --- latency and throughput, every model resident and interleaved ------------------------
        if not args.skip_latency:
            with tracker.stage("timing, round-robin"):
                registry = ModelRegistry(
                    models_dir=settings.models_dir,
                    data_dir=settings.data_dir,
                    baseline=settings.baseline_model,
                    device=LATENCY_DEVICE,
                )
                registry.load()
                registry.warmup()
                _time_everything(registry, names, results, tracker)

        with tracker.stage("writing"):
            payload = _assemble(names, results, infos, settings, split, args)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {args.out} ({len(payload['models'])} models, {len(texts)} rows)")
    for entry in payload["models"]:
        latency = entry.get("latency")
        shown = f"{latency['median_ms']:.1f} ms median" if latency else "latency not measured"
        print(f"  {entry['name']:15s} accuracy {entry['accuracy']:.4f}   {shown}")
    print(f"\ncorpus {DEFAULT_CSV.name} sha256 {file_sha256(DEFAULT_CSV)[:12]}")
    return 0


def _time_everything(
    registry: Any, names: list[str], results: dict[str, dict[str, Any]], tracker: Any
) -> None:
    """Latency and throughput for every model, interleaved so none of them pays for going last."""
    from app.services.latency import latency_stats, throughput_stats, time_round_robin

    live = [name for name in names if registry.get(name) is not None and name in results]
    if not live:
        return

    predictors = {name: registry.get(name).predictor for name in live}
    single = {
        name: (lambda p=predictor: p.predict(LATENCY_REVIEW))
        for name, predictor in predictors.items()
    }
    for name, timing in time_round_robin(
        single, warmup=LATENCY_WARMUP, repeats=LATENCY_REPEATS
    ).items():
        tokens = predictors[name].predict(LATENCY_REVIEW).n_tokens
        results[name]["latency"] = latency_stats(
            timing, warmup=LATENCY_WARMUP, n_tokens=tokens
        ).model_dump()

    batch = [LATENCY_REVIEW] * THROUGHPUT_BATCH
    batched = {
        name: (lambda p=predictor: p.predict_many(batch)) for name, predictor in predictors.items()
    }
    for name, timing in time_round_robin(batched, warmup=1, repeats=THROUGHPUT_REPEATS).items():
        tokens = results[name]["latency"]["n_tokens"]
        results[name]["throughput"] = throughput_stats(
            timing, warmup=1, batch_size=THROUGHPUT_BATCH, n_tokens=tokens
        ).model_dump()
    tracker.write(f"timed {len(live)} models round-robin, {LATENCY_REPEATS} repeats each")


def _assemble(
    names: list[str],
    results: dict[str, dict[str, Any]],
    infos: dict[str, Any],
    settings: Any,
    split: Any,
    args: Any,
) -> dict[str, Any]:
    """Put the measurements together with the protocol that produced them."""
    import torch

    from training.dataset import DEFAULT_CSV, file_sha256

    rows = args.limit if args.limit is not None else len(split.test)
    protocol = _describe_environment(
        split.test_index_sha256, file_sha256(DEFAULT_CSV), rows, int(torch.get_num_threads())
    )

    models = []
    for name in names:
        if name not in results:
            continue
        info = infos[name]
        models.append(
            {
                "name": info.name,
                "label": info.label,
                "method": info.method,
                "n_layers": info.n_layers,
                "max_length": info.max_length,
                "hf_repo": info.hf_repo,
                "hf_revision": info.hf_revision,
                **cost_of(info, settings.models_dir),
                **results[name],
            }
        )
    models.sort(key=lambda entry: -(entry["params"] or 0))

    return {
        "schema": 1,
        "generated_by": "training/benchmark.py",
        "note": (
            "Measured by this project, not quoted from the notebooks. Latency and throughput are "
            "separate fields on purpose: the figure the original published as ms/example was a "
            "batch time divided by the batch size."
        ),
        "protocol": protocol.to_dict(),
        "models": models,
    }


if __name__ == "__main__":
    raise SystemExit(main())
