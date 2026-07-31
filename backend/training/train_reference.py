"""Fine-tune the controls the searched architectures have to be compared against.

Three teammates comparing their own three models to each other establishes which of the three is
best and nothing else. The question a reader actually has is whether *searching* bought anything
over keeping every third layer, and only a control can answer it. That is what this file trains.

**The protocol is fixed by the data, not chosen.** It is read straight out of the notebooks that
produced the shipped Random Search and BANANAS checkpoints, which agree with each other to the
digit: one epoch over the full 28 000 training rows, 512 tokens, batch 16, AdamW at 3e-5 with
weight decay 0.01 and a linear schedule after a 10% warm-up. Nothing here is tuned. Shortening the
training or the sequence would bias every comparison in this project's favour, which is the one
thing a control must not do. AlphaNAS is the exception - it received two epochs - and that is a
footnote on its row rather than a licence to give the controls two.

**The order is fixed too**, so that a run cut short by a laptop lid still produced the most
informative subset it could:

    1. TF-IDF + logistic regression      how much of this task needs a transformer at all
    2. evenly-spaced masks, k=4,5        did searching beat "keep every third layer"
    3. first-k and last-k, k=4,5         does it matter which end the layers come from
    4. DistilBERT                        how does distillation compare at a similar size
    5. random masks, k=4, five seeds     is the found mask in the tail or the middle
    6. random masks, k=5, five seeds     the same question at the other size

Run (from `backend/`, inside .venv):
    python -m training.train_reference --time-probe            # what a night buys, measured
    python -m training.train_reference --time-probe --budget-hours 8
    python -m training.train_reference --only tfidf
    python -m training.train_reference --budget-hours 8        # work down the list, then stop
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

_BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND_DIR))

DEFAULT_OUT = _BACKEND_DIR / "data" / "controls.json"

BASE_MODEL = "bert-base-uncased"
DISTIL_MODEL = "distilbert-base-uncased"

# Transcribed from the eval notebooks that produced the shipped checkpoints. They agree with each
# other exactly; this is their recipe, not a tuned one.
EPOCHS = 1
MAX_LENGTH = 512
BATCH_SIZE = 16
# Scoring needs no gradients, so it can take a bigger batch than training does.
EVAL_BATCH = 32
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

N_LAYERS = 12
RANDOM_SEEDS = (0, 1, 2, 3, 4)

# DistilBERT is six layers, each the same width as BERT's, so a step of it costs what a six-layer
# mask costs and the probe can price it without downloading it.
DISTILBERT_LAYERS = 6

# The depths the plan actually contains, and therefore the depths the probe measures. Nothing is
# extrapolated: see `probe` for the reading that made extrapolation untenable.
PROBE_DEPTHS = (4, 5, DISTILBERT_LAYERS)

# Batches of the scoring pass to time per depth. Enough to leave the first-batch cost behind,
# few enough that the probe stays a probe.
PROBE_EVAL_BATCHES = 8

Kind = Literal["tfidf", "mask", "distilbert"]


@dataclass(frozen=True)
class DepthCost:
    """What one encoder depth costs, in the two passes a control actually pays for."""

    train_seconds_per_step: float
    eval_seconds_per_row: float


@dataclass(frozen=True)
class Control:
    """One control run: what to train, and the question it exists to answer."""

    key: str
    label: str
    kind: Kind
    question: str
    priority: int
    # Encoder layers to keep, for the masked runs. None for the two that are not masked BERTs.
    layers: tuple[int, ...] | None = None
    seed: int | None = None

    @property
    def n_layers(self) -> int | None:
        return None if self.layers is None else len(self.layers)


def evenly_spaced(k: int, total: int = N_LAYERS) -> tuple[int, ...]:
    """`k` layers spread across the stack, the obvious thing to do without searching at all.

    Rounded from an even division rather than taken from a table, so the rule is visible: this is
    the control that has to be genuinely naive for the comparison to mean anything.
    """
    return tuple(round(index * (total - 1) / (k - 1)) for index in range(k))


def random_mask(k: int, seed: int, total: int = N_LAYERS) -> tuple[int, ...]:
    """`k` layers drawn without replacement. Seeded, so the run is reproducible from its name."""
    import random

    return tuple(sorted(random.Random(seed).sample(range(total), k)))


def controls() -> list[Control]:
    """The full list, in the order it should be worked through."""
    items: list[Control] = [
        Control(
            key="tfidf",
            label="TF-IDF + logistic regression",
            kind="tfidf",
            question="How much of this task needs a transformer at all?",
            priority=1,
        )
    ]
    for k in (4, 5):
        items.append(
            Control(
                key=f"uniform-{k}",
                label=f"Evenly spaced, {k} layers",
                kind="mask",
                question="Did searching beat keeping every third layer?",
                priority=2,
                layers=evenly_spaced(k),
            )
        )
    for k in (4, 5):
        items.append(
            Control(
                key=f"first-{k}",
                label=f"First {k} layers",
                kind="mask",
                question="Does it matter which end of the stack the layers come from?",
                priority=3,
                layers=tuple(range(k)),
            )
        )
        items.append(
            Control(
                key=f"last-{k}",
                label=f"Last {k} layers",
                kind="mask",
                question="Does it matter which end of the stack the layers come from?",
                priority=3,
                layers=tuple(range(N_LAYERS - k, N_LAYERS)),
            )
        )
    items.append(
        Control(
            key="distilbert",
            label="DistilBERT",
            kind="distilbert",
            question="How does distillation compare with search at a similar size?",
            priority=4,
        )
    )
    for k, priority in ((4, 5), (5, 6)):
        for seed in RANDOM_SEEDS:
            items.append(
                Control(
                    key=f"random-{k}-seed{seed}",
                    label=f"Random {k} layers, seed {seed}",
                    kind="mask",
                    question="Is the found mask in the tail of the distribution, or the middle?",
                    priority=priority,
                    layers=random_mask(k, seed),
                    seed=seed,
                )
            )
    return sorted(items, key=lambda control: (control.priority, control.key))


# --- building the models -------------------------------------------------------------------------


def masked_bert(layers: tuple[int, ...]) -> Any:
    """A pretrained BERT with only `layers` kept, ready to be fine-tuned.

    This is the NAS protocol and not the explorer's: the mask is applied to the *pretrained*
    encoder and the result is then trained. Masking a fine-tuned model and not retraining is a
    different and much worse thing, which is exactly the point the explorer page will make.
    """
    import torch
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    kept = [model.bert.encoder.layer[index] for index in layers]
    model.bert.encoder.layer = torch.nn.ModuleList(kept)
    model.config.num_hidden_layers = len(kept)
    return model


# --- the training loop ---------------------------------------------------------------------------


def train_transformer(
    model: Any,
    tokenizer: Any,
    frame: Any,
    device: str,
    tracker: Any,
    label: str,
    *,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """One epoch of the notebooks' recipe. Returns what it cost, in seconds and in steps.

    `max_batches` stops early and is what `--time-probe` uses: the loop is otherwise identical, so
    the seconds-per-step it measures are the real ones rather than an estimate of them.
    """
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import get_linear_schedule_with_warmup

    encoded = tokenizer(
        frame["review"].tolist(),
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    dataset = TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        torch.tensor(frame["label"].tolist(), dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    total_steps = len(loader) * EPOCHS
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(  # type: ignore[no-untyped-call]
        optimizer, int(total_steps * WARMUP_RATIO), total_steps
    )

    model.to(device)
    model.train()
    planned = total_steps if max_batches is None else min(max_batches, len(loader))
    started = time.perf_counter()
    steps = 0
    running = 0.0

    with tracker.sub(planned, label) as counter:
        for _ in range(EPOCHS):
            for input_ids, attention_mask, labels in loader:
                outputs = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    labels=labels.to(device),
                )
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                running += float(outputs.loss.detach())
                steps += 1
                counter.update(1)
                if max_batches is not None and steps >= max_batches:
                    break
            if max_batches is not None and steps >= max_batches:
                break

    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "steps": steps,
        "steps_per_epoch": len(loader),
        "seconds": round(elapsed, 2),
        "seconds_per_step": elapsed / max(steps, 1),
        "train_loss": round(running / max(steps, 1), 4),
    }


# --- the time probe ------------------------------------------------------------------------------


def probe(device: str, frame: Any, tracker: Any, batches: int) -> dict[int, DepthCost]:
    """Measure what a step and a scored row cost at every depth the plan actually contains.

    The first version measured 4 and 12 layers and drew a line through them, on the reasoning that
    a step is near-linear in depth. It is not, at this batch and sequence length: 4 layers came out
    at 832 ms and 12 at 6576, which is 7.9x the cost for 3x the depth, because a twelve-layer
    backward pass over 16 sequences of 512 tokens runs into unified memory and the rest is paging.
    Extrapolating a budget from that overstated every control by half.

    So there is no extrapolation now. The plan holds three depths - 4 and 5 for the masks, 6 for
    DistilBERT - and all three are measured. It is also cheaper than the version it replaces: the
    twelve-layer probe alone cost more than these three together.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    measured: dict[int, DepthCost] = {}
    for depth in PROBE_DEPTHS:
        with tracker.stage(f"probing {depth} layers"):
            model = masked_bert(tuple(range(depth)))
            cost = train_transformer(
                model, tokenizer, frame, device, tracker, f"{depth} layers", max_batches=batches
            )
            per_row = _eval_seconds_per_row(model, tokenizer, frame, device)
            measured[depth] = DepthCost(
                train_seconds_per_step=cost["seconds_per_step"], eval_seconds_per_row=per_row
            )
            tracker.write(
                f"{depth} layers: {cost['seconds_per_step'] * 1000:.0f} ms/step training "
                f"over {cost['steps']} steps, {per_row * 1000:.1f} ms/row scoring"
            )
            del model
    return measured


def _eval_seconds_per_row(model: Any, tokenizer: Any, frame: Any, device: str) -> float:
    """Seconds per row of the scoring pass, which every control pays after it finishes training.

    Left out of the first schedule, and it is not a rounding error: scoring 15 000 rows takes about
    three minutes per control, which over seventeen of them is an hour that the budget was told
    nothing about. A schedule that omits it will report that all eighteen fit in a night when they
    do not, and the runs it silently drops are the last ones in priority order.
    """
    import torch

    texts = frame["review"].tolist()[: EVAL_BATCH * PROBE_EVAL_BATCHES]
    model.eval()
    started = time.perf_counter()
    for start in range(0, len(texts), EVAL_BATCH):
        batch = texts[start : start + EVAL_BATCH]
        encoded = tokenizer(
            batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
        )
        with torch.inference_mode():
            model(**{k: v.to(device) for k, v in encoded.items()})
    if device == "mps":
        torch.mps.synchronize()
    model.train()
    return (time.perf_counter() - started) / max(len(texts), 1)


def project(
    measured: dict[int, DepthCost], steps_per_epoch: int, test_rows: int
) -> list[tuple[Control, float]]:
    """Wall-clock for each control, from the depth that was measured for it.

    Every depth in the plan is in `measured`, so this reads a number rather than fitting one. A
    depth that is somehow missing falls back to the nearest measured one, which is a worse estimate
    but an honest one - the alternative is a line through two points that were never on a line.

    Training and scoring are both counted, because a control is not done until it has a number.
    """
    projected: list[tuple[Control, float]] = []
    for control in controls():
        if control.kind == "tfidf":
            # Minutes on a CPU, and dominated by vectorising rather than by fitting.
            projected.append((control, 3 * 60))
            continue
        # DistilBERT is six layers, and each is the same width as BERT's.
        depth = control.n_layers if control.n_layers is not None else DISTILBERT_LAYERS
        nearest = min(measured, key=lambda known: abs(known - depth))
        cost = measured.get(depth, measured[nearest])
        seconds = (
            cost.train_seconds_per_step * steps_per_epoch * EPOCHS
            + cost.eval_seconds_per_row * test_rows
        )
        projected.append((control, seconds))
    return projected


def print_schedule(projected: list[tuple[Control, float]], budget_hours: float | None) -> None:
    """What fits, in priority order, and what does not."""
    print(f"\n{'control':24s} {'layers':>7s} {'projected':>11s} {'cumulative':>12s}")
    print("-" * 58)
    running = 0.0
    fits = 0
    for control, seconds in projected:
        running += seconds
        depth = "-" if control.n_layers is None else str(control.n_layers)
        marker = ""
        if budget_hours is not None:
            if running <= budget_hours * 3600:
                fits += 1
            else:
                marker = "  (over budget)"
        print(
            f"{control.key:24s} {depth:>7s} {seconds / 60:>9.0f} m "
            f"{running / 3600:>10.1f} h{marker}"
        )
    print("-" * 58)
    print(f"all {len(projected)} controls: {running / 3600:.1f} h")
    if budget_hours is not None:
        print(f"in {budget_hours:g} h: the first {fits} of them")
    print(
        f"\nEach depth in the plan was measured, not interpolated: {', '.join(str(d) for d in PROBE_DEPTHS)} "
        "layers. Training and scoring are both counted, one epoch at the notebooks' protocol "
        f"({MAX_LENGTH} tokens, batch {BATCH_SIZE}). Whatever is actually run is what "
        "controls.json records - the file names the runs, not the plan."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--time-probe", action="store_true", help="measure the real cost of a step and project"
    )
    parser.add_argument("--probe-batches", type=int, default=20)
    parser.add_argument("--budget-hours", type=float, default=None)
    parser.add_argument("--only", action="append", help="run just this control (repeatable)")
    parser.add_argument("--list", action="store_true", help="print the controls and exit")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default=None, help="default: mps when available, else cpu")
    args = parser.parse_args()

    if args.list:
        for control in controls():
            layers = "-" if control.layers is None else ",".join(str(i) for i in control.layers)
            print(f"  {control.priority}  {control.key:24s} {layers:28s} {control.question}")
        return 0

    import torch

    from training.dataset import load_split
    from training.progress import StageTracker

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    split = load_split()

    if args.time_probe:
        steps_per_epoch = (len(split.train) + BATCH_SIZE - 1) // BATCH_SIZE
        stages = [f"probing {depth} layers" for depth in PROBE_DEPTHS]
        with StageTracker(stages, desc="time probe") as tracker:
            tracker.write(
                f"{len(split.train)} training rows, {steps_per_epoch} steps per epoch on {device}; "
                f"scoring {len(split.test)} test rows after each control"
            )
            measured = probe(device, split.train, tracker, args.probe_batches)
        print_schedule(project(measured, steps_per_epoch, len(split.test)), args.budget_hours)
        return 0

    selected = [c for c in controls() if not args.only or c.key in args.only]
    if not selected:
        print("no such control; try --list", file=sys.stderr)
        return 1

    stages = [control.key for control in selected]
    done: list[dict[str, Any]] = []
    budget = None if args.budget_hours is None else args.budget_hours * 3600
    started = time.perf_counter()

    with StageTracker(stages, desc="controls") as tracker:
        tracker.write(
            f"{len(split.train)} training rows, {len(split.test)} test rows, {EPOCHS} epoch, "
            f"{MAX_LENGTH} tokens, batch {BATCH_SIZE}, lr {LEARNING_RATE} on {device}"
        )
        for control in selected:
            spent = time.perf_counter() - started
            if budget is not None and spent >= budget:
                # Every control after this one is left undone, and the file says which.
                tracker.skip(control.key)
                continue
            with tracker.stage(control.key):
                result = run_control(control, split, device, tracker)
            done.append(result)
            tracker.write(
                f"{control.key}: accuracy {result['accuracy']:.4f} "
                f"({result['train']['seconds'] / 60:.0f} m training)"
            )
            # Written after every run, so a laptop lid keeps whatever finished.
            _write(args.out, selected, done, split, device)

    print(f"\nwrote {args.out}: {len(done)} of {len(selected)} controls")
    for entry in done:
        print(f"  {entry['key']:24s} accuracy {entry['accuracy']:.4f}")
    return 0


def run_control(control: Control, split: Any, device: str, tracker: Any) -> dict[str, Any]:
    """Train one control and score it on the same test split every other model is scored on."""
    from training.benchmark import score_predictions

    if control.kind == "tfidf":
        predictions, cost = _fit_tfidf(split, tracker)
    else:
        predictions, cost = _fit_transformer(control, split, device, tracker)

    labels = [int(label) for label in split.test["label"].tolist()]
    return {
        "key": control.key,
        "label": control.label,
        "question": control.question,
        "priority": control.priority,
        "kind": control.kind,
        "layers": list(control.layers) if control.layers else None,
        "n_layers": control.n_layers,
        "seed": control.seed,
        "train": cost,
        **score_predictions(predictions, labels),
    }


def _fit_tfidf(split: Any, tracker: Any) -> tuple[list[int], dict[str, Any]]:
    """The control that costs minutes and asks whether any of this needed a transformer."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    started = time.perf_counter()
    with tracker.sub(2, "tfidf") as counter:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000)
        train_x = vectorizer.fit_transform(split.train["review"].tolist())
        counter.update(1)
        model = LogisticRegression(max_iter=1000, C=4.0)
        model.fit(train_x, split.train["label"].tolist())
        counter.update(1)

    predictions = [
        int(p) for p in model.predict(vectorizer.transform(split.test["review"].tolist()))
    ]
    return predictions, {
        "seconds": round(time.perf_counter() - started, 2),
        "steps": None,
        "features": int(train_x.shape[1]),
        "note": "not a transformer, so none of the transformer protocol applies to it",
    }


def _fit_transformer(
    control: Control, split: Any, device: str, tracker: Any
) -> tuple[list[int], dict[str, Any]]:
    """Fine-tune under the notebooks' protocol, then score the test split."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if control.kind == "distilbert":
        tokenizer = AutoTokenizer.from_pretrained(DISTIL_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(DISTIL_MODEL, num_labels=2)
    else:
        assert control.layers is not None
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = masked_bert(control.layers)

    cost = train_transformer(model, tokenizer, split.train, device, tracker, control.key)
    cost["params"] = sum(p.numel() for p in model.parameters())

    model.eval()
    predictions: list[int] = []
    texts = split.test["review"].tolist()
    with tracker.sub(len(texts), f"{control.key} eval") as counter:
        for start in range(0, len(texts), EVAL_BATCH):
            batch = texts[start : start + EVAL_BATCH]
            encoded = tokenizer(
                batch,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                logits = model(**{k: v.to(device) for k, v in encoded.items()}).logits
            predictions.extend(int(index) for index in logits.argmax(dim=-1).cpu())
            counter.update(len(batch))
    return predictions, cost


def _write(
    out: Path, selected: list[Control], done: list[dict[str, Any]], split: Any, device: str
) -> None:
    """The controls file: what ran, what did not, and under exactly which protocol."""
    import torch

    finished = {entry["key"] for entry in done}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema": 1,
                "generated_by": "training/train_reference.py",
                "note": (
                    "Controls, so that the searched architectures have something to be compared "
                    "against other than each other. Trained under the protocol the shipped "
                    "checkpoints were, read out of the eval notebooks and not tuned here."
                ),
                "protocol": {
                    "epochs": EPOCHS,
                    "train_rows": len(split.train),
                    "test_rows": len(split.test),
                    "test_index_sha256": split.test_index_sha256,
                    "max_length": MAX_LENGTH,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "warmup_ratio": WARMUP_RATIO,
                    "base_model": BASE_MODEL,
                    "device": device,
                    "torch": str(torch.__version__),
                    "source": (
                        "notebooks/*/\\*_Eval.ipynb, which agree on every value. AlphaNAS received "
                        "two epochs rather than one; that is a footnote on its row, not a licence "
                        "to give the controls two."
                    ),
                },
                # Stated rather than implied. A list of six results where eighteen were planned
                # reads as "these are the controls" unless the plan is written down beside it.
                "planned": [control.key for control in selected],
                "not_run": [c.key for c in selected if c.key not in finished],
                "weights_saved": False,
                "weights_note": (
                    "The controls answer a question and the answer is a number; eighteen "
                    "fine-tuned BERTs are seven gigabytes. Re-run this script to reproduce one."
                ),
                "controls": done,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
