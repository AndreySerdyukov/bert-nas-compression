"""Extract the data spine - search trajectories and shipped architectures - from the notebooks.

Why parse rather than transcribe: the charts in the methodology section plot what the searches
actually did, and a hand-typed table would drift from its source the first time anyone touched
either. Here the notebooks stay the primary record, and `--check` in CI re-parses them and fails if
a committed file no longer matches - so the two cannot disagree without someone noticing.

Two files come out of this:

  data/search_trajectories.json  every candidate each search evaluated, from the *_Selection
                                 notebooks. Three searches printed three formats, so there is one
                                 parser each; what they share is the shape of a candidate - a
                                 12-bit layer mask, its accuracy, its parameter count, and the
                                 fitness the search ranked it by.

  data/architectures.json        the architecture each method actually shipped, read from the
                                 `layers_flag` literal in the *_Eval notebooks - the notebooks that
                                 trained and uploaded the checkpoints. This is deliberately a
                                 different source from the trajectories, because for Random Search
                                 the two disagree, and that disagreement is a published finding
                                 rather than something to reconcile away.

Run (from the backend directory, inside .venv):
    python training/extract_notebook_data.py            # rewrite both files
    python training/extract_notebook_data.py --check    # verify them, write nothing
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

_BACKEND_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_DIR.parent
TRAJECTORIES_OUT = _BACKEND_DIR / "data" / "search_trajectories.json"
ARCHITECTURES_OUT = _BACKEND_DIR / "data" / "architectures.json"

# Every candidate's parameter count is fully determined by how many layers it keeps. Recomputing it
# here and comparing against what the notebook printed is a free consistency check on both.
NON_LAYER_PARAMS = 24_429_314
PARAMS_PER_LAYER = 7_087_872

NOTEBOOKS = {
    "random-search": "notebooks/random_search/RandomSearch_Selection.ipynb",
    "alphanas": "notebooks/alphanas/AlphaNas_Selection.ipynb",
    "bananas": "notebooks/bananas/BANANAS_Selection.ipynb",
}


def params_for(n_layers: int) -> int:
    """Parameter count of a BERT-base classifier keeping `n_layers` encoder layers."""
    return NON_LAYER_PARAMS + PARAMS_PER_LAYER * n_layers


def cell_output_lines(path: Path) -> list[str]:
    """Every line of stdout the notebook recorded, in order."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    lines: list[str] = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", []))
            if not text and "data" in output:
                text = "".join(output["data"].get("text/plain", []))
            lines.extend(text.splitlines())
    return lines


def _mask_from_indices(indices: list[int]) -> list[int]:
    mask = [0] * 12
    for index in indices:
        mask[index] = 1
    return mask


def _layers_from_mask(mask: list[int]) -> list[int]:
    return [i for i, bit in enumerate(mask) if bit]


def _candidate(
    mask: list[int], accuracy: float, params: int, fitness: float | None, **extra: Any
) -> dict[str, Any]:
    """One evaluated architecture, with the printed parameter count checked against the formula."""
    layers = _layers_from_mask(mask)
    expected = params_for(len(layers))
    if params != expected:
        raise ValueError(
            f"mask {mask} has {len(layers)} layers: printed {params}, formula {expected}"
        )
    return {
        "mask": mask,
        "layers": layers,
        "n_layers": len(layers),
        "accuracy": accuracy,
        "params": params,
        "fitness": fitness,
        **extra,
    }


# --- Random Search -------------------------------------------------------------------------

_RS_ARCH = re.compile(r"архитекура с параметрами (\d+): \{'layer_indices': \[([\d, ]*)\]\}")
_RS_SCORE = re.compile(
    r"Accuracy = ([\d.]+), Params = ([\d,]+), Time = ([\d.]+)s, Custom Score = ([\d.]+)"
)


def parse_random_search(lines: list[str]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    pending: list[int] | None = None
    for line in lines:
        arch = _RS_ARCH.search(line)
        if arch:
            pending = [int(x) for x in arch.group(2).split(",") if x.strip()]
            continue
        score = _RS_SCORE.search(line)
        if score and pending is not None:
            candidates.append(
                _candidate(
                    _mask_from_indices(pending),
                    accuracy=float(score.group(1)),
                    params=int(score.group(2).replace(",", "")),
                    fitness=float(score.group(4)),
                    trial=len(candidates) + 1,
                    train_seconds=float(score.group(3)),
                )
            )
            pending = None
    return {
        "candidates": candidates,
        "stages": [{"name": f"trial {c['trial']}", "index": c["trial"]} for c in candidates],
    }


# --- AlphaNAS ------------------------------------------------------------------------------

_AN_GENERATION = re.compile(r"Поколение (\d+)")
_AN_REJECT = re.compile(r"Отменяем архитектуру: параметров модели (\d+) > трешхолда")
_AN_CANDIDATE = re.compile(
    r"Кандидат \(mask = \[([\d, ]+)\]\): аккураси: ([\d.]+), параметров: (\d+), "
    r"кастомная метрика: (-?[\d.]+|-inf)"
)
_AN_BEST = re.compile(r"Лучшая архитектура: \[([\d, ]+)\]")


def parse_alphanas(lines: list[str]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    generation = 0
    rejected_next = False
    best_seen = False
    for line in lines:
        gen = _AN_GENERATION.search(line)
        if gen:
            generation = int(gen.group(1))
            continue
        if _AN_REJECT.search(line):
            # The rejection is printed just before the candidate it refers to.
            rejected_next = True
            continue
        if _AN_BEST.search(line):
            # Everything after this line is the final re-evaluation, which passed the validation
            # loader as both train and eval set. Recorded, and flagged, rather than dropped.
            best_seen = True
            continue
        found = _AN_CANDIDATE.search(line)
        if not found:
            continue
        fitness = -math.inf if found.group(4) == "-inf" else float(found.group(4))
        candidates.append(
            _candidate(
                [int(x) for x in found.group(1).split(",")],
                accuracy=float(found.group(2)),
                params=int(found.group(3)),
                fitness=None if fitness == -math.inf else fitness,
                generation=generation,
                rejected=rejected_next,
                rejection_reason="parameter cap of 1e8 exceeded" if rejected_next else None,
                final_reevaluation=best_seen,
            )
        )
        rejected_next = False
    return {
        "candidates": candidates,
        "stages": [
            {"name": f"generation {g}", "index": g}
            for g in sorted({c["generation"] for c in candidates if not c["final_reevaluation"]})
        ],
    }


# --- BANANAS -------------------------------------------------------------------------------

_BA_EVALUATED = re.compile(
    r"Конфигурация \(config = \[([\d, ]+)\]\): точность: ([\d.]+), параметров: (\d+), "
    r"метрика: (-?[\d.]+)"
)
_BA_PREDICTED = re.compile(r"Конфигурация: \[([\d, ]+)\], предсказанная метрика: (-?[\d.]+)")
_BA_INITIAL = re.compile(r"Начальная конфигурация (\d+):")


def parse_bananas(lines: list[str]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    # The surrogate's predictions for each acquisition round. Keeping them is what lets the
    # methodology chapter plot predicted fitness against the fitness the candidate actually got -
    # the only direct look at whether the surrogate was learning anything.
    proposals: list[dict[str, Any]] = []
    round_index = 0
    initial_phase = True
    for line in lines:
        predicted = _BA_PREDICTED.search(line)
        if predicted:
            mask = [int(x) for x in predicted.group(1).split(",")]
            proposals.append(
                {
                    "round": round_index,
                    "mask": mask,
                    "n_layers": sum(mask),
                    "predicted_fitness": float(predicted.group(2)),
                }
            )
            continue
        if _BA_INITIAL.search(line):
            continue
        evaluated = _BA_EVALUATED.search(line)
        if not evaluated:
            continue
        mask = [int(x) for x in evaluated.group(1).split(",")]
        if not initial_phase or len(candidates) >= 5:
            initial_phase = False
        candidates.append(
            _candidate(
                mask,
                accuracy=float(evaluated.group(2)),
                params=int(evaluated.group(3)),
                fitness=float(evaluated.group(4)),
                index=len(candidates) + 1,
                phase="initial" if initial_phase else "bayesian",
            )
        )
        if not initial_phase:
            round_index += 1
        if len(candidates) == 5:
            initial_phase = False
            round_index = 1
    return {"candidates": candidates, "proposals": proposals, "stages": []}


PARSERS = {
    "random-search": parse_random_search,
    "alphanas": parse_alphanas,
    "bananas": parse_bananas,
}

# Context the charts need and the printed logs do not carry. Every number here is stated in the
# notebook's own code cells; see notebooks/README.md for why they are not comparable to each other.
BUDGETS = {
    "random-search": {
        "train_rows": 2800,
        "val_rows": 700,
        "max_length": 512,
        "epochs_per_candidate": 1,
        "fitness": "alpha * accuracy + beta * (1 - log(params + 1) / log(max_params))",
        "fitness_params": {"alpha": 0.3, "beta": 0.7, "max_params": 110_000_000},
    },
    "alphanas": {
        "train_rows": 5600,
        "val_rows": 1400,
        "max_length": 512,
        "epochs_per_candidate": 1,
        "fitness": "accuracy - 1e-8 * params, rejected if accuracy < 0.75 or params > 1e8",
        "fitness_params": {"penalty": 1e-8, "min_accuracy": 0.75, "max_params": 1e8},
    },
    "bananas": {
        "train_rows": 560,
        "val_rows": 140,
        "max_length": 128,
        "epochs_per_candidate": 1,
        "fitness": "accuracy - 1e-8 * params",
        "fitness_params": {"penalty": 1e-8},
        "surrogate": "ensemble of 3 MLPs (12 -> 16 -> 16 -> 1), MSE, independent Thompson sampling",
    },
}


# --- what each method shipped -------------------------------------------------------------

_LAYERS_FLAG = re.compile(r"layers?_flag\s*=\s*\[([01, ]+)\]")
_TRAIN_CALL = re.compile(r"num_epochs\s*=\s*(\d+)")

# The eval notebooks are what trained and uploaded the published checkpoints, so the mask they set
# is the architecture that exists on the Hub - whatever the write-up says.
EVAL_NOTEBOOKS = {
    "random-search": "notebooks/random_search/RandomSearch_Eval.ipynb",
    "alphanas": "notebooks/alphanas/AlphaNas_Eval.ipynb",
    "bananas": "notebooks/bananas/BANANAS_Eval.ipynb",
}

HUB_REPOS = {
    "random-search": "AndreySerdyukov/random-search",
    "alphanas": "AndreySerdyukov/alphanas",
    "bananas": "alinaselivanets/bananas-bert",
}

METHOD_LABELS = {
    "random-search": "Random Search",
    "alphanas": "AlphaNAS",
    "bananas": "BANANAS",
}


def notebook_sources(path: Path) -> list[str]:
    """Every code cell's source, in order."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def parse_shipped_architecture(path: Path) -> dict[str, Any]:
    """Read the layer mask and the fine-tuning length out of an eval notebook."""
    sources = notebook_sources(path)
    masks = [
        [int(bit) for bit in found.group(1).split(",")]
        for source in sources
        for found in [_LAYERS_FLAG.search(source)]
        if found
    ]
    if len(masks) != 1:
        raise ValueError(
            f"{path.name}: expected exactly one layers_flag literal, found {len(masks)}"
        )
    mask = masks[0]
    if len(mask) != 12:
        raise ValueError(f"{path.name}: layers_flag has {len(mask)} entries, expected 12")

    # The last `num_epochs=` in the notebook is the one on the training call; earlier ones are
    # defaults in the class signature.
    epochs = [int(m.group(1)) for source in sources for m in _TRAIN_CALL.finditer(source)]
    layers = _layers_from_mask(mask)
    return {
        "mask": mask,
        "layers": layers,
        "n_layers": len(layers),
        "params": params_for(len(layers)),
        "finetune_epochs": epochs[-1] if epochs else None,
    }


def build_architectures(trajectories: dict[str, Any]) -> dict[str, Any]:
    """The shipped architectures, cross-checked against what each search reported."""
    methods: dict[str, Any] = {}
    for name, relative in EVAL_NOTEBOOKS.items():
        shipped = parse_shipped_architecture(_REPO_ROOT / relative)

        # What the search itself concluded, so the two can be compared rather than conflated.
        candidates = trajectories["methods"][name]["candidates"]
        ranked = [c for c in candidates if c["fitness"] is not None and not c.get("rejected")]
        best = max(ranked, key=lambda c: c["fitness"]) if ranked else None

        methods[name] = {
            "label": METHOD_LABELS[name],
            "hf_repo": HUB_REPOS[name],
            "eval_notebook": relative,
            "selection_notebook": trajectories["methods"][name]["notebook"],
            "shipped": shipped,
            "search_best": None
            if best is None
            else {"mask": best["mask"], "layers": best["layers"]},
            # True for Random Search and nothing else: the notebook that trained the published
            # model used a different mask from the one the search reported as its winner. Recorded
            # as data so the UI and a unit test can both point at it.
            "shipped_matches_search": best is not None and best["mask"] == shipped["mask"],
        }

    # AdaBERT is not a layer mask at all, so it carries no mask and says why.
    methods["adabert"] = {
        "label": "AdaBERT",
        "hf_repo": "ilkonz/dnas",
        "eval_notebook": "notebooks/dnas/Differentiable_NAS.ipynb",
        "selection_notebook": "notebooks/dnas/Differentiable_NAS.ipynb",
        "shipped": {
            "mask": None,
            "layers": None,
            "n_layers": None,
            # 30522*256 embeddings + 256*2 + 2 classifier. Every searched operation was avg_pool,
            # which has no parameters, so this is the whole model.
            "params": 30522 * 256 + 256 * 2 + 2,
            "finetune_epochs": 3,
        },
        "search_best": None,
        "shipped_matches_search": None,
        "note": (
            "Gumbel-softmax selected avg_pool for all five surviving cells. avg_pool is "
            "parameter-free, so the shipped network is Embedding(30522, 256) -> mean over tokens "
            "-> Linear(256, 2): a bag of embeddings, not a distilled transformer."
        ),
    }

    return {
        "schema": 1,
        "source": "parsed from the eval notebooks by training/extract_notebook_data.py",
        "baseline": {
            "label": "BERT-base",
            "hf_repo": "AndreySerdyukov/bert-imdb",
            "n_layers": 12,
            "params": params_for(12),
            "eval_notebook": "notebooks/bert_finetune/Bert_Finetune_Andrey.ipynb",
        },
        "params_formula": {"base": NON_LAYER_PARAMS, "per_layer": PARAMS_PER_LAYER},
        "methods": methods,
    }


def build_trajectories() -> dict[str, Any]:
    """Parse all three selection notebooks into the committed structure."""
    methods: dict[str, Any] = {}
    for name, relative in NOTEBOOKS.items():
        path = _REPO_ROOT / relative
        parsed = PARSERS[name](cell_output_lines(path))
        methods[name] = {"notebook": relative, "budget": BUDGETS[name], **parsed}
    return {
        "schema": 1,
        "source": "parsed from the selection notebooks by training/extract_notebook_data.py",
        "params_formula": {"base": NON_LAYER_PARAMS, "per_layer": PARAMS_PER_LAYER},
        "methods": methods,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed files match the notebooks; write nothing",
    )
    args = parser.parse_args()

    trajectories = build_trajectories()
    architectures = build_architectures(trajectories)
    outputs = {TRAJECTORIES_OUT: trajectories, ARCHITECTURES_OUT: architectures}

    if args.check:
        for path, built in outputs.items():
            serialized = json.dumps(built, indent=2, ensure_ascii=False) + "\n"
            if not path.exists():
                print(
                    f"{path.name} is missing; run training/extract_notebook_data.py",
                    file=sys.stderr,
                )
                return 1
            if path.read_text(encoding="utf-8") != serialized:
                print(
                    f"{path.name} no longer matches the notebooks it was parsed from.\n"
                    "Re-run training/extract_notebook_data.py and commit the result.",
                    file=sys.stderr,
                )
                return 1
        counts = ", ".join(
            f"{k}: {len(v['candidates'])}" for k, v in trajectories["methods"].items()
        )
        print(f"search_trajectories.json matches the notebooks ({counts})")
        print(f"architectures.json matches the notebooks ({len(architectures['methods'])} methods)")
        return 0

    for path, built in outputs.items():
        path.write_text(json.dumps(built, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for name, method in trajectories["methods"].items():
        print(f"  {name:14s} {len(method['candidates']):2d} candidates from {method['notebook']}")
    print(f"  -> {TRAJECTORIES_OUT.name}")
    for name, method in architectures["methods"].items():
        shipped = method["shipped"]
        match = method["shipped_matches_search"]
        flag = "" if match is not False else "   <- differs from what the search reported"
        print(f"  {name:14s} shipped {shipped['layers']}{flag}")
    print(f"  -> {ARCHITECTURES_OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
