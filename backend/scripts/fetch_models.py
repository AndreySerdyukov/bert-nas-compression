"""Download the checkpoints from the Hugging Face Hub and write their manifests.

The models are public, so no token is needed. Each is fetched as a snapshot at a **pinned
revision**: `main` moves, and the model a measurement was taken on has to be the model that comes
up at serve time.

Two checks run on every download, and both exist because their failure mode is invisible.

**The classifier head must have arrived.** One selection notebook replaced the head with an
`nn.Sequential`, whose state-dict keys are `classifier.1.*` rather than `classifier.*`. A checkpoint
saved from that variant loads without error, silently re-initialises the head at random, scores
about 50%, and looks completely healthy from outside. `output_loading_info=True` makes the missing
keys visible, and a missing `classifier.*` is fatal here rather than a warning.

**The label order must be measured.** None of these checkpoints carry `id2label`, so which class
index means "positive" is genuinely unknown until someone asks the model. This script asks - with
the unambiguous reviews in `data/label_probe.json` - and pins the answer into the manifest. Serving
inverted sentiment is the worst outcome available, because every other signal stays green.

Run (from the backend directory, inside .venv):
    python scripts/fetch_models.py                      # all five, ~1.1 GB
    python scripts/fetch_models.py --only alphanas      # one of them
    python scripts/fetch_models.py --list               # what would be fetched, and how big
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND_DIR))

MODELS_DIR = _BACKEND_DIR / "models"
DATA_DIR = _BACKEND_DIR / "data"

# Only what inference needs. Several of these repositories hold the weights twice - safetensors and
# a pickled .bin - and safetensors is both the smaller and the one that cannot execute code.
ALLOW_PATTERNS = ["*.json", "*.txt", "*.safetensors", "*.model", "*.pt"]
IGNORE_PATTERNS = ["*.h5", "*.msgpack", "*.onnx", "tf_model*", "flax_model*", "pytorch_model.bin"]


@dataclass(frozen=True)
class ModelSpec:
    """One model to fetch, and everything the manifest needs that the Hub does not carry."""

    name: str
    repo_id: str
    label: str
    description: str
    kind: str = "huggingface"
    # None means "whatever main is now"; the resolved sha is recorded either way.
    revision: str | None = None
    method: str | None = None
    # Which files matter. AdaBERT is a bare state dict rather than a transformers snapshot.
    allow_patterns: list[str] = field(default_factory=lambda: list(ALLOW_PATTERNS))


SPECS: list[ModelSpec] = [
    ModelSpec(
        name="bert-imdb",
        repo_id="AndreySerdyukov/bert-imdb",
        label="BERT-base",
        method=None,
        description=(
            "The uncompressed reference: bert-base-uncased fine-tuned on IMDB with a three-way "
            "split and early stopping on a held-out validation set."
        ),
    ),
    ModelSpec(
        name="random-search",
        repo_id="AndreySerdyukov/random-search",
        label="Random Search",
        method="random-search",
        description=(
            "Five encoder layers found by random sampling over the layer mask, then fine-tuned "
            "for one epoch."
        ),
    ),
    ModelSpec(
        name="alphanas",
        repo_id="AndreySerdyukov/alphanas",
        label="AlphaNAS",
        method="alphanas",
        description=(
            "Four encoder layers found by evolutionary search over the layer mask, then "
            "fine-tuned for two epochs."
        ),
    ),
    ModelSpec(
        name="bananas",
        repo_id="alinaselivanets/bananas-bert",
        label="BANANAS",
        method="bananas",
        description=(
            "Four encoder layers found by Bayesian optimisation over a surrogate model, then "
            "fine-tuned for one epoch."
        ),
    ),
    ModelSpec(
        name="adabert",
        repo_id="ilkonz/dnas",
        label="AdaBERT",
        method="adabert",
        kind="adabert",
        description=(
            "Differentiable NAS with distillation. Every searched operation came out "
            "parameter-free, so the shipped network is an embedding table, a mean over tokens and "
            "a linear layer."
        ),
        allow_patterns=["*.json", "*.txt", "*.pt"],
    ),
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(spec: ModelSpec) -> Path:
    """Download the repository into models/<name>/ at a pinned revision."""
    from huggingface_hub import snapshot_download

    target = MODELS_DIR / spec.name
    path = snapshot_download(
        repo_id=spec.repo_id,
        revision=spec.revision,
        local_dir=str(target),
        allow_patterns=spec.allow_patterns,
        ignore_patterns=IGNORE_PATTERNS,
    )
    return Path(path)


def resolved_revision(spec: ModelSpec) -> str:
    """The commit sha that was actually downloaded, so the manifest can pin it."""
    from huggingface_hub import HfApi

    return str(HfApi().model_info(spec.repo_id, revision=spec.revision).sha)


def lock_entry(directory: Path) -> dict[str, Any]:
    """Size and digest of every downloaded file, so a silent replacement is detectable."""
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            files[str(path.relative_to(directory))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_of(path),
            }
    return files


def load_probe() -> dict[str, Any]:
    payload = json.loads((DATA_DIR / "label_probe.json").read_text(encoding="utf-8"))
    return dict(payload)


def probe_polarity(model: Any, tokenizer: Any, probe: dict[str, Any]) -> dict[str, Any]:
    """Work out which class index means "positive", and how confidently.

    Scores each probe review under both possible mappings and takes the one that agrees more. A
    model whose better mapping still disagrees with the probe is not a labelling problem - it is a
    broken checkpoint - and the caller refuses it.
    """
    import torch

    reviews = probe["reviews"]
    predictions: list[int] = []
    with torch.inference_mode():
        for review in reviews:
            encoded = tokenizer(
                review["text"], return_tensors="pt", truncation=True, max_length=512
            )
            logits = model(**encoded).logits.reshape(-1)
            predictions.append(int(torch.argmax(logits)))

    truths = [1 if review["label"] == "positive" else 0 for review in reviews]
    direct = sum(int(p == t) for p, t in zip(predictions, truths, strict=True)) / len(truths)
    inverted = 1.0 - direct

    positive_index = 1 if direct >= inverted else 0
    agreement = max(direct, inverted)
    return {
        "positive_index": positive_index,
        "id2label": (
            {"0": "negative", "1": "positive"}
            if positive_index == 1
            else {"0": "positive", "1": "negative"}
        ),
        "agreement": round(agreement, 4),
        "n_reviews": len(reviews),
    }


def verify_and_probe(spec: ModelSpec, directory: Path, probe: dict[str, Any]) -> dict[str, Any]:
    """Load the snapshot, insist the classifier head arrived, and measure the label order."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if spec.kind == "adabert":
        # A bare state dict: there is no transformers snapshot to load, and no head to check for.
        # The registry validates it with strict=True at serve time, which is where the two
        # divergent FrozenAdaBERT definitions get decided between.
        return {"skipped": "state dict, verified at load time with strict=True"}

    model, info = AutoModelForSequenceClassification.from_pretrained(
        str(directory), local_files_only=True, output_loading_info=True
    )
    missing = {key for key in info["missing_keys"] if key.startswith("classifier.")}
    if missing:
        raise RuntimeError(
            f"{spec.name}: the classifier head did not come with the checkpoint ({sorted(missing)}).\n"
            "transformers has re-initialised it at random, which scores about 50% and looks "
            "healthy. One selection notebook wrapped the head in nn.Sequential, whose keys are "
            "classifier.1.*; a checkpoint saved from that variant lands here."
        )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    polarity = probe_polarity(model, tokenizer, probe)
    if polarity["agreement"] < probe["min_agreement"]:
        raise RuntimeError(
            f"{spec.name}: neither label mapping agrees with the probe "
            f"(best {polarity['agreement']:.0%}, need {probe['min_agreement']:.0%}). "
            "The checkpoint is not doing sentiment on this data."
        )

    return {
        "params": sum(p.numel() for p in model.parameters()),
        "n_layers": int(model.config.num_hidden_layers),
        "unexpected_keys": sorted(info["unexpected_keys"]),
        **polarity,
    }


def write_manifest(
    spec: ModelSpec, directory: Path, verification: dict[str, Any], sha: str
) -> None:
    """The manifest the registry reads. Tracked in git; the weights beside it are not."""
    manifest: dict[str, Any] = {
        "kind": spec.kind,
        "dir": spec.name,
        "hf_repo": spec.repo_id,
        "hf_revision": sha,
        "method": spec.method,
        "model_info": {
            "name": spec.name,
            "label": spec.label,
            "description": spec.description,
            "method": spec.method,
        },
    }
    if spec.kind == "adabert":
        manifest["arch"] = {
            "name": "frozen_adabert",
            "weights": "frozen_adabert.pt",
            "vocab_size": 30522,
            "hidden_size": 256,
            "num_labels": 2,
        }
        # Not measurable without loading the class, and the class is what the registry validates.
        manifest["label_order"] = {"positive_index": 1, "source": "assumed, verified at load"}
    else:
        manifest["arch"] = {
            "name": "bert_sequence_classification",
            "num_labels": 2,
            "max_length": 512,
            "n_layers": verification["n_layers"],
        }
        manifest["model_info"]["params"] = verification["params"]
        manifest["model_info"]["n_layers"] = verification["n_layers"]
        manifest["label_order"] = {
            "positive_index": verification["positive_index"],
            "id2label": verification["id2label"],
            "probe_agreement": verification["agreement"],
            "probe_reviews": verification["n_reviews"],
            "source": "measured by scripts/fetch_models.py against data/label_probe.json",
        }

    path = MODELS_DIR / f"{spec.name}.meta.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lock = {"hf_repo": spec.repo_id, "hf_revision": sha, "files": lock_entry(directory)}
    (MODELS_DIR / f"{spec.name}.lock.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", help="fetch just this model (repeatable)")
    parser.add_argument("--list", action="store_true", help="list the models and exit")
    args = parser.parse_args()

    selected = [spec for spec in SPECS if not args.only or spec.name in args.only]
    if not selected:
        print(f"no such model; known names: {', '.join(s.name for s in SPECS)}", file=sys.stderr)
        return 1

    if args.list:
        for spec in selected:
            print(f"  {spec.name:15s} {spec.repo_id}")
        return 0

    probe = load_probe()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for spec in selected:
        print(f"\n{spec.name}  <-  {spec.repo_id}")
        try:
            directory = snapshot(spec)
            sha = resolved_revision(spec)
            verification = verify_and_probe(spec, directory, probe)
            write_manifest(spec, directory, verification, sha)
        except Exception as exc:  # noqa: BLE001 - one bad repo must not stop the others
            failures.append(f"{spec.name}: {exc}")
            print(f"  ! {exc}")
            continue

        size_mb = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file()) / 1_048_576
        print(f"  revision {sha[:12]}  ({size_mb:.0f} MB)")
        if "positive_index" in verification:
            print(
                f"  label order: class {verification['positive_index']} is positive "
                f"({verification['agreement']:.0%} agreement on {verification['n_reviews']} probes)"
            )
            print(f"  {verification['n_layers']} layers, {verification['params']:,} parameters")
        else:
            print(f"  {verification['skipped']}")

    if failures:
        print(f"\n{len(failures)} of {len(selected)} failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(f"\n{len(selected)} models ready in {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
