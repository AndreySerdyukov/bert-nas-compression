"""The model registry: manifests in, loaded predictors out.

`models/<name>.meta.json` is the only place that knows where a model comes from and what is true
about it. `kind` picks the wrapper - a transformers snapshot or AdaBERT's bare state dict - and
everything else in the manifest was measured at download time by `scripts/fetch_models.py`.

The unusual part is that **a model that fails to load is kept, not forgotten**. Weights are
gitignored and fetched separately, so an empty registry is the normal state of a fresh clone rather
than a fault; the app has to come up either way. But answering as though a checkpoint had never
existed would make "you have not downloaded it yet" indistinguishable from "it broke". So each
failure is recorded with its reason and, where there is one, the command that fixes it, and
`/api/models` reports both lists.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.schemas.models import ModelInfo, SkippedModel
from app.serving.base import ModelPredictor
from app.serving.probe import LabelProbe

logger = logging.getLogger(__name__)

FETCH_COMMAND = "python scripts/fetch_models.py"


@dataclass(frozen=True)
class LoadedModel:
    """A predictor and the description that travels with it."""

    predictor: ModelPredictor
    info: ModelInfo


@dataclass
class ModelRegistry:
    """Every manifest on disk, each either loaded or skipped with a reason."""

    models_dir: Path
    data_dir: Path
    baseline: str
    device: str = "cpu"

    _models: dict[str, LoadedModel] = field(default_factory=dict, init=False)
    _skipped: dict[str, SkippedModel] = field(default_factory=dict, init=False)

    def load(self, only: str | None = None) -> None:
        """Read every manifest and build what can be built.

        `only` builds a single model and ignores the rest. The benchmark uses it to load one
        checkpoint at a time when it measures how much memory each one costs - a figure that means
        nothing if five of them are resident at once.
        """
        self._models.clear()
        self._skipped.clear()
        if not self.models_dir.exists():
            logger.warning("no models directory at %s", self.models_dir)
            return

        probe = LabelProbe.load(self.data_dir)
        for path in sorted(self.models_dir.glob("*.meta.json")):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            name = str(manifest.get("dir") or path.name.removesuffix(".meta.json"))
            if only is not None and name != only:
                continue
            try:
                info = self._describe(manifest)
            except (KeyError, TypeError, ValueError) as exc:
                self._skip(name, name, f"the manifest is malformed: {exc}", None)
                continue
            predictor = self._build(manifest, info, probe)
            if predictor is not None:
                self._models[info.name] = LoadedModel(predictor=predictor, info=info)

    def _describe(self, manifest: dict[str, Any]) -> ModelInfo:
        """Manifest -> the description the API publishes. Raises if the manifest is incoherent."""
        from app.serving.bert_classifier import resolve_positive_index

        model_info = dict(manifest["model_info"])
        name = str(model_info["name"])
        return ModelInfo(
            name=name,
            label=str(model_info["label"]),
            description=str(model_info.get("description", "")),
            method=model_info.get("method"),
            params=model_info.get("params"),
            n_layers=model_info.get("n_layers"),
            max_length=int(manifest["arch"]["max_length"]),
            hf_repo=str(manifest["hf_repo"]),
            hf_revision=str(manifest["hf_revision"]),
            positive_index=resolve_positive_index(manifest, name),
            is_baseline=name == self.baseline,
        )

    def _build(
        self, manifest: dict[str, Any], info: ModelInfo, probe: LabelProbe
    ) -> ModelPredictor | None:
        """Construct the predictor for a manifest kind, or record why it could not be built.

        Every branch sits inside the `try`, including the raise for an unknown kind. Outside it, a
        typo in one manifest's `kind` would take down not that model but the whole startup: the
        exception escapes into `create_app()`.
        """
        kind = str(manifest.get("kind", ""))
        try:
            # Both imports are deferred: transformers and torch take seconds to import, and the
            # methodology half of this application needs neither.
            if kind == "huggingface":
                from app.serving.bert_classifier import build_bert_classifier

                return build_bert_classifier(
                    manifest, info, self.models_dir, probe, device=self.device
                )
            if kind == "adabert":
                from app.serving.adabert import build_adabert

                return build_adabert(manifest, info, self.models_dir, probe, device=self.device)
            raise ValueError(f"unsupported model kind {kind!r}")
        except FileNotFoundError as exc:
            self._skip(info.name, info.label, f"the weights are not downloaded ({exc})", None)
        except (KeyError, ValueError, RuntimeError, OSError, ImportError) as exc:
            # RuntimeError covers a state dict that does not fit the architecture; OSError, a
            # half-downloaded snapshot; ImportError, a serving dependency that is not installed
            # while the rest of the app still has work to do.
            self._skip(info.name, info.label, str(exc), None)
        return None

    def warmup(self) -> None:
        """Run every predictor's warm-up, dropping any model that fails it.

        For these models warm-up is the polarity check, so a failure here means the model would
        have served inverted sentiment. Dropping it is the whole point: coming up without a model
        beats coming up with a confidently wrong one.
        """
        broken: list[str] = []
        for name, model in self._models.items():
            try:
                model.predictor.warmup()
            except Exception as exc:  # noqa: BLE001 - at startup, coming up beats falling over
                logger.warning("warmup failed for %r, dropping it: %s", name, exc)
                self._skip(name, model.info.label, str(exc), None)
                broken.append(name)
        for name in broken:
            del self._models[name]

    def _skip(self, name: str, label: str, reason: str, remedy: str | None) -> None:
        if remedy is None and "not downloaded" in reason:
            remedy = f"{FETCH_COMMAND} --only {name}"
        logger.warning("skipping model %r: %s", name, reason)
        self._skipped[name] = SkippedModel(name=name, label=label, reason=reason, remedy=remedy)

    # --- reading -------------------------------------------------------------------------------

    def list_infos(self) -> list[ModelInfo]:
        """Loaded models, baseline first, then in manifest order."""
        infos = [model.info for model in self._models.values()]
        return sorted(infos, key=lambda info: (not info.is_baseline, info.name))

    def get(self, name: str) -> LoadedModel | None:
        """A loaded model, or None if it is unknown or was skipped."""
        return self._models.get(name)

    def is_known(self, name: str) -> bool:
        """Whether a manifest exists for this name, loaded or not.

        This is what separates a 404 from a 503: an unknown name is a mistake by the caller, a
        known name whose weights are absent is a missing download.
        """
        return name in self._models or name in self._skipped

    @property
    def skipped(self) -> dict[str, SkippedModel]:
        """Models with a manifest that are not being served, keyed by name."""
        return dict(self._skipped)
