"""The single, versioned source of truth for downloadable model artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from .settings import REPOSITORY_ROOT, RuntimeSettings


@dataclass(frozen=True)
class ModelArtifact:
    key: str
    label: str
    fold: int
    readout: str
    descriptor_dim: int
    filename: str
    config: str
    description: str
    artifact_version: str
    release_url: str
    benchmark: dict[str, Any]

    def config_path(self) -> Path:
        return REPOSITORY_ROOT / self.config

    def checkpoint_path(self, settings: RuntimeSettings) -> Path:
        return settings.weights_dir / self.filename

    def public_dict(self, settings: RuntimeSettings) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "fold": self.fold,
            "readout": self.readout,
            "descriptor_dim": self.descriptor_dim,
            "artifact_version": self.artifact_version,
            "available": self.checkpoint_path(settings).is_file(),
            "benchmark": self.benchmark,
        }


def load_manifest() -> tuple[dict[str, Any], dict[str, ModelArtifact]]:
    payload = json.loads(files("effiped").joinpath("model_manifest.json").read_text(encoding="utf-8"))
    artifacts = {}
    for row in payload["models"]:
        artifact = ModelArtifact(
            **row,
            artifact_version=payload["artifact_version"],
            release_url=payload["release_url"],
        )
        artifacts[artifact.key] = artifact
    return payload, artifacts

