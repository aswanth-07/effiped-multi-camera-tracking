"""Portable filesystem and runtime settings for EffiPed."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in os.getenv(name, default).split(",") if value.strip())


@dataclass(frozen=True)
class RuntimeSettings:
    weights_dir: Path
    runtime_dir: Path
    device: str
    max_upload_mb: int
    allowed_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        default_runtime = Path.home() / ".cache" / "effiped" / "runtime"
        default_weights = Path.home() / ".cache" / "effiped" / "weights"
        return cls(
            weights_dir=Path(os.getenv("EFFIPED_WEIGHTS_DIR", default_weights)).expanduser().resolve(),
            runtime_dir=Path(os.getenv("EFFIPED_RUNTIME_DIR", default_runtime)).expanduser().resolve(),
            device=os.getenv("EFFIPED_DEVICE", "auto").strip().lower(),
            max_upload_mb=max(1, int(os.getenv("EFFIPED_MAX_UPLOAD_MB", "512"))),
            allowed_origins=_csv_env(
                "EFFIPED_ALLOWED_ORIGINS",
                "http://127.0.0.1:5173,http://localhost:5173",
            ),
        )

