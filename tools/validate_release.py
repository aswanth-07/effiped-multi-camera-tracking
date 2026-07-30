"""Repository hygiene checks for a public portfolio release."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".json", ".yaml", ".yml", ".md", ".toml", ".html", ".css", ".cff"}
SKIP_PARTS = {".git", ".venv", "node_modules", "dist", "tmp"}
FORBIDDEN_NAMES = {"datasets", "checkpoints", "runs_compact", "runs_fivefold", "asu_reid"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".mp4", ".mov", ".avi", ".mkv"}


def validate() -> list[str]:
    errors: list[str] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        if path.is_dir() and path.name.lower() in FORBIDDEN_NAMES:
            errors.append(f"Forbidden directory: {relative}")
            continue
        if not path.is_file():
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"Forbidden binary artifact: {relative}")
        if path.stat().st_size > 20 * 1024 * 1024:
            errors.append(f"File exceeds 20 MB: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES or path.name == "LICENSE":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"[A-Za-z]:\\(?:Users|Lenovo_data)\\", text):
            errors.append(f"Absolute Windows path: {relative}")
        if re.search(r"/(?:home|Users)/[^/\s]+/", text):
            errors.append(f"Absolute user path: {relative}")
        if "sys.path.append" in text or "sys.path.insert" in text:
            errors.append(f"sys.path manipulation: {relative}")
        if re.search(r"\bASU-ReID\b|\basu_reid\b", text, re.IGNORECASE):
            errors.append(f"Unrelated project reference: {relative}")
        if "paper/configs/compact" in text or "runs_compact" in text or "runs_fivefold" in text:
            errors.append(f"Stale research path: {relative}")
    return errors


if __name__ == "__main__":
    problems = validate()
    if problems:
        raise SystemExit("\n".join(f"- {problem}" for problem in problems))
    print("Release hygiene checks passed.")
