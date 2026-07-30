"""Repository hygiene checks for a public portfolio release."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".json", ".yaml", ".yml", ".md", ".toml", ".html", ".css", ".cff"}
SKIP_PARTS = {".git", ".venv", "node_modules", "dist", "tmp"}
FORBIDDEN_NAMES = {"datasets", "checkpoints", "runs_compact", "runs_fivefold", "asu_reid"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".mp4", ".mov", ".avi", ".mkv"}
ARCHITECTURE_SVG = Path("docs/architecture/effiped-architecture.svg")


def validate_svg(path: Path, relative: Path) -> list[str]:
    """Reject mislabeled raster exports and malformed SVG documents."""

    errors: list[str] = []
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return [f"PNG bytes mislabeled as SVG: {relative}"]
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        return [f"Malformed SVG: {relative} ({exc})"]
    if root.tag.rsplit("}", 1)[-1] != "svg":
        errors.append(f"SVG root element missing: {relative}")
    if relative == ARCHITECTURE_SVG:
        child_names = {child.tag.rsplit("}", 1)[-1] for child in root}
        for required in {"title", "desc"}:
            if required not in child_names:
                errors.append(f"Architecture SVG missing accessible {required}: {relative}")
    return errors


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
        if path.suffix.lower() == ".svg":
            errors.extend(validate_svg(path, relative))
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
