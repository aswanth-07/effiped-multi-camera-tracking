"""Verify attribution coverage and hashes for every P-DESTRE-derived asset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"
MANIFEST = MEDIA / "ASSET_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate() -> list[str]:
    errors: list[str] = []
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    listed = {row["path"]: row for row in payload["assets"]}
    committed = {
        path.relative_to(MEDIA).as_posix(): path
        for path in (MEDIA / "pdestre").rglob("*")
        if path.is_file()
    }
    if listed.keys() != committed.keys():
        errors.append(f"Manifest coverage differs: listed={sorted(listed)} committed={sorted(committed)}")
    for relative, path in committed.items():
        actual = sha256(path)
        if listed.get(relative, {}).get("sha256") != actual:
            errors.append(f"Hash mismatch: {relative}")
        for field in ("source", "transformations", "purpose"):
            if not listed.get(relative, {}).get(field):
                errors.append(f"{relative} is missing {field}")
    if payload.get("license") != "CC BY-NC-SA 4.0":
        errors.append("Media license changed.")
    return errors


if __name__ == "__main__":
    problems = validate()
    if problems:
        raise SystemExit("\n".join(f"- {problem}" for problem in problems))
    print("Media attribution manifest and SHA-256 hashes are valid.")
