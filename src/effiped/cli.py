"""Console entry points for the portfolio repository."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .manifest import load_manifest
from .settings import REPOSITORY_ROOT, RuntimeSettings


def _run(command: list[str]) -> None:
    raise SystemExit(subprocess.run(command, check=False).returncode)


def app() -> None:
    """Launch the local FastAPI-backed identity-review application."""
    from apps.api.main import run

    run()


def train() -> None:
    """Run the packaged trainer while preserving its public CLI."""
    _run([sys.executable, "-m", "effiped.train", *sys.argv[1:]])


def evaluate() -> None:
    """Run the contest evidence validator or a supplied model evaluation."""
    runner = REPOSITORY_ROOT / "tools" / "validate_results.py"
    if not runner.is_file():
        raise SystemExit("effiped-eval requires a repository checkout containing tools/validate_results.py")
    _run([sys.executable, str(runner), *sys.argv[1:]])


def demo() -> None:
    """Launch the precomputed portfolio demo locally."""
    web_root = REPOSITORY_ROOT / "apps" / "web"
    _run(["npm", "run", "dev", "--prefix", str(web_root)])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_weights() -> None:
    """Download versioned release artifacts and verify their published hashes."""
    parser = argparse.ArgumentParser(description="Download EffiPed v1 inference artifacts.")
    parser.add_argument("--weights-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    settings = RuntimeSettings.from_env()
    destination = (args.weights_dir or settings.weights_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest, artifacts = load_manifest()
    base_url = manifest["release_url"].replace("/tag/", "/download/")

    checksums_url = f"{base_url}/SHA256SUMS"
    try:
        checksums_text = urllib.request.urlopen(checksums_url, timeout=30).read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            "Weights are not published. The dataset-rights gate is documented in DATA_LICENSES.md."
        ) from exc

    expected = {}
    for line in checksums_text.splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            expected[parts[1].lstrip("*")] = parts[0].lower()

    for artifact in artifacts.values():
        output = destination / artifact.filename
        if not output.exists() or args.force:
            print(f"Downloading {artifact.filename} …")
            urllib.request.urlretrieve(f"{base_url}/{artifact.filename}", output)
        actual = _sha256(output)
        if expected.get(artifact.filename) != actual:
            output.unlink(missing_ok=True)
            raise SystemExit(f"Checksum mismatch for {artifact.filename}")
        print(f"Verified {artifact.filename} ({actual[:12]}…)")

