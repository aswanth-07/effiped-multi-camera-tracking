"""Validate the single-source evidence fixture and its public claim boundaries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "research" / "results" / "summary.json"

EXPECTED = {
    ("verified_contest_system", "pdestre", "validation", "rank1_cross"): 62.8,
    ("verified_contest_system", "pdestre", "test", "rank1_cross"): 61.3,
    ("verified_contest_system", "pdestre", "validation", "detection_map50"): 90.74,
    ("verified_contest_system", "pdestre", "test", "detection_map50"): 88.4,
    ("verified_contest_system", "mot17", "mota"): 64.08,
    ("verified_contest_system", "mot17", "idf1"): 74.24,
    ("verified_contest_system", "mot17", "hota"): 61.34,
    ("verified_contest_system", "footprint", "parameters_m"): 7.78,
    ("verified_contest_system", "footprint", "pipeline_fps_approx"): 18,
    ("post_contest_evolution", "partjde", "matched_part_readout_gain_pp"): 6.66,
    ("post_contest_evolution", "boxjde", "source_detected_rank1_gain_pp"): 13.64,
    ("post_contest_evolution", "boxjde", "source_detected_map_gain_pp"): 12.94,
    ("post_contest_evolution", "boxjde", "natural_predicted_rank1_gain_pp"): 13.31,
    ("post_contest_evolution", "boxjde", "natural_predicted_map_gain_pp"): 12.29,
    ("post_contest_evolution", "boxjde", "natural_e2e_rank1_gain_pp"): 13.01,
    ("post_contest_evolution", "boxjde", "natural_e2e_map_gain_pp"): 12.0,
}


def nested(payload: dict, keys: tuple[str, ...]):
    value = payload
    for key in keys:
        value = value[key]
    return value


def validate(config: Path | None = None) -> list[str]:
    errors: list[str] = []
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for keys, expected in EXPECTED.items():
        actual = nested(data, keys)
        if actual != expected:
            errors.append(f"{'.'.join(keys)}: expected {expected}, found {actual}")

    title = data["project"]["title"]
    if title != "Multi-Camera Pedestrian Detection, Tracking & Re-Identification using Joint ConvNeXt V2 Architecture":
        errors.append("The exact contest title changed.")
    if "not official Task 4" not in data["post_contest_evolution"]["boxjde"]["protocol"]:
        errors.append("BoxJDE protocol boundary is missing.")
    if "not presented as a pure part-only ablation" not in data["contest_submission_snapshot"]["note"]:
        errors.append("The +16.2 pp reconciliation is missing.")

    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [ROOT / "README.md", ROOT / "RESULTS.md", ROOT / "apps" / "web" / "src" / "App.tsx"]
    )
    forbidden = [
        (r"\bstate[- ]of[- ]the[- ]art\b", "unsupported state-of-the-art claim"),
        (r"\bGDPR compatible\b", "unsupported GDPR claim"),
        (r"\banonymous embeddings cannot be reversed\b", "unsupported irreversibility claim"),
        (r"canonical.{0,80}\b22\s*FPS\b", "stale canonical 22 FPS headline"),
        (r"pure part.{0,50}\+16\.2", "stale pure part-only +16.2 claim"),
    ]
    for pattern, label in forbidden:
        if re.search(pattern, public_text, re.IGNORECASE | re.DOTALL):
            errors.append(label)

    if config is not None and not config.is_file():
        errors.append(f"Config does not exist: {config}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args, _unknown = parser.parse_known_args()
    errors = validate(args.config)
    if errors:
        raise SystemExit("\n".join(f"- {error}" for error in errors))
    print(f"Validated {len(EXPECTED)} canonical metrics and public claim boundaries.")


if __name__ == "__main__":
    main()
