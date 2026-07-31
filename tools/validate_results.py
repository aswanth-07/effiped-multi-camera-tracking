"""Validate the single-source evidence fixture and public claim boundaries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "research" / "results" / "summary.json"

EXPECTED = {
    ("system_benchmarks", "pdestre", "validation", "rank1_cross"): 62.8,
    ("system_benchmarks", "pdestre", "test", "rank1_cross"): 61.3,
    ("system_benchmarks", "pdestre", "validation", "detection_map50"): 90.74,
    ("system_benchmarks", "pdestre", "test", "detection_map50"): 88.4,
    ("system_benchmarks", "mot17", "mota"): 64.08,
    ("system_benchmarks", "mot17", "idf1"): 74.24,
    ("system_benchmarks", "mot17", "hota"): 61.34,
    ("system_benchmarks", "footprint", "parameters_m"): 7.78,
    ("system_benchmarks", "footprint", "pipeline_fps_approx"): 18,
    ("research_extensions", "partjde", "matched_part_readout_gain_pp"): 6.66,
    ("research_extensions", "boxjde", "source_detected_rank1_gain_pp"): 13.64,
    ("research_extensions", "boxjde", "source_detected_map_gain_pp"): 12.94,
    ("research_extensions", "boxjde", "natural_predicted_rank1_gain_pp"): 13.31,
    ("research_extensions", "boxjde", "natural_predicted_map_gain_pp"): 12.29,
    ("research_extensions", "boxjde", "natural_e2e_rank1_gain_pp"): 13.01,
    ("research_extensions", "boxjde", "natural_e2e_map_gain_pp"): 12.0,
}

PUBLIC_SURFACES = [
    ROOT / "README.md",
    ROOT / "RESULTS.md",
    ROOT / "MODEL_CARD.md",
    ROOT / "apps" / "web" / "index.html",
    ROOT / "apps" / "web" / "src" / "App.tsx",
    ROOT / "apps" / "web" / "src" / "components" / "DemoConsole.tsx",
]


def nested(payload: dict, keys: tuple[str, ...]):
    value = payload
    for key in keys:
        value = value[key]
    return value


def validate(config: Path | None = None) -> list[str]:
    errors: list[str] = []
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))

    if data.get("schema_version") != 2:
        errors.append("Evidence fixture must use schema version 2.")

    for keys, expected in EXPECTED.items():
        actual = nested(data, keys)
        if actual != expected:
            errors.append(f"{'.'.join(keys)}: expected {expected}, found {actual}")

    title = data["project"]["title"]
    expected_title = (
        "Multi-Camera Pedestrian Detection, Tracking & Re-Identification "
        "using Joint ConvNeXt V2 Architecture"
    )
    if title != expected_title:
        errors.append("The public project title changed.")
    if "not official Task 4" not in data["research_extensions"]["boxjde"]["protocol"]:
        errors.append("BoxJDE protocol boundary is missing.")

    videos = data["demo_case"]["videos"]
    if len(videos) != 2 or {video["id"] for video in videos} != {"tracking", "cross-camera"}:
        errors.append("The hosted demo must expose both archived replay variants.")
    if any(len(subject["candidates"]) != 4 for subject in data["demo_case"]["subjects"]):
        errors.append("Every demo subject must include four ranked candidates.")

    public_text = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_SURFACES)
    event_terms = ["con" + "test", "compe" + "tition", "pr" + "ize", "aw" + "ard", "SI" + "PC"]
    forbidden = [
        (r"\bstate[- ]of[- ]the[- ]art\b", "unsupported state-of-the-art claim"),
        (r"\bGDPR compatible\b", "unsupported GDPR claim"),
        (r"\banonymous embeddings cannot be reversed\b", "unsupported irreversibility claim"),
        (rf"\b(?:{'|'.join(event_terms)})\b", "non-standalone event framing"),
        (r"canonical.{0,80}\b22\s*FPS\b", "stale canonical 22 FPS headline"),
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
