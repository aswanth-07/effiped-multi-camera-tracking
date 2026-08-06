"""Refresh ASSET_MANIFEST.json entries for the session_3 workbench media.

`tools/validate_media.py` requires an attributed, hashed entry for every file
under docs/media/pdestre. The session_3 demo adds a few hundred generated
crops, so those rows are produced here rather than hand-written. Re-run after
regenerating any session_3 asset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"
MANIFEST = MEDIA / "ASSET_MANIFEST.json"
SESSION = MEDIA / "pdestre" / "session3"

SOURCE_CLIPS = "P-DESTRE-derived PedestrianTracker demo_clips/session_3 source footage"
SOURCE_RENDER = "P-DESTRE-derived PedestrianTracker cross_camera_demo/session_3_4cam application output"
INDEX_NOTE = (
    "Person-search index computed offline with the BoxJDE research checkpoint "
    "(paper/runs_boxjde/boxjde_fold0_full_roi); EffiPed Tier-1 weights are withheld "
    "pending dataset rights review"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows() -> list[dict]:
    out: list[dict] = []

    for index in range(1, 5):
        cam = f"cam{index}"
        out.append(
            {
                "path": f"pdestre/session3/{cam}-source.webm",
                "source": f"{SOURCE_CLIPS} ({cam}.mp4)",
                "transformations": (
                    "Unaltered 15-second 960x540 10 FPS view re-encoded to VP9 CRF 50; audio removed"
                ),
                "purpose": f"Pre-attached source clip for camera {index} in the workbench demo",
            }
        )
        out.append(
            {
                "path": f"pdestre/session3/{cam}-tracked.webm",
                "source": f"{SOURCE_RENDER} (cross_camera_demo.mp4)",
                "transformations": (
                    f"Cropped the 960x540 quadrant for sequence {index} out of the 1920x1080 2x2 grid "
                    "and re-encoded to VP9 CRF 50; original detector and tracker overlays retained; "
                    "audio removed"
                ),
                "purpose": f"Single-camera tracked output for camera {index}",
            }
        )
        out.append(
            {
                "path": f"pdestre/session3/{cam}-poster.webp",
                "source": f"{SOURCE_RENDER} (cross_camera_demo.mp4)",
                "transformations": (
                    f"Frame at 00:06 from the sequence {index} quadrant, resized to 640 px wide and "
                    "encoded as WebP"
                ),
                "purpose": f"Poster frame for the camera {index} clips",
            }
        )

    out.append(
        {
            "path": "pdestre/session3/detection-input.webp",
            "source": f"{SOURCE_CLIPS} (cam1.mp4)",
            "transformations": "Frame at 00:06 encoded as WebP quality 85; no annotations",
            "purpose": "Input image for the image-detection panel",
        }
    )
    out.append(
        {
            "path": "pdestre/session3/detection-output.webp",
            "source": f"{SOURCE_RENDER} (cross_camera_demo.mp4)",
            "transformations": (
                "Frame at 00:06 from the sequence 1 quadrant encoded as WebP quality 85; original "
                "detector annotations retained"
            ),
            "purpose": "Annotated result for the image-detection panel",
        }
    )

    for crop in sorted((SESSION / "crops").glob("*.webp")):
        stem = crop.stem.replace("_crop", "")
        video_token = stem.split("_", 1)[0]
        try:
            video_index = int(video_token.lstrip("v")) + 1
        except ValueError:
            video_index = 0
        out.append(
            {
                "path": f"pdestre/session3/crops/{crop.name}",
                "source": f"{SOURCE_CLIPS} (cam{video_index}.mp4)",
                "transformations": (
                    "Person bounding-box crop resized to 192 px tall and encoded as WebP quality 82; "
                    f"{INDEX_NOTE}"
                ),
                "purpose": "Detected-person thumbnail in the person-search gallery",
            }
        )

    for row in out:
        row["sha256"] = sha256(MEDIA / row["path"])
    return out


def main() -> int:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    kept = [row for row in payload["assets"] if not row["path"].startswith("pdestre/session3/")]
    payload["assets"] = kept + rows()
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"manifest now lists {len(payload['assets'])} assets ({len(payload['assets']) - len(kept)} session_3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
