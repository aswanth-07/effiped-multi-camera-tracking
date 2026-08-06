"""Rebuild apps/web/src/data/person-search.json from the session_3 clips.

This is an **offline, one-off** provenance tool. It is deliberately not wired
into CI: it needs a GPU, a research checkout, and a checkpoint that is not
distributed with this repository.

Why it exists
-------------
The hosted build is static, so the person-search panel replays a precomputed
index. This script records exactly how that index was produced.

Prerequisites
-------------
1. A PedestrianTracker checkout providing ``src.paper_runtime`` (with
   ``PaperRuntime`` and ``preset_from_key``) and ``src.tracker``.
2. Its ``backend/person_search_service.py`` — the ``PersonSearchManager`` used
   here. ``apps/api/person_search_service.py`` in this repo is the same class
   wired to ``effiped.runtime`` instead; that path needs the EffiPed Tier-1
   weights, which are withheld pending dataset rights review. The published
   fixture was therefore generated with the BoxJDE research checkpoint
   (``paper/runs_boxjde/boxjde_fold0_full_roi/best_model.pth``), and every
   surface that shows it says so.
3. ``BOXJDE_PAPER_ROOT`` pointing at the research ``paper/`` tree.

Usage
-----
Point PYTHONPATH at the PedestrianTracker checkout (for ``src.*``) and at the
directory holding the ``backend`` package, then::

    PYTHONPATH=/path/to/PedestrianTracker:/path/to/checkout \
    BOXJDE_PAPER_ROOT=/path/to/paper \
    python tools/build_person_search_fixture.py \
        --clips /path/to/demo_clips/session_3 \
        --out apps/web/src/data/person-search.json \
        --assets docs/media/pdestre/session3/crops

Afterwards run ``tools/build_session3_manifest.py`` to refresh the attribution
hashes, then ``tools/validate_media.py``.

Design notes
------------
* Scene images are **not** exported. A full-frame JPEG per appearance came to
  roughly 48 MB; instead each appearance carries its ``bbox`` and frame index
  and the browser redraws the frame from the shipped clip (see
  ``apps/web/src/components/frameGrabber.ts``). That is ~0.7 MB of crops total.
* Crops are re-encoded to WebP quality 82.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

CROP_BASE = "/media/pdestre/session3/crops"
SESSION_BASE = "/media/pdestre/session3"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--assets", required=True, type=Path)
    ap.add_argument("--model-key", default="boxjde_full_fold0")
    ap.add_argument("--decode-thresh", type=float, default=0.05)
    ap.add_argument("--track-thresh", type=float, default=0.30)
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--max-people", type=int, default=40)
    ap.add_argument("--max-views", type=int, default=6)
    ap.add_argument("--topk-detections", type=int, default=300)
    ap.add_argument("--topk-matches", type=int, default=12)
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    from PIL import Image

    try:
        from backend.person_search_service import (  # type: ignore[import-not-found]
            PersonSearchJob,
            PersonSearchManager,
            _normalize_settings,
        )
    except ImportError as exc:  # pragma: no cover - operator guidance
        print(
            f"{exc}\n\n"
            "Put the PedestrianTracker checkout (providing src.*) and the directory\n"
            "containing the backend package on PYTHONPATH before running this tool.\n"
            "See the module docstring for the full invocation.",
            file=sys.stderr,
        )
        return 2

    clips = sorted(args.clips.resolve().glob("cam*.mp4"))
    if not clips:
        print(f"no cam*.mp4 under {args.clips}", file=sys.stderr)
        return 2

    assets_out = args.assets.resolve()
    if assets_out.exists():
        shutil.rmtree(assets_out)
    assets_out.mkdir(parents=True, exist_ok=True)

    runtime_root = Path(".person-search-build").resolve()
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    manager = PersonSearchManager(runtime_root)

    available = [preset for preset in manager.model_presets() if preset["available"]]
    if not any(preset["key"] == args.model_key for preset in available):
        print(f"preset {args.model_key} unavailable; have {[p['key'] for p in available]}", file=sys.stderr)
        return 3

    settings = _normalize_settings(
        {
            "model_key": args.model_key,
            "decode_thresh": args.decode_thresh,
            "track_thresh": args.track_thresh,
            "max_frames": 0,
            "frame_stride": args.frame_stride,
            "max_people": args.max_people,
            "max_views_per_person": args.max_views,
            "topk": args.topk_detections,
        }
    )

    job_id = uuid.uuid4().hex[:12]
    job_root = manager.jobs_root / job_id
    job_root.mkdir(parents=True, exist_ok=True)
    job = PersonSearchJob(
        job_id=job_id,
        root=job_root,
        upload_paths=list(clips),
        upload_names=[clip.name for clip in clips],
        settings=settings,
    )
    manager.jobs[job_id] = job

    print(f"indexing {[c.name for c in clips]} ...")
    manager._run_job_locked(job)
    if job.status == "error":
        print(f"index failed: {job.message}", file=sys.stderr)
        return 4
    print(f"{job.status}: {job.message}")

    people_out = manager.people(job_id)
    crops: dict[str, str] = {}

    def crop_url(asset_id: str | None) -> str | None:
        if not asset_id:
            return None
        if asset_id not in crops:
            try:
                src = manager.asset_path(asset_id)
            except KeyError:
                return None
            stem = (asset_id.split("/", 1)[1] if "/" in asset_id else asset_id).replace("/", "_")
            name = Path(stem).with_suffix(".webp").name
            with Image.open(src) as image:
                image.convert("RGB").save(assets_out / name, "WEBP", quality=82, method=6)
            crops[asset_id] = name
        return f"{CROP_BASE}/{crops[asset_id]}"

    def person_record(record: dict) -> dict:
        out = dict(record)
        out["crop"] = crop_url(record.get("best_crop_asset"))
        out.pop("best_crop_asset", None)
        out.pop("best_scene_asset", None)
        return out

    people = [person_record(p.model_dump()) for p in people_out.people]

    details: dict = {}
    matches: dict = {}
    for person in people:
        pid = person["id"]
        geometry = {
            sample["id"]: {
                "bbox": [round(float(v), 2) for v in sample["bbox"]],
                "frame_idx": int(sample["frame_idx"]),
            }
            for sample in job.people[pid]["appearances"]
        }
        appearances = []
        for appearance in manager.person_detail(job_id, pid).appearances:
            row = appearance.model_dump()
            row["crop"] = crop_url(row.get("crop_asset"))
            row.pop("crop_asset", None)
            row.pop("scene_asset", None)  # redrawn client-side from the clip
            row.update(geometry.get(row["id"], {}))
            appearances.append(row)
        details[pid] = {"appearances": appearances}
        matches[pid] = [
            {
                "similarity": round(float(m.similarity), 4),
                "same_video": bool(m.same_video),
                "person": person_record(m.person.model_dump()),
            }
            for m in manager.matches(job_id, pid, topk=args.topk_matches)
        ]

    videos = []
    for source in manager.videos(job_id):
        row = source.model_dump()
        index = int(row["id"])
        cam = f"cam{index + 1}"
        row.update(
            {
                "label": f"Camera {index + 1}",
                "file_name": row["name"],
                "source": f"{SESSION_BASE}/{cam}-source.webm",
                "tracked": f"{SESSION_BASE}/{cam}-tracked.webm",
                "poster": f"{SESSION_BASE}/{cam}-poster.webp",
            }
        )
        videos.append(row)

    fixture = {
        "schema_version": 1,
        "generated_with": {
            "checkpoint": next(p for p in available if p["key"] == args.model_key)["label"],
            "checkpoint_key": args.model_key,
            "note": (
                "Person-search index computed offline with the BoxJDE research checkpoint "
                "over the four session_3 clips; EffiPed Tier-1 weights are withheld pending "
                "dataset rights review."
            ),
            "settings": settings,
        },
        "job": people_out.job.model_dump(),
        "videos": videos,
        "people": people,
        "details": details,
        "matches": matches,
        "detection": {
            "input": f"{SESSION_BASE}/detection-input.webp",
            "output": f"{SESSION_BASE}/detection-output.webp",
        },
    }

    args.out.resolve().write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    shutil.rmtree(runtime_root, ignore_errors=True)
    print(f"wrote {args.out} — {len(people)} people, {len(crops)} crops")
    print("next: tools/build_session3_manifest.py && tools/validate_media.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
