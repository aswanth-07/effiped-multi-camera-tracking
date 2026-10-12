"""Rebuild apps/web/src/data/frame-detections.json from the session_3 clips.

Offline, one-off provenance tool, the companion of
``tools/build_person_search_fixture.py`` and run with the same environment.

Why it exists
-------------
The person-search index keeps only the strongest few views of a capped number
of tracks per camera, which is right for retrieval and wrong for showing a
frame: most people in any given frame have no stored view, so a detection view
drawn from the index shows a handful of boxes where the detector found many.
This tool records what the detector actually returned on every processed frame,
with the track each box was associated to, so the Detection tab can redraw a
frame faithfully.

It replays the index run exactly (same checkpoint, decode threshold, frame
stride and tracker configuration), so track ids line up with
``person-search.json``.

Usage
-----
::

    PYTHONPATH=/path/to/checkout:/path/to/paper \
    python tools/build_detection_fixture.py \
        --paper-root /path/to/paper \
        --clips /path/to/demo_clips/session_3 \
        --out apps/web/src/data/frame-detections.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--paper-root", required=True, type=Path, help="The research paper/ tree, also on PYTHONPATH.")
    ap.add_argument("--model-key", default="boxjde_full_fold0")
    ap.add_argument("--config", default="configs/boxjde/fold0_boxjde_full_roi.yaml")
    ap.add_argument("--checkpoint", default="runs_boxjde/boxjde_fold0_full_roi/best_model.pth")
    ap.add_argument("--decode-thresh", type=float, default=0.05)
    ap.add_argument("--track-thresh", type=float, default=0.30)
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--topk-detections", type=int, default=300)
    ap.add_argument(
        "--store-floor",
        type=float,
        default=0.30,
        help="Lowest score written out; matches the bottom of the console's confidence slider.",
    )
    return ap.parse_args()


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main() -> int:
    args = parse_args()

    import dataclasses

    import cv2

    try:
        from backend.person_search_service import (  # type: ignore[import-not-found]
            PersonSearchManager,
            _track_observation_tlbr,
        )
        from src.paper_runtime import preset_from_key  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - operator guidance
        print(f"{exc}\n\nSee the module docstring for the PYTHONPATH this tool needs.", file=sys.stderr)
        return 2

    clips = sorted(args.clips.resolve().glob("cam*.mp4"))
    if not clips:
        print(f"no cam*.mp4 under {args.clips}", file=sys.stderr)
        return 2

    runtime_root = Path(".detection-build").resolve()
    manager = PersonSearchManager(runtime_root)
    # The checkpoint the published index was built from, wherever this
    # checkout's presets happen to point.
    preset = dataclasses.replace(
        preset_from_key(manager.presets, args.model_key),
        config_path=str(args.paper_root.resolve() / args.config),
        checkpoint_path=str(args.paper_root.resolve() / args.checkpoint),
    )
    manager.runtime.ensure_loaded(preset)

    # Every camera advances one frame at a time in a single batch, in the same
    # order as the index run: the tracker's id counter is shared across
    # cameras, so replaying them one after another would renumber the tracks.
    caps = [cv2.VideoCapture(str(clip)) for clip in clips]
    fps = [float(cap.get(cv2.CAP_PROP_FPS)) or 10.0 for cap in caps]
    trackers = [manager._build_tracker(rate, args.track_thresh, args.decode_thresh) for rate in fps]
    frames: list[dict[str, list]] = [{} for _ in clips]
    frame_idx = -1
    while True:
        frame_idx += 1
        batch_in = {}
        any_read = False
        for vid_idx, cap in enumerate(caps):
            ok, frame = cap.read()
            any_read = any_read or ok
            if ok and frame_idx % args.frame_stride == 0:
                batch_in[vid_idx] = frame
        if not any_read:
            break
        if not batch_in:
            continue
        batch = manager.runtime.process_batch(
            batch_in,
            conf_thresh=args.decode_thresh,
            topk=args.topk_detections,
            min_box_height=10.0,
            min_box_area=40.0,
            return_part_features=preset.return_part_features,
        )
        for vid_idx, frame in batch_in.items():
            dets, _ = batch[vid_idx]
            boxes = dets.get("boxes")
            scores = dets.get("scores")
            raw = [
                ([float(v) for v in boxes[i][:4]], float(scores[i]))
                for i in range(0 if boxes is None else len(boxes))
                if float(scores[i]) >= args.store_floor
            ]
            tracker = trackers[vid_idx]
            tracker.update(dets)
            tracks = []
            for track in tracker.tracked_stracks:
                if getattr(track, "frame_id", None) != tracker.frame_id or not track.is_activated:
                    continue
                box = _track_observation_tlbr(track, frame.shape[:2])
                if box is not None:
                    tracks.append((int(track.track_id), box))

            rows = []
            for box, score in raw:
                best = max(tracks, key=lambda t: iou(box, t[1]), default=None)
                track_id = best[0] if best is not None and iou(box, best[1]) >= 0.5 else None
                rows.append([*(round(v) for v in box), round(score, 3), track_id])
            rows.sort(key=lambda row: -row[4])
            frames[vid_idx][str(frame_idx)] = rows

    videos = []
    for vid_idx, (clip, cap) in enumerate(zip(clips, caps)):
        cap.release()
        total = sum(len(rows) for rows in frames[vid_idx].values())
        print(f"{clip.name}: {len(frames[vid_idx])} frames, {total} detections >= {args.store_floor}")
        videos.append({"video_index": vid_idx, "fps": fps[vid_idx], "frames": frames[vid_idx]})

    fixture = {
        "schema_version": 1,
        "generated_with": {
            "checkpoint_key": args.model_key,
            "decode_thresh": args.decode_thresh,
            "track_thresh": args.track_thresh,
            "frame_stride": args.frame_stride,
            "store_floor": args.store_floor,
        },
        "row": ["x1", "y1", "x2", "y2", "score", "track_id"],
        "videos": videos,
    }
    args.out.resolve().write_text(json.dumps(fixture, separators=(",", ":")), encoding="utf-8")
    shutil.rmtree(runtime_root, ignore_errors=True)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
