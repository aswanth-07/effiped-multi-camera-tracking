"""Person-search job service for the EffiPed web prototype.

The service imports the installed EffiPed runtime and tracker package.
It replaces the Gradio state callbacks with an explicit job store, asset files,
and JSON/WebSocket events suitable for a React frontend.
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from fastapi import UploadFile

from effiped.runtime import EffiPedRuntime, default_presets, preset_from_key
from effiped.settings import RuntimeSettings
from effiped.tracking.unified_tracker import UnifiedTracker

from .schemas import (
    AppearanceOut,
    DetectionMatchOut,
    DetectionOut,
    JobStatusOut,
    MatchOut,
    PeopleOut,
    PersonDetailOut,
    PersonSummaryOut,
    SearchByExampleIn,
    TrackOut,
    VideoSourceOut,
)

SEARCH_CROP_HEIGHT = 192
SEARCH_SCENE_MAX_WIDTH = 1120


def _read_video_info(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1 or fps > 120:
        fps = 30.0
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": fps,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    }
    return cap, info


def _frame_limit(frame_count: int, max_frames: int) -> int:
    max_frames = int(max_frames)
    if frame_count <= 0:
        return max_frames if max_frames > 0 else 100000
    return min(frame_count, max_frames) if max_frames > 0 else frame_count


def _resize_crop(crop_rgb: np.ndarray, target_h: int = SEARCH_CROP_HEIGHT) -> np.ndarray:
    h, w = crop_rgb.shape[:2]
    if h <= 0 or w <= 0:
        return crop_rgb
    scale = target_h / float(h)
    return cv2.resize(crop_rgb, (max(1, int(round(w * scale))), target_h), interpolation=cv2.INTER_AREA)


def _downscale_scene(frame_rgb: np.ndarray, max_w: int = SEARCH_SCENE_MAX_WIDTH):
    h, w = frame_rgb.shape[:2]
    if w <= max_w:
        return frame_rgb.copy(), 1.0
    scale = max_w / float(w)
    return cv2.resize(frame_rgb, (max_w, int(round(h * scale))), interpolation=cv2.INTER_AREA), scale


def _draw_label(frame, x: int, y: int, text: str, color: Tuple[int, int, int]):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - 8)
    cv2.rectangle(frame, (x, y0), (x + tw + 8, y0 + th + 8), color, -1)
    cv2.putText(frame, text, (x + 4, y0 + th + 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_person_on_scene(
    scene_rgb: np.ndarray,
    bbox: Sequence[float],
    *,
    label: str,
    color: Tuple[int, int, int] = (0, 180, 80),
) -> np.ndarray:
    vis = scene_rgb.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
    if label:
        _draw_label(vis, x1, y1, label, color)
    return vis


def _crop_from_box(frame_bgr: np.ndarray, bbox: Sequence[float]) -> Optional[np.ndarray]:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 12:
        return None
    crop_bgr = frame_bgr[y1:y2, x1:x2]
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)


def _track_embedding(track) -> Optional[np.ndarray]:
    feat = getattr(track, "smooth_feat", None)
    if feat is None:
        feat = getattr(track, "curr_feat", None)
    if feat is None:
        return None
    feat = np.asarray(feat, dtype=np.float32)
    norm = np.linalg.norm(feat)
    if norm <= 1e-8:
        return None
    return feat / norm


def _track_observation_tlbr(track, frame_shape: Tuple[int, int]) -> Optional[List[float]]:
    """Return the latest detector-observed box in original frame pixels.

    Kalman-smoothed boxes are useful for tracking continuity, but they can lag
    behind moving pedestrians in an investigation overlay. For the UI, the box
    should mark the actual current detection.
    """
    obs = getattr(track, "last_observation", None)
    if obs is not None:
        obs = np.asarray(obs, dtype=np.float32).copy()
        if obs.shape[0] >= 4:
            x1, y1, w, h = [float(v) for v in obs[:4]]
            box = [x1, y1, x1 + w, y1 + h]
        else:
            box = [float(v) for v in track.tlbr]
    else:
        box = [float(v) for v in track.tlbr]

    frame_h, frame_w = frame_shape
    x1, y1, x2, y2 = box
    x1 = min(max(0.0, x1), float(frame_w - 1))
    y1 = min(max(0.0, y1), float(frame_h - 1))
    x2 = min(max(0.0, x2), float(frame_w - 1))
    y2 = min(max(0.0, y2), float(frame_h - 1))
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None
    return [x1, y1, x2, y2]


def _appearance_caption(sample: dict) -> str:
    t = sample["frame_idx"] / max(sample["fps"], 1e-6)
    return f"Video {sample['video_index'] + 1} | frame {sample['frame_idx']} | {t:.1f}s"


def _person_caption(person: dict) -> str:
    t0 = person["first_frame"] / max(person["fps"], 1e-6)
    t1 = person["last_frame"] / max(person["fps"], 1e-6)
    return (
        f"Video {person['video_index'] + 1} | Track {person['track_id']} | "
        f"{person['num_samples']} views | {t0:.1f}-{t1:.1f}s"
    )


def _save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 88])


def _normalize_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_key": str(settings.get("model_key") or "effiped_tier1_contest"),
        "decode_thresh": float(settings.get("decode_thresh", 0.05)),
        "track_thresh": float(settings.get("track_thresh", 0.25)),
        "max_frames": int(settings.get("max_frames", 0)),
        "frame_stride": max(1, int(settings.get("frame_stride", 1))),
        "max_people": max(1, int(settings.get("max_people", 120))),
        "max_views_per_person": max(1, int(settings.get("max_views_per_person", 16))),
        "topk": int(settings.get("topk", 500)),
    }


@dataclass
class PersonSearchJob:
    job_id: str
    root: Path
    upload_paths: List[Path]
    upload_names: List[str]
    settings: Dict[str, Any]
    status: str = "queued"
    message: str = "Queued"
    progress: float = 0.0
    processed_frame_sets: int = 0
    total_frame_sets: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    video_sources: List[dict] = field(default_factory=list)
    people: Dict[str, dict] = field(default_factory=dict)
    detections: Dict[str, dict] = field(default_factory=dict)
    events: List[dict] = field(default_factory=list)
    assets: Dict[str, Path] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)


class PersonSearchManager:
    def __init__(self, runtime_root: Path, settings: RuntimeSettings | None = None):
        self.settings = settings or RuntimeSettings.from_env()
        self.runtime_root = runtime_root
        self.jobs_root = runtime_root / "jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, PersonSearchJob] = {}
        self.jobs_lock = threading.Lock()
        self.runtime_lock = threading.Lock()
        self.runtime = EffiPedRuntime(self.settings)
        self.presets = default_presets(self.settings)

    def model_presets(self) -> List[dict]:
        return [
            {
                "key": preset.key,
                "label": preset.label,
                "description": preset.description,
                "fold": preset.fold,
                "readout": preset.readout,
                "descriptor_dim": preset.descriptor_dim,
                "artifact_version": preset.artifact_version,
                "available": preset.available,
                "benchmark": preset.benchmark,
            }
            for preset in self.presets.values()
        ]

    def create_job(self, files: List[UploadFile], settings: Dict[str, Any]) -> PersonSearchJob:
        if not files:
            raise ValueError("Upload at least one video.")
        if len(files) > 4:
            raise ValueError("Upload at most four videos.")
        allowed_suffixes = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in allowed_suffixes:
                raise ValueError(f"Unsupported video type: {suffix or 'missing extension'}")
            upload.file.seek(0, 2)
            size = upload.file.tell()
            upload.file.seek(0)
            if size <= 0:
                raise ValueError(f"Empty upload: {upload.filename or 'video'}")
            if size > max_bytes:
                raise ValueError(
                    f"Upload exceeds EFFIPED_MAX_UPLOAD_MB={self.settings.max_upload_mb}: "
                    f"{upload.filename or 'video'}"
                )

        normalized = _normalize_settings(settings)
        preset = preset_from_key(self.presets, normalized["model_key"])
        if not preset.available:
            raise FileNotFoundError(f"Model preset is unavailable: {preset.label}")

        job_id = uuid.uuid4().hex[:12]
        job_root = self.jobs_root / job_id
        upload_root = job_root / "uploads"
        upload_root.mkdir(parents=True, exist_ok=True)

        upload_paths: List[Path] = []
        upload_names: List[str] = []
        for idx, upload in enumerate(files):
            suffix = Path(upload.filename or "").suffix or ".mp4"
            dst = upload_root / f"video_{idx}{suffix}"
            upload.file.seek(0)
            with dst.open("wb") as out:
                shutil.copyfileobj(upload.file, out)
            upload_paths.append(dst)
            upload_names.append(upload.filename or dst.name)

        job = PersonSearchJob(
            job_id=job_id,
            root=job_root,
            upload_paths=upload_paths,
            upload_names=upload_names,
            settings=normalized,
        )
        with self.jobs_lock:
            self.jobs[job_id] = job
        self._emit(job, "job_queued", {"job": self.job_status(job).model_dump()})

        thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        thread.start()
        return job

    def get_job(self, job_id: str) -> PersonSearchJob:
        with self.jobs_lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(f"Unknown job: {job_id}")
        return job

    def delete_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        with job.lock:
            if job.status in {"queued", "running"}:
                raise RuntimeError("A running job cannot be deleted.")
        resolved_root = job.root.resolve()
        jobs_root = self.jobs_root.resolve()
        if jobs_root not in resolved_root.parents:
            raise RuntimeError("Refusing to delete a path outside the runtime jobs directory.")
        with self.jobs_lock:
            self.jobs.pop(job_id, None)
        shutil.rmtree(resolved_root, ignore_errors=False)

    def asset_path(self, asset_id: str) -> Path:
        job_id = asset_id.split("/", 1)[0]
        job = self.get_job(job_id)
        with job.lock:
            path = job.assets.get(asset_id)
        if path is None:
            raise KeyError(f"Unknown asset: {asset_id}")
        return path

    def job_status(self, job: PersonSearchJob) -> JobStatusOut:
        with job.lock:
            return JobStatusOut(
                job_id=job.job_id,
                status=job.status,
                message=job.message,
                progress=float(job.progress),
                processed_frame_sets=int(job.processed_frame_sets),
                total_frame_sets=int(job.total_frame_sets),
                people_count=len(job.people),
            )

    def events_since(self, job_id: str, seq: int) -> List[dict]:
        job = self.get_job(job_id)
        with job.lock:
            return [event for event in job.events if int(event["seq"]) >= seq]

    def people(self, job_id: str) -> PeopleOut:
        job = self.get_job(job_id)
        with job.lock:
            people = [self._person_summary_unlocked(person) for person in self._sorted_people_unlocked(job)]
        return PeopleOut(job=self.job_status(job), people=people)

    def person_detail(self, job_id: str, person_id: str) -> PersonDetailOut:
        job = self.get_job(job_id)
        with job.lock:
            person = job.people[person_id]
            return PersonDetailOut(
                person=self._person_summary_unlocked(person),
                appearances=[self._appearance_unlocked(person, sample) for sample in person["appearances"]],
            )

    def matches(self, job_id: str, person_id: str, topk: int = 36) -> List[MatchOut]:
        job = self.get_job(job_id)
        with job.lock:
            people = job.people
            query = people[person_id]
            q_emb = query["embedding"]
            rows = []
            for other_id, other in people.items():
                if other_id == person_id:
                    continue
                if other["video_index"] == query["video_index"]:
                    continue
                sim = float(np.dot(q_emb, other["embedding"]))
                rows.append((other_id, sim))
            rows.sort(key=lambda item: item[1], reverse=True)
            return [
                MatchOut(
                    person=self._person_summary_unlocked(people[other_id]),
                    similarity=sim,
                    same_video=people[other_id]["video_index"] == query["video_index"],
                )
                for other_id, sim in rows[: max(1, int(topk))]
            ]

    def videos(self, job_id: str) -> List[VideoSourceOut]:
        job = self.get_job(job_id)
        with job.lock:
            return [VideoSourceOut(**source) for source in job.video_sources]

    def detections(
        self,
        job_id: str,
        *,
        video_index: Optional[int] = None,
        time_start_s: Optional[float] = None,
        time_end_s: Optional[float] = None,
        min_confidence: Optional[float] = None,
    ) -> List[DetectionOut]:
        job = self.get_job(job_id)
        with job.lock:
            rows = self._filtered_detection_rows_unlocked(
                job,
                video_indices=[video_index] if video_index is not None else None,
                time_start_s=time_start_s,
                time_end_s=time_end_s,
                min_confidence=min_confidence,
            )
            return [self._detection_unlocked(row) for row in rows]

    def frame_detections(
        self,
        job_id: str,
        *,
        video_index: int,
        frame_index: Optional[int] = None,
        timestamp_s: Optional[float] = None,
    ) -> List[DetectionOut]:
        job = self.get_job(job_id)
        with job.lock:
            fps = 30.0
            for source in job.video_sources:
                if int(source["id"]) == int(video_index):
                    fps = max(float(source["fps"]), 1e-6)
                    break
            target_frame = int(frame_index) if frame_index is not None else int(round(float(timestamp_s or 0.0) * fps))
            rows = [
                det for det in job.detections.values()
                if int(det["video_index"]) == int(video_index)
                and int(det["frame_idx"]) == target_frame
                and det.get("person_id") in job.people
            ]
            return [self._detection_unlocked(row) for row in sorted(rows, key=lambda det: det["track_id"])]

    def tracks(self, job_id: str) -> List[TrackOut]:
        job = self.get_job(job_id)
        with job.lock:
            return [self._track_unlocked(job, person) for person in self._sorted_people_unlocked(job)]

    def search_by_example(self, job_id: str, query: SearchByExampleIn) -> List[DetectionMatchOut]:
        job = self.get_job(job_id)
        with job.lock:
            if query.detection_id not in job.detections:
                raise KeyError(f"Unknown detection: {query.detection_id}")

            q_det = job.detections[query.detection_id]
            q_emb = np.asarray(q_det["embedding"], dtype=np.float32)
            selected = None
            if query.selected_video_indices is not None:
                selected = [int(idx) for idx in query.selected_video_indices]

            best_by_person: Dict[str, Tuple[dict, float]] = {}
            for det in self._filtered_detection_rows_unlocked(
                job,
                video_indices=selected,
                time_start_s=query.time_start_s,
                time_end_s=query.time_end_s,
                min_confidence=None,
            ):
                if det["id"] == query.detection_id:
                    continue
                if not query.include_same_video and int(det["video_index"]) == int(q_det["video_index"]):
                    continue
                sim = float(np.dot(q_emb, det["embedding"]))
                if sim < float(query.min_similarity):
                    continue
                person_id = str(det["person_id"])
                current = best_by_person.get(person_id)
                if current is None or sim > current[1]:
                    best_by_person[person_id] = (det, sim)

            rows = sorted(best_by_person.values(), key=lambda item: item[1], reverse=True)
            results: List[DetectionMatchOut] = []
            for det, sim in rows[: max(1, int(query.topk))]:
                person = job.people.get(det["person_id"])
                if person is None:
                    continue
                band = "strong" if sim >= 0.70 else "possible" if sim >= 0.50 else "low"
                results.append(
                    DetectionMatchOut(
                        detection=self._detection_unlocked(det),
                        person=self._person_summary_unlocked(person),
                        similarity=sim,
                        band=band,
                    )
                )
            return results

    def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        try:
            with self.runtime_lock:
                self._run_job_locked(job)
        except Exception as exc:
            with job.lock:
                job.status = "error"
                job.message = str(exc)
                job.updated_at = time.time()
            self._emit(job, "error", {"message": str(exc), "job": self.job_status(job).model_dump()})
        finally:
            upload_root = job.root / "uploads"
            if upload_root.is_dir():
                shutil.rmtree(upload_root, ignore_errors=True)

    def _run_job_locked(self, job: PersonSearchJob) -> None:
        settings = job.settings
        preset = preset_from_key(self.presets, settings["model_key"])
        self.runtime.ensure_loaded(preset)

        with job.lock:
            job.status = "running"
            job.message = "Loading videos"
            job.updated_at = time.time()
        self._emit(job, "job_started", {"job": self.job_status(job).model_dump(), "model": preset.label})

        caps = {}
        info = {}
        trackers = {}
        for vid_idx, path in enumerate(job.upload_paths):
            cap, cap_info = _read_video_info(str(path))
            caps[vid_idx] = cap
            info[vid_idx] = cap_info
            trackers[vid_idx] = self._build_tracker(cap_info["fps"], settings["track_thresh"], settings["decode_thresh"])
        with job.lock:
            job.video_sources = [
                {
                    "id": int(vid_idx),
                    "name": job.upload_names[vid_idx] if vid_idx < len(job.upload_names) else path.name,
                    "duration_s": float(info[vid_idx]["frames"] / max(info[vid_idx]["fps"], 1e-6)) if info[vid_idx]["frames"] else 0.0,
                    "fps": float(info[vid_idx]["fps"]),
                    "frame_count": int(info[vid_idx]["frames"]),
                    "width": int(info[vid_idx]["width"]),
                    "height": int(info[vid_idx]["height"]),
                    "status": "indexing",
                }
                for vid_idx, path in enumerate(job.upload_paths)
            ]

        try:
            raw_total = max(data["frames"] for data in info.values())
            limit = _frame_limit(raw_total, settings["max_frames"])
            frame_stride = settings["frame_stride"]
            with job.lock:
                job.total_frame_sets = max(1, int(np.ceil(limit / frame_stride)))
                job.message = "Indexing people"

            frame_times: List[float] = []
            for frame_idx in range(limit):
                frames = {}
                for vid_idx, cap in caps.items():
                    if info[vid_idx]["frames"] > 0 and frame_idx >= info[vid_idx]["frames"]:
                        continue
                    ok, frame = cap.read()
                    if ok and frame_idx % frame_stride == 0:
                        frames[vid_idx] = frame
                if not frames:
                    continue

                batch = self.runtime.process_batch(
                    frames,
                    conf_thresh=settings["decode_thresh"],
                    topk=settings["topk"],
                    min_box_height=10.0,
                    min_box_area=40.0,
                    return_part_features=preset.return_part_features,
                )

                for vid_idx, frame in frames.items():
                    dets, elapsed = batch[vid_idx]
                    frame_times.append(elapsed)
                    tracker = trackers[vid_idx]
                    tracker.update(dets)
                    current_tracks = [
                        track for track in tracker.tracked_stracks
                        if getattr(track, "frame_id", None) == tracker.frame_id
                    ]
                    self._ingest_tracks(job, current_tracks, frame, frame_idx, info[vid_idx], vid_idx)

                with job.lock:
                    job.processed_frame_sets += 1
                    job.progress = min(1.0, job.processed_frame_sets / max(job.total_frame_sets, 1))
                    job.updated_at = time.time()
                    status = self.job_status(job).model_dump()
                self._emit(job, "progress", {"job": status})

            avg_fps = 1.0 / max(float(np.mean(frame_times)), 1e-6) if frame_times else 0.0
            self._finalize_people(job)
            with job.lock:
                for source in job.video_sources:
                    source["status"] = "indexed"
                job.status = "complete"
                job.progress = 1.0
                job.message = f"Indexed {len(job.people)} people. Average model FPS: {avg_fps:.2f}"
                job.updated_at = time.time()
            self._emit(job, "job_complete", {"job": self.job_status(job).model_dump()})
        finally:
            for cap in caps.values():
                cap.release()

    def _build_tracker(self, fps: float, track_thresh: float, low_thresh: float):
        tracker_cfg = self.runtime.tracker_config()
        tracker_cfg["det_thresh"] = float(track_thresh)
        tracker_cfg["low_thresh"] = float(low_thresh)
        tracker_cfg["track_buffer"] = max(90, int(tracker_cfg.get("track_buffer", 30)))
        tracker_cfg["new_track_thresh"] = max(0.45, float(track_thresh) + 0.10, float(tracker_cfg.get("new_track_thresh", 0.40)))
        tracker_cfg["min_hits"] = min(2, int(tracker_cfg.get("min_hits", 3)))
        tracker_cfg["match_thresh"] = max(0.50, float(tracker_cfg.get("match_thresh", 0.40)))
        tracker_cfg["step2_match_thresh"] = max(0.60, float(tracker_cfg.get("step2_match_thresh", 0.50)))
        tracker_cfg["step3_match_thresh"] = max(0.60, float(tracker_cfg.get("step3_match_thresh", 0.50)))
        tracker_cfg["unconfirmed_match_thresh"] = max(0.75, float(tracker_cfg.get("unconfirmed_match_thresh", 0.70)))
        tracker_cfg["embedding_fusion_alpha"] = min(0.70, float(tracker_cfg.get("embedding_fusion_alpha", 0.90)))
        tracker_cfg["appearance_thresh"] = max(0.65, float(tracker_cfg.get("appearance_thresh", 0.60)))
        tracker_cfg["reid_thresh"] = min(0.35, float(tracker_cfg.get("reid_thresh", 0.40)))
        return UnifiedTracker(config=tracker_cfg, frame_rate=max(1, int(round(fps))))

    def _ingest_tracks(
        self,
        job: PersonSearchJob,
        tracks: Iterable[object],
        frame_bgr: np.ndarray,
        frame_idx: int,
        info: dict,
        vid_idx: int,
    ) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        scene, scale = _downscale_scene(frame_rgb)

        for track in tracks:
            if not getattr(track, "is_activated", False):
                continue
            emb = _track_embedding(track)
            if emb is None:
                continue
            bbox = _track_observation_tlbr(track, frame_bgr.shape[:2])
            if bbox is None:
                continue
            crop = _crop_from_box(frame_bgr, bbox)
            if crop is None:
                continue

            key = f"v{vid_idx}_t{int(track.track_id)}"
            bbox_scene = [bbox[0] * scale, bbox[1] * scale, bbox[2] * scale, bbox[3] * scale]
            sample_idx = uuid.uuid4().hex[:10]
            detection_id = f"{key}_d{sample_idx}"
            crop_resized = _resize_crop(crop)
            drawn_scene = _draw_person_on_scene(
                scene,
                bbox_scene,
                label=f"Track {int(track.track_id)}",
                color=(0, 180, 80),
            )

            crop_asset = self._save_asset(job, f"{key}_{sample_idx}_crop.jpg", crop_resized)
            scene_asset = self._save_asset(job, f"{key}_{sample_idx}_scene.jpg", drawn_scene)
            sample = {
                "id": detection_id,
                "person_id": key,
                "crop_asset": crop_asset,
                "scene_asset": scene_asset,
                "bbox": bbox,
                "bbox_scene": bbox_scene,
                "frame_idx": int(frame_idx),
                "fps": float(info["fps"]),
                "score": float(track.score),
                "embedding": emb.astype(np.float32),
                "video_index": int(vid_idx),
                "track_id": int(track.track_id),
            }

            with job.lock:
                is_new = key not in job.people
                if is_new:
                    job.people[key] = {
                        "id": key,
                        "video_index": int(vid_idx),
                        "track_id": int(track.track_id),
                        "fps": float(info["fps"]),
                        "first_frame": int(frame_idx),
                        "last_frame": int(frame_idx),
                        "num_samples": 0,
                        "best_score": -1.0,
                        "best_crop_asset": None,
                        "best_scene_asset": None,
                        "first_bbox": bbox,
                        "last_bbox": bbox,
                        "embedding_sum": np.zeros_like(sample["embedding"]),
                        "embedding": sample["embedding"],
                        "appearances": [],
                    }

                person = job.people[key]
                person["last_frame"] = int(frame_idx)
                person["last_bbox"] = bbox
                person["num_samples"] += 1
                person["embedding_sum"] += sample["embedding"]
                if sample["score"] > person["best_score"]:
                    person["best_score"] = sample["score"]
                    person["best_crop_asset"] = crop_asset
                    person["best_scene_asset"] = scene_asset

                appearances = person["appearances"]
                max_views = job.settings["max_views_per_person"]
                if len(appearances) < max_views:
                    appearances.append(sample)
                else:
                    weakest_idx = int(np.argmin([a["score"] for a in appearances]))
                    if sample["score"] > appearances[weakest_idx]["score"]:
                        appearances[weakest_idx] = sample
                person["appearances"].sort(key=lambda item: (item["frame_idx"], -item["score"]))

                emb_mean = person["embedding_sum"] / max(person["num_samples"], 1)
                norm = np.linalg.norm(emb_mean)
                person["embedding"] = (emb_mean / max(norm, 1e-8)).astype(np.float32)
                job.detections[detection_id] = sample
                summary = self._person_summary_unlocked(person).model_dump()

            if is_new or person["num_samples"] % 3 == 0:
                self._emit(job, "person_upsert", {"person": summary})

    def _finalize_people(self, job: PersonSearchJob) -> None:
        with job.lock:
            for person in job.people.values():
                emb = person["embedding_sum"] / max(person["num_samples"], 1)
                norm = np.linalg.norm(emb)
                person["embedding"] = (emb / max(norm, 1e-8)).astype(np.float32)
                person["appearances"].sort(key=lambda item: (item["frame_idx"], -item["score"]))

            self._merge_fragmented_people_unlocked(job)
            keep_people = self._balanced_people_keep_unlocked(job, job.settings["max_people"])
            keep_ids = {person["id"] for person in keep_people}
            job.people = {key: person for key, person in job.people.items() if key in keep_ids}
            job.detections = {
                key: det for key, det in job.detections.items()
                if det.get("person_id") in keep_ids
            }
            summaries = [self._person_summary_unlocked(person).model_dump() for person in self._sorted_people_unlocked(job)]

        self._emit(job, "people_snapshot", {"people": summaries})

    def _balanced_people_keep_unlocked(self, job: PersonSearchJob, max_people: int) -> List[dict]:
        people = self._sorted_people_unlocked(job)
        if len(people) <= max_people:
            return people

        by_video: Dict[int, List[dict]] = {}
        for person in people:
            by_video.setdefault(int(person["video_index"]), []).append(person)

        video_count = max(len(by_video), 1)
        per_video_cap = max(1, int(np.ceil(max_people / video_count)))
        selected: List[dict] = []
        selected_ids = set()

        for video_id in sorted(by_video):
            video_people = sorted(
                by_video[video_id],
                key=lambda person: (-float(person["best_score"]), -int(person["num_samples"]), int(person["first_frame"])),
            )
            for person in video_people[:per_video_cap]:
                if len(selected) >= max_people:
                    break
                selected.append(person)
                selected_ids.add(person["id"])

        if len(selected) < max_people:
            remaining = [
                person for person in people
                if person["id"] not in selected_ids
            ]
            remaining.sort(key=lambda person: (-float(person["best_score"]), -int(person["num_samples"])))
            for person in remaining:
                if len(selected) >= max_people:
                    break
                selected.append(person)

        return sorted(selected, key=lambda person: (person["video_index"], person["first_frame"], person["track_id"]))

    def _merge_fragmented_people_unlocked(self, job: PersonSearchJob) -> None:
        """Conservatively merge short same-video ID fragments for app presentation.

        The tracker remains the source of detections, but the demo should not
        present obvious same-person fragments as separate people after brief
        detector/tracker gaps. This merge is same-video only, non-overlapping,
        bounded by a short temporal gap, and requires high descriptor agreement.
        """
        people = self._sorted_people_unlocked(job)
        merged: List[dict] = []
        max_gap_s = 6.0
        strong_sim = 0.80
        short_gap_sim = 0.72

        for person in people:
            target = None
            for existing in reversed(merged):
                if existing["video_index"] != person["video_index"]:
                    continue
                if person["first_frame"] <= existing["last_frame"]:
                    continue
                fps = max(float(existing["fps"]), 1e-6)
                gap_frames = person["first_frame"] - existing["last_frame"]
                if gap_frames > int(round(max_gap_s * fps)):
                    continue
                sim = float(np.dot(existing["embedding"], person["embedding"]))
                if sim >= strong_sim or (gap_frames <= int(round(1.5 * fps)) and sim >= short_gap_sim):
                    target = existing
                    break

            if target is None:
                merged.append(person)
                continue

            target["last_frame"] = max(target["last_frame"], person["last_frame"])
            target["last_bbox"] = person.get("last_bbox", target.get("last_bbox"))
            target["num_samples"] += person["num_samples"]
            target["embedding_sum"] += person["embedding_sum"]
            emb = target["embedding_sum"] / max(target["num_samples"], 1)
            target["embedding"] = (emb / max(np.linalg.norm(emb), 1e-8)).astype(np.float32)
            if person["best_score"] > target["best_score"]:
                target["best_score"] = person["best_score"]
                target["best_crop_asset"] = person["best_crop_asset"]
                target["best_scene_asset"] = person["best_scene_asset"]
            for sample in person["appearances"]:
                sample["person_id"] = target["id"]
            for det in job.detections.values():
                if det.get("person_id") == person["id"]:
                    det["person_id"] = target["id"]
            target["appearances"].extend(person["appearances"])
            target["appearances"].sort(key=lambda item: (item["frame_idx"], -item["score"]))
            max_views = job.settings["max_views_per_person"]
            if len(target["appearances"]) > max_views:
                best_by_score = sorted(target["appearances"], key=lambda item: item["score"], reverse=True)[:max_views]
                target["appearances"] = sorted(best_by_score, key=lambda item: (item["frame_idx"], -item["score"]))

        job.people = {person["id"]: person for person in merged}

    def _save_asset(self, job: PersonSearchJob, filename: str, image_rgb: np.ndarray) -> str:
        asset_id = f"{job.job_id}/{filename}"
        path = job.root / "assets" / filename
        _save_rgb(path, image_rgb)
        with job.lock:
            job.assets[asset_id] = path
        return asset_id

    def _emit(self, job: PersonSearchJob, event_type: str, payload: Dict[str, Any]) -> None:
        with job.lock:
            event = {
                "seq": len(job.events),
                "type": event_type,
                "job_id": job.job_id,
                "timestamp": time.time(),
                "payload": payload,
            }
            job.events.append(event)
            job.updated_at = time.time()

    def _sorted_people_unlocked(self, job: PersonSearchJob) -> List[dict]:
        return sorted(
            job.people.values(),
            key=lambda person: (person["video_index"], person["first_frame"], person["track_id"]),
        )

    def _person_summary_unlocked(self, person: dict) -> PersonSummaryOut:
        first_time = person["first_frame"] / max(person["fps"], 1e-6)
        last_time = person["last_frame"] / max(person["fps"], 1e-6)
        return PersonSummaryOut(
            id=person["id"],
            video_index=int(person["video_index"]),
            track_id=int(person["track_id"]),
            first_frame=int(person["first_frame"]),
            last_frame=int(person["last_frame"]),
            first_time_s=float(first_time),
            last_time_s=float(last_time),
            num_samples=int(person["num_samples"]),
            best_score=float(person["best_score"]),
            best_crop_asset=person.get("best_crop_asset"),
            best_scene_asset=person.get("best_scene_asset"),
            caption=_person_caption(person),
        )

    def _appearance_unlocked(self, person: dict, sample: dict) -> AppearanceOut:
        return AppearanceOut(
            id=str(sample["id"]),
            video_index=int(sample["video_index"]),
            track_id=int(sample["track_id"]),
            frame_index=int(sample["frame_idx"]),
            time_s=float(sample["frame_idx"] / max(sample["fps"], 1e-6)),
            score=float(sample["score"]),
            crop_asset=str(sample["crop_asset"]),
            scene_asset=str(sample["scene_asset"]),
            caption=_appearance_caption(sample),
        )

    def _filtered_detection_rows_unlocked(
        self,
        job: PersonSearchJob,
        *,
        video_indices: Optional[Sequence[Optional[int]]] = None,
        time_start_s: Optional[float] = None,
        time_end_s: Optional[float] = None,
        min_confidence: Optional[float] = None,
    ) -> List[dict]:
        selected = None
        if video_indices is not None:
            selected = {int(idx) for idx in video_indices if idx is not None}
        rows = []
        for det in job.detections.values():
            if det.get("person_id") not in job.people:
                continue
            if selected is not None and int(det["video_index"]) not in selected:
                continue
            time_s = float(det["frame_idx"] / max(det["fps"], 1e-6))
            if time_start_s is not None and time_s < float(time_start_s):
                continue
            if time_end_s is not None and time_s > float(time_end_s):
                continue
            if min_confidence is not None and float(det["score"]) < float(min_confidence):
                continue
            rows.append(det)
        return sorted(rows, key=lambda item: (item["video_index"], item["frame_idx"], item["track_id"]))

    def _detection_unlocked(self, det: dict) -> DetectionOut:
        x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
        return DetectionOut(
            id=str(det["id"]),
            person_id=str(det["person_id"]),
            video_index=int(det["video_index"]),
            track_id=int(det["track_id"]),
            frame_index=int(det["frame_idx"]),
            time_s=float(det["frame_idx"] / max(det["fps"], 1e-6)),
            bbox={
                "x": x1,
                "y": y1,
                "width": max(0.0, x2 - x1),
                "height": max(0.0, y2 - y1),
            },
            confidence=float(det["score"]),
            crop_asset=str(det.get("crop_asset") or ""),
            scene_asset=str(det.get("scene_asset") or ""),
            embedding_id=str(det["id"]),
        )

    def _track_unlocked(self, job: PersonSearchJob, person: dict) -> TrackOut:
        detections = [
            det for det in job.detections.values()
            if det.get("person_id") == person["id"]
        ]
        detections.sort(key=lambda det: det["frame_idx"])
        avg_conf = float(np.mean([det["score"] for det in detections])) if detections else float(person["best_score"])
        return TrackOut(
            id=str(person["id"]),
            person_id=str(person["id"]),
            video_index=int(person["video_index"]),
            track_id=int(person["track_id"]),
            detection_ids=[str(det["id"]) for det in detections],
            start_time_s=float(person["first_frame"] / max(person["fps"], 1e-6)),
            end_time_s=float(person["last_frame"] / max(person["fps"], 1e-6)),
            start_frame=int(person["first_frame"]),
            end_frame=int(person["last_frame"]),
            representative_crop=person.get("best_crop_asset"),
            detection_count=len(detections),
            average_confidence=avg_conf,
        )
