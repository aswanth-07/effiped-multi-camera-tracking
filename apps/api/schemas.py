"""Typed response schemas for the EffiPed web prototype backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelPresetOut(BaseModel):
    key: str
    label: str
    description: str
    fold: int
    readout: str
    descriptor_dim: int
    artifact_version: str
    available: bool
    benchmark: Dict[str, Any]


class JobCreateOut(BaseModel):
    job_id: str


class JobStatusOut(BaseModel):
    job_id: str
    status: str
    message: str
    progress: float = Field(ge=0.0, le=1.0)
    processed_frame_sets: int = 0
    total_frame_sets: int = 0
    people_count: int = 0


class PersonSummaryOut(BaseModel):
    id: str
    video_index: int
    track_id: int
    first_frame: int
    last_frame: int
    first_time_s: float
    last_time_s: float
    num_samples: int
    best_score: float
    best_crop_asset: Optional[str]
    best_scene_asset: Optional[str]
    caption: str


class AppearanceOut(BaseModel):
    id: str
    video_index: int
    track_id: int
    frame_index: int
    time_s: float
    score: float
    crop_asset: str
    scene_asset: str
    caption: str


class PersonDetailOut(BaseModel):
    person: PersonSummaryOut
    appearances: List[AppearanceOut]


class MatchOut(BaseModel):
    person: PersonSummaryOut
    similarity: float
    same_video: bool


class PeopleOut(BaseModel):
    job: JobStatusOut
    people: List[PersonSummaryOut]


class VideoSourceOut(BaseModel):
    id: int
    name: str
    duration_s: float
    fps: float
    frame_count: int
    width: int
    height: int
    status: str = "indexed"


class DetectionOut(BaseModel):
    id: str
    person_id: str
    video_index: int
    track_id: int
    frame_index: int
    time_s: float
    bbox: Dict[str, float]
    confidence: float
    crop_asset: Optional[str] = None
    scene_asset: Optional[str] = None
    embedding_id: Optional[str] = None


class TrackOut(BaseModel):
    id: str
    person_id: str
    video_index: int
    track_id: int
    detection_ids: List[str]
    start_time_s: float
    end_time_s: float
    start_frame: int
    end_frame: int
    representative_crop: Optional[str] = None
    detection_count: int
    average_confidence: float


class SearchByExampleIn(BaseModel):
    detection_id: str
    selected_video_indices: Optional[List[int]] = None
    time_start_s: Optional[float] = None
    time_end_s: Optional[float] = None
    min_similarity: float = 0.35
    topk: int = 24
    include_same_video: bool = False


class DetectionMatchOut(BaseModel):
    detection: DetectionOut
    person: PersonSummaryOut
    similarity: float
    band: str


class EventOut(BaseModel):
    seq: int
    type: str
    job_id: str
    timestamp: float
    payload: Dict[str, Any]
