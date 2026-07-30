"""FastAPI service for EffiPed Identity Review."""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from effiped.settings import REPOSITORY_ROOT, RuntimeSettings

from .person_search_service import PersonSearchManager
from .schemas import JobCreateOut, ModelPresetOut, SearchByExampleIn

settings = RuntimeSettings.from_env()
manager = PersonSearchManager(settings.runtime_dir, settings)
frontend_dist = REPOSITORY_ROOT / "apps" / "web" / "dist"

app = FastAPI(
    title="EffiPed Identity Review API",
    version="1.0.0",
    description="Local-first multi-camera person-search inference.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0.0", "device": settings.device}


@app.get("/api/models", response_model=List[ModelPresetOut])
def models():
    return manager.model_presets()


@app.post("/api/person-search/jobs", response_model=JobCreateOut)
def create_person_search_job(
    files: List[UploadFile] = File(...),
    model_key: str = Form("effiped_tier1_contest"),
    decode_thresh: float = Form(0.05),
    track_thresh: float = Form(0.25),
    max_frames: int = Form(0),
    frame_stride: int = Form(1),
    max_people: int = Form(120),
    max_views_per_person: int = Form(16),
    topk: int = Form(500),
):
    try:
        job = manager.create_job(
            files,
            {
                "model_key": model_key,
                "decode_thresh": decode_thresh,
                "track_thresh": track_thresh,
                "max_frames": max_frames,
                "frame_stride": frame_stride,
                "max_people": max_people,
                "max_views_per_person": max_views_per_person,
                "topk": topk,
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JobCreateOut(job_id=job.job_id)


@app.get("/api/person-search/jobs/{job_id}")
def job_status(job_id: str):
    try:
        return manager.job_status(manager.get_job(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/person-search/jobs/{job_id}")
def delete_job(job_id: str):
    try:
        manager.delete_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"deleted": True, "job_id": job_id}


@app.get("/api/person-search/jobs/{job_id}/people")
def job_people(job_id: str):
    try:
        return manager.people(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/person-search/jobs/{job_id}/people/{person_id}")
def person_detail(job_id: str, person_id: str):
    try:
        return manager.person_detail(job_id, person_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/person-search/jobs/{job_id}/people/{person_id}/matches")
def person_matches(job_id: str, person_id: str, topk: int = 36):
    try:
        return manager.matches(job_id, person_id, topk=topk)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/person-search/jobs/{job_id}/videos")
def job_videos(job_id: str):
    try:
        return manager.videos(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/person-search/jobs/{job_id}/detections")
def job_detections(
    job_id: str,
    video_index: int | None = None,
    time_start_s: float | None = None,
    time_end_s: float | None = None,
    min_confidence: float | None = None,
):
    try:
        return manager.detections(
            job_id,
            video_index=video_index,
            time_start_s=time_start_s,
            time_end_s=time_end_s,
            min_confidence=min_confidence,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/person-search/jobs/{job_id}/detections/frame")
def job_frame_detections(
    job_id: str,
    video_index: int,
    frame_index: int | None = None,
    timestamp_s: float | None = None,
):
    try:
        return manager.frame_detections(
            job_id,
            video_index=video_index,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/person-search/jobs/{job_id}/tracks")
def job_tracks(job_id: str):
    try:
        return manager.tracks(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/person-search/jobs/{job_id}/search-by-example")
def job_search_by_example(job_id: str, query: SearchByExampleIn):
    try:
        return manager.search_by_example(job_id, query)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.websocket("/api/person-search/jobs/{job_id}/stream")
async def job_stream(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        manager.get_job(job_id)
    except KeyError:
        await websocket.send_json({"type": "error", "payload": {"message": f"Unknown job: {job_id}"}})
        await websocket.close(code=1008)
        return

    next_seq = 0
    try:
        while True:
            events = manager.events_since(job_id, next_seq)
            for event in events:
                await websocket.send_json(event)
                next_seq = int(event["seq"]) + 1
            status = manager.job_status(manager.get_job(job_id)).status
            if status in {"complete", "error"} and not events:
                await asyncio.sleep(0.2)
                break
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return
    finally:
        with suppress(Exception):
            await websocket.close()


@app.get("/api/assets/{asset_id:path}")
def asset(asset_id: str):
    try:
        path = manager.asset_path(asset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path)


if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


def run() -> None:
    import uvicorn

    port = int(os.environ.get("EFFIPED_APP_PORT", "8000"))
    uvicorn.run("apps.api.main:app", host="127.0.0.1", port=port, reload=False)


if __name__ == "__main__":
    run()
