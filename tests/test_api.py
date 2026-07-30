from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)


def test_health_and_portable_model_metadata():
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    response = client.get("/api/models")
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 1
    assert rows[0]["key"] == "effiped_tier1"
    assert rows[0]["descriptor_dim"] == 256
    assert rows[0]["available"] is False
    assert "path" not in str(rows[0]).lower()


def test_unknown_job_and_asset_are_404():
    assert client.get("/api/person-search/jobs/missing").status_code == 404
    assert client.delete("/api/person-search/jobs/missing").status_code == 404
    assert client.get("/api/assets/missing/file.jpg").status_code == 404


def test_unavailable_weight_behavior_and_upload_validation():
    empty = client.post(
        "/api/person-search/jobs",
        files={"files": ("camera.mp4", b"", "video/mp4")},
        data={"model_key": "effiped_tier1"},
    )
    assert empty.status_code == 400
    assert "Empty upload" in empty.json()["detail"]

    unavailable = client.post(
        "/api/person-search/jobs",
        files={"files": ("camera.mp4", b"not-a-video", "video/mp4")},
        data={"model_key": "effiped_tier1"},
    )
    assert unavailable.status_code == 400
    assert "unavailable" in unavailable.json()["detail"].lower()
