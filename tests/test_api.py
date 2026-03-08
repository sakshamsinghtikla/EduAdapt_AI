from fastapi.testclient import TestClient

from app.api.main import app


client = TestClient(app)


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "EduAdapt-AI is running"


def test_debug_endpoints() -> None:
    response = client.get("/debug/students")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] > 0
    assert "s000" in payload["student_ids"]

    response = client.get("/debug/questions")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] > 0
    assert any(qid.startswith("q") for qid in payload["question_ids"])


def test_start_session_and_next_question() -> None:
    response = client.post("/start_session", json={"student_id": "s000"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["student_id"] == "s000"
    assert "first_question" in payload

    response = client.get("/next_question/s000")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["recommendations"]) > 0


def test_submit_answer_and_event_logging() -> None:
    first = client.post("/start_session", json={"student_id": "s001"}).json()
    qid = first["first_question"]["question_id"]

    response = client.post(
        "/submit_answer",
        json={
            "student_id": "s001",
            "question_id": qid,
            "selected_option": "A",
            "response_time": 11.0,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "updated_mastery" in payload
    assert "next_recommendations" in payload
    assert "event" in payload
    assert payload["event"]["student_id"] == "s001"
    assert payload["event"]["question_id"] == qid

    debug_events = client.get("/debug/events")
    assert debug_events.status_code == 200
    events_payload = debug_events.json()
    assert events_payload["event_count"] >= 1
    assert len(events_payload["recent_events"]) >= 1


def test_build_temporal_dataset() -> None:
    first = client.post("/start_session", json={"student_id": "s002"}).json()
    qid = first["first_question"]["question_id"]

    submit = client.post(
        "/submit_answer",
        json={
            "student_id": "s002",
            "question_id": qid,
            "selected_option": "B",
            "response_time": 9.5,
        },
    )
    assert submit.status_code == 200

    response = client.post("/admin/build_dataset?min_history=0")
    assert response.status_code == 200
    payload = response.json()

    assert payload["message"] == "Temporal dataset built"
    assert payload["summary"]["event_count"] >= 1
    assert payload["summary"]["sample_count"] >= 1

    preview = client.get("/debug/dataset")
    assert preview.status_code == 200
    preview_payload = preview.json()
    assert "summary" in preview_payload
    assert "samples" in preview_payload
    assert len(preview_payload["samples"]) >= 1
    
    def test_train_baseline_after_dataset_build() -> None:
    first = client.post("/start_session", json={"student_id": "s003"}).json()
    qid = first["first_question"]["question_id"]

    submit = client.post(
        "/submit_answer",
        json={
            "student_id": "s003",
            "question_id": qid,
            "selected_option": "A",
            "response_time": 10.0,
        },
    )
    assert submit.status_code == 200

    build = client.post("/admin/build_dataset?min_history=0")
    assert build.status_code == 200

    train = client.post("/admin/train_baseline")
    assert train.status_code == 200
    payload = train.json()

    assert payload["message"] == "Baseline model trained successfully"
    assert "metrics" in payload
    assert "accuracy" in payload["metrics"]

    metrics = client.get("/debug/baseline_metrics")
    assert metrics.status_code == 200