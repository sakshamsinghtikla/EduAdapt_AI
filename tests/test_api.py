from fastapi.testclient import TestClient

from app.api.main import app


client = TestClient(app)


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "EduAdapt-AI is running"


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


def test_submit_answer() -> None:
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
