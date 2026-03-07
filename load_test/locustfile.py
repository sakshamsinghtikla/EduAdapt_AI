from __future__ import annotations

from locust import HttpUser, between, task


class StudentUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self) -> None:
        self.student_id = "s000"
        response = self.client.post("/start_session", json={"student_id": self.student_id})
        if response.ok:
            payload = response.json()
            self.current_question_id = payload["first_question"]["question_id"]
        else:
            self.current_question_id = "q000"

    @task(2)
    def fetch_next_question(self) -> None:
        response = self.client.get(f"/next_question/{self.student_id}")
        if response.ok and response.json()["recommendations"]:
            self.current_question_id = response.json()["recommendations"][0]["question_id"]

    @task(3)
    def submit_answer(self) -> None:
        self.client.post(
            "/submit_answer",
            json={
                "student_id": self.student_id,
                "question_id": getattr(self, "current_question_id", "q000"),
                "selected_option": "A",
                "response_time": 9.5,
            },
        )
