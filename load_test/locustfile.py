from __future__ import annotations

import random
from typing import Any

from locust import HttpUser, between, task


class StudentUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        self.student_id = random.choice(
            [f"s{idx:03d}" for idx in range(10)]
        )
        self.current_question_id = None
        response = self.client.post(
            "/start_session",
            json={"student_id": self.student_id},
            name="/start_session",
        )
        if response.ok:
            payload = response.json()
            self.current_question_id = payload["first_question"]["question_id"]

    @task(2)
    def fetch_next_question(self) -> None:
        response = self.client.get(
            f"/next_question/{self.student_id}",
            name="/next_question",
        )
        if response.ok:
            payload = response.json()
            recommendations = payload.get("recommendations", [])
            if recommendations:
                self.current_question_id = recommendations[0]["question_id"]

    @task(3)
    def submit_answer(self) -> None:
        if not self.current_question_id:
            response = self.client.post(
                "/start_session",
                json={"student_id": self.student_id},
                name="/start_session",
            )
            if response.ok:
                payload = response.json()
                self.current_question_id = payload["first_question"]["question_id"]
            else:
                return

        selected_option = random.choice(["A", "B", "C", "D"])
        response_time = round(random.uniform(5.0, 16.0), 2)

        response = self.client.post(
            "/submit_answer",
            json={
                "student_id": self.student_id,
                "question_id": self.current_question_id,
                "selected_option": selected_option,
                "response_time": response_time,
            },
            name="/submit_answer",
        )

        if response.ok:
            payload = response.json()
            next_recommendations = payload.get("next_recommendations", [])
            if next_recommendations:
                self.current_question_id = next_recommendations[0]["question_id"]

    @task(1)
    def fetch_student_state(self) -> None:
        self.client.get(
            f"/student_state/{self.student_id}",
            name="/student_state",
        )

    @task(1)
    def fetch_metrics(self) -> None:
        self.client.get("/metrics", name="/metrics")