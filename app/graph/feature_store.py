from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class InMemoryFeatureStore:
    def __init__(self) -> None:
        self.students: dict[str, dict[str, Any]] = {}
        self.questions: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self._load_seed_data()

    def _load_seed_data(self) -> None:
        students = json.loads((DATA_DIR / "seed_students.json").read_text())
        questions = json.loads((DATA_DIR / "question_bank.json").read_text())
        self.students = {student["student_id"]: student for student in students}
        self.questions = {question["question_id"]: question for question in questions}

    def get_student(self, student_id: str) -> dict[str, Any]:
        return self.students[student_id]

    def get_question(self, question_id: str) -> dict[str, Any]:
        return self.questions[question_id]

    def list_questions(self) -> list[dict[str, Any]]:
        return list(self.questions.values())

    def append_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)


feature_store = InMemoryFeatureStore()
