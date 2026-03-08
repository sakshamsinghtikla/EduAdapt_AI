from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RUNTIME_DIR = DATA_DIR / "runtime"
EVENT_LOG_PATH = RUNTIME_DIR / "interactions.jsonl"


class InMemoryFeatureStore:
    def __init__(self) -> None:
        self.students: dict[str, dict[str, Any]] = {}
        self.questions: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self._load_seed_data()

    def _load_seed_data(self) -> None:
        students = json.loads((DATA_DIR / "seed_students.json").read_text(encoding="utf-8"))
        questions = json.loads((DATA_DIR / "question_bank.json").read_text(encoding="utf-8"))
        self.students = {student["student_id"]: student for student in students}
        self.questions = {question["question_id"]: question for question in questions}

    def get_student(self, student_id: str) -> dict[str, Any]:
        return self.students[student_id]

    def get_question(self, question_id: str) -> dict[str, Any]:
        return self.questions[question_id]

    def list_questions(self) -> list[dict[str, Any]]:
        return list(self.questions.values())

    def list_student_ids(self) -> list[str]:
        return sorted(self.students.keys())

    def list_question_ids(self) -> list[str]:
        return sorted(self.questions.keys())

    def list_recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        return self.events[-limit:]

    def append_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        with EVENT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    @property
    def event_log_path(self) -> Path:
        return EVENT_LOG_PATH


feature_store = InMemoryFeatureStore()
