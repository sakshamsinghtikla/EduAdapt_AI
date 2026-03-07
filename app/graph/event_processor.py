from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.graph.feature_store import feature_store
from app.models.baseline import baseline_model
from app.utils.metrics import metrics_store


class EventProcessor:
    def process_answer(
        self,
        student_id: str,
        question_id: str,
        selected_option: str,
        response_time: float,
    ) -> dict[str, Any]:
        student = feature_store.get_student(student_id)
        question = feature_store.get_question(question_id)
        is_correct = int(selected_option == question["correct_option"])
        baseline_model.update_student(student, question, is_correct, response_time)
        event = {
            "student_id": student_id,
            "question_id": question_id,
            "concept_id": question["concept_id"],
            "is_correct": is_correct,
            "response_time": response_time,
            "difficulty": question["difficulty"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        feature_store.append_event(event)
        metrics_store.events_processed += 1
        return {
            "is_correct": is_correct,
            "correct_option": question["correct_option"],
            "updated_student": student,
            "event": event,
        }


event_processor = EventProcessor()
