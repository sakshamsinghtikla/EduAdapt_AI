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

        mastery_before = float(student["mastery"].get(question["concept_id"], 0.5))
        predicted_correctness_before_update = baseline_model.predict_proba(student, question)

        is_correct = int(selected_option == question["correct_option"])

        baseline_model.update_student(student, question, is_correct, response_time)

        mastery_after = float(student["mastery"].get(question["concept_id"], 0.5))

        event = {
            "student_id": student_id,
            "question_id": question_id,
            "concept_id": question["concept_id"],
            "selected_option": selected_option,
            "correct_option": question["correct_option"],
            "is_correct": is_correct,
            "response_time": float(response_time),
            "difficulty": float(question["difficulty"]),
            "predicted_correctness_before_update": round(predicted_correctness_before_update, 4),
            "mastery_before": round(mastery_before, 4),
            "mastery_after": round(mastery_after, 4),
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
