from __future__ import annotations

import math
from typing import Any

import numpy as np


class BaselineKnowledgeTracer:
    def __init__(self, concept_lr: float = 0.06, ability_lr: float = 0.04) -> None:
        self.concept_lr = concept_lr
        self.ability_lr = ability_lr

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def predict_proba(self, student: dict[str, Any], question: dict[str, Any]) -> float:
        mastery = student["mastery"].get(question["concept_id"], 0.5)
        ability = float(student.get("ability", 0.0))
        difficulty = float(question.get("difficulty", 0.0))
        recent_accuracy = float(student.get("recent_accuracy", 0.5))
        response_penalty = min(float(student.get("avg_response_time", 12.0)) / 30.0, 0.6)
        logit = ability + 2.0 * (mastery - 0.5) + 0.8 * (recent_accuracy - 0.5) - difficulty - response_penalty
        return float(np.clip(self._sigmoid(logit), 1e-4, 1 - 1e-4))

    def update_student(self, student: dict[str, Any], question: dict[str, Any], is_correct: int, response_time: float) -> dict[str, Any]:
        concept_id = question["concept_id"]
        current_mastery = float(student["mastery"].get(concept_id, 0.5))
        delta = self.concept_lr if is_correct else -self.concept_lr / 2
        new_mastery = float(np.clip(current_mastery + delta, 0.05, 0.95))
        student["mastery"][concept_id] = new_mastery

        ability_delta = self.ability_lr if is_correct else -self.ability_lr / 2
        student["ability"] = float(np.clip(student.get("ability", 0.0) + ability_delta, -3.0, 3.0))

        history = student.setdefault("history", [])
        history.append(
            {
                "question_id": question["question_id"],
                "concept_id": concept_id,
                "is_correct": int(is_correct),
                "response_time": float(response_time),
            }
        )
        window = history[-20:]
        student["history"] = window
        student["recent_accuracy"] = float(np.mean([item["is_correct"] for item in window]))
        student["avg_response_time"] = float(np.mean([item["response_time"] for item in window]))
        return student


baseline_model = BaselineKnowledgeTracer()
