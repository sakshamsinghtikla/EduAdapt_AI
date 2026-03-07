from __future__ import annotations

from typing import Any


class AdaptiveRecommender:
    def __init__(self, alpha: float = 0.6, beta: float = 0.3, gamma: float = 0.1, target_prob: float = 0.65) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.target_prob = target_prob

    def rank_questions(
        self,
        student: dict[str, Any],
        questions: list[dict[str, Any]],
        predictor,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        seen_questions = {item["question_id"] for item in student.get("history", [])}
        scored = []
        for question in questions:
            p = predictor.predict_proba(student, question)
            concept_mastery = student["mastery"].get(question["concept_id"], 0.5)
            learning_gain = 1.0 - abs(concept_mastery - 0.7)
            repetition_penalty = 1.0 if question["question_id"] in seen_questions else 0.0
            score = self.alpha * (1.0 - abs(p - self.target_prob)) + self.beta * learning_gain - self.gamma * repetition_penalty
            scored.append(
                {
                    "question_id": question["question_id"],
                    "concept_id": question["concept_id"],
                    "difficulty": question["difficulty"],
                    "text": question["text"],
                    "predicted_correctness": round(float(p), 4),
                    "score": round(float(score), 4),
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]


recommender = AdaptiveRecommender()
