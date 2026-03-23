from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from app.graph.feature_store import feature_store
from app.models.baseline import baseline_model
from app.models.recommender import recommender


RUNTIME_DIR = Path(__file__).resolve().parents[1] / "data" / "runtime"
MODEL_DIR = RUNTIME_DIR / "models"
RECOMMENDATION_METRICS_PATH = MODEL_DIR / "recommendation_metrics.json"


class RecommendationEvaluator:
    def __init__(self, top_k: int = 5, target_prob: float = 0.65) -> None:
        self.top_k = top_k
        self.target_prob = target_prob

    def _entropy(self, counts: dict[str, int]) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p, 2)
        return entropy

    def evaluate(self) -> dict[str, Any]:
        students = list(feature_store.students.values())
        questions = feature_store.list_questions()

        if not students or not questions:
            raise ValueError("Students or questions are not available for recommendation evaluation.")

        all_recommendations: list[dict[str, Any]] = []
        concept_counts: dict[str, int] = {}
        total_recommendations = 0
        in_target_zone = 0
        repeated_questions = 0
        total_score = 0.0
        total_predicted_correctness = 0.0

        for student in students:
            ranked = recommender.rank_questions(
                student=student,
                questions=questions,
                predictor=baseline_model,
                top_k=self.top_k,
            )

            seen_questions = {item["question_id"] for item in student.get("history", [])}

            for rec in ranked:
                total_recommendations += 1
                all_recommendations.append(rec)

                p = float(rec["predicted_correctness"])
                total_predicted_correctness += p
                total_score += float(rec["score"])

                if 0.55 <= p <= 0.75:
                    in_target_zone += 1

                if rec["question_id"] in seen_questions:
                    repeated_questions += 1

                concept_id = rec["concept_id"]
                concept_counts[concept_id] = concept_counts.get(concept_id, 0) + 1

        if total_recommendations == 0:
            raise ValueError("No recommendations were generated for evaluation.")

        concept_entropy = self._entropy(concept_counts)
        max_entropy = math.log(len(concept_counts), 2) if len(concept_counts) > 1 else 1.0
        normalized_concept_diversity = concept_entropy / max_entropy if max_entropy > 0 else 0.0

        metrics = {
            "student_count_evaluated": len(students),
            "top_k": self.top_k,
            "total_recommendations": total_recommendations,
            "target_zone_rate": round(in_target_zone / total_recommendations, 4),
            "repetition_rate": round(repeated_questions / total_recommendations, 4),
            "avg_recommendation_score": round(total_score / total_recommendations, 4),
            "avg_predicted_correctness": round(total_predicted_correctness / total_recommendations, 4),
            "concept_diversity_entropy": round(concept_entropy, 4),
            "concept_diversity_normalized": round(normalized_concept_diversity, 4),
            "concept_distribution": concept_counts,
            "interpretation": self._build_interpretation(
                target_zone_rate=in_target_zone / total_recommendations,
                repetition_rate=repeated_questions / total_recommendations,
                normalized_concept_diversity=normalized_concept_diversity,
            ),
        }

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        RECOMMENDATION_METRICS_PATH.write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
        return metrics

    def _build_interpretation(
        self,
        target_zone_rate: float,
        repetition_rate: float,
        normalized_concept_diversity: float,
    ) -> str:
        parts = []

        if target_zone_rate >= 0.7:
            parts.append("Recommendations are well aligned with the target challenge zone.")
        elif target_zone_rate >= 0.5:
            parts.append("Recommendations are moderately aligned with the target challenge zone.")
        else:
            parts.append("Recommendations need better challenge calibration.")

        if repetition_rate <= 0.15:
            parts.append("Question repetition is low.")
        else:
            parts.append("Question repetition is relatively high and may need tuning.")

        if normalized_concept_diversity >= 0.75:
            parts.append("Concept diversity is strong.")
        elif normalized_concept_diversity >= 0.5:
            parts.append("Concept diversity is acceptable.")
        else:
            parts.append("Concept diversity is limited.")

        return " ".join(parts)

    def load_metrics(self) -> dict[str, Any]:
        if not RECOMMENDATION_METRICS_PATH.exists():
            return {}
        return json.loads(RECOMMENDATION_METRICS_PATH.read_text(encoding="utf-8"))


recommendation_evaluator = RecommendationEvaluator()