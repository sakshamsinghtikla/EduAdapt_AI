from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import math
import random
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent


@dataclass
class InteractionEvent:
    student_id: str
    question_id: str
    concept_id: str
    is_correct: int
    response_time: float
    difficulty: float
    timestamp: str


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def generate_seed_data(
    n_students: int = 50,
    n_questions: int = 120,
    n_concepts: int = 8,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    concepts = [f"concept_{idx}" for idx in range(n_concepts)]

    students = []
    for idx in range(n_students):
        mastery = {concept: float(np.clip(np_rng.normal(0.5, 0.15), 0.1, 0.9)) for concept in concepts}
        students.append(
            {
                "student_id": f"s{idx:03d}",
                "ability": float(np.clip(np_rng.normal(0.0, 0.8), -2.0, 2.0)),
                "mastery": mastery,
                "recent_accuracy": 0.5,
                "avg_response_time": 14.0,
                "history": [],
            }
        )

    questions = []
    for idx in range(n_questions):
        concept = rng.choice(concepts)
        difficulty = float(np.clip(np_rng.normal(0.0, 1.0), -2.0, 2.0))
        questions.append(
            {
                "question_id": f"q{idx:03d}",
                "concept_id": concept,
                "difficulty": difficulty,
                "discrimination": float(np.clip(np_rng.normal(1.0, 0.2), 0.5, 1.5)),
                "text": f"Practice question {idx} for {concept}",
                "options": ["A", "B", "C", "D"],
                "correct_option": rng.choice(["A", "B", "C", "D"]),
            }
        )
    return students, questions


def persist_seed_files() -> None:
    students, questions = generate_seed_data()
    (DATA_DIR / "seed_students.json").write_text(json.dumps(students, indent=2))
    (DATA_DIR / "question_bank.json").write_text(json.dumps(questions, indent=2))


def simulate_response(student: dict[str, Any], question: dict[str, Any], rng_seed: int | None = None) -> tuple[int, float]:
    rng = random.Random(rng_seed)
    concept_mastery = student["mastery"].get(question["concept_id"], 0.5)
    ability = student["ability"]
    logit = ability + 2.2 * (concept_mastery - 0.5) - question["difficulty"]
    probability = _sigmoid(logit)
    is_correct = 1 if rng.random() < probability else 0
    response_time = max(3.0, rng.normalvariate(10.0 + 3.0 * abs(question["difficulty"] - concept_mastery), 1.5))
    return is_correct, round(response_time, 2)


if __name__ == "__main__":
    persist_seed_files()
    print("Seed files written to app/data/")
