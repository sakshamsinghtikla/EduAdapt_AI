from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from app.graph.feature_store import EVENT_LOG_PATH, RUNTIME_DIR


DATASET_JSONL_PATH = RUNTIME_DIR / "temporal_dataset.jsonl"
DATASET_CSV_PATH = RUNTIME_DIR / "temporal_dataset.csv"
DATASET_SUMMARY_PATH = RUNTIME_DIR / "temporal_dataset_summary.json"


def _safe_mean(values: list[float], default: float) -> float:
    if not values:
        return default
    return round(float(mean(values)), 4)


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


class TemporalDatasetBuilder:
    def load_events(self) -> list[dict[str, Any]]:
        if not EVENT_LOG_PATH.exists():
            return []

        events: list[dict[str, Any]] = []
        with EVENT_LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line))

        events.sort(key=lambda event: _parse_timestamp(event["timestamp"]))
        return events

    def _build_sample(
        self,
        event: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        same_concept_history = [
            previous for previous in history
            if previous["concept_id"] == event["concept_id"]
        ]

        recent_history = history[-5:]
        previous_event = history[-1] if history else None

        if previous_event is None:
            time_since_last_interaction_sec = -1.0
        else:
            current_ts = _parse_timestamp(event["timestamp"])
            previous_ts = _parse_timestamp(previous_event["timestamp"])
            time_since_last_interaction_sec = round(
                (current_ts - previous_ts).total_seconds(),
                4,
            )

        sample = {
            "student_id": event["student_id"],
            "question_id": event["question_id"],
            "concept_id": event["concept_id"],
            "timestamp": event["timestamp"],
            "target_is_correct": int(event["is_correct"]),
            "question_difficulty": float(event["difficulty"]),
            "predicted_correctness_before_update": float(
                event["predicted_correctness_before_update"]
            ),
            "mastery_before": float(event["mastery_before"]),
            "mastery_after": float(event["mastery_after"]),
            "response_time": float(event["response_time"]),
            "history_len": len(history),
            "student_global_accuracy": _safe_mean(
                [float(item["is_correct"]) for item in history],
                0.5,
            ),
            "student_recent_accuracy_5": _safe_mean(
                [float(item["is_correct"]) for item in recent_history],
                0.5,
            ),
            "student_avg_response_time": _safe_mean(
                [float(item["response_time"]) for item in history],
                0.0,
            ),
            "student_recent_avg_response_time_5": _safe_mean(
                [float(item["response_time"]) for item in recent_history],
                0.0,
            ),
            "same_concept_attempts": len(same_concept_history),
            "same_concept_accuracy": _safe_mean(
                [float(item["is_correct"]) for item in same_concept_history],
                0.5,
            ),
            "time_since_last_interaction_sec": time_since_last_interaction_sec,
        }
        return sample

    def _write_jsonl(self, samples: list[dict[str, Any]]) -> None:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        with DATASET_JSONL_PATH.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

    def _write_csv(self, samples: list[dict[str, Any]]) -> None:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "student_id",
            "question_id",
            "concept_id",
            "timestamp",
            "target_is_correct",
            "question_difficulty",
            "predicted_correctness_before_update",
            "mastery_before",
            "mastery_after",
            "response_time",
            "history_len",
            "student_global_accuracy",
            "student_recent_accuracy_5",
            "student_avg_response_time",
            "student_recent_avg_response_time_5",
            "same_concept_attempts",
            "same_concept_accuracy",
            "time_since_last_interaction_sec",
        ]

        with DATASET_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sample in samples:
                writer.writerow(sample)

    def _write_summary(
        self,
        events: list[dict[str, Any]],
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        summary = {
            "event_count": len(events),
            "sample_count": len(samples),
            "student_count": len({event["student_id"] for event in events}),
            "question_count": len({event["question_id"] for event in events}),
            "concept_count": len({event["concept_id"] for event in events}),
            "positive_rate": _safe_mean(
                [float(sample["target_is_correct"]) for sample in samples],
                0.0,
            ),
            "output_jsonl_path": str(DATASET_JSONL_PATH),
            "output_csv_path": str(DATASET_CSV_PATH),
        }

        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        DATASET_SUMMARY_PATH.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary

    def build_dataset(self, min_history: int = 0) -> dict[str, Any]:
        events = self.load_events()
        per_student_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        samples: list[dict[str, Any]] = []

        for event in events:
            student_history = per_student_history[event["student_id"]]

            if len(student_history) >= min_history:
                samples.append(self._build_sample(event, student_history))

            student_history.append(event)

        self._write_jsonl(samples)
        self._write_csv(samples)
        return self._write_summary(events, samples)

    def load_summary(self) -> dict[str, Any]:
        if not DATASET_SUMMARY_PATH.exists():
            return {
                "event_count": 0,
                "sample_count": 0,
                "student_count": 0,
                "question_count": 0,
                "concept_count": 0,
                "positive_rate": 0.0,
                "output_jsonl_path": str(DATASET_JSONL_PATH),
                "output_csv_path": str(DATASET_CSV_PATH),
            }
        return json.loads(DATASET_SUMMARY_PATH.read_text(encoding="utf-8"))

    def preview_samples(self, limit: int = 5) -> list[dict[str, Any]]:
        if not DATASET_JSONL_PATH.exists():
            return []

        samples: list[dict[str, Any]] = []
        with DATASET_JSONL_PATH.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
        return samples


dataset_builder = TemporalDatasetBuilder()