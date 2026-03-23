
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from app.graph.feature_store import feature_store
from app.models.dynamic_gnn import (
    DynamicGNNTrainer,
    DynamicTemporalHeteroGNN,
    build_hetero_graph_snapshot,
)


RUNTIME_DIR = Path(__file__).resolve().parents[1] / "data" / "runtime"
DYNAMIC_GNN_DIR = RUNTIME_DIR / "models"
DYNAMIC_GNN_MODEL_PATH = DYNAMIC_GNN_DIR / "dynamic_gnn.pt"
DYNAMIC_GNN_METRICS_PATH = DYNAMIC_GNN_DIR / "dynamic_gnn_metrics.json"


class DynamicGNNTrainingPipeline:
    def __init__(self, hidden_dim: int = 32, lr: float = 1e-3, epochs: int = 25) -> None:
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs

    def _load_students_questions(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        students = list(feature_store.students.values())
        questions = list(feature_store.questions.values())
        concept_names = sorted({question["concept_id"] for question in questions})
        return students, questions, concept_names

    def _load_events(self) -> list[dict[str, Any]]:
        events = list(feature_store.events)

        if not events and feature_store.event_log_path.exists():
            with feature_store.event_log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))

        events.sort(key=lambda item: item["timestamp"])
        return events

    def _build_concept_targets(
        self,
        students: list[dict[str, Any]],
        concept_names: list[str],
    ) -> torch.Tensor:
        targets = []
        for concept in concept_names:
            avg_mastery = sum(
                float(student["mastery"].get(concept, 0.5))
                for student in students
            ) / max(len(students), 1)
            targets.append(avg_mastery)
        return torch.tensor(targets, dtype=torch.float)

    def _evaluate(
        self,
        model: DynamicTemporalHeteroGNN,
        data,
        edge_labels: torch.Tensor,
        concept_targets: torch.Tensor,
    ) -> dict[str, float]:
        model.eval()
        bce_loss = torch.nn.BCEWithLogitsLoss()
        mse_loss = torch.nn.MSELoss()

        with torch.no_grad():
            _, logits, mastery = model(data)
            correctness_loss = bce_loss(logits, edge_labels.float())
            mastery_loss = mse_loss(mastery, concept_targets.float())
            total_loss = correctness_loss + 0.4 * mastery_loss

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            accuracy = (preds == edge_labels.float()).float().mean().item()

        return {
            "loss": float(total_loss.item()),
            "correctness_loss": float(correctness_loss.item()),
            "mastery_loss": float(mastery_loss.item()),
            "accuracy": round(float(accuracy), 4),
        }

    def train(self) -> dict[str, Any]:
        students, questions, concept_names = self._load_students_questions()
        events = self._load_events()

        if len(events) < 10:
            raise ValueError(
                "Need at least 10 interaction events before training the dynamic GNN."
            )

        split_idx = max(1, int(len(events) * 0.8))
        train_events = events[:split_idx]
        val_events = events[split_idx:]

        if len(val_events) == 0:
            val_events = train_events[-1:]
            train_events = train_events[:-1]

        if len(train_events) == 0:
            raise ValueError("Not enough events to create train/validation splits.")

        train_data = build_hetero_graph_snapshot(
            students=students,
            questions=questions,
            concept_names=concept_names,
            events=train_events,
        )
        val_data = build_hetero_graph_snapshot(
            students=students,
            questions=questions,
            concept_names=concept_names,
            events=val_events,
        )

        train_edge_labels = train_data["student", "attempted", "question"].edge_label
        val_edge_labels = val_data["student", "attempted", "question"].edge_label
        concept_targets = self._build_concept_targets(students, concept_names)

        model = DynamicTemporalHeteroGNN(
            student_in_dim=4,
            question_in_dim=3,
            concept_in_dim=2,
            hidden_dim=self.hidden_dim,
        )
        trainer = DynamicGNNTrainer(model=model, lr=self.lr)

        history: list[dict[str, float]] = []

        for epoch in range(1, self.epochs + 1):
            train_metrics = trainer.train_step(
                train_data,
                train_edge_labels,
                concept_targets,
            )
            val_metrics = self._evaluate(
                model,
                val_data,
                val_edge_labels,
                concept_targets,
            )

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 4),
                "train_correctness_loss": round(train_metrics["correctness_loss"], 4),
                "train_mastery_loss": round(train_metrics["mastery_loss"], 4),
                "val_loss": round(val_metrics["loss"], 4),
                "val_correctness_loss": round(val_metrics["correctness_loss"], 4),
                "val_mastery_loss": round(val_metrics["mastery_loss"], 4),
                "val_accuracy": round(val_metrics["accuracy"], 4),
            }
            history.append(epoch_metrics)

        final_metrics = {
            "message": "Dynamic GNN trained successfully",
            "train_event_count": len(train_events),
            "val_event_count": len(val_events),
            "epochs": self.epochs,
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1]["val_loss"],
            "final_val_accuracy": history[-1]["val_accuracy"],
            "history": history,
            "model_path": str(DYNAMIC_GNN_MODEL_PATH),
            "metrics_path": str(DYNAMIC_GNN_METRICS_PATH),
        }

        DYNAMIC_GNN_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), DYNAMIC_GNN_MODEL_PATH)
        DYNAMIC_GNN_METRICS_PATH.write_text(
            json.dumps(final_metrics, indent=2),
            encoding="utf-8",
        )

        return final_metrics

    def load_metrics(self) -> dict[str, Any]:
        if not DYNAMIC_GNN_METRICS_PATH.exists():
            return {}
        return json.loads(DYNAMIC_GNN_METRICS_PATH.read_text(encoding="utf-8"))


dynamic_gnn_training_pipeline = DynamicGNNTrainingPipeline()