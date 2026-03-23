from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RUNTIME_DIR = Path(__file__).resolve().parents[1] / "data" / "runtime"
MODEL_DIR = RUNTIME_DIR / "models"

BASELINE_METRICS_PATH = MODEL_DIR / "baseline_metrics.json"
DYNAMIC_GNN_METRICS_PATH = MODEL_DIR / "dynamic_gnn_metrics.json"
MODEL_COMPARISON_PATH = MODEL_DIR / "model_comparison.json"


class ModelComparison:
    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def compare(self) -> dict[str, Any]:
        baseline = self._load_json(BASELINE_METRICS_PATH)
        gnn = self._load_json(DYNAMIC_GNN_METRICS_PATH)

        if not baseline:
            raise FileNotFoundError("Baseline metrics not found.")
        if not gnn:
            raise FileNotFoundError("Dynamic GNN metrics not found.")

        baseline_accuracy = baseline.get("accuracy")
        baseline_f1 = baseline.get("f1")
        baseline_auc = baseline.get("roc_auc")

        gnn_val_accuracy = gnn.get("final_val_accuracy")
        gnn_train_loss = gnn.get("final_train_loss")
        gnn_val_loss = gnn.get("final_val_loss")

        comparison = {
            "baseline": {
                "accuracy": baseline_accuracy,
                "f1": baseline_f1,
                "roc_auc": baseline_auc,
                "train_samples": baseline.get("train_samples"),
                "test_samples": baseline.get("test_samples"),
            },
            "dynamic_gnn": {
                "val_accuracy": gnn_val_accuracy,
                "train_loss": gnn_train_loss,
                "val_loss": gnn_val_loss,
                "train_event_count": gnn.get("train_event_count"),
                "val_event_count": gnn.get("val_event_count"),
                "epochs": gnn.get("epochs"),
            },
            "summary": {
                "baseline_vs_gnn_accuracy_gap": round(float(gnn_val_accuracy - baseline_accuracy), 4)
                if baseline_accuracy is not None and gnn_val_accuracy is not None
                else None,
                "gnn_overfit_signal": bool(
                    gnn_train_loss is not None
                    and gnn_val_loss is not None
                    and gnn_val_loss > gnn_train_loss
                ),
                "winner_by_accuracy": (
                    "dynamic_gnn"
                    if baseline_accuracy is not None and gnn_val_accuracy is not None and gnn_val_accuracy > baseline_accuracy
                    else "baseline"
                ),
                "interpretation": self._interpret(
                    baseline_accuracy=baseline_accuracy,
                    baseline_auc=baseline_auc,
                    gnn_val_accuracy=gnn_val_accuracy,
                    gnn_train_loss=gnn_train_loss,
                    gnn_val_loss=gnn_val_loss,
                ),
            },
        }

        MODEL_COMPARISON_PATH.write_text(
            json.dumps(comparison, indent=2),
            encoding="utf-8",
        )
        return comparison

    def _interpret(
        self,
        baseline_accuracy: float | None,
        baseline_auc: float | None,
        gnn_val_accuracy: float | None,
        gnn_train_loss: float | None,
        gnn_val_loss: float | None,
    ) -> str:
        if baseline_accuracy is None or gnn_val_accuracy is None:
            return "Insufficient metrics to compare models."

        if gnn_val_accuracy > baseline_accuracy:
            return "Dynamic GNN outperformed the baseline on validation accuracy."

        if gnn_train_loss is not None and gnn_val_loss is not None and gnn_val_loss > gnn_train_loss:
            return (
                "Dynamic GNN trained successfully but shows signs of overfitting. "
                "Baseline remains more reliable on this dataset."
            )

        if baseline_auc is not None and baseline_auc >= 0.7:
            return (
                "Baseline remains competitive and better calibrated on the current small, imbalanced dataset."
            )

        return "Models are close in performance; more data is needed for a decisive comparison."

    def load_comparison(self) -> dict[str, Any]:
        if not MODEL_COMPARISON_PATH.exists():
            return {}
        return json.loads(MODEL_COMPARISON_PATH.read_text(encoding="utf-8"))


model_comparison = ModelComparison()