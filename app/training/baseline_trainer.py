from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.graph.feature_store import RUNTIME_DIR


DATASET_CSV_PATH = RUNTIME_DIR / "temporal_dataset.csv"
MODEL_DIR = RUNTIME_DIR / "models"
BASELINE_MODEL_PATH = MODEL_DIR / "baseline_logreg.joblib"
BASELINE_METRICS_PATH = MODEL_DIR / "baseline_metrics.json"


class BaselineTrainer:
    feature_columns = [
        "concept_id",
        "question_difficulty",
        "predicted_correctness_before_update",
        "mastery_before",
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

    target_column = "target_is_correct"

    categorical_features = ["concept_id"]
    numeric_features = [
        "question_difficulty",
        "predicted_correctness_before_update",
        "mastery_before",
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

    def load_dataset(self) -> pd.DataFrame:
        if not DATASET_CSV_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found at {DATASET_CSV_PATH}. Build the temporal dataset first."
            )
        return pd.read_csv(DATASET_CSV_PATH)

    def train(self) -> dict[str, Any]:
        df = self.load_dataset()

        if len(df) < 5:
            raise ValueError(
                "Need at least 5 samples in temporal_dataset.csv before training the baseline model."
            )

        if df[self.target_column].nunique() < 2:
            raise ValueError(
                "Baseline training requires both classes in the dataset. "
                "Submit a few more mixed correct/incorrect answers first."
            )

        df = df.sort_values("timestamp").reset_index(drop=True)

        split_idx = max(1, int(len(df) * 0.8))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        if len(test_df) == 0:
            test_df = train_df.tail(1).copy()
            train_df = train_df.iloc[:-1].copy()

        if len(train_df) == 0:
            raise ValueError("Not enough chronological samples to create a train/test split.")

        X_train = train_df[self.feature_columns]
        y_train = train_df[self.target_column]
        X_test = test_df[self.feature_columns]
        y_test = test_df[self.target_column]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    self.numeric_features,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    self.categorical_features,
                ),
            ]
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics: dict[str, Any] = {
            "train_samples": int(len(train_df)),
            "test_samples": int(len(test_df)),
            "feature_count": int(len(self.feature_columns)),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        }

        if len(set(y_test)) > 1:
            metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_proba)), 4)
        else:
            metrics["roc_auc"] = None

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, BASELINE_MODEL_PATH)
        BASELINE_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        return {
            "message": "Baseline model trained successfully",
            "model_path": str(BASELINE_MODEL_PATH),
            "metrics_path": str(BASELINE_METRICS_PATH),
            "metrics": metrics,
        }

    def load_metrics(self) -> dict[str, Any]:
        if not BASELINE_METRICS_PATH.exists():
            return {}
        return json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))


baseline_trainer = BaselineTrainer()