from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import GATConv, HeteroConv, Linear
except Exception:  # pragma: no cover
    HeteroData = None
    GATConv = None
    HeteroConv = None
    Linear = None


@dataclass
class TemporalBatch:
    student_ids: list[str]
    question_ids: list[str]
    concept_ids: list[str]
    correctness: torch.Tensor


class DynamicTemporalHeteroGNN(nn.Module):
    """Starter model for the main AI component.

    This is intentionally a real PyTorch model with a heterogeneous graph design,
    while still being light enough to keep the starter repo understandable.
    """

    def __init__(
        self,
        student_in_dim: int = 4,
        question_in_dim: int = 3,
        concept_in_dim: int = 2,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if Linear is None or HeteroConv is None or GATConv is None:
            raise ImportError(
                "torch_geometric is not installed. Install PyTorch Geometric to use the dynamic GNN module."
            )

        self.student_encoder = Linear(student_in_dim, hidden_dim)
        self.question_encoder = Linear(question_in_dim, hidden_dim)
        self.concept_encoder = Linear(concept_in_dim, hidden_dim)
        self.time_gate = nn.GRUCell(hidden_dim, hidden_dim)

        self.conv1 = HeteroConv(
            {
                ("student", "attempted", "question"): GATConv((hidden_dim, hidden_dim), hidden_dim, add_self_loops=False),
                ("question", "rev_attempted", "student"): GATConv((hidden_dim, hidden_dim), hidden_dim, add_self_loops=False),
                ("question", "tagged_with", "concept"): GATConv((hidden_dim, hidden_dim), hidden_dim, add_self_loops=False),
                ("concept", "rev_tagged_with", "question"): GATConv((hidden_dim, hidden_dim), hidden_dim, add_self_loops=False),
                ("concept", "prerequisite", "concept"): GATConv((hidden_dim, hidden_dim), hidden_dim, add_self_loops=False),
                ("student", "mastery_estimate", "concept"): GATConv((hidden_dim, hidden_dim), hidden_dim, add_self_loops=False),
                ("concept", "rev_mastery_estimate", "student"): GATConv((hidden_dim, hidden_dim), hidden_dim, add_self_loops=False),
            },
            aggr="sum",
        )

        self.conv2 = HeteroConv(
            {
                ("student", "attempted", "question"): GATConv((hidden_dim, hidden_dim), hidden_dim, add_self_loops=False),
                ("question", "rev_attempted", "student"): GATConv((hidden_dim, hidden_dim), hidden_dim, add_self_loops=False),
                ("question", "tagged_with", "concept"): GATConv((hidden_dim, hidden_dim), hidden_dim, add_self_loops=False),
                ("concept", "rev_tagged_with", "question"): GATConv((hidden_dim, hidden_dim), hidden_dim, add_self_loops=False),
                ("concept", "prerequisite", "concept"): GATConv((hidden_dim, hidden_dim), hidden_dim, add_self_loops=False),
                ("student", "mastery_estimate", "concept"): GATConv((hidden_dim, hidden_dim), hidden_dim, add_self_loops=False),
                ("concept", "rev_mastery_estimate", "student"): GATConv((hidden_dim, hidden_dim), hidden_dim, add_self_loops=False),
            },
            aggr="sum",
        )

        self.correctness_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.mastery_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, data: "HeteroData") -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        x_dict = {
            "student": self.student_encoder(data["student"].x),
            "question": self.question_encoder(data["question"].x),
            "concept": self.concept_encoder(data["concept"].x),
        }

        if "prev_h" in data["student"]:
            x_dict["student"] = self.time_gate(x_dict["student"], data["student"].prev_h)

        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {key: value.relu() for key, value in x_dict.items()}
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {key: value.relu() for key, value in x_dict.items()}

        student_idx = data["student", "attempted", "question"].edge_label_index[0]
        question_idx = data["student", "attempted", "question"].edge_label_index[1]
        pair_embedding = torch.cat([x_dict["student"][student_idx], x_dict["question"][question_idx]], dim=-1)
        logits = self.correctness_head(pair_embedding).squeeze(-1)
        mastery = self.mastery_head(x_dict["concept"]).squeeze(-1)
        return x_dict, logits, mastery


class DynamicGNNTrainer:
    def __init__(self, model: DynamicTemporalHeteroGNN, lr: float = 1e-3) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def train_step(self, data: "HeteroData", edge_labels: torch.Tensor, concept_targets: torch.Tensor) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        _, logits, mastery = self.model(data)
        loss_correctness = self.bce_loss(logits, edge_labels.float())
        loss_mastery = self.mse_loss(mastery, concept_targets.float())
        loss = loss_correctness + 0.4 * loss_mastery
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "correctness_loss": float(loss_correctness.item()),
            "mastery_loss": float(loss_mastery.item()),
        }


def build_hetero_graph_snapshot(
    students: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    concept_names: list[str],
    events: list[dict[str, Any]],
) -> "HeteroData":
    if HeteroData is None:
        raise ImportError("torch_geometric is not installed.")

    data = HeteroData()
    student_index = {student["student_id"]: idx for idx, student in enumerate(students)}
    question_index = {question["question_id"]: idx for idx, question in enumerate(questions)}
    concept_index = {concept: idx for idx, concept in enumerate(concept_names)}

    student_x = []
    for student in students:
        student_x.append(
            [
                float(student.get("ability", 0.0)),
                float(student.get("recent_accuracy", 0.5)),
                float(student.get("avg_response_time", 12.0)) / 30.0,
                float(len(student.get("history", []))) / 20.0,
            ]
        )

    question_x = []
    for question in questions:
        question_x.append(
            [
                float(question.get("difficulty", 0.0)),
                float(question.get("discrimination", 1.0)),
                float(hash(question.get("concept_id", "")) % 100) / 100.0,
            ]
        )

    concept_x = []
    for concept in concept_names:
        avg_mastery = sum(student["mastery"].get(concept, 0.5) for student in students) / max(len(students), 1)
        concept_x.append([float(avg_mastery), float(concept_index[concept]) / max(len(concept_names), 1)])

    data["student"].x = torch.tensor(student_x, dtype=torch.float)
    data["student"].prev_h = torch.zeros((len(student_x), 32), dtype=torch.float)
    data["question"].x = torch.tensor(question_x, dtype=torch.float)
    data["concept"].x = torch.tensor(concept_x, dtype=torch.float)

    attempted_src, attempted_dst, attempted_labels = [], [], []
    mastery_src, mastery_dst = [], []
    tagged_src, tagged_dst = [], []

    for question in questions:
        tagged_src.append(question_index[question["question_id"]])
        tagged_dst.append(concept_index[question["concept_id"]])

    for student in students:
        for concept, mastery in student["mastery"].items():
            if concept in concept_index:
                mastery_src.append(student_index[student["student_id"]])
                mastery_dst.append(concept_index[concept])

    for event in events:
        if event["student_id"] in student_index and event["question_id"] in question_index:
            attempted_src.append(student_index[event["student_id"]])
            attempted_dst.append(question_index[event["question_id"]])
            attempted_labels.append(float(event["is_correct"]))

    if not attempted_src:
        attempted_src = [0]
        attempted_dst = [0]
        attempted_labels = [0.0]

    data[("student", "attempted", "question")].edge_index = torch.tensor([attempted_src, attempted_dst], dtype=torch.long)
    data[("question", "rev_attempted", "student")].edge_index = torch.tensor([attempted_dst, attempted_src], dtype=torch.long)
    data[("student", "attempted", "question")].edge_label_index = torch.tensor([attempted_src, attempted_dst], dtype=torch.long)
    data[("question", "tagged_with", "concept")].edge_index = torch.tensor([tagged_src, tagged_dst], dtype=torch.long)
    data[("concept", "rev_tagged_with", "question")].edge_index = torch.tensor([tagged_dst, tagged_src], dtype=torch.long)
    data[("student", "mastery_estimate", "concept")].edge_index = torch.tensor([mastery_src, mastery_dst], dtype=torch.long)
    data[("concept", "rev_mastery_estimate", "student")].edge_index = torch.tensor([mastery_dst, mastery_src], dtype=torch.long)
    data[("concept", "prerequisite", "concept")].edge_index = torch.empty((2, 0), dtype=torch.long)
    data[("student", "attempted", "question")].edge_label = torch.tensor(attempted_labels, dtype=torch.float)
    return data
