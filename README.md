# EduAdapt-AI

EduAdapt-AI is a full-stack adaptive learning system that delivers real-time question recommendations based on student performance, concept mastery, and historical interaction patterns.

The project combines:

- a **FastAPI backend**
- a **React frontend**
- a **baseline machine learning model**
- a **dynamic heterogeneous temporal GNN**
- **recommendation quality evaluation**
- **model comparison**
- **load benchmarking**

It is designed as an end-to-end AI project for adaptive educational assessment rather than only a model demo.

---

## Project goal

Build an adaptive quiz engine that:

- models students, questions, and concepts as a dynamic graph
- updates student state after every interaction
- predicts the probability of a correct response
- recommends the next best question
- evaluates recommendation quality
- compares baseline and graph-based models
- measures system performance under concurrent usage

---

## Current project scope

EduAdapt-AI now includes:

### Backend
- student session start
- next-question recommendation
- answer submission and correctness checking
- student mastery state tracking
- event logging
- temporal dataset generation
- baseline model training
- dynamic GNN training
- model comparison
- recommendation evaluation
- benchmark summary loading

### Frontend
- home page
- student session page
- answer feedback panel
- student dashboard
- admin panel for model workflows

### Evaluation
- baseline classification metrics
- dynamic GNN validation metrics
- recommendation quality metrics
- system load/latency benchmarking

---

## Key features

- **Real educational content** instead of placeholder concepts
- **Adaptive recommendation engine** based on target challenge level
- **Concept mastery tracking** across student interactions
- **Baseline model** for correctness prediction
- **Dynamic GNN model** for graph-based learning
- **Model comparison pipeline**
- **Recommendation quality evaluation**
- **Locust-based concurrency benchmarking**
- **Frontend UI + Admin controls**

---

## Real educational concepts used

The current real question bank includes topics such as:

- Linear Equations
- Quadratic Equations
- Fractions
- Percentages
- Probability
- Geometry
- Ratios and Proportions
- Algebra Basics

---

## System architecture

### Student flow
1. Student starts a session
2. System recommends the first question
3. Student submits an answer
4. Backend checks correctness and updates mastery
5. System recommends the next best question
6. Student dashboard displays updated mastery and metrics

### Admin flow
1. Build temporal dataset
2. Train baseline model
3. Train dynamic GNN
4. Compare models
5. Evaluate recommendations
6. Inspect benchmark and report summaries

---

## Repository structure

```text
EduAdapt-AI/
├── app/
│   ├── api/
│   │   └── main.py
│   ├── data/
│   │   ├── question_bank.json
│   │   ├── seed_students.json
│   │   └── runtime/
│   ├── graph/
│   │   ├── event_processor.py
│   │   └── feature_store.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── dynamic_gnn.py
│   │   └── recommender.py
│   ├── training/
│   │   ├── baseline_trainer.py
│   │   ├── benchmark_loader.py
│   │   ├── dataset_builder.py
│   │   ├── dynamic_gnn_trainer.py
│   │   ├── model_comparison.py
│   │   └── recommendation_evaluator.py
│   └── utils/
│       └── metrics.py
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── load_test/
│   └── locustfile.py
├── scripts/
│   ├── benchmark_summary.py
│   ├── compare_models.py
│   ├── evaluate_recommendations.py
│   ├── train_baseline.py
│   └── train_dynamic_gnn.py
├── tests/
│   └── test_api.py
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── README.md
