# EduAdapt-AI

A separate Git-ready AI project for **real-time adaptive assessment** using a **dynamic heterogeneous temporal GNN**.

## Project goal

Build an adaptive quiz engine that:
- models students, questions, and concepts as a dynamic graph
- updates student state after every interaction
- predicts the probability of a correct next response
- recommends the next best question
- remains measurable under concurrent multi-student usage

This repo starts with a working **baseline model + FastAPI loop** and includes the first **dynamic GNN module** so the project is a complete AI project rather than only a systems demo.

## Repo structure

```text
EduAdapt-AI/
├── app/
│   ├── api/
│   │   └── main.py
│   ├── data/
│   │   ├── simulator.py
│   │   ├── question_bank.json
│   │   └── seed_students.json
│   ├── graph/
│   │   ├── event_processor.py
│   │   └── feature_store.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── dynamic_gnn.py
│   │   └── recommender.py
│   └── utils/
│       ├── config.py
│       └── metrics.py
├── load_test/
│   └── locustfile.py
├── tests/
│   └── test_api.py
├── .gitignore
├── docker-compose.yml
└── requirements.txt
```

## Current scope

### Working now
- synthetic students and questions
- baseline correctness predictor
- question recommender
- FastAPI endpoints:
  - `POST /start_session`
  - `GET /next_question/{student_id}`
  - `POST /submit_answer`
  - `GET /student_state/{student_id}`
  - `GET /metrics`
- in-memory feature store and event history
- Locust starter for concurrency testing

### Included for the AI model track
- heterogeneous temporal GNN module in `app/models/dynamic_gnn.py`
- graph data builder inside the same file
- student/question/concept node design that matches the project problem statement

## Setup

Create a fresh Git repo for this project only:

```bash
git init
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are installing PyTorch and PyTorch Geometric manually for your hardware, install them first and then run:

```bash
pip install fastapi uvicorn pydantic pandas numpy scikit-learn locust pytest httpx redis
```

## Run the API

```bash
uvicorn app.api.main:app --reload
```

Open:
- API docs: `http://127.0.0.1:8000/docs`

## Run tests

```bash
pytest -q
```

## Run load test

```bash
locust -f load_test/locustfile.py
```

Then open Locust UI at `http://127.0.0.1:8089`.

## Core modeling plan

### Baseline
Logistic-style student ability vs question difficulty:

\[
p = \sigma(\theta_{student} - b_{question} + w^T x)
\]

### Main model
Dynamic heterogeneous temporal GNN with node types:
- student
- question
- concept

Edge types:
- student -> question: attempted
- question -> concept: tagged_with
- concept -> concept: prerequisite
- student -> concept: mastery_estimate

Targets:
1. probability of correctness for the next interaction
2. updated mastery vector over concepts

## Recommended next Git commits

1. `init repo with simulator, api, baseline, recommender`
2. `add event processor and metrics collection`
3. `add dynamic heterogeneous temporal gnn module`
4. `add notebook for baseline training`
5. `add notebook for dynamic graph training`
6. `add redis stream integration`
7. `add dashboard and benchmark plots`

## Evaluation to report later

### ML metrics
- accuracy
- F1
- ROC-AUC
- calibration
- recommendation success rate
- mastery prediction error

### System metrics
- mean latency
- P95 latency
- requests per second
- throughput
- error rate

### Real-time adaptation metrics
- state update time
- next-question generation time
- performance under burst traffic
- baseline vs dynamic-GNN comparison
