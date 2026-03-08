from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.graph.event_processor import event_processor
from app.graph.feature_store import feature_store
from app.models.baseline import baseline_model
from app.models.recommender import recommender
from app.utils.metrics import metrics_store


app = FastAPI(title="EduAdapt-AI", version="0.1.0")


class StartSessionRequest(BaseModel):
    student_id: str = Field(
        ...,
        description="Existing seeded student id such as s000",
        examples=["s000"],
    )


class SubmitAnswerRequest(BaseModel):
    student_id: str = Field(..., examples=["s000"])
    question_id: str = Field(..., examples=["q000"])
    selected_option: str = Field(..., examples=["A"])
    response_time: float = Field(..., examples=[11.2])


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "EduAdapt-AI is running"}


@app.post("/start_session")
def start_session(payload: StartSessionRequest) -> dict[str, Any]:
    if payload.student_id not in feature_store.students:
        raise HTTPException(status_code=404, detail="Student not found")

    student = feature_store.get_student(payload.student_id)
    recommendations = recommender.rank_questions(
        student,
        feature_store.list_questions(),
        baseline_model,
        top_k=1,
    )

    return {
        "student_id": payload.student_id,
        "first_question": recommendations[0],
        "message": "Session started",
    }


@app.get("/next_question/{student_id}")
def next_question(student_id: str) -> dict[str, Any]:
    if student_id not in feature_store.students:
        raise HTTPException(status_code=404, detail="Student not found")

    student = feature_store.get_student(student_id)
    ranked = recommender.rank_questions(
        student,
        feature_store.list_questions(),
        baseline_model,
        top_k=5,
    )

    return {
        "student_id": student_id,
        "recommendations": ranked,
    }


@app.post("/submit_answer")
def submit_answer(payload: SubmitAnswerRequest) -> dict[str, Any]:
    if payload.student_id not in feature_store.students:
        raise HTTPException(status_code=404, detail="Student not found")

    if payload.question_id not in feature_store.questions:
        raise HTTPException(status_code=404, detail="Question not found")

    result = event_processor.process_answer(
        student_id=payload.student_id,
        question_id=payload.question_id,
        selected_option=payload.selected_option,
        response_time=payload.response_time,
    )

    ranked = recommender.rank_questions(
        result["updated_student"],
        feature_store.list_questions(),
        baseline_model,
        top_k=3,
    )

    return {
        "student_id": payload.student_id,
        "question_id": payload.question_id,
        "is_correct": result["is_correct"],
        "correct_option": result["correct_option"],
        "event": result["event"],
        "next_recommendations": ranked,
        "updated_mastery": result["updated_student"]["mastery"],
    }


@app.get("/student_state/{student_id}")
def student_state(student_id: str) -> dict[str, Any]:
    if student_id not in feature_store.students:
        raise HTTPException(status_code=404, detail="Student not found")
    return feature_store.get_student(student_id)


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return metrics_store.snapshot()


@app.get("/debug/students")
def debug_students(limit: int = 10) -> dict[str, Any]:
    safe_limit = max(1, min(limit, 100))
    return {
        "count": len(feature_store.students),
        "student_ids": feature_store.list_student_ids()[:safe_limit],
    }


@app.get("/debug/questions")
def debug_questions(limit: int = 10) -> dict[str, Any]:
    safe_limit = max(1, min(limit, 100))
    return {
        "count": len(feature_store.questions),
        "question_ids": feature_store.list_question_ids()[:safe_limit],
    }


@app.get("/debug/events")
def debug_events(limit: int = 10) -> dict[str, Any]:
    safe_limit = max(1, min(limit, 100))
    return {
        "event_count": len(feature_store.events),
        "event_log_path": str(feature_store.event_log_path),
        "recent_events": feature_store.list_recent_events(safe_limit),
    }
