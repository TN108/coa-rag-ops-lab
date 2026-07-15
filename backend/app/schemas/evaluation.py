from typing import Any

from pydantic import BaseModel, Field


class EvaluationRunRequest(BaseModel):
    dataset_path: str = Field(
        default="data/evaluation/evaluation_dataset.json"
    )
    top_k: int = Field(default=5, ge=1, le=20)
    min_retrieval_score: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
    )


class EvaluationQuestionResult(BaseModel):
    question_id: str
    question: str
    expected_answer: str
    generated_answer: str
    expected_document: str | None
    expected_pages: list[int]
    retrieved_sources: list[dict[str, Any]]
    answerable: bool
    success: bool
    error: str | None = None


class EvaluationRunResponse(BaseModel):
    message: str
    total_questions: int
    successful_questions: int
    failed_questions: int
    top_k: int
    min_retrieval_score: float
    results: list[EvaluationQuestionResult]