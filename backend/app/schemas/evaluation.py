from typing import Any

from pydantic import BaseModel, Field


class EvaluationRunRequest(BaseModel):
    dataset_path: str = Field(
        default="data/evaluation/evaluation_dataset.json",
        min_length=1,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    min_retrieval_score: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
    )


class QuestionMetrics(BaseModel):
    retrieval_hit: float = 0.0
    reciprocal_rank: float = 0.0
    page_recall: float = 0.0
    exact_match: float = 0.0
    token_f1: float = 0.0
    semantic_similarity: float = 0.0
    unanswerable_correct: float = 0.0


class LatencyMetrics(BaseModel):
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    filtering_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class LatencySummary(BaseModel):
    average_embedding_ms: float = 0.0
    average_retrieval_ms: float = 0.0
    average_filtering_ms: float = 0.0
    average_generation_ms: float = 0.0
    average_total_ms: float = 0.0


class EvaluationSummary(BaseModel):
    retrieval_hit_rate: float = 0.0
    mean_reciprocal_rank: float = 0.0
    mean_page_recall: float = 0.0
    mean_exact_match: float = 0.0
    mean_token_f1: float = 0.0
    mean_semantic_similarity: float = 0.0
    answerable_semantic_similarity: float = 0.0
    unanswerable_semantic_similarity: float = 0.0
    unanswerable_accuracy: float = 0.0

    latency: LatencySummary = Field(
        default_factory=LatencySummary
    )


class EvaluationQuestionResult(BaseModel):
    question_id: str
    question: str
    expected_answer: str
    generated_answer: str

    expected_document: str | None = None

    expected_pages: list[int] = Field(
        default_factory=list
    )

    retrieved_sources: list[
        dict[str, Any]
    ] = Field(
        default_factory=list
    )

    answerable: bool

    metrics: QuestionMetrics = Field(
        default_factory=QuestionMetrics
    )

    latency: LatencyMetrics = Field(
        default_factory=LatencyMetrics
    )

    success: bool
    error: str | None = None


class EvaluationRunResponse(BaseModel):
    message: str
    total_questions: int
    successful_questions: int
    failed_questions: int
    answerable_questions: int
    unanswerable_questions: int
    top_k: int
    min_retrieval_score: float

    summary: EvaluationSummary

    results: list[
        EvaluationQuestionResult
    ] = Field(
        default_factory=list
    )