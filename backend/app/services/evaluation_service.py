import json
from pathlib import Path
from typing import Any, Callable

from app.services.metrics_service import (
    calculate_question_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class EvaluationService:
    def __init__(
        self,
        rag_answer_function: Callable[
            ..., dict[str, Any]
        ],
    ) -> None:
        self.rag_answer_function = rag_answer_function

    def load_dataset(
        self,
        dataset_path: str,
    ) -> list[dict[str, Any]]:
        path = Path(dataset_path)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not path.exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            dataset = json.load(file)

        if not isinstance(dataset, list):
            raise ValueError(
                "Evaluation dataset root must be a JSON list."
            )

        return dataset

    def evaluate_question(
        self,
        record: dict[str, Any],
        top_k: int,
        min_retrieval_score: float,
    ) -> dict[str, Any]:
        try:
            rag_result = self.rag_answer_function(
                question=record["question"],
                top_k=top_k,
                min_retrieval_score=(
                    min_retrieval_score
                ),
            )

            generated_answer = rag_result.get(
                "answer",
                "",
            )

            retrieved_sources = rag_result.get(
                "sources",
                [],
            )

            metrics = calculate_question_metrics(
                expected_answer=record[
                    "expected_answer"
                ],
                generated_answer=generated_answer,
                expected_pages=record.get(
                    "expected_pages",
                    [],
                ),
                retrieved_sources=(
                    retrieved_sources
                ),
                answerable=record[
                    "answerable"
                ],
            )

            return {
                "question_id": record[
                    "question_id"
                ],
                "question": record[
                    "question"
                ],
                "expected_answer": record[
                    "expected_answer"
                ],
                "generated_answer": (
                    generated_answer
                ),
                "expected_document": record.get(
                    "expected_document"
                ),
                "expected_pages": record.get(
                    "expected_pages",
                    [],
                ),
                "retrieved_sources": (
                    retrieved_sources
                ),
                "answerable": record[
                    "answerable"
                ],
                "metrics": metrics,
                "success": True,
                "error": None,
            }

        except Exception as exc:
            return {
                "question_id": record.get(
                    "question_id",
                    "unknown",
                ),
                "question": record.get(
                    "question",
                    "",
                ),
                "expected_answer": record.get(
                    "expected_answer",
                    "",
                ),
                "generated_answer": "",
                "expected_document": record.get(
                    "expected_document"
                ),
                "expected_pages": record.get(
                    "expected_pages",
                    [],
                ),
                "retrieved_sources": [],
                "answerable": record.get(
                    "answerable",
                    False,
                ),
                "metrics": {},
                "success": False,
                "error": str(exc),
            }

    def run_evaluation(
        self,
        dataset_path: str,
        top_k: int,
        min_retrieval_score: float,
    ) -> dict[str, Any]:
        dataset = self.load_dataset(
            dataset_path
        )

        results = [
            self.evaluate_question(
                record=record,
                top_k=top_k,
                min_retrieval_score=(
                    min_retrieval_score
                ),
            )
            for record in dataset
        ]

        successful_results = [
            result
            for result in results
            if result["success"]
        ]

        successful_questions = len(
            successful_results
        )

        failed_questions = (
            len(results)
            - successful_questions
        )

        def average_metric(
            metric_name: str,
        ) -> float:
            if not successful_results:
                return 0.0

            values = [
                result["metrics"].get(
                    metric_name,
                    0.0,
                )
                for result in successful_results
            ]

            return (
                sum(values)
                / len(values)
            )

        answerable_results = [
            result
            for result in successful_results
            if result["answerable"]
        ]

        unanswerable_results = [
            result
            for result in successful_results
            if not result["answerable"]
        ]

        def average_metric_for_results(
            selected_results: list[
                dict[str, Any]
            ],
            metric_name: str,
        ) -> float:
            if not selected_results:
                return 0.0

            values = [
                result["metrics"].get(
                    metric_name,
                    0.0,
                )
                for result in selected_results
            ]

            return (
                sum(values)
                / len(values)
            )

        summary = {
            "retrieval_hit_rate": (
                average_metric_for_results(
                    answerable_results,
                    "retrieval_hit",
                )
            ),
            "mean_reciprocal_rank": (
                average_metric_for_results(
                    answerable_results,
                    "reciprocal_rank",
                )
            ),
            "mean_page_recall": (
                average_metric_for_results(
                    answerable_results,
                    "page_recall",
                )
            ),
            "mean_exact_match": (
                average_metric(
                    "exact_match"
                )
            ),
            "mean_token_f1": (
                average_metric(
                    "token_f1"
                )
            ),
            "mean_semantic_similarity": (
                average_metric(
                    "semantic_similarity"
                )
            ),
            "answerable_semantic_similarity": (
                average_metric_for_results(
                    answerable_results,
                    "semantic_similarity",
                )
            ),
            "unanswerable_semantic_similarity": (
                average_metric_for_results(
                    unanswerable_results,
                    "semantic_similarity",
                )
            ),
            "unanswerable_accuracy": (
                average_metric_for_results(
                    unanswerable_results,
                    "unanswerable_correct",
                )
            ),
        }

        return {
            "message": (
                "Evaluation completed."
            ),
            "total_questions": len(
                results
            ),
            "successful_questions": (
                successful_questions
            ),
            "failed_questions": (
                failed_questions
            ),
            "answerable_questions": len(
                answerable_results
            ),
            "unanswerable_questions": len(
                unanswerable_results
            ),
            "top_k": top_k,
            "min_retrieval_score": (
                min_retrieval_score
            ),
            "summary": summary,
            "results": results,
        }