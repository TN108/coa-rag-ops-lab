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

    @staticmethod
    def average_values(
        values: list[float],
    ) -> float:
        """
        Return the average of a list of numbers.

        Returns 0.0 when the list is empty.
        """

        if not values:
            return 0.0

        return round(
            sum(values) / len(values),
            2,
        )

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

            latency = rag_result.get(
                "latency",
                {},
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
                "latency": {
                    "embedding_ms": float(
                        latency.get(
                            "embedding_ms",
                            0.0,
                        )
                    ),
                    "retrieval_ms": float(
                        latency.get(
                            "retrieval_ms",
                            0.0,
                        )
                    ),
                    "filtering_ms": float(
                        latency.get(
                            "filtering_ms",
                            0.0,
                        )
                    ),
                    "generation_ms": float(
                        latency.get(
                            "generation_ms",
                            0.0,
                        )
                    ),
                    "total_ms": float(
                        latency.get(
                            "total_ms",
                            0.0,
                        )
                    ),
                },
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
                "latency": {
                    "embedding_ms": 0.0,
                    "retrieval_ms": 0.0,
                    "filtering_ms": 0.0,
                    "generation_ms": 0.0,
                    "total_ms": 0.0,
                },
                "success": False,
                "error": str(exc),
            }

    def calculate_latency_summary(
        self,
        successful_results: list[
            dict[str, Any]
        ],
    ) -> dict[str, float]:
        """
        Calculate average latency for every RAG stage.
        """

        embedding_latencies = []
        retrieval_latencies = []
        filtering_latencies = []
        generation_latencies = []
        total_latencies = []

        for result in successful_results:
            latency = result.get(
                "latency",
                {},
            )

            embedding_latencies.append(
                float(
                    latency.get(
                        "embedding_ms",
                        0.0,
                    )
                )
            )

            retrieval_latencies.append(
                float(
                    latency.get(
                        "retrieval_ms",
                        0.0,
                    )
                )
            )

            filtering_latencies.append(
                float(
                    latency.get(
                        "filtering_ms",
                        0.0,
                    )
                )
            )

            generation_latencies.append(
                float(
                    latency.get(
                        "generation_ms",
                        0.0,
                    )
                )
            )

            total_latencies.append(
                float(
                    latency.get(
                        "total_ms",
                        0.0,
                    )
                )
            )

        return {
            "average_embedding_ms": (
                self.average_values(
                    embedding_latencies
                )
            ),
            "average_retrieval_ms": (
                self.average_values(
                    retrieval_latencies
                )
            ),
            "average_filtering_ms": (
                self.average_values(
                    filtering_latencies
                )
            ),
            "average_generation_ms": (
                self.average_values(
                    generation_latencies
                )
            ),
            "average_total_ms": (
                self.average_values(
                    total_latencies
                )
            ),
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
                float(
                    result["metrics"].get(
                        metric_name,
                        0.0,
                    )
                )
                for result in successful_results
            ]

            return round(
                sum(values) / len(values),
                4,
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
                float(
                    result["metrics"].get(
                        metric_name,
                        0.0,
                    )
                )
                for result in selected_results
            ]

            return round(
                sum(values) / len(values),
                4,
            )

        latency_summary = (
            self.calculate_latency_summary(
                successful_results
            )
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
            "latency": latency_summary,
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