from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from app.agents.coordinator_agent import create_coa_graph
from app.services.llm_service import (
    FALLBACK_ANSWER,
    normalise_text,
)


class COAEvaluationService:
    """
    Run the complete COA graph against an evaluation dataset and
    calculate retrieval, answer, critic, safety, and latency metrics.
    """

    def __init__(
        self,
        dataset_path: str,
    ) -> None:
        self.dataset_path = Path(
            dataset_path
        )

        self.dataset = self._load_dataset(
            self.dataset_path
        )

        # Create the graph once.
        # Do not recreate it for every question.
        self.graph = create_coa_graph()

    # ============================================================
    # Dataset loading
    # ============================================================

    @staticmethod
    def _load_dataset(
        dataset_path: Path,
    ) -> list[dict[str, Any]]:
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found: "
                f"{dataset_path}"
            )

        with dataset_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        # Support both:
        # [...]
        #
        # and:
        # {"questions": [...]}
        if isinstance(data, list):
            dataset = data

        elif isinstance(data, dict):
            dataset = data.get(
                "questions",
                [],
            )

        else:
            raise ValueError(
                "Evaluation dataset must be a JSON list "
                "or an object containing a 'questions' list."
            )

        if not dataset:
            raise ValueError(
                "Evaluation dataset contains no questions."
            )

        for index, item in enumerate(
            dataset,
            start=1,
        ):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Dataset item {index} must be an object."
                )

            if not str(
                item.get("question") or ""
            ).strip():
                raise ValueError(
                    f"Dataset item {index} has no question."
                )

        return dataset

    # ============================================================
    # Main evaluation
    # ============================================================

    def run(
        self,
        top_k: int = 5,
        min_retrieval_score: float = 0.1,
        save_report: bool = True,
    ) -> dict[str, Any]:
        if top_k < 1:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if not 0.0 <= min_retrieval_score <= 1.0:
            raise ValueError(
                "min_retrieval_score must be between 0 and 1."
            )

        results: list[dict[str, Any]] = []

        evaluation_start = time.perf_counter()

        total_questions = len(
            self.dataset
        )

        for index, item in enumerate(
            self.dataset,
            start=1,
        ):
            question_id = str(
                item.get("question_id")
                or f"q{index:03d}"
            )

            question = str(
                item["question"]
            ).strip()

            print(
                f"Evaluating {index}/{total_questions}: "
                f"{question_id} - {question}"
            )

            question_result = (
                self._evaluate_question(
                    item=item,
                    top_k=top_k,
                    min_retrieval_score=(
                        min_retrieval_score
                    ),
                )
            )

            results.append(
                question_result
            )

        total_evaluation_latency_ms = (
            time.perf_counter()
            - evaluation_start
        ) * 1000

        summary = self._calculate_summary(
            results
        )

        report: dict[str, Any] = {
            "message": "COA evaluation completed.",
            "dataset_path": str(
                self.dataset_path
            ),
            "configuration": {
                "top_k": top_k,
                "min_retrieval_score": (
                    min_retrieval_score
                ),
            },
            "dataset_size": total_questions,
            "answerable_questions": sum(
                1
                for item in self.dataset
                if bool(
                    item.get(
                        "answerable",
                        True,
                    )
                )
            ),
            "unanswerable_questions": sum(
                1
                for item in self.dataset
                if not bool(
                    item.get(
                        "answerable",
                        True,
                    )
                )
            ),
            "total_evaluation_latency_ms": round(
                total_evaluation_latency_ms,
                2,
            ),
            "average_evaluation_latency_ms": round(
                (
                    total_evaluation_latency_ms
                    / total_questions
                ),
                2,
            ),
            "summary": summary,
            "results": results,
        }

        if save_report:
            report_path = self._save_report(
                report
            )

            report["report_path"] = (
                report_path
            )

        return report

    # ============================================================
    # Single-question evaluation
    # ============================================================

    def _evaluate_question(
        self,
        item: dict[str, Any],
        top_k: int,
        min_retrieval_score: float,
    ) -> dict[str, Any]:
        question_id = str(
            item.get("question_id") or ""
        ).strip()

        question = str(
            item.get("question") or ""
        ).strip()

        answerable = bool(
            item.get(
                "answerable",
                True,
            )
        )

        expected_answer = str(
            item.get("expected_answer") or ""
        ).strip()

        expected_document = (
            item.get("expected_document")
        )

        expected_pages = {
            int(page)
            for page in (
                item.get("expected_pages")
                or []
            )
        }

        state: dict[str, Any] = {
            "question": question,

            # Retrieval configuration
            "top_k": top_k,
            "requested_top_k": top_k,
            "min_retrieval_score": (
                min_retrieval_score
            ),

            # Retrieval state
            "retrieved_chunks": [],
            "retrieval_confidence": 0.0,
            "retrieval_latency_ms": 0.0,

            # Reasoning state
            "reasoning": "",
            "structured_facts": [],

            # Critic state
            "critic_results": [],
            "critic_feedback": "",

            # Final state
            "final_answer": "",
            "latency": {},
            "error": None,
        }

        start_time = time.perf_counter()

        try:
            result_state = self.graph.invoke(
                state
            )

            if result_state is None:
                result_state = {}

            error = result_state.get(
                "error"
            )

        except Exception as exception:
            result_state = {}
            error = (
                f"{type(exception).__name__}: "
                f"{exception}"
            )

        total_latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        final_answer = str(
            result_state.get(
                "final_answer"
            )
            or ""
        ).strip()

        retrieved_chunks = (
            result_state.get(
                "retrieved_chunks"
            )
            or []
        )

        structured_facts = (
            result_state.get(
                "structured_facts"
            )
            or []
        )

        critic_results = (
            result_state.get(
                "critic_results"
            )
            or []
        )

        latency = (
            result_state.get("latency")
            or {}
        )

        fallback_returned = (
            normalise_text(final_answer)
            == normalise_text(
                FALLBACK_ANSWER
            )
        )

        retrieval_hit = (
            self._calculate_retrieval_hit(
                answerable=answerable,
                expected_document=(
                    expected_document
                ),
                expected_pages=(
                    expected_pages
                ),
                retrieved_chunks=(
                    retrieved_chunks
                ),
            )
        )

        reciprocal_rank = (
            self._calculate_reciprocal_rank(
                answerable=answerable,
                expected_document=(
                    expected_document
                ),
                expected_pages=(
                    expected_pages
                ),
                retrieved_chunks=(
                    retrieved_chunks
                ),
            )
        )

        exact_match = (
            normalise_text(final_answer)
            == normalise_text(
                expected_answer
            )
        )

        token_f1 = self._token_f1(
            prediction=final_answer,
            reference=expected_answer,
        )

        approved_claims = [
            decision
            for decision in critic_results
            if (
                decision.get(
                    "supported"
                )
                is True
                and decision.get(
                    "relevant_to_question"
                )
                is True
            )
        ]

        rejected_claims = [
            decision
            for decision in critic_results
            if (
                decision.get(
                    "supported"
                )
                is False
                or decision.get(
                    "relevant_to_question"
                )
                is False
            )
        ]

        rejected_claim_leaked = (
            self._rejected_claim_leaked(
                final_answer=final_answer,
                critic_results=(
                    critic_results
                ),
            )
        )

        retrieved_pages = [
            chunk.get("page_number")
            for chunk in retrieved_chunks
            if chunk.get(
                "page_number"
            )
            is not None
        ]

        retrieved_documents = list(
            dict.fromkeys(
                str(
                    chunk.get(
                        "document_name"
                    )
                    or ""
                )
                for chunk in retrieved_chunks
                if chunk.get(
                    "document_name"
                )
            )
        )

        return {
            "question_id": question_id,
            "question": question,
            "category": item.get(
                "category"
            ),
            "difficulty": item.get(
                "difficulty"
            ),
            "answerable": answerable,
            "notes": item.get(
                "notes",
                "",
            ),

            # Expected result
            "expected_answer": (
                expected_answer
            ),
            "expected_document": (
                expected_document
            ),
            "expected_pages": sorted(
                expected_pages
            ),

            # Actual result
            "final_answer": final_answer,
            "retrieved_documents": (
                retrieved_documents
            ),
            "retrieved_pages": (
                retrieved_pages
            ),
            "structured_facts": (
                structured_facts
            ),
            "critic_results": (
                critic_results
            ),
            "critic_feedback": (
                result_state.get(
                    "critic_feedback",
                    "",
                )
            ),

            # Retrieval metrics
            "retrieval_hit": (
                retrieval_hit
            ),
            "reciprocal_rank": round(
                reciprocal_rank,
                4,
            ),

            # Answer metrics
            "exact_match": exact_match,
            "token_f1": round(
                token_f1,
                4,
            ),
            "fallback_returned": (
                fallback_returned
            ),
            "answerable_response_returned": (
                answerable
                and not fallback_returned
                and bool(final_answer)
            ),
            "unanswerable_correct": (
                not answerable
                and fallback_returned
            ),

            # COA metrics
            "total_claims": len(
                structured_facts
            ),
            "approved_claims": len(
                approved_claims
            ),
            "rejected_claims": len(
                rejected_claims
            ),
            "direct_match_decisions": sum(
                1
                for decision
                in critic_results
                if decision.get(
                    "verification_method"
                )
                == "direct_match"
            ),
            "llm_critic_decisions": sum(
                1
                for decision
                in critic_results
                if decision.get(
                    "verification_method"
                )
                == "llm"
            ),
            "empty_reasoning": (
                len(
                    structured_facts
                )
                == 0
            ),
            "rejected_claim_leaked": (
                rejected_claim_leaked
            ),

            # Latency
            "retrieval_latency_ms": round(
                float(
                    result_state.get(
                        "retrieval_latency_ms",
                        0.0,
                    )
                    or 0.0
                ),
                2,
            ),
            "reasoning_latency_ms": round(
                float(
                    latency.get(
                        "reasoning_ms",
                        0.0,
                    )
                    or 0.0
                ),
                2,
            ),
            "critic_latency_ms": round(
                float(
                    latency.get(
                        "critic_ms",
                        0.0,
                    )
                    or 0.0
                ),
                2,
            ),
            "synthesis_latency_ms": round(
                float(
                    latency.get(
                        "synthesis_ms",
                        0.0,
                    )
                    or 0.0
                ),
                2,
            ),
            "total_latency_ms": round(
                total_latency_ms,
                2,
            ),

            "error": error,
        }

    # ============================================================
    # Retrieval metrics
    # ============================================================

    @staticmethod
    def _chunk_is_relevant(
        chunk: dict[str, Any],
        expected_document: str | None,
        expected_pages: set[int],
    ) -> bool:
        document_name = str(
            chunk.get(
                "document_name"
            )
            or ""
        )

        page_number = chunk.get(
            "page_number"
        )

        document_matches = (
            expected_document is None
            or document_name
            == expected_document
        )

        page_matches = (
            not expected_pages
            or page_number
            in expected_pages
        )

        return (
            document_matches
            and page_matches
        )

    def _calculate_retrieval_hit(
        self,
        answerable: bool,
        expected_document: str | None,
        expected_pages: set[int],
        retrieved_chunks: list[dict[str, Any]],
    ) -> bool:
        if not answerable:
            return False

        return any(
            self._chunk_is_relevant(
                chunk=chunk,
                expected_document=(
                    expected_document
                ),
                expected_pages=(
                    expected_pages
                ),
            )
            for chunk in retrieved_chunks
        )

    def _calculate_reciprocal_rank(
        self,
        answerable: bool,
        expected_document: str | None,
        expected_pages: set[int],
        retrieved_chunks: list[dict[str, Any]],
    ) -> float:
        if not answerable:
            return 0.0

        for rank, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            if self._chunk_is_relevant(
                chunk=chunk,
                expected_document=(
                    expected_document
                ),
                expected_pages=(
                    expected_pages
                ),
            ):
                return 1.0 / rank

        return 0.0

    # ============================================================
    # Answer metrics
    # ============================================================

    @staticmethod
    def _tokenise(
        text: str,
    ) -> list[str]:
        return re.findall(
            r"\b\w+\b",
            normalise_text(text),
        )

    def _token_f1(
        self,
        prediction: str,
        reference: str,
    ) -> float:
        prediction_tokens = (
            self._tokenise(
                prediction
            )
        )

        reference_tokens = (
            self._tokenise(
                reference
            )
        )

        if (
            not prediction_tokens
            and not reference_tokens
        ):
            return 1.0

        if (
            not prediction_tokens
            or not reference_tokens
        ):
            return 0.0

        prediction_counts: dict[
            str,
            int,
        ] = {}

        reference_counts: dict[
            str,
            int,
        ] = {}

        for token in prediction_tokens:
            prediction_counts[token] = (
                prediction_counts.get(
                    token,
                    0,
                )
                + 1
            )

        for token in reference_tokens:
            reference_counts[token] = (
                reference_counts.get(
                    token,
                    0,
                )
                + 1
            )

        overlap = sum(
            min(
                count,
                reference_counts.get(
                    token,
                    0,
                ),
            )
            for token, count
            in prediction_counts.items()
        )

        if overlap == 0:
            return 0.0

        precision = (
            overlap
            / len(prediction_tokens)
        )

        recall = (
            overlap
            / len(reference_tokens)
        )

        return (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    @staticmethod
    def _rejected_claim_leaked(
        final_answer: str,
        critic_results: list[
            dict[str, Any]
        ],
    ) -> bool:
        normalised_answer = (
            normalise_text(
                final_answer
            )
        )

        if not normalised_answer:
            return False

        for decision in critic_results:
            rejected = (
                decision.get(
                    "supported"
                )
                is False
                or decision.get(
                    "relevant_to_question"
                )
                is False
            )

            if not rejected:
                continue

            claim = normalise_text(
                str(
                    decision.get(
                        "claim"
                    )
                    or ""
                )
            )

            if (
                claim
                and claim
                in normalised_answer
            ):
                return True

        return False

    # ============================================================
    # Summary
    # ============================================================

    def _calculate_summary(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        answerable_results = [
            result
            for result in results
            if result["answerable"]
        ]

        unanswerable_results = [
            result
            for result in results
            if not result["answerable"]
        ]

        successful_results = [
            result
            for result in results
            if result["error"] is None
        ]

        total_claims = sum(
            result["total_claims"]
            for result in results
        )

        approved_claims = sum(
            result["approved_claims"]
            for result in results
        )

        rejected_claims = sum(
            result["rejected_claims"]
            for result in results
        )

        return {
            "successful_questions": len(
                successful_results
            ),
            "failed_questions": (
                len(results)
                - len(
                    successful_results
                )
            ),

            # Retrieval metrics:
            # calculated only over answerable questions.
            "retrieval_hit_rate": (
                self._safe_mean(
                    [
                        float(
                            result[
                                "retrieval_hit"
                            ]
                        )
                        for result
                        in answerable_results
                    ]
                )
            ),
            "mrr": self._safe_mean(
                [
                    result[
                        "reciprocal_rank"
                    ]
                    for result
                    in answerable_results
                ]
            ),

            # Answer metrics
            "answerable_response_rate": (
                self._safe_mean(
                    [
                        float(
                            result[
                                "answerable_response_returned"
                            ]
                        )
                        for result
                        in answerable_results
                    ]
                )
            ),
            "unanswerable_accuracy": (
                self._safe_mean(
                    [
                        float(
                            result[
                                "unanswerable_correct"
                            ]
                        )
                        for result
                        in unanswerable_results
                    ]
                )
            ),
            "answerable_exact_match_rate": (
                self._safe_mean(
                    [
                        float(
                            result[
                                "exact_match"
                            ]
                        )
                        for result
                        in answerable_results
                    ]
                )
            ),
            "mean_answerable_token_f1": (
                self._safe_mean(
                    [
                        result[
                            "token_f1"
                        ]
                        for result
                        in answerable_results
                    ]
                )
            ),

            # COA metrics
            "total_claims": total_claims,
            "approved_claims": (
                approved_claims
            ),
            "rejected_claims": (
                rejected_claims
            ),
            "claim_approval_rate": (
                round(
                    approved_claims
                    / total_claims,
                    4,
                )
                if total_claims
                else 0.0
            ),
            "direct_match_decisions": sum(
                result[
                    "direct_match_decisions"
                ]
                for result in results
            ),
            "llm_critic_decisions": sum(
                result[
                    "llm_critic_decisions"
                ]
                for result in results
            ),
            "empty_reasoning_count": sum(
                int(
                    result[
                        "empty_reasoning"
                    ]
                )
                for result in results
            ),
            "fallback_count": sum(
                int(
                    result[
                        "fallback_returned"
                    ]
                )
                for result in results
            ),
            "rejected_claim_leakage_count": sum(
                int(
                    result[
                        "rejected_claim_leaked"
                    ]
                )
                for result in results
            ),
            "rejected_claim_leakage_rate": (
                self._safe_mean(
                    [
                        float(
                            result[
                                "rejected_claim_leaked"
                            ]
                        )
                        for result in results
                    ]
                )
            ),

            # Latency
            "average_retrieval_latency_ms": (
                self._safe_mean(
                    [
                        result[
                            "retrieval_latency_ms"
                        ]
                        for result in results
                    ]
                )
            ),
            "average_reasoning_latency_ms": (
                self._safe_mean(
                    [
                        result[
                            "reasoning_latency_ms"
                        ]
                        for result in results
                    ]
                )
            ),
            "average_critic_latency_ms": (
                self._safe_mean(
                    [
                        result[
                            "critic_latency_ms"
                        ]
                        for result in results
                    ]
                )
            ),
            "average_synthesis_latency_ms": (
                self._safe_mean(
                    [
                        result[
                            "synthesis_latency_ms"
                        ]
                        for result in results
                    ]
                )
            ),
            "average_total_latency_ms": (
                self._safe_mean(
                    [
                        result[
                            "total_latency_ms"
                        ]
                        for result in results
                    ]
                )
            ),
            "p50_total_latency_ms": (
                self._percentile(
                    [
                        result[
                            "total_latency_ms"
                        ]
                        for result in results
                    ],
                    50,
                )
            ),
            "p95_total_latency_ms": (
                self._percentile(
                    [
                        result[
                            "total_latency_ms"
                        ]
                        for result in results
                    ],
                    95,
                )
            ),
        }

    @staticmethod
    def _safe_mean(
        values: list[float],
    ) -> float:
        if not values:
            return 0.0

        return round(
            mean(values),
            4,
        )

    @staticmethod
    def _percentile(
        values: list[float],
        percentile: int,
    ) -> float:
        if not values:
            return 0.0

        sorted_values = sorted(
            values
        )

        if len(sorted_values) == 1:
            return round(
                sorted_values[0],
                2,
            )

        position = (
            percentile
            / 100
        ) * (
            len(sorted_values)
            - 1
        )

        lower_index = int(
            position
        )

        upper_index = min(
            lower_index + 1,
            len(sorted_values) - 1,
        )

        fraction = (
            position
            - lower_index
        )

        value = (
            sorted_values[
                lower_index
            ]
            + (
                sorted_values[
                    upper_index
                ]
                - sorted_values[
                    lower_index
                ]
            )
            * fraction
        )

        return round(
            value,
            2,
        )

    # ============================================================
    # Report saving
    # ============================================================

    def _save_report(
        self,
        report: dict[str, Any],
    ) -> str:
        project_root = (
            Path(__file__)
            .resolve()
            .parents[3]
        )

        output_directory = (
            project_root
            / "data"
            / "evaluation"
            / "results"
            / "coa"
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        output_path = (
            output_directory
            / (
                "coa_evaluation_"
                f"{timestamp}.json"
            )
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return str(
            output_path
        )