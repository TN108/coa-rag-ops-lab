import json
from pathlib import Path
from typing import Any, Callable

from app.services.experiment_service import (
    experiment_service,
)

from app.services.metrics_service import (
    calculate_question_metrics,
)

from app.services.planner_service import (
    planner_service,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class EvaluationService:

    def __init__(
        self,
        rag_answer_function: Callable[
            ..., dict[str, Any]
        ],
    ) -> None:

        self.rag_answer_function = (
            rag_answer_function
        )
    

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


            generated_answer = (
                rag_result.get(
                    "answer",
                    "",
                )
            )


            retrieved_sources = (
                rag_result.get(
                    "sources",
                    [],
                )
            )


            latency = (
                rag_result.get(
                    "latency",
                    {},
                )
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

                "latency": self.extract_latency(
                    latency
                ),

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

                "latency": self.extract_latency(
                    {}
                ),

                "success": False,

                "error": str(exc),
            }


    @staticmethod
    def extract_latency(
        latency: dict[str, Any],
    ) -> dict[str, float]:

        return {

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

            "reranking_ms": float(
                latency.get(
                    "reranking_ms",
                    0.0,
                )
            ),

            "neighbor_retrieval_ms": float(
                latency.get(
                    "neighbor_retrieval_ms",
                    0.0,
                )
            ),

            "llm_generation_ms": float(
                latency.get(
                    "llm_generation_ms",
                    latency.get(
                        "generation_ms",
                        0.0,
                    ),
                )
            ),

            "total_ms": float(
                latency.get(
                    "total_ms",
                    0.0,
                )
            ),
        }


    def calculate_latency_summary(
        self,
        successful_results: list[
            dict[str, Any]
        ],
    ) -> dict[str, float]:


        mapping = {

            "average_embedding_ms":
                "embedding_ms",

            "average_retrieval_ms":
                "retrieval_ms",

            "average_filtering_ms":
                "filtering_ms",

            "average_reranking_ms":
                "reranking_ms",

            "average_neighbor_retrieval_ms":
                "neighbor_retrieval_ms",

            "average_llm_generation_ms":
                "llm_generation_ms",

            "average_total_ms":
                "total_ms",
        }


        summary = {}


        for output_key, latency_key in mapping.items():

            values = [

                result["latency"].get(
                    latency_key,
                    0.0,
                )

                for result in successful_results

            ]


            summary[output_key] = (
                self.average_values(
                    values
                )
            )


        return summary
    def run_evaluation(
        self,
        dataset_path: str,
        top_k: int,
        min_retrieval_score: float,
    ) -> dict[str, Any]:

        experiment_id = (
            experiment_service.generate_experiment_id()
        )

        experiment_configuration = {
            "dataset_path": dataset_path,
            "evaluation_type": "baseline_rag",
            "top_k": top_k,
            "min_retrieval_score": (
                min_retrieval_score
            ),
        }


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


        def average_metric(
            selected_results,
            metric_name,
        ):

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


        summary = {

            "retrieval_hit_rate":
                average_metric(
                    answerable_results,
                    "retrieval_hit",
                ),

            "mean_reciprocal_rank":
                average_metric(
                    answerable_results,
                    "reciprocal_rank",
                ),

            "mean_page_recall":
                average_metric(
                    answerable_results,
                    "page_recall",
                ),

            "mean_exact_match":
                average_metric(
                    successful_results,
                    "exact_match",
                ),

            "mean_token_f1":
                average_metric(
                    successful_results,
                    "token_f1",
                ),

            "mean_semantic_similarity":
                average_metric(
                    successful_results,
                    "semantic_similarity",
                ),

            "answerable_semantic_similarity":
                average_metric(
                    answerable_results,
                    "semantic_similarity",
                ),

            "unanswerable_accuracy":
                average_metric(
                    unanswerable_results,
                    "unanswerable_correct",
                ),

            "latency":
                self.calculate_latency_summary(
                    successful_results
                ),
        }


        evaluation_result = {

            "message":
                "Evaluation completed.",

            "experiment_id":
                experiment_id,

            "experiment_configuration":
                experiment_configuration,

            "total_questions":
                len(results),

            "successful_questions":
                len(successful_results),

            "summary":
                summary,

            "results":
                results,
        }


        saved_path = (
            experiment_service.save_experiment(
                experiment_id=experiment_id,
                configuration=(
                    experiment_configuration
                ),
                results=evaluation_result,
                experiment_type="baseline_rag",
            )
        )


        evaluation_result[
            "saved_result_path"
        ] = str(saved_path)


        return evaluation_result



    def evaluate_adaptive_question(
        self,
        record: dict[str, Any],
        decision: dict[str, Any],
        planner_result: dict[str, Any],
    ) -> dict[str, Any]:

        try:

            rag_result = self.rag_answer_function(
                question=record["question"],

                top_k=decision[
                    "top_k"
                ],

                min_retrieval_score=decision[
                    "min_retrieval_score"
                ],

                chunking_method=decision[
                    "chunking_method"
                ],

                page_window=decision[
                    "neighbor_window"
                ],
            )


            generated_answer = (
                rag_result.get(
                    "answer",
                    "",
                )
            )


            retrieved_sources = (
                rag_result.get(
                    "sources",
                    [],
                )
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

                "planner_decision": decision,

                "planner_ms":
                    planner_result[
                        "planner_ms"
                    ],

                "generated_answer":
                    generated_answer,

                "retrieved_sources":
                    retrieved_sources,

                "answerable":
                    record[
                        "answerable"
                    ],

                "metrics":
                    metrics,

                "latency":
                    self.extract_latency(
                        rag_result.get(
                            "latency",
                            {},
                        )
                    ),
                    

                "success":
                    True,

                "error":
                    None,
                "planner_ms":
                    planner_result["planner_ms"],
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

                "planner_decision":
                    decision,

                "planner_ms":
                    planner_result.get(
                        "planner_ms",
                        0.0,
                    ),

                "generated_answer":
                    "",

                "retrieved_sources":
                    [],

                "answerable":
                    record.get(
                        "answerable",
                        False,
                    ),

                "metrics":
                    {},

                "latency":
                    self.extract_latency(
                        {}
                    ),

                "success":
                    False,

                "error":
                    str(exc),
            }



    def run_adaptive_evaluation(
        self,
        dataset_path: str,
    ) -> dict[str, Any]:

        experiment_id = (
            experiment_service.generate_experiment_id()
        )


        experiment_configuration = {

            "dataset_path":
                dataset_path,

            "evaluation_type":
                "adaptive_rag",

            "planner_model":
                planner_service.model,
        }


        dataset = self.load_dataset(
            dataset_path
        )


        results = []

        planner_latencies = []


        for record in dataset:

            planner_result = (
                planner_service.plan(
                    record["question"]
                )
            )


            decision = (
                planner_result[
                    "planner_decision"
                ]
            )


            planner_latencies.append(
                planner_result[
                    "planner_ms"
                ]
            )


            result = (
                self.evaluate_adaptive_question(
                    record=record,
                    decision=decision,
                    planner_result=planner_result,
                )
            )


            results.append(result)



        successful_results = [
            result
            for result in results
            if result["success"]
        ]


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


        def average_metric(
            selected_results,
            metric_name,
        ):

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


        summary = {

            "retrieval_hit_rate":
                average_metric(
                    answerable_results,
                    "retrieval_hit",
                ),

            "mean_reciprocal_rank":
                average_metric(
                    answerable_results,
                    "reciprocal_rank",
                ),

            "mean_page_recall":
                average_metric(
                    answerable_results,
                    "page_recall",
                ),

            "mean_exact_match":
                average_metric(
                    successful_results,
                    "exact_match",
                ),

            "mean_token_f1":
                average_metric(
                    successful_results,
                    "token_f1",
                ),

            "mean_semantic_similarity":
                average_metric(
                    successful_results,
                    "semantic_similarity",
                ),

            "unanswerable_accuracy":
                average_metric(
                    unanswerable_results,
                    "unanswerable_correct",
                ),

            "planner_latency": {

                "average_planner_ms":
                    self.average_values(
                        planner_latencies
                    )
            },

            "latency":
                self.calculate_latency_summary(
                    successful_results
                ),
        }


        evaluation_result = {

            "message":
                "Adaptive evaluation completed.",

            "experiment_id":
                experiment_id,

            "experiment_configuration":
                experiment_configuration,

            "evaluation_type":
                "adaptive_rag",

            "total_questions":
                len(results),

            "successful_questions":
                len(successful_results),

            "summary":
                summary,

            "results":
                results,
        }


        saved_path = (
            experiment_service.save_experiment(
                experiment_id=experiment_id,
                configuration=(
                    experiment_configuration
                ),
                results=evaluation_result,
                experiment_type="adaptive_rag",
            )
        )


        evaluation_result[
            "saved_result_path"
        ] = str(saved_path)


        return evaluation_result


