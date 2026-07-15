import math
import re
import string
from collections import Counter
from typing import Any

from app.services.embedding_service import generate_embedding


REFUSAL_PHRASES = [
    "does not contain enough information",
    "does not contain this information",
    "not enough information",
    "cannot determine",
    "cannot be determined",
    "not specified",
    "does not specify",
    "not mentioned",
    "no information",
    "does not contain information relevant to the question",
    "retrieved context does not contain information",
    "context does not contain information",
    "no relevant information",
    "unsupported premise",
    "not available in the context",
]

def normalize_text(text: str) -> str:
    text = text.lower().strip()

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation,
        )
    )

    text = re.sub(r"\s+", " ", text)

    return text


def exact_match_score(
    expected_answer: str,
    generated_answer: str,
) -> float:
    expected = normalize_text(expected_answer)
    generated = normalize_text(generated_answer)

    return float(expected == generated)


def token_f1_score(
    expected_answer: str,
    generated_answer: str,
) -> float:
    expected_tokens = normalize_text(
        expected_answer
    ).split()

    generated_tokens = normalize_text(
        generated_answer
    ).split()

    if not expected_tokens and not generated_tokens:
        return 1.0

    if not expected_tokens or not generated_tokens:
        return 0.0

    expected_counter = Counter(expected_tokens)
    generated_counter = Counter(generated_tokens)

    common_tokens = (
        expected_counter & generated_counter
    )

    overlap_count = sum(common_tokens.values())

    if overlap_count == 0:
        return 0.0

    precision = (
        overlap_count / len(generated_tokens)
    )

    recall = (
        overlap_count / len(expected_tokens)
    )

    return (
        2 * precision * recall
        / (precision + recall)
    )


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    if not vector_a or not vector_b:
        return 0.0

    if len(vector_a) != len(vector_b):
        raise ValueError(
            "Embedding vectors must have the same dimension."
        )

    dot_product = sum(
        value_a * value_b
        for value_a, value_b in zip(
            vector_a,
            vector_b,
        )
    )

    magnitude_a = math.sqrt(
        sum(value * value for value in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(value * value for value in vector_b)
    )

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    similarity = (
        dot_product
        / (magnitude_a * magnitude_b)
    )

    return max(
        0.0,
        min(1.0, similarity),
    )


def semantic_similarity_score(
    expected_answer: str,
    generated_answer: str,
) -> float:
    if not expected_answer.strip():
        return 0.0

    if not generated_answer.strip():
        return 0.0

    expected_embedding = generate_embedding(
        expected_answer
    )

    generated_embedding = generate_embedding(
        generated_answer
    )

    return cosine_similarity(
        expected_embedding,
        generated_embedding,
    )


def extract_retrieved_pages(
    retrieved_sources: list[dict[str, Any]],
) -> list[int]:
    pages: list[int] = []

    for source in retrieved_sources:
        page_number = source.get("page_number")

        if isinstance(page_number, int):
            pages.append(page_number)

    return pages


def retrieval_hit_score(
    expected_pages: list[int],
    retrieved_sources: list[dict[str, Any]],
    answerable: bool,
) -> float:
    if not answerable:
        return 1.0

    if not expected_pages:
        return 0.0

    retrieved_pages = extract_retrieved_pages(
        retrieved_sources
    )

    return float(
        bool(
            set(expected_pages)
            & set(retrieved_pages)
        )
    )


def reciprocal_rank_score(
    expected_pages: list[int],
    retrieved_sources: list[dict[str, Any]],
    answerable: bool,
) -> float:
    if not answerable:
        return 1.0

    expected_page_set = set(expected_pages)

    for rank, source in enumerate(
        retrieved_sources,
        start=1,
    ):
        page_number = source.get("page_number")

        if page_number in expected_page_set:
            return 1.0 / rank

    return 0.0


def page_recall_score(
    expected_pages: list[int],
    retrieved_sources: list[dict[str, Any]],
    answerable: bool,
) -> float:
    if not answerable:
        return 1.0

    if not expected_pages:
        return 0.0

    retrieved_pages = set(
        extract_retrieved_pages(
            retrieved_sources
        )
    )

    expected_page_set = set(expected_pages)

    matched_pages = (
        expected_page_set & retrieved_pages
    )

    return (
        len(matched_pages)
        / len(expected_page_set)
    )


def is_refusal_answer(
    generated_answer: str,
) -> bool:
    normalized_answer = normalize_text(
        generated_answer
    )

    return any(
        phrase in normalized_answer
        for phrase in REFUSAL_PHRASES
    )


def unanswerable_correctness_score(
    answerable: bool,
    generated_answer: str,
) -> float:
    if answerable:
        return 1.0

    return float(
        is_refusal_answer(
            generated_answer
        )
    )


def calculate_question_metrics(
    expected_answer: str,
    generated_answer: str,
    expected_pages: list[int],
    retrieved_sources: list[dict[str, Any]],
    answerable: bool,
) -> dict[str, float]:
    return {
        "retrieval_hit": retrieval_hit_score(
            expected_pages=expected_pages,
            retrieved_sources=retrieved_sources,
            answerable=answerable,
        ),
        "reciprocal_rank": reciprocal_rank_score(
            expected_pages=expected_pages,
            retrieved_sources=retrieved_sources,
            answerable=answerable,
        ),
        "page_recall": page_recall_score(
            expected_pages=expected_pages,
            retrieved_sources=retrieved_sources,
            answerable=answerable,
        ),
        "exact_match": exact_match_score(
            expected_answer=expected_answer,
            generated_answer=generated_answer,
        ),
        "token_f1": token_f1_score(
            expected_answer=expected_answer,
            generated_answer=generated_answer,
        ),
        "semantic_similarity": (
            semantic_similarity_score(
                expected_answer=expected_answer,
                generated_answer=generated_answer,
            )
        ),
        "unanswerable_correct": (
            unanswerable_correctness_score(
                answerable=answerable,
                generated_answer=generated_answer,
            )
        ),
    }