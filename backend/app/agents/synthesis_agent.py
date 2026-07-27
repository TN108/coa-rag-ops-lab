# backend/app/agents/synthesis_agent.py

import re

from app.agents.state import COAState
from app.services.llm_service import (
    FALLBACK_ANSWER,
    is_definition_question,
    is_yes_no_question,
)


def capitalise_sentence(
    text: str,
) -> str:
    """
    Capitalise the first character of a claim.
    """

    cleaned = str(text or "").strip()

    if not cleaned:
        return ""

    return cleaned[0].upper() + cleaned[1:]


def normalise_words(
    text: str,
) -> set[str]:
    """
    Return meaningful words for claim-overlap comparison.
    """

    words = re.findall(
        r"\b[a-z0-9]+\b",
        str(text or "").casefold(),
    )

    ignored_words = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "in",
        "of",
        "to",
        "and",
        "or",
        "that",
        "this",
        "it",
    }

    return {
        word
        for word in words
        if word not in ignored_words
    }


def claims_overlap(
    first_claim: str,
    second_claim: str,
    threshold: float = 0.65,
) -> bool:
    """
    Check whether two claims express substantially the same fact.

    Overlap is measured against the smaller claim so that a more
    detailed restatement can be identified as duplicate information.
    """

    first_words = normalise_words(
        first_claim
    )

    second_words = normalise_words(
        second_claim
    )

    if not first_words or not second_words:
        return False

    overlap_count = len(
        first_words & second_words
    )

    smaller_claim_size = min(
        len(first_words),
        len(second_words),
    )

    overlap_ratio = (
        overlap_count
        / smaller_claim_size
    )

    return overlap_ratio >= threshold


def synthesis_agent(
    state: COAState,
) -> COAState:
    """
    Build the final answer from critic-approved claims only.

    No LLM call is used. This prevents synthesis hallucination
    and keeps synthesis latency effectively zero.
    """

    print("Running Synthesis Agent")

    question = str(
        state.get("question") or ""
    ).strip()

    critic_results = (
        state.get("critic_results") or []
    )

    state.setdefault(
        "latency",
        {},
    )

    approved_claims: list[str] = []

    for result in critic_results:
        supported = (
            result.get("supported") is True
        )

        relevant = (
            result.get(
                "relevant_to_question"
            ) is True
        )

        if not supported or not relevant:
            continue

        claim = capitalise_sentence(
            result.get("claim") or ""
        )

        if not claim:
            continue

        exact_duplicate = (
            claim in approved_claims
        )

        overlapping_duplicate = any(
            claims_overlap(
                existing_claim,
                claim,
            )
            for existing_claim in approved_claims
        )

        if (
            exact_duplicate
            or overlapping_duplicate
        ):
            continue

        approved_claims.append(
            claim
        )

    if not approved_claims:
        state["final_answer"] = (
            FALLBACK_ANSWER
        )

    elif is_yes_no_question(question):
        # Yes/no questions need one direct answer.
        state["final_answer"] = (
            approved_claims[0]
        )

    elif is_definition_question(question):
        # Definition questions use the highest-priority approved claim
        # to avoid repetitive definitions.
        state["final_answer"] = (
            approved_claims[0]
        )

    else:
        state["final_answer"] = " ".join(
            approved_claims
        )

    state["latency"]["synthesis_ms"] = 0.0

    return state