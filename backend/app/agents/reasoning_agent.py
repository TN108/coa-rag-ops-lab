# backend/app/agents/reasoning_agent.py

import json
import time

from app.agents.state import COAState
from app.services.llm_service import generate_coa_facts


def get_reasoning_chunk_limit(
    question: str,
) -> int:
    """
    Select the number of retrieved chunks available to reasoning.

    Simple definition and authorship questions use three chunks.
    More complex questions use five chunks.
    """

    normalized_question = question.strip().lower()

    simple_definition_prefixes = (
        "what is a ",
        "what is an ",
        "what is langgraph",
        "who authored",
    )

    if normalized_question.startswith(
        simple_definition_prefixes
    ):
        return 3

    return 5


def reasoning_agent(
    state: COAState,
) -> COAState:
    """
    Extract evidence-grounded claims from retrieved chunks.

    The reasoning agent:
    1. Reads the question and retrieved chunks.
    2. Selects an appropriate reasoning context limit.
    3. Calls the structured COA fact generator.
    4. Stores validated claims in structured_facts.
    5. Stores a JSON representation in reasoning for debugging.
    6. Records reasoning latency.
    """

    print("Running Reasoning Agent")

    question = str(
        state.get("question") or ""
    ).strip()

    chunks = (
        state.get("retrieved_chunks") or []
    )

    state.setdefault("latency", {})
    state.setdefault("error", None)

    if not question:
        state["reasoning"] = json.dumps(
            {"claims": []},
            ensure_ascii=False,
        )
        state["structured_facts"] = []
        state["latency"]["reasoning_ms"] = 0.0
        state["error"] = (
            "Reasoning agent received an empty question."
        )
        return state

    if not chunks:
        state["reasoning"] = json.dumps(
            {"claims": []},
            ensure_ascii=False,
        )
        state["structured_facts"] = []
        state["latency"]["reasoning_ms"] = 0.0
        return state

    reasoning_chunk_limit = (
        get_reasoning_chunk_limit(question)
    )

    print(
        "Reasoning chunk limit:",
        reasoning_chunk_limit,
    )

    start = time.perf_counter()

    try:
        reasoning_output = generate_coa_facts(
            question=question,
            retrieved_chunks=chunks,
            max_chunks=reasoning_chunk_limit,
        )

    except Exception as error:
        reasoning_ms = (
            time.perf_counter() - start
        ) * 1000

        state["reasoning"] = json.dumps(
            {"claims": []},
            ensure_ascii=False,
        )
        state["structured_facts"] = []
        state["latency"]["reasoning_ms"] = round(
            reasoning_ms,
            2,
        )
        state["error"] = (
            "Reasoning agent failed to generate "
            f"structured facts: {error}"
        )
        return state

    reasoning_ms = (
        time.perf_counter() - start
    ) * 1000

    structured_facts = [
        claim.model_dump()
        for claim in reasoning_output.claims
    ]

    # Machine-readable facts consumed by the critic.
    state["structured_facts"] = structured_facts

    # Valid JSON string retained for logs and API debugging.
    state["reasoning"] = json.dumps(
        {
            "claims": structured_facts,
        },
        ensure_ascii=False,
    )

    state["latency"]["reasoning_ms"] = round(
        reasoning_ms,
        2,
    )

    # Clear a previous reasoning error after a successful run.
    state["error"] = None

    return state