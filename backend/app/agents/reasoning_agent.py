# backend/app/agents/reasoning_agent.py

import json
import time

from app.agents.state import COAState
from app.services.llm_service import generate_coa_facts


def reasoning_agent(
    state: COAState,
) -> COAState:
    """
    Extract evidence-grounded claims from retrieved chunks.

    The reasoning agent:
    1. Reads the question and retrieved chunks.
    2. Calls the structured COA fact generator.
    3. Stores validated claims in structured_facts.
    4. Stores a JSON representation in reasoning for debugging.
    5. Records reasoning latency.
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

    start = time.perf_counter()

    try:
        reasoning_output = generate_coa_facts(
            question=question,
            retrieved_chunks=chunks,
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

    # Valid JSON string retained only for logs and API debugging.
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