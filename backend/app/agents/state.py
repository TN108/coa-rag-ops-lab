# backend/app/agents/state.py

from typing import Any, Literal, TypedDict


class ReasoningFact(TypedDict):
    """
    One evidence-grounded claim produced by the reasoning agent.
    """

    claim: str
    evidence_chunk_ids: list[str]


class CriticResult(TypedDict):
    """
    Verification result for one reasoning claim.
    """

    claim_index: int
    claim: str
    supported: bool
    relevant_to_question: bool
    verified_chunk_ids: list[str]
    feedback: str

    # Explains how the claim was checked.
    verification_method: Literal[
        "direct_match",
        "llm",
        "validation",
        "error",
    ]


class AgentLatency(TypedDict, total=False):
    """
    Latency measurements for individual pipeline stages.
    """

    reasoning_ms: float
    critic_ms: float
    synthesis_ms: float


class COAState(TypedDict, total=False):
    """
    Shared state passed between the COA graph nodes.

    total=False is required because each agent gradually adds
    its own output fields.
    """

    # Initial input
    question: str

    # Retrieval agent output
    retrieved_chunks: list[dict[str, Any]]
    retrieval_confidence: float
    retrieval_latency_ms: float

    # Reasoning agent output
    reasoning: str
    structured_facts: list[ReasoningFact]

    # Critic agent output
    critic_results: list[CriticResult]
    critic_feedback: str

    # Synthesis agent output
    final_answer: str

    # Shared metadata
    latency: AgentLatency
    error: str | None