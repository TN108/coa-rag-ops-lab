from typing import TypedDict, List, Dict, Any


class ReasoningFact(TypedDict):
    claim: str
    evidence_chunk_ids: List[str]

class COAState(TypedDict):

    question: str

    retrieved_chunks: List[Dict[str, Any]]

    retrieval_confidence: float

    retrieval_latency_ms: float

    reasoning: str

    structured_facts: List[ReasoningFact]  # NEW

    critic_feedback: str

    final_answer: str