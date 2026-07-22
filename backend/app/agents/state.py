from typing import TypedDict, List, Dict, Any


class COAState(TypedDict):

    question: str

    retrieved_chunks: List[Dict[str, Any]]

    reasoning: str

    critic_feedback: str

    final_answer: str