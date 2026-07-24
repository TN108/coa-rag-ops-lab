# backend/app/api/v1/coa_router.py
from fastapi import APIRouter
from app.agents.coordinator_agent import create_coa_graph

router = APIRouter()

@router.post("/coa")
def run_coa(question: str):
    """
    Run the Chain-of-Agents pipeline for a given question.
    """
    graph = create_coa_graph()
    result = graph.invoke({
        "question": question,
        "retrieved_chunks": [],
        "retrieval_confidence": 0.0,
        "retrieval_latency_ms": 0.0,
        "reasoning": "",
        "structured_facts": [],
        "critic_feedback": "",
        "final_answer": "",
        "latency": {}
    })
    return result