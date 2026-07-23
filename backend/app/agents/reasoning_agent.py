# backend/app/agents/reasoning_agent.py
from app.services.llm_service import generate_rag_answer
from app.agents.state import COAState

def reasoning_agent(state: COAState) -> COAState:
    print("Running Reasoning Agent")

    chunks = state["retrieved_chunks"]

    # Generate reasoning from the LLM using the existing RAG prompt
    answer = generate_rag_answer(state["question"], chunks)

    # Store reasoning text
    state["reasoning"] = answer

    # Optional structured facts placeholder
    state["structured_facts"] = [
        {
            "claim": answer,
            "evidence_chunk_ids": [c["chunk_id"] for c in chunks]
        }
    ]

    return state