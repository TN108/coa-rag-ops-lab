from time import perf_counter
from app.agents.state import COAState

def critic_agent(state: COAState) -> COAState:
    print("Running Critic Agent")

    start_time = perf_counter()

    structured_facts = state.get("structured_facts", [])
    retrieved_chunks = state.get("retrieved_chunks", [])

    # Map chunk_id to text for quick lookup
    chunk_map = {c["chunk_id"]: c["text"] for c in retrieved_chunks}

    feedback_list = []

    for fact in structured_facts:
        claim = fact.get("claim", "")
        evidence_ids = fact.get("evidence_chunk_ids", [])
        supported = all(eid in chunk_map for eid in evidence_ids)

        if supported:
            feedback_list.append(f"Claim supported by evidence: '{claim[:80]}...'")
        else:
            feedback_list.append(f"Claim may be unsupported: '{claim[:80]}...'")

    # Aggregate feedback
    state["critic_feedback"] = "\n".join(feedback_list)

    # Record latency
    if "latency" not in state:
        state["latency"] = {}
    state["latency"]["critic_ms"] = round((perf_counter() - start_time) * 1000, 2)

    return state