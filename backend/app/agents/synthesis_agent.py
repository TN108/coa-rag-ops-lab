from time import perf_counter
from app.agents.state import COAState

def synthesis_agent(state: COAState) -> COAState:
    print("Running Synthesis Agent")

    start_time = perf_counter()

    reasoning = state.get("reasoning", "")
    critic_feedback = state.get("critic_feedback", "")

    # Combine reasoning + critic into final answer
    final_answer = reasoning
    if critic_feedback:
        final_answer += "\n\nCritic notes:\n" + critic_feedback

    state["final_answer"] = final_answer

    # Record latency
    if "latency" not in state:
        state["latency"] = {}
    state["latency"]["synthesis_ms"] = round((perf_counter() - start_time) * 1000, 2)

    return state