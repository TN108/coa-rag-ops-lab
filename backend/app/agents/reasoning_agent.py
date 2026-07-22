from .state import COAState


def reasoning_agent(state: COAState):

    print("Running Reasoning Agent")


    chunks = state["retrieved_chunks"]


    state["reasoning"] = (
        f"Based on {len(chunks)} chunks, "
        "the answer can be generated."
    )


    return state