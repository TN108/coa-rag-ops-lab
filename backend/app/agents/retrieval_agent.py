from .state import COAState


def retrieval_agent(state: COAState):

    print("Running Retrieval Agent")

    state["retrieved_chunks"] = [
        {
            "text": "Example retrieved document chunk",
            "score": 0.9
        }
    ]

    return state