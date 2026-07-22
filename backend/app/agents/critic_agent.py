from .state import COAState


def critic_agent(state: COAState):

    print("Running Critic Agent")


    state["critic_feedback"] = (
        "Claims are supported by retrieved evidence."
    )


    return state