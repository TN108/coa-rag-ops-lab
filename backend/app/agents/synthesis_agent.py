from .state import COAState


def synthesis_agent(state: COAState):

    print("Running Synthesis Agent")


    state["final_answer"] = (
        "Final answer generated using reasoning "
        "and critic validation."
    )


    return state