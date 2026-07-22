from langgraph.graph import StateGraph, END

from .state import COAState

from .retrieval_agent import retrieval_agent
from .reasoning_agent import reasoning_agent
from .critic_agent import critic_agent
from .synthesis_agent import synthesis_agent



def create_coa_graph():

    graph = StateGraph(COAState)


    graph.add_node(
        "retrieval",
        retrieval_agent
    )

    graph.add_node(
        "reasoning",
        reasoning_agent
    )

    graph.add_node(
        "critic",
        critic_agent
    )

    graph.add_node(
        "synthesis",
        synthesis_agent
    )


    graph.set_entry_point(
        "retrieval"
    )


    graph.add_edge(
        "retrieval",
        "reasoning"
    )


    graph.add_edge(
        "reasoning",
        "critic"
    )


    graph.add_edge(
        "critic",
        "synthesis"
    )


    graph.add_edge(
        "synthesis",
        END
    )


    return graph.compile()