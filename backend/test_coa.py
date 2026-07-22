from app.agents.coordinator_agent import create_coa_graph


graph = create_coa_graph()


result = graph.invoke(
    {
        "question": "What is LangGraph?",
        "retrieved_chunks": [],
        "reasoning": "",
        "critic_feedback": "",
        "final_answer": ""
    }
)


print(result)