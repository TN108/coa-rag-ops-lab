from app.agents.coordinator_agent import create_coa_graph


graph = create_coa_graph()


for i in range(3):

    print(f"\nRUN {i+1}")

    result = graph.invoke(
        {
            "question": "What is LangGraph?",
            "retrieved_chunks": [],
            "retrieval_confidence": 0.0,
            "retrieval_latency_ms": 0.0,
            "reasoning": "",
            "critic_feedback": "",
            "final_answer": "",
        }
    )