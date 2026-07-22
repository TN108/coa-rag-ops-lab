from app.services.planner_service import planner_service


def test_factoid_question():
    result = planner_service.plan(
        "What is LangGraph?"
    )

    decision = result["planner_decision"]

    assert decision["query_type"] == "factoid"
    assert decision["top_k"] == 3
    assert decision["min_retrieval_score"] == 0.2
    assert decision["chunking_method"] == "semantic"
    assert decision["neighbor_window"] == 0


def test_explanation_question():
    result = planner_service.plan(
        "Explain how LangGraph manages state."
    )

    decision = result["planner_decision"]

    assert decision["query_type"] == "explanation"
    assert decision["top_k"] == 5
    assert decision["min_retrieval_score"] == 0.1
    assert decision["neighbor_window"] == 1


def test_broad_question():
    result = planner_service.plan(
        "Summarize the complete document."
    )

    decision = result["planner_decision"]

    assert decision["query_type"] == "broad"
    assert decision["top_k"] == 8
    assert decision["min_retrieval_score"] == 0.05
    assert decision["neighbor_window"] == 2