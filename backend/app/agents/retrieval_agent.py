from time import perf_counter
from typing import Any

from app.agents.state import COAState

from app.services.embedding_service import (
    generate_embedding,
)

from app.services.qdrant_service import (
    search_similar_chunks,
)

from app.services.reranker_service import (
    rerank_chunks,
)


def retrieval_agent(
    state: COAState,
) -> COAState:

    print("Running Retrieval Agent")


    start_time = perf_counter()


    question = state["question"]


    # -----------------------------
    # Step 1: Generate query embedding
    # -----------------------------

    query_embedding = generate_embedding(
        question
    )


    # -----------------------------
    # Step 2: Retrieve candidates
    # -----------------------------

    candidate_chunks = search_similar_chunks(
        query_embedding=query_embedding,
        top_k=15,
    )


    # -----------------------------
    # Step 3: Rerank candidates
    # -----------------------------

    reranked_chunks = rerank_chunks(
        question=question,
        chunks=candidate_chunks,
    )


    # -----------------------------
    # Step 4: Keep top chunks
    # -----------------------------

    final_chunks = reranked_chunks[:5]


    latency = (
        perf_counter() - start_time
    ) * 1000


    # -----------------------------
    # Update COA state
    # -----------------------------

    state["retrieved_chunks"] = (
        final_chunks
    )


    if final_chunks:

        state["retrieval_confidence"] = float(
            final_chunks[0].get(
                "score",
                0.0,
            )
        )

    else:

        state["retrieval_confidence"] = 0.0


    state["retrieval_latency_ms"] = round(
        latency,
        2,
    )


    return state