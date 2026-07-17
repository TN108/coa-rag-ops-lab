from functools import lru_cache
from typing import Any

from sentence_transformers import CrossEncoder


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """
    Load the CrossEncoder only once.
    """
    return CrossEncoder(RERANKER_MODEL)


def rerank_chunks(
    question: str,
    chunks: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Reorder retrieved chunks using a CrossEncoder.
    """

    if not chunks:
        return []

    pairs = []

    for chunk in chunks:
        text = str(
            chunk.get("text")
            or chunk.get("content")
            or chunk.get("page_content")
            or ""
        ).strip()

        pairs.append([question, text])

    model = get_reranker()

    scores = model.predict(
        pairs,
        show_progress_bar=False,
    )

    reranked_chunks = []

    for chunk, reranker_score in zip(chunks, scores):
        updated_chunk = dict(chunk)

        updated_chunk["retrieval_score"] = updated_chunk.get("score")
        updated_chunk["reranker_score"] = float(reranker_score)

        reranked_chunks.append(updated_chunk)

    reranked_chunks.sort(
        key=lambda chunk: chunk["reranker_score"],
        reverse=True,
    )

    return reranked_chunks[:top_k]