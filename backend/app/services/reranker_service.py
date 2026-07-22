from functools import lru_cache
from typing import Any

from sentence_transformers import CrossEncoder


RERANKER_MODEL = (
    "cross-encoder/ms-marco-TinyBERT-L-2-v2"
)


@lru_cache(maxsize=1)
def get_reranker():

    print(
        f"Loading reranker: {RERANKER_MODEL}"
    )

    return CrossEncoder(
        RERANKER_MODEL
    )


def rerank_chunks(
    question: str,
    chunks: list[dict[str, Any]],
    top_k: int = 5,
):

    if not chunks:
        return []


    model = get_reranker()


    pairs = []

    for chunk in chunks:

        text = str(
            chunk.get("text")
            or ""
        )

        pairs.append(
            [
                question,
                text
            ]
        )


    scores = model.predict(
        pairs,
        show_progress_bar=False
    )


    results=[]


    for chunk, score in zip(
        chunks,
        scores
    ):

        item = dict(chunk)

        item["retrieval_score"] = (
            item.get("score")
        )

        item["reranker_score"] = (
            float(score)
        )

        results.append(item)


    results.sort(
        key=lambda x:
        x["reranker_score"],
        reverse=True
    )


    return results[:top_k]