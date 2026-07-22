from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load embedding model only once.
    """

    print("LOADING EMBEDDING MODEL")

    return SentenceTransformer(
        settings.EMBEDDING_MODEL_NAME
    )


def generate_embedding(
    text: str,
) -> List[float]:

    if not text or not text.strip():
        return []

    model = get_embedding_model()

    embedding = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embedding.tolist()



def generate_embeddings(
    texts: List[str],
) -> List[List[float]]:

    if not texts:
        return []

    cleaned_texts = [
        text.strip()
        for text in texts
        if text and text.strip()
    ]

    if not cleaned_texts:
        return []

    model = get_embedding_model()

    embeddings = model.encode(
        cleaned_texts,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()



def get_embedding_dimension() -> int:
    """
    Return embedding vector size.
    Used when creating Qdrant collections.
    """

    model = get_embedding_model()

    return model.get_sentence_embedding_dimension()