from typing import List

from sentence_transformers import SentenceTransformer

from app.config import settings


_model = None


def get_embedding_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

    return _model


def generate_embedding(text: str) -> List[float]:
    if not text or not text.strip():
        return []

    model = get_embedding_model()

    embedding = model.encode(
        text,
        normalize_embeddings=True
    )

    return embedding.tolist()


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    cleaned_texts = [text for text in texts if text and text.strip()]

    if not cleaned_texts:
        return []

    model = get_embedding_model()

    embeddings = model.encode(
        cleaned_texts,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.tolist()


def get_embedding_dimension() -> int:
    model = get_embedding_model()
    return model.get_sentence_embedding_dimension()