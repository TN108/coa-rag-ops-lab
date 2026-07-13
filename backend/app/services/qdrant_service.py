import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.config import settings


_client = None


def get_qdrant_client() -> QdrantClient:
    global _client

    if _client is None:
        _client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )

    return _client


def ensure_collection_exists():
    client = get_qdrant_client()

    try:
        client.get_collection(settings.QDRANT_COLLECTION_NAME)
        return
    except Exception:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            )
        )


def build_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def upsert_chunks_to_qdrant(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    document_name: str,
    extraction_method: str,
    chunking_method: str
):
    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks and embeddings must match.")

    ensure_collection_exists()
    client = get_qdrant_client()

    points = []

    for chunk, embedding in zip(chunks, embeddings):
        point_id = build_point_id(chunk["chunk_id"])

        payload = {
            "chunk_id": chunk["chunk_id"],
            "document_name": document_name,
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "global_chunk_index": chunk["global_chunk_index"],
            "character_count": chunk["character_count"],
            "chunking_method": chunk["chunking_method"],
            "selected_chunking_method": chunking_method,
            "extraction_method": extraction_method,
            "text": chunk["text"]
        }

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
        )

    client.upsert(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points=points
    )

    collection_count = client.count(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        exact=True
    )

    return {
        "collection_name": settings.QDRANT_COLLECTION_NAME,
        "stored_points": len(points),
        "total_points_in_collection": collection_count.count
    }


def get_collection_info():
    ensure_collection_exists()
    client = get_qdrant_client()

    collection_info = client.get_collection(settings.QDRANT_COLLECTION_NAME)
    collection_count = client.count(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        exact=True
    )

    return {
        "collection_name": settings.QDRANT_COLLECTION_NAME,
        "vector_size": settings.EMBEDDING_DIMENSION,
        "distance": "cosine",
        "total_points": collection_count.count,
        "status": str(collection_info.status)
    }