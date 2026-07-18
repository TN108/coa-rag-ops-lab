import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from app.config import settings


_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """
    Create the Qdrant client once and reuse it.
    """

    global _client

    if _client is None:
        _client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )

    return _client


def validate_chunking_method(
    chunking_method: str | None = None,
) -> str:
    """
    Validate and normalize the selected chunking method.
    """

    method = (
        chunking_method
        or getattr(
            settings,
            "CHUNKING_METHOD",
            "semantic",
        )
    ).strip().lower()

    if method not in {"fixed", "semantic"}:
        raise ValueError(
            "chunking_method must be either "
            "'fixed' or 'semantic'."
        )

    return method


def get_collection_name(
    chunking_method: str | None = None,
) -> str:
    """
    Return the Qdrant collection for the selected
    chunking strategy.
    """

    method = validate_chunking_method(
        chunking_method
    )

    if hasattr(settings, "get_collection_name"):
        return settings.get_collection_name(method)

    if method == "fixed":
        return settings.QDRANT_FIXED_COLLECTION

    return settings.QDRANT_SEMANTIC_COLLECTION


def ensure_collection_exists(
    chunking_method: str | None = None,
) -> str:
    """
    Create the selected Qdrant collection when it does
    not already exist.

    Returns the collection name.
    """

    collection_name = get_collection_name(
        chunking_method
    )

    client = get_qdrant_client()

    try:
        client.get_collection(collection_name)
        return collection_name

    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )

    return collection_name


def build_point_id(
    chunk_id: str,
    collection_name: str,
) -> str:
    """
    Build a stable UUID from the collection and chunk ID.

    Including the collection name prevents fixed and
    semantic chunks from sharing the same generated point
    identifier.
    """

    stable_value = (
        f"{collection_name}:{chunk_id}"
    )

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_DNS,
            stable_value,
        )
    )


def _payload_to_chunk(
    payload: dict[str, Any],
    score: float | None = None,
) -> dict[str, Any]:
    """
    Convert a Qdrant payload into the chunk structure used
    by the RAG pipeline.
    """

    text = payload.get("text") or ""

    return {
        "score": score,
        "retrieval_score": score,
        "chunk_id": payload.get("chunk_id"),
        "document_name": payload.get(
            "document_name"
        ),
        "page_number": payload.get(
            "page_number"
        ),
        "chunk_index": payload.get(
            "chunk_index"
        ),
        "global_chunk_index": payload.get(
            "global_chunk_index"
        ),
        "character_count": payload.get(
            "character_count",
            len(text),
        ),
        "chunking_method": payload.get(
            "chunking_method"
        ),
        "selected_chunking_method": payload.get(
            "selected_chunking_method"
        ),
        "extraction_method": payload.get(
            "extraction_method"
        ),
        "text": text,
    }


def upsert_chunks_to_qdrant(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    document_name: str,
    extraction_method: str,
    chunking_method: str,
) -> dict[str, Any]:
    """
    Store embedded chunks in the collection associated
    with the selected chunking method.
    """

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks and embeddings must match."
        )

    method = validate_chunking_method(
        chunking_method
    )

    collection_name = ensure_collection_exists(
        method
    )

    client = get_qdrant_client()

    points: list[PointStruct] = []

    for chunk, embedding in zip(
        chunks,
        embeddings,
    ):
        chunk_id = str(chunk["chunk_id"])

        point_id = build_point_id(
            chunk_id=chunk_id,
            collection_name=collection_name,
        )

        payload = {
            "chunk_id": chunk_id,
            "document_name": document_name,
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "global_chunk_index": chunk[
                "global_chunk_index"
            ],
            "character_count": chunk[
                "character_count"
            ],
            "chunking_method": chunk[
                "chunking_method"
            ],
            "selected_chunking_method": method,
            "extraction_method": extraction_method,
            "text": chunk["text"],
        }

        if "start_char" in chunk:
            payload["start_char"] = chunk[
                "start_char"
            ]

        if "end_char" in chunk:
            payload["end_char"] = chunk[
                "end_char"
            ]

        if "block_count" in chunk:
            payload["block_count"] = chunk[
                "block_count"
            ]

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
        )

    if points:
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

    collection_count = client.count(
        collection_name=collection_name,
        exact=True,
    )

    return {
        "collection_name": collection_name,
        "chunking_method": method,
        "stored_points": len(points),
        "total_points_in_collection": (
            collection_count.count
        ),
    }


def get_collection_info(
    chunking_method: str | None = None,
) -> dict[str, Any]:
    """
    Return basic information about the selected Qdrant
    collection.
    """

    method = validate_chunking_method(
        chunking_method
    )

    collection_name = ensure_collection_exists(
        method
    )

    client = get_qdrant_client()

    collection_info = client.get_collection(
        collection_name
    )

    collection_count = client.count(
        collection_name=collection_name,
        exact=True,
    )

    return {
        "collection_name": collection_name,
        "chunking_method": method,
        "vector_size": (
            settings.EMBEDDING_DIMENSION
        ),
        "distance": "cosine",
        "total_points": collection_count.count,
        "status": str(collection_info.status),
    }


def get_all_collection_info() -> dict[str, Any]:
    """
    Return information for both experiment collections.
    """

    return {
        "fixed": get_collection_info("fixed"),
        "semantic": get_collection_info(
            "semantic"
        ),
    }


def search_similar_chunks(
    query_embedding: list[float],
    top_k: int = 5,
    chunking_method: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search the selected Qdrant collection for chunks
    semantically similar to the question embedding.
    """

    method = validate_chunking_method(
        chunking_method
    )

    collection_name = ensure_collection_exists(
        method
    )

    client = get_qdrant_client()

    try:
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
        )

    except AttributeError:
        response = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        search_results = response.points

    results: list[dict[str, Any]] = []

    for result in search_results:
        payload = result.payload or {}

        results.append(
            _payload_to_chunk(
                payload=payload,
                score=float(result.score),
            )
        )

    return results


def get_chunks_from_page_range(
    document_name: str,
    start_page: int,
    end_page: int,
    limit: int = 100,
    chunking_method: str | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve every stored chunk belonging to a document
    within an inclusive page range from the selected
    collection.

    Example:

        start_page = 7
        end_page = 9

    retrieves chunks from pages 7, 8, and 9.
    """

    if not document_name:
        return []

    start_page = max(1, start_page)

    if end_page < start_page:
        return []

    method = validate_chunking_method(
        chunking_method
    )

    collection_name = ensure_collection_exists(
        method
    )

    client = get_qdrant_client()

    page_filter = Filter(
        must=[
            FieldCondition(
                key="document_name",
                match=MatchValue(
                    value=document_name
                ),
            ),
            FieldCondition(
                key="page_number",
                range=Range(
                    gte=start_page,
                    lte=end_page,
                ),
            ),
        ]
    )

    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=page_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    chunks: list[dict[str, Any]] = []

    for point in points:
        payload = point.payload or {}

        chunk = _payload_to_chunk(
            payload=payload,
            score=None,
        )

        chunk["is_neighbor"] = True
        chunks.append(chunk)

    chunks.sort(
        key=lambda chunk: (
            chunk.get("page_number") or 0,
            chunk.get("chunk_index") or 0,
        )
    )

    return chunks


def get_neighboring_chunks(
    seed_chunks: list[dict[str, Any]],
    page_window: int = 1,
    limit_per_seed: int = 100,
    chunking_method: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch neighboring pages around every reranked seed
    chunk from the same collection used for retrieval.

    With page_window=1:

        seed page 8  -> pages 7, 8, and 9
        seed page 20 -> pages 19, 20, and 21

    Distant seed pages are not merged into one large page
    range. Duplicate chunks are removed by chunk_id, and
    original seed chunks are excluded from the returned
    neighbor list.
    """

    if not seed_chunks:
        return []

    if page_window < 0:
        raise ValueError(
            "page_window cannot be negative."
        )

    method = validate_chunking_method(
        chunking_method
    )

    seed_chunk_ids: set[str] = {
        str(chunk_id)
        for chunk in seed_chunks
        if (chunk_id := chunk.get("chunk_id"))
    }

    processed_seed_ranges: set[
        tuple[str, int, int]
    ] = set()

    seen_neighbor_ids: set[str] = set()

    neighboring_chunks: list[
        dict[str, Any]
    ] = []

    for seed_chunk in seed_chunks:
        document_name = seed_chunk.get(
            "document_name"
        )
        page_number = seed_chunk.get(
            "page_number"
        )

        if (
            not document_name
            or page_number is None
        ):
            continue

        page_number = int(page_number)

        start_page = max(
            1,
            page_number - page_window,
        )

        end_page = (
            page_number + page_window
        )

        seed_range = (
            str(document_name),
            start_page,
            end_page,
        )

        if seed_range in processed_seed_ranges:
            continue

        processed_seed_ranges.add(seed_range)

        local_chunks = get_chunks_from_page_range(
            document_name=str(document_name),
            start_page=start_page,
            end_page=end_page,
            limit=limit_per_seed,
            chunking_method=method,
        )

        for chunk in local_chunks:
            chunk_id = chunk.get("chunk_id")

            if not chunk_id:
                continue

            chunk_id = str(chunk_id)

            if chunk_id in seed_chunk_ids:
                continue

            if chunk_id in seen_neighbor_ids:
                continue

            seen_neighbor_ids.add(chunk_id)
            neighboring_chunks.append(chunk)

    neighboring_chunks.sort(
        key=lambda chunk: (
            chunk.get("document_name") or "",
            chunk.get("page_number") or 0,
            chunk.get("chunk_index") or 0,
        )
    )

    return neighboring_chunks