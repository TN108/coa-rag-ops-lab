from typing import Any

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from pydantic import BaseModel, Field

from app.config import settings
from app.services.chunking_service import (
    create_chunks_from_pages,
    create_semantic_chunks_from_pages,
)
from app.services.embedding_service import (
    generate_embedding,
    generate_embeddings,
    get_embedding_dimension,
)
from app.services.evaluation_service import EvaluationService
from app.services.llm_service import generate_rag_answer
from app.services.pdf_service import extract_text_from_pdf
from app.services.qdrant_service import (
    get_collection_info,
    search_similar_chunks,
    upsert_chunks_to_qdrant,
)
from app.services.reranker_service import rerank_chunks


router = APIRouter(
    prefix="/api/v1/rag",
    tags=["RAG"],
)


class EvaluationRunRequest(BaseModel):
    dataset_path: str = Field(
        default="data/evaluation/evaluation_dataset.json"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )
    min_retrieval_score: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
    )


@router.post("/chunk-pdf")
async def chunk_pdf(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    extraction_result = await extract_text_from_pdf(file)

    chunks = create_chunks_from_pages(
        pages=extraction_result["pages"],
        document_name=extraction_result["filename"],
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    return {
        "message": "PDF extracted and chunked successfully.",
        "filename": extraction_result["filename"],
        "extraction_method": extraction_result[
            "extraction_method"
        ],
        "chunking_method": "fixed_character_overlap",
        "total_pages": extraction_result["total_pages"],
        "total_characters": extraction_result[
            "total_characters"
        ],
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "total_chunks": len(chunks),
        "chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "global_chunk_index": chunk[
                    "global_chunk_index"
                ],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "character_count": chunk[
                    "character_count"
                ],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "chunking_method": chunk[
                    "chunking_method"
                ],
                "preview": chunk["text"][:300],
            }
            for chunk in chunks
        ],
    }


@router.post("/semantic-chunk-pdf")
async def semantic_chunk_pdf(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    extraction_result = await extract_text_from_pdf(file)

    chunks = create_semantic_chunks_from_pages(
        pages=extraction_result["pages"],
        document_name=extraction_result["filename"],
        max_chunk_size=settings.SEMANTIC_CHUNK_MAX_SIZE,
        min_chunk_size=settings.SEMANTIC_CHUNK_MIN_SIZE,
        overlap_paragraphs=(
            settings.SEMANTIC_CHUNK_OVERLAP_PARAGRAPHS
        ),
    )

    return {
        "message": (
            "PDF extracted and semantically chunked "
            "successfully."
        ),
        "filename": extraction_result["filename"],
        "extraction_method": extraction_result[
            "extraction_method"
        ],
        "chunking_method": "paragraph_section_semantic",
        "total_pages": extraction_result["total_pages"],
        "total_characters": extraction_result[
            "total_characters"
        ],
        "semantic_chunk_max_size": (
            settings.SEMANTIC_CHUNK_MAX_SIZE
        ),
        "semantic_chunk_min_size": (
            settings.SEMANTIC_CHUNK_MIN_SIZE
        ),
        "semantic_chunk_overlap_paragraphs": (
            settings.SEMANTIC_CHUNK_OVERLAP_PARAGRAPHS
        ),
        "total_chunks": len(chunks),
        "chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "global_chunk_index": chunk[
                    "global_chunk_index"
                ],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "character_count": chunk[
                    "character_count"
                ],
                "block_count": chunk["block_count"],
                "chunking_method": chunk[
                    "chunking_method"
                ],
                "preview": chunk["text"][:500],
            }
            for chunk in chunks
        ],
    }


@router.post("/embed-pdf")
async def embed_pdf(
    file: UploadFile = File(...),
    chunking_method: str = Query(
        default="semantic",
        description="Choose 'fixed' or 'semantic'",
    ),
) -> dict[str, Any]:
    extraction_result = await extract_text_from_pdf(file)

    if chunking_method == "fixed":
        chunks = create_chunks_from_pages(
            pages=extraction_result["pages"],
            document_name=extraction_result["filename"],
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    elif chunking_method == "semantic":
        chunks = create_semantic_chunks_from_pages(
            pages=extraction_result["pages"],
            document_name=extraction_result["filename"],
            max_chunk_size=(
                settings.SEMANTIC_CHUNK_MAX_SIZE
            ),
            min_chunk_size=(
                settings.SEMANTIC_CHUNK_MIN_SIZE
            ),
            overlap_paragraphs=(
                settings.SEMANTIC_CHUNK_OVERLAP_PARAGRAPHS
            ),
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid chunking_method. "
                "Use 'fixed' or 'semantic'."
            ),
        )

    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(chunk_texts)
    embedding_dimension = get_embedding_dimension()

    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "global_chunk_index": chunk[
                    "global_chunk_index"
                ],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "character_count": chunk[
                    "character_count"
                ],
                "chunking_method": chunk[
                    "chunking_method"
                ],
                "embedding_dimension": len(embedding),
                "embedding_preview": embedding[:5],
                "text_preview": chunk["text"][:300],
            }
        )

    return {
        "message": (
            "PDF extracted, chunked, and embedded "
            "successfully."
        ),
        "filename": extraction_result["filename"],
        "extraction_method": extraction_result[
            "extraction_method"
        ],
        "chunking_method": chunking_method,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "embedding_dimension": embedding_dimension,
        "total_pages": extraction_result["total_pages"],
        "total_characters": extraction_result[
            "total_characters"
        ],
        "total_chunks": len(chunks),
        "total_embeddings": len(embeddings),
        "chunks": embedded_chunks,
    }


@router.post("/store-pdf")
async def store_pdf(
    file: UploadFile = File(...),
    chunking_method: str = Query(
        default="semantic",
        description="Choose 'fixed' or 'semantic'",
    ),
) -> dict[str, Any]:
    extraction_result = await extract_text_from_pdf(file)

    if chunking_method == "fixed":
        chunks = create_chunks_from_pages(
            pages=extraction_result["pages"],
            document_name=extraction_result["filename"],
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    elif chunking_method == "semantic":
        chunks = create_semantic_chunks_from_pages(
            pages=extraction_result["pages"],
            document_name=extraction_result["filename"],
            max_chunk_size=(
                settings.SEMANTIC_CHUNK_MAX_SIZE
            ),
            min_chunk_size=(
                settings.SEMANTIC_CHUNK_MIN_SIZE
            ),
            overlap_paragraphs=(
                settings.SEMANTIC_CHUNK_OVERLAP_PARAGRAPHS
            ),
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid chunking_method. "
                "Use 'fixed' or 'semantic'."
            ),
        )

    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(chunk_texts)

    storage_result = upsert_chunks_to_qdrant(
        chunks=chunks,
        embeddings=embeddings,
        document_name=extraction_result["filename"],
        extraction_method=extraction_result[
            "extraction_method"
        ],
        chunking_method=chunking_method,
    )

    return {
        "message": (
            "PDF extracted, chunked, embedded, and "
            "stored in Qdrant successfully."
        ),
        "filename": extraction_result["filename"],
        "extraction_method": extraction_result[
            "extraction_method"
        ],
        "chunking_method": chunking_method,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "embedding_dimension": settings.EMBEDDING_DIMENSION,
        "total_pages": extraction_result["total_pages"],
        "total_characters": extraction_result[
            "total_characters"
        ],
        "total_chunks": len(chunks),
        "total_embeddings": len(embeddings),
        "qdrant": storage_result,
        "stored_chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "global_chunk_index": chunk[
                    "global_chunk_index"
                ],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "character_count": chunk[
                    "character_count"
                ],
                "chunking_method": chunk[
                    "chunking_method"
                ],
                "text_preview": chunk["text"][:300],
            }
            for chunk in chunks
        ],
    }


@router.get("/collection-info")
def collection_info() -> dict[str, Any]:
    return get_collection_info()


@router.get("/retrieve")
def retrieve_chunks(
    question: str = Query(
        ...,
        description=(
            "User question to search relevant chunks"
        ),
    ),
    top_k: int = Query(
        default=5,
        ge=1,
        le=10,
        description=(
            "Number of relevant chunks to retrieve"
        ),
    ),
) -> dict[str, Any]:
    question_embedding = generate_embedding(question)

    results = search_similar_chunks(
        query_embedding=question_embedding,
        top_k=top_k,
    )

    return {
        "message": "Relevant chunks retrieved successfully.",
        "question": question,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "embedding_dimension": len(question_embedding),
        "top_k": top_k,
        "total_results": len(results),
        "results": [
            {
                "rank": index + 1,
                "score": result["score"],
                "document_name": result[
                    "document_name"
                ],
                "page_number": result["page_number"],
                "chunk_id": result["chunk_id"],
                "chunk_index": result["chunk_index"],
                "global_chunk_index": result[
                    "global_chunk_index"
                ],
                "chunking_method": result[
                    "chunking_method"
                ],
                "extraction_method": result[
                    "extraction_method"
                ],
                "text_preview": (
                    result["text"][:500]
                    if result["text"]
                    else None
                ),
            }
            for index, result in enumerate(results)
        ],
    }


def run_rag_question(
    question: str,
    top_k: int = 5,
    min_retrieval_score: float = 0.10,
) -> dict[str, Any]:
    """
    Run the reusable RAG question-answering pipeline.

    Processing sequence:

    1. Generate an embedding for the question.
    2. Retrieve a larger candidate set from Qdrant.
    3. Filter weak or very short chunks.
    4. Rerank the remaining chunks with a CrossEncoder.
    5. Pass the final top-k chunks to the LLM.

    This function is used by both the normal /ask endpoint
    and the automatic evaluation endpoint.
    """

    question_embedding = generate_embedding(question)

    candidate_top_k = max(top_k * 3, 10)
    min_chunk_characters = 80

    candidate_chunks = search_similar_chunks(
        query_embedding=question_embedding,
        top_k=candidate_top_k,
    )

    filtered_chunks = [
        chunk
        for chunk in candidate_chunks
        if chunk["score"] >= min_retrieval_score
        and chunk.get("text")
        and len(chunk["text"]) >= min_chunk_characters
    ]

    retrieved_chunks = rerank_chunks(
        question=question,
        chunks=filtered_chunks,
        top_k=top_k,
    )

    if not retrieved_chunks:
        return {
            "message": (
                "No sufficiently relevant chunks found."
            ),
            "question": question,
            "answer": (
                "The provided document context does not "
                "contain enough information to answer "
                "this question."
            ),
            "embedding_model": (
                settings.EMBEDDING_MODEL_NAME
            ),
            "llm_model": settings.OLLAMA_MODEL,
            "requested_top_k": top_k,
            "candidate_top_k": candidate_top_k,
            "total_candidate_chunks": len(
                candidate_chunks
            ),
            "total_filtered_chunks": len(
                filtered_chunks
            ),
            "total_retrieved_chunks": 0,
            "reranking_enabled": True,
            "reranker_model": (
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            "filters": {
                "min_retrieval_score": (
                    min_retrieval_score
                ),
                "min_chunk_characters": (
                    min_chunk_characters
                ),
            },
            "sources": [],
        }

    answer = generate_rag_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    return {
        "message": "Answer generated successfully.",
        "question": question,
        "answer": answer,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "llm_model": settings.OLLAMA_MODEL,
        "requested_top_k": top_k,
        "candidate_top_k": candidate_top_k,
        "total_candidate_chunks": len(candidate_chunks),
        "total_filtered_chunks": len(filtered_chunks),
        "total_retrieved_chunks": len(
            retrieved_chunks
        ),
        "reranking_enabled": True,
        "reranker_model": (
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        "filters": {
            "min_retrieval_score": (
                min_retrieval_score
            ),
            "min_chunk_characters": (
                min_chunk_characters
            ),
        },
        "sources": [
            {
                "rank": index + 1,
                "score": chunk.get(
                    "retrieval_score",
                    chunk.get("score"),
                ),
                "retrieval_score": chunk.get(
                    "retrieval_score",
                    chunk.get("score"),
                ),
                "reranker_score": chunk.get(
                    "reranker_score"
                ),
                "document_name": chunk[
                    "document_name"
                ],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "global_chunk_index": chunk[
                    "global_chunk_index"
                ],
                "character_count": chunk[
                    "character_count"
                ],
                "text_preview": (
                    chunk["text"][:300]
                    if chunk["text"]
                    else None
                ),
            }
            for index, chunk in enumerate(
                retrieved_chunks
            )
        ],
    }


@router.get("/ask")
def ask_question(
    question: str = Query(
        ...,
        description=(
            "Question to answer from stored PDF chunks"
        ),
    ),
    top_k: int = Query(
        default=5,
        ge=1,
        le=10,
        description=(
            "Number of final reranked chunks "
            "to send to the LLM"
        ),
    ),
    min_retrieval_score: float = Query(
        default=0.10,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum similarity score for retrieved chunks"
        ),
    ),
) -> dict[str, Any]:
    return run_rag_question(
        question=question,
        top_k=top_k,
        min_retrieval_score=min_retrieval_score,
    )


@router.post("/evaluation/run")
def run_evaluation(
    request: EvaluationRunRequest,
) -> dict[str, Any]:
    try:
        evaluation_service = EvaluationService(
            rag_answer_function=run_rag_question,
        )

        return evaluation_service.run_evaluation(
            dataset_path=request.dataset_path,
            top_k=request.top_k,
            min_retrieval_score=(
                request.min_retrieval_score
            ),
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {exc}",
        ) from exc