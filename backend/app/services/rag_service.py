from time import perf_counter
from typing import Any, Callable

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
    get_neighboring_chunks,
    search_similar_chunks,
    upsert_chunks_to_qdrant,
)
from app.services.reranker_service import rerank_chunks
from app.services.planner_service import planner_service


router = APIRouter(
    prefix="/api/v1/rag",
    tags=["RAG"],
)


def validate_chunking_method(chunking_method: str) -> str:
    method = chunking_method.strip().lower()

    if method not in {"fixed", "semantic"}:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid chunking_method. "
                "Use 'fixed' or 'semantic'."
            ),
        )

    return method


def build_chunks(
    extraction_result: dict[str, Any],
    chunking_method: str,
) -> list[dict[str, Any]]:
    method = validate_chunking_method(chunking_method)

    if method == "fixed":
        return create_chunks_from_pages(
            pages=extraction_result["pages"],
            document_name=extraction_result["filename"],
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    return create_semantic_chunks_from_pages(
        pages=extraction_result["pages"],
        document_name=extraction_result["filename"],
        max_chunk_size=settings.SEMANTIC_CHUNK_MAX_SIZE,
        min_chunk_size=settings.SEMANTIC_CHUNK_MIN_SIZE,
        overlap_paragraphs=(
            settings.SEMANTIC_CHUNK_OVERLAP_PARAGRAPHS
        ),
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

    chunking_method: str = Field(
        default="semantic",
        description="Choose 'fixed' or 'semantic'.",
    )

    adaptive: bool = Field(
        default=True,
        description=(
            "True runs Adaptive RAG evaluation. "
            "False runs baseline RAG evaluation."
        ),
    )


@router.post("/chunk-pdf")
async def chunk_pdf(
    file: UploadFile = File(...),
    chunking_method: str = Query(
        default="fixed",
        description="Choose 'fixed' or 'semantic'.",
    ),
) -> dict[str, Any]:
    method = validate_chunking_method(chunking_method)
    extraction_result = await extract_text_from_pdf(file)
    chunks = build_chunks(extraction_result, method)

    return {
        "message": "PDF extracted and chunked successfully.",
        "filename": extraction_result["filename"],
        "extraction_method": extraction_result[
            "extraction_method"
        ],
        "chunking_method": method,
        "collection_name": settings.get_collection_name(method),
        "total_pages": extraction_result["total_pages"],
        "total_characters": extraction_result[
            "total_characters"
        ],
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
                "chunking_method": chunk[
                    "chunking_method"
                ],
                "preview": chunk["text"][:500],
            }
            for chunk in chunks
        ],
    }


@router.post("/semantic-chunk-pdf")
async def semantic_chunk_pdf(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    extraction_result = await extract_text_from_pdf(file)
    chunks = build_chunks(
        extraction_result=extraction_result,
        chunking_method="semantic",
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
        "chunking_method": "semantic",
        "collection_name": settings.get_collection_name(
            "semantic"
        ),
        "total_pages": extraction_result["total_pages"],
        "total_characters": extraction_result[
            "total_characters"
        ],
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
                "block_count": chunk.get("block_count"),
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
        description="Choose 'fixed' or 'semantic'.",
    ),
) -> dict[str, Any]:
    method = validate_chunking_method(chunking_method)
    extraction_result = await extract_text_from_pdf(file)
    chunks = build_chunks(extraction_result, method)

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
        "chunking_method": method,
        "collection_name": settings.get_collection_name(method),
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
        description="Choose 'fixed' or 'semantic'.",
    ),
) -> dict[str, Any]:
    method = validate_chunking_method(chunking_method)
    extraction_result = await extract_text_from_pdf(file)
    chunks = build_chunks(extraction_result, method)

    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(chunk_texts)

    storage_result = upsert_chunks_to_qdrant(
        chunks=chunks,
        embeddings=embeddings,
        document_name=extraction_result["filename"],
        extraction_method=extraction_result[
            "extraction_method"
        ],
        chunking_method=method,
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
        "chunking_method": method,
        "collection_name": settings.get_collection_name(method),
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
def collection_info(
    chunking_method: str = Query(
        default="semantic",
        description="Choose 'fixed' or 'semantic'.",
    ),
) -> dict[str, Any]:
    method = validate_chunking_method(chunking_method)

    return get_collection_info(
        chunking_method=method,
    )


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
    chunking_method: str = Query(
        default="semantic",
        description="Choose 'fixed' or 'semantic'.",
    ),
) -> dict[str, Any]:
    method = validate_chunking_method(chunking_method)
    question_embedding = generate_embedding(question)

    results = search_similar_chunks(
        query_embedding=question_embedding,
        top_k=top_k,
        chunking_method=method,
    )

    return {
        "message": "Relevant chunks retrieved successfully.",
        "question": question,
        "chunking_method": method,
        "collection_name": settings.get_collection_name(method),
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
    chunking_method: str = "semantic",
    page_window: int = 1,
) -> dict[str, Any]:
    method = chunking_method.strip().lower()

    if method not in {"fixed", "semantic"}:
        raise ValueError(
            "chunking_method must be either "
            "'fixed' or 'semantic'."
        )

    total_start = perf_counter()

    embedding_start = perf_counter()
    question_embedding = generate_embedding(question)
    embedding_ms = (
        perf_counter() - embedding_start
    ) * 1000

    candidate_top_k = max(top_k * 3, 10)
    min_chunk_characters = 80

    retrieval_start = perf_counter()
    candidate_chunks = search_similar_chunks(
        query_embedding=question_embedding,
        top_k=candidate_top_k,
        chunking_method=method,
    )
    retrieval_ms = (
        perf_counter() - retrieval_start
    ) * 1000

    filtering_start = perf_counter()

    filtered_chunks = [
        chunk
        for chunk in candidate_chunks
        if chunk["score"] >= min_retrieval_score
        and chunk.get("text")
        and len(chunk["text"]) >= min_chunk_characters
    ]

    filtering_ms = (
        perf_counter() - filtering_start
    ) * 1000

    reranking_start = perf_counter()
    reranked_chunks = rerank_chunks(
        question=question,
        chunks=filtered_chunks,
        top_k=top_k,
    )
    reranking_ms = (
        perf_counter() - reranking_start
    ) * 1000

    base_response = {
        "question": question,
        "chunking_method": method,
        "collection_name": settings.get_collection_name(method),
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "llm_model": settings.OLLAMA_MODEL,
        "requested_top_k": top_k,
        "candidate_top_k": candidate_top_k,
        "total_candidate_chunks": len(
            candidate_chunks
        ),
        "total_filtered_chunks": len(
            filtered_chunks
        ),
        "reranking_enabled": True,
        "reranker_model": (
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        "neighbor_page_retrieval": True,
        "page_window": page_window,
        "filters": {
            "min_retrieval_score": (
                min_retrieval_score
            ),
            "min_chunk_characters": (
                min_chunk_characters
            ),
        },
    }

    if not reranked_chunks:
        total_ms = (
            perf_counter() - total_start
        ) * 1000

        return {
            **base_response,
            "message": (
                "No sufficiently relevant chunks found."
            ),
            "answer": (
                "The provided document context does not "
                "contain enough information to answer "
                "this question."
            ),
            "total_reranked_chunks": 0,
            "total_neighbor_chunks": 0,
            "total_retrieved_chunks": 0,
            "latency": {
                "embedding_ms": round(
                    embedding_ms,
                    2,
                ),
                "retrieval_ms": round(
                    retrieval_ms,
                    2,
                ),
                "filtering_ms": round(
                    filtering_ms,
                    2,
                ),
                "reranking_ms": round(
                    reranking_ms,
                    2,
                ),
                "neighbor_retrieval_ms": 0.0,
                "llm_generation_ms": 0.0,
                "total_ms": round(
                    total_ms,
                    2,
                ),
            },
            "sources": [],
        }

    

    neighbor_retrieval_start = perf_counter()

    neighbor_chunks = get_neighboring_chunks(
        seed_chunks=reranked_chunks,
        page_window=page_window,
        chunking_method=method,
    )

    neighbor_chunks = [
        chunk
        for chunk in neighbor_chunks
        if chunk.get("text")
        and len(chunk["text"].strip())
        >= min_chunk_characters
    ]

    neighbor_retrieval_ms = (
        perf_counter() - neighbor_retrieval_start
    ) * 1000

    retrieved_chunks: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()

    for chunk in reranked_chunks + neighbor_chunks:
        chunk_id = chunk.get("chunk_id")

        if not chunk_id:
            continue

        if chunk_id in seen_chunk_ids:
            continue

        seen_chunk_ids.add(chunk_id)
        retrieved_chunks.append(chunk)

    llm_generation_start = perf_counter()

    answer = generate_rag_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    llm_generation_ms = (
        perf_counter() - llm_generation_start
    ) * 1000

    total_ms = (
        perf_counter() - total_start
    ) * 1000

    return {
        **base_response,
        "message": "Answer generated successfully.",
        "answer": answer,
        "total_reranked_chunks": len(
            reranked_chunks
        ),
        "total_neighbor_chunks": len(
            neighbor_chunks
        ),
        "total_retrieved_chunks": len(
            retrieved_chunks
        ),
        "latency": {
            "embedding_ms": round(
                embedding_ms,
                2,
            ),
            "retrieval_ms": round(
                retrieval_ms,
                2,
            ),
            "filtering_ms": round(
                filtering_ms,
                2,
            ),
            "reranking_ms": round(
                reranking_ms,
                2,
            ),
            "neighbor_retrieval_ms": round(
                neighbor_retrieval_ms,
                2,
            ),
            "llm_generation_ms": round(
                llm_generation_ms,
                2,
            ),
            "total_ms": round(
                total_ms,
                2,
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
                "is_neighbor": chunk.get(
                    "is_neighbor",
                    False,
                ),
                "document_name": chunk[
                    "document_name"
                ],
                "page_number": chunk[
                    "page_number"
                ],
                "chunk_id": chunk[
                    "chunk_id"
                ],
                "chunk_index": chunk[
                    "chunk_index"
                ],
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
    chunking_method: str = Query(
        default="semantic",
        description="Choose 'fixed' or 'semantic'.",
    ),
) -> dict[str, Any]:
    method = validate_chunking_method(chunking_method)

    return run_rag_question(
        question=question,
        top_k=top_k,
        min_retrieval_score=min_retrieval_score,
        chunking_method=method,
    )


@router.post("/evaluation/run")
def run_evaluation(
    request: EvaluationRunRequest,
) -> dict[str, Any]:

    try:

        method = (
            request.chunking_method
            .strip()
            .lower()
        )


        if method not in {
            "fixed",
            "semantic",
        }:
            raise ValueError(
                "chunking_method must be either "
                "'fixed' or 'semantic'."
            )


        rag_answer_function: Callable[
            ..., dict[str, Any]
        ] = (
            lambda question,
            top_k=5,
            min_retrieval_score=0.10,
            chunking_method="semantic",
            page_window=1:

            run_rag_question(
                question=question,
                top_k=top_k,
                min_retrieval_score=(
                    min_retrieval_score
                ),
                chunking_method=(
                    chunking_method
                ),
                page_window=(
                    page_window
                ),
            )
        )


        evaluation_service = EvaluationService(
            rag_answer_function=(
                rag_answer_function
            )
        )


        if request.adaptive:

            result = (
                evaluation_service
                .run_adaptive_evaluation(
                    dataset_path=(
                        request.dataset_path
                    )
                )
            )

        else:

            result = (
                evaluation_service
                .run_evaluation(
                    dataset_path=(
                        request.dataset_path
                    ),
                    top_k=request.top_k,
                    min_retrieval_score=(
                        request.min_retrieval_score
                    ),
                )
            )


        result["chunking_method"] = method

        result["collection_name"] = (
            settings.get_collection_name(
                method
            )
        )


        return result


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
            detail=(
                f"Evaluation failed: {exc}"
            ),
        ) from exc
@router.post("/adaptive-ask")
def adaptive_ask_question(
    question: str = Query(
        ...,
        description=(
            "Question answered using Adaptive RAG planner"
        ),
    ),
) -> dict[str, Any]:

    total_start = perf_counter()

    planner_result = planner_service.plan(
        question
    )

    decision = planner_result[
        "planner_decision"
    ]

    result = run_rag_question(
        question=question,
        top_k=decision["top_k"],
        min_retrieval_score=(
            decision["min_retrieval_score"]
        ),
        chunking_method=(
            decision["chunking_method"]
        ),
        page_window=(
            decision["neighbor_window"]
        ),
    )


    total_ms = (
        perf_counter()
        -
        total_start
    ) * 1000


    return {
        "question": question,

        "planner_model": planner_result[
            "planner_model"
        ],

        "planner_decision": decision,

        "planner_fallback_used": planner_result[
            "fallback_used"
        ],

        "answer": result["answer"],

        "sources": result["sources"],

        "latency": {
            "planner_ms": planner_result[
                "planner_ms"
            ],

            "embedding_ms": result[
                "latency"
            ]["embedding_ms"],

            "retrieval_ms": result[
                "latency"
            ]["retrieval_ms"],

            "reranking_ms": result[
                "latency"
            ]["reranking_ms"],

            "generation_ms": result[
                "latency"
            ]["llm_generation_ms"],

            "total_ms": round(
                total_ms,
                2,
            ),
        },

        "rag_metadata": {
            "chunking_method": decision[
                "chunking_method"
            ],
            "top_k": decision[
                "top_k"
            ],
            "min_retrieval_score": decision[
                "min_retrieval_score"
            ],
            "neighbor_window": decision[
                "neighbor_window"
            ],
        },
    }