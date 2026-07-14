from fastapi import APIRouter, UploadFile, File, Query

from app.config import settings
from app.services.pdf_service import extract_text_from_pdf
from app.services.chunking_service import (
    create_chunks_from_pages,
    create_semantic_chunks_from_pages
)
from app.services.embedding_service import (
    generate_embedding,
    generate_embeddings,
    get_embedding_dimension
)

from app.services.qdrant_service import (
    upsert_chunks_to_qdrant,
    get_collection_info,
    search_similar_chunks
)

router = APIRouter(
    prefix="/api/v1/rag",
    tags=["RAG"]
)


@router.post("/chunk-pdf")
async def chunk_pdf(file: UploadFile = File(...)):
    extraction_result = await extract_text_from_pdf(file)

    chunks = create_chunks_from_pages(
        pages=extraction_result["pages"],
        document_name=extraction_result["filename"],
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )

    return {
        "message": "PDF extracted and chunked successfully.",
        "filename": extraction_result["filename"],
        "extraction_method": extraction_result["extraction_method"],
        "chunking_method": "fixed_character_overlap",
        "total_pages": extraction_result["total_pages"],
        "total_characters": extraction_result["total_characters"],
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "total_chunks": len(chunks),
        "chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "global_chunk_index": chunk["global_chunk_index"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "character_count": chunk["character_count"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "chunking_method": chunk["chunking_method"],
                "preview": chunk["text"][:300]
            }
            for chunk in chunks
        ]
    }


@router.post("/semantic-chunk-pdf")
async def semantic_chunk_pdf(file: UploadFile = File(...)):
    extraction_result = await extract_text_from_pdf(file)

    chunks = create_semantic_chunks_from_pages(
        pages=extraction_result["pages"],
        document_name=extraction_result["filename"],
        max_chunk_size=settings.SEMANTIC_CHUNK_MAX_SIZE,
        min_chunk_size=settings.SEMANTIC_CHUNK_MIN_SIZE,
        overlap_paragraphs=settings.SEMANTIC_CHUNK_OVERLAP_PARAGRAPHS
    )

    return {
        "message": "PDF extracted and semantically chunked successfully.",
        "filename": extraction_result["filename"],
        "extraction_method": extraction_result["extraction_method"],
        "chunking_method": "paragraph_section_semantic",
        "total_pages": extraction_result["total_pages"],
        "total_characters": extraction_result["total_characters"],
        "semantic_chunk_max_size": settings.SEMANTIC_CHUNK_MAX_SIZE,
        "semantic_chunk_min_size": settings.SEMANTIC_CHUNK_MIN_SIZE,
        "semantic_chunk_overlap_paragraphs": settings.SEMANTIC_CHUNK_OVERLAP_PARAGRAPHS,
        "total_chunks": len(chunks),
        "chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "global_chunk_index": chunk["global_chunk_index"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "character_count": chunk["character_count"],
                "block_count": chunk["block_count"],
                "chunking_method": chunk["chunking_method"],
                "preview": chunk["text"][:500]
            }
            for chunk in chunks
        ]
    }


@router.post("/embed-pdf")
async def embed_pdf(
    file: UploadFile = File(...),
    chunking_method: str = Query(
        default="semantic",
        description="Choose 'fixed' or 'semantic'"
    )
):
    extraction_result = await extract_text_from_pdf(file)

    if chunking_method == "fixed":
        chunks = create_chunks_from_pages(
            pages=extraction_result["pages"],
            document_name=extraction_result["filename"],
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
    elif chunking_method == "semantic":
        chunks = create_semantic_chunks_from_pages(
            pages=extraction_result["pages"],
            document_name=extraction_result["filename"],
            max_chunk_size=settings.SEMANTIC_CHUNK_MAX_SIZE,
            min_chunk_size=settings.SEMANTIC_CHUNK_MIN_SIZE,
            overlap_paragraphs=settings.SEMANTIC_CHUNK_OVERLAP_PARAGRAPHS
        )
    else:
        return {
            "error": "Invalid chunking_method. Use 'fixed' or 'semantic'."
        }

    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(chunk_texts)
    embedding_dimension = get_embedding_dimension()

    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "global_chunk_index": chunk["global_chunk_index"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "character_count": chunk["character_count"],
            "chunking_method": chunk["chunking_method"],
            "embedding_dimension": len(embedding),
            "embedding_preview": embedding[:5],
            "text_preview": chunk["text"][:300]
        })

    return {
        "message": "PDF extracted, chunked, and embedded successfully.",
        "filename": extraction_result["filename"],
        "extraction_method": extraction_result["extraction_method"],
        "chunking_method": chunking_method,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "embedding_dimension": embedding_dimension,
        "total_pages": extraction_result["total_pages"],
        "total_characters": extraction_result["total_characters"],
        "total_chunks": len(chunks),
        "total_embeddings": len(embeddings),
        "chunks": embedded_chunks
    }


@router.post("/store-pdf")
async def store_pdf(
    file: UploadFile = File(...),
    chunking_method: str = Query(
        default="semantic",
        description="Choose 'fixed' or 'semantic'"
    )
):
    extraction_result = await extract_text_from_pdf(file)

    if chunking_method == "fixed":
        chunks = create_chunks_from_pages(
            pages=extraction_result["pages"],
            document_name=extraction_result["filename"],
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
    elif chunking_method == "semantic":
        chunks = create_semantic_chunks_from_pages(
            pages=extraction_result["pages"],
            document_name=extraction_result["filename"],
            max_chunk_size=settings.SEMANTIC_CHUNK_MAX_SIZE,
            min_chunk_size=settings.SEMANTIC_CHUNK_MIN_SIZE,
            overlap_paragraphs=settings.SEMANTIC_CHUNK_OVERLAP_PARAGRAPHS
        )
    else:
        return {
            "error": "Invalid chunking_method. Use 'fixed' or 'semantic'."
        }

    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(chunk_texts)

    storage_result = upsert_chunks_to_qdrant(
        chunks=chunks,
        embeddings=embeddings,
        document_name=extraction_result["filename"],
        extraction_method=extraction_result["extraction_method"],
        chunking_method=chunking_method
    )

    return {
        "message": "PDF extracted, chunked, embedded, and stored in Qdrant successfully.",
        "filename": extraction_result["filename"],
        "extraction_method": extraction_result["extraction_method"],
        "chunking_method": chunking_method,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "embedding_dimension": settings.EMBEDDING_DIMENSION,
        "total_pages": extraction_result["total_pages"],
        "total_characters": extraction_result["total_characters"],
        "total_chunks": len(chunks),
        "total_embeddings": len(embeddings),
        "qdrant": storage_result,
        "stored_chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "global_chunk_index": chunk["global_chunk_index"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "character_count": chunk["character_count"],
                "chunking_method": chunk["chunking_method"],
                "text_preview": chunk["text"][:300]
            }
            for chunk in chunks
        ]
    }


@router.get("/collection-info")
def collection_info():
    return get_collection_info()

@router.get("/retrieve")
def retrieve_chunks(
    question: str = Query(..., description="User question to search relevant chunks"),
    top_k: int = Query(
        default=5,
        ge=1,
        le=10,
        description="Number of relevant chunks to retrieve"
    )
):
    question_embedding = generate_embedding(question)

    results = search_similar_chunks(
        query_embedding=question_embedding,
        top_k=top_k
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
                "document_name": result["document_name"],
                "page_number": result["page_number"],
                "chunk_id": result["chunk_id"],
                "chunk_index": result["chunk_index"],
                "global_chunk_index": result["global_chunk_index"],
                "chunking_method": result["chunking_method"],
                "extraction_method": result["extraction_method"],
                "text_preview": result["text"][:500] if result["text"] else None
            }
            for index, result in enumerate(results)
        ]
    }