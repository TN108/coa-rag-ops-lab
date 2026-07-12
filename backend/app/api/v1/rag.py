from fastapi import APIRouter, UploadFile, File, Query

from app.config import settings
from app.services.pdf_service import extract_text_from_pdf
from app.services.chunking_service import (
    create_chunks_from_pages,
    create_semantic_chunks_from_pages
)
from app.services.embedding_service import (
    generate_embeddings,
    get_embedding_dimension
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