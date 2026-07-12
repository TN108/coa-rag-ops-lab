from fastapi import APIRouter, UploadFile, File

from app.config import settings
from app.services.pdf_service import extract_text_from_pdf
from app.services.chunking_service import (
    create_chunks_from_pages,
    create_semantic_chunks_from_pages
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