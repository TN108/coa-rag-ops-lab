from fastapi import APIRouter, UploadFile, File

from app.config import settings
from app.services.pdf_service import extract_text_from_pdf
from app.services.chunking_service import create_chunks_from_pages

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
                "preview": chunk["text"][:300]
            }
            for chunk in chunks
        ]
    }