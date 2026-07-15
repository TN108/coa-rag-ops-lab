from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import settings
from app.services.chunking_service import create_chunks_from_pages
from app.services.evaluation_service import EvaluationService
from app.services.pdf_service import extract_text_from_pdf
from app.services.rag_service import generate_rag_answer


router = APIRouter(
    prefix="/api/v1/rag",
    tags=["RAG"],
)


class EvaluationRunRequest(BaseModel):
    dataset_path: str = Field(
        default="data/evaluation/evaluation_dataset.json"
    )
    top_k: int = Field(default=5, ge=1, le=20)
    min_retrieval_score: float = Field(
        default=0.1,
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
                "preview": chunk["text"][:300],
            }
            for chunk in chunks
        ],
    }


@router.post("/evaluation/run")
def run_evaluation(
    request: EvaluationRunRequest,
) -> dict[str, Any]:
    try:
        evaluation_service = EvaluationService(
            rag_answer_function=generate_rag_answer,
        )

        return evaluation_service.run_evaluation(
            dataset_path=request.dataset_path,
            top_k=request.top_k,
            min_retrieval_score=request.min_retrieval_score,
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