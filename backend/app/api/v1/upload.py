from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from app.config import settings
from app.services.chunking_service import (
    create_chunks_from_pages,
    create_semantic_chunks_from_pages,
)
from app.services.embedding_service import (
    generate_embeddings,
)
from app.services.pdf_service import (
    extract_text_from_pdf,
)
from app.services.qdrant_service import (
    upsert_chunks_to_qdrant,
)


router = APIRouter(
    prefix="/api/v1/upload",
    tags=["Upload"],
)


@router.post("/pdf")
async def upload_pdf(
    file: UploadFile = File(...),
):
    try:
        result = await extract_text_from_pdf(
            file
        )

        pages = result["pages"]
        document_name = result["filename"]
        extraction_method = result.get(
            "extraction_method",
            "pdf_text_extraction",
        )

        chunking_method = (
            settings.CHUNKING_METHOD
        )

        if chunking_method == "fixed":
            chunks = create_chunks_from_pages(
                pages=pages,
                document_name=document_name,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=(
                    settings.CHUNK_OVERLAP
                ),
            )

        elif chunking_method == "semantic":
            chunks = (
                create_semantic_chunks_from_pages(
                    pages=pages,
                    document_name=document_name,
                    max_chunk_size=(
                        settings
                        .SEMANTIC_CHUNK_MAX_SIZE
                    ),
                    min_chunk_size=(
                        settings
                        .SEMANTIC_CHUNK_MIN_SIZE
                    ),
                    overlap_paragraphs=(
                        settings
                        .SEMANTIC_CHUNK_OVERLAP_PARAGRAPHS
                    ),
                )
            )

        else:
            raise ValueError(
                "CHUNKING_METHOD must be either "
                "'fixed' or 'semantic'."
            )

        if not chunks:
            raise ValueError(
                "No chunks were generated from "
                "the uploaded PDF."
            )

        embeddings = generate_embeddings(
            [
                chunk["text"]
                for chunk in chunks
            ]
        )

        storage_result = (
            upsert_chunks_to_qdrant(
                chunks=chunks,
                embeddings=embeddings,
                document_name=document_name,
                extraction_method=(
                    extraction_method
                ),
                chunking_method=(
                    chunking_method
                ),
            )
        )

        return {
            "message": (
                "PDF processed and stored "
                "successfully."
            ),
            "filename": document_name,
            "total_pages": result[
                "total_pages"
            ],
            "total_characters": result[
                "total_characters"
            ],
            "extraction_method": (
                extraction_method
            ),
            "chunking_method": (
                chunking_method
            ),
            "collection_name": (
                storage_result[
                    "collection_name"
                ]
            ),
            "total_chunks": len(chunks),
            "stored_points": (
                storage_result[
                    "stored_points"
                ]
            ),
            "total_points_in_collection": (
                storage_result[
                    "total_points_in_collection"
                ]
            ),
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc