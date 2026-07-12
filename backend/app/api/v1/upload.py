from fastapi import APIRouter, UploadFile, File
from app.services.pdf_service import extract_text_from_pdf

router = APIRouter(
    prefix="/api/v1/upload",
    tags=["Upload"]
)


@router.post("/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    result = await extract_text_from_pdf(file)

    return {
        "message": "PDF uploaded and text extracted successfully.",
        "filename": result["filename"],
        "total_pages": result["total_pages"],
        "total_characters": result["total_characters"],
        "extraction_method": result["extraction_method"],
        "preview": result["preview"],
        "pages": [
            {
                "page_number": page["page_number"],
                "character_count": page["character_count"],
                "extraction_method": page["extraction_method"]
            }
            for page in result["pages"]
        ]
    }