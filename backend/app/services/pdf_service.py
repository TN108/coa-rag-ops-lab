from io import BytesIO
from fastapi import UploadFile, HTTPException
from pypdf import PdfReader


async def extract_text_from_pdf(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    try:
        pdf_reader = PdfReader(BytesIO(file_bytes))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read PDF file."
        )

    pages_data = []
    full_text_parts = []

    for page_number, page in enumerate(pdf_reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = page_text.strip()

        pages_data.append({
            "page_number": page_number,
            "character_count": len(page_text),
            "text": page_text
        })

        if page_text:
            full_text_parts.append(page_text)

    full_text = "\n\n".join(full_text_parts).strip()

    if not full_text:
        raise HTTPException(
            status_code=400,
            detail="No readable text found in this PDF. It may be a scanned PDF."
        )

    return {
        "filename": file.filename,
        "total_pages": len(pdf_reader.pages),
        "total_characters": len(full_text),
        "preview": full_text[:1000],
        "pages": pages_data
    }