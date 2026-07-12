from io import BytesIO

import fitz
import pytesseract
from PIL import Image, ImageOps, ImageFilter
from fastapi import UploadFile, HTTPException
from pypdf import PdfReader

from app.config import settings


pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD


def extract_text_with_pypdf(pdf_reader: PdfReader):
    pages_data = []
    full_text_parts = []

    for page_number, page in enumerate(pdf_reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = page_text.strip()

        pages_data.append({
            "page_number": page_number,
            "character_count": len(page_text),
            "text": page_text,
            "extraction_method": "pypdf"
        })

        if page_text:
            full_text_parts.append(page_text)

    full_text = "\n\n".join(full_text_parts).strip()

    return full_text, pages_data


def get_ocr_image_variants(image: Image.Image):
    """
    Create multiple image versions for OCR.
    Different scanned PDFs work better with different preprocessing.
    """

    variants = []

    # Original image
    variants.append(image)

    # Grayscale
    gray = image.convert("L")
    variants.append(gray)

    # Auto contrast
    contrast = ImageOps.autocontrast(gray)
    variants.append(contrast)

    # Sharpened contrast
    sharpened = contrast.filter(ImageFilter.SHARPEN)
    variants.append(sharpened)

    # Light threshold
    threshold_light = contrast.point(lambda x: 0 if x < 160 else 255, "1")
    variants.append(threshold_light)

    # Medium threshold
    threshold_medium = contrast.point(lambda x: 0 if x < 180 else 255, "1")
    variants.append(threshold_medium)

    # Strong threshold
    threshold_strong = contrast.point(lambda x: 0 if x < 200 else 255, "1")
    variants.append(threshold_strong)

    return variants


def run_tesseract_ocr(image: Image.Image) -> str:
    """
    Run Tesseract with multiple OCR settings and keep the best result.
    """

    ocr_configs = [
        "--oem 3 --psm 3",   # Fully automatic page segmentation
        "--oem 3 --psm 4",   # Single column of text
        "--oem 3 --psm 6",   # Uniform block of text
        "--oem 3 --psm 11",  # Sparse text
        "--oem 3 --psm 12",  # Sparse text with orientation/script detection
    ]

    best_text = ""

    image_variants = get_ocr_image_variants(image)

    for variant in image_variants:
        for config in ocr_configs:
            try:
                text = pytesseract.image_to_string(
                    variant,
                    config=config
                ).strip()

                if len(text) > len(best_text):
                    best_text = text

            except Exception:
                continue

    return best_text


def extract_text_with_ocr(file_bytes: bytes):
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not open PDF for OCR."
        )

    pages_data = []
    full_text_parts = []

    zoom = settings.OCR_DPI / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page_index in range(len(pdf_document)):
        page = pdf_document[page_index]

        pix = page.get_pixmap(matrix=matrix, alpha=False)

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        page_text = run_tesseract_ocr(image)
        page_text = page_text.strip()

        pages_data.append({
            "page_number": page_index + 1,
            "character_count": len(page_text),
            "text": page_text,
            "extraction_method": "ocr"
        })

        if page_text:
            full_text_parts.append(page_text)

    pdf_document.close()

    full_text = "\n\n".join(full_text_parts).strip()

    return full_text, pages_data


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

    full_text, pages_data = extract_text_with_pypdf(pdf_reader)
    extraction_method = "pypdf"

    if not full_text:
        if not settings.ENABLE_OCR:
            raise HTTPException(
                status_code=400,
                detail="No readable text found in this PDF. It may be a scanned PDF."
            )

        full_text, pages_data = extract_text_with_ocr(file_bytes)
        extraction_method = "ocr"

    if not full_text:
        raise HTTPException(
            status_code=400,
            detail="No readable text found even after OCR."
        )

    return {
        "filename": file.filename,
        "total_pages": len(pages_data),
        "total_characters": len(full_text),
        "preview": full_text[:1000],
        "extraction_method": extraction_method,
        "pages": pages_data
    }