from app.config import settings
from app.utils.text_cleaner import clean_chunk_text


def create_chunks_from_text(
    text: str,
    page_number: int,
    document_name: str,
    chunk_size: int = None,
    chunk_overlap: int = None
):
    if chunk_size is None:
        chunk_size = settings.CHUNK_SIZE

    if chunk_overlap is None:
        chunk_overlap = settings.CHUNK_OVERLAP

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    cleaned_text = clean_chunk_text(text)

    if not cleaned_text:
        return []

    chunks = []
    start = 0
    chunk_index = 1
    text_length = len(cleaned_text)

    while start < text_length:
        end = start + chunk_size
        chunk_text = cleaned_text[start:end].strip()

        if chunk_text:
            chunks.append({
                "chunk_id": f"{document_name}_page_{page_number}_chunk_{chunk_index}",
                "document_name": document_name,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "character_count": len(chunk_text),
                "start_char": start,
                "end_char": min(end, text_length)
            })

        start = end - chunk_overlap
        chunk_index += 1

        if start >= text_length:
            break

    return chunks


def create_chunks_from_pages(
    pages: list,
    document_name: str,
    chunk_size: int = None,
    chunk_overlap: int = None
):
    all_chunks = []

    for page in pages:
        page_number = page.get("page_number")
        page_text = page.get("text", "")

        page_chunks = create_chunks_from_text(
            text=page_text,
            page_number=page_number,
            document_name=document_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        all_chunks.extend(page_chunks)

    for global_index, chunk in enumerate(all_chunks, start=1):
        chunk["global_chunk_index"] = global_index

    return all_chunks