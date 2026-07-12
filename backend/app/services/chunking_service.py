import re

from app.config import settings
from app.utils.text_cleaner import clean_chunk_text, clean_text


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
                "end_char": min(end, text_length),
                "chunking_method": "fixed_character_overlap"
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


def split_text_into_semantic_blocks(text: str):
    """
    Split text into meaningful blocks using paragraph breaks and sentence boundaries.
    This is a lightweight semantic chunking method.
    """

    cleaned_text = clean_text(text)

    if not cleaned_text:
        return []

    paragraphs = re.split(r"\n\s*\n", cleaned_text)

    blocks = []

    for paragraph in paragraphs:
        paragraph = clean_chunk_text(paragraph)

        if not paragraph:
            continue

        if len(paragraph) <= settings.SEMANTIC_CHUNK_MAX_SIZE:
            blocks.append(paragraph)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)

            current_block = ""

            for sentence in sentences:
                sentence = sentence.strip()

                if not sentence:
                    continue

                if len(current_block) + len(sentence) + 1 <= settings.SEMANTIC_CHUNK_MAX_SIZE:
                    current_block = f"{current_block} {sentence}".strip()
                else:
                    if current_block:
                        blocks.append(current_block)
                    current_block = sentence

            if current_block:
                blocks.append(current_block)

    return blocks


def create_semantic_chunks_from_text(
    text: str,
    page_number: int,
    document_name: str,
    max_chunk_size: int = None,
    min_chunk_size: int = None,
    overlap_paragraphs: int = None
):
    if max_chunk_size is None:
        max_chunk_size = settings.SEMANTIC_CHUNK_MAX_SIZE

    if min_chunk_size is None:
        min_chunk_size = settings.SEMANTIC_CHUNK_MIN_SIZE

    if overlap_paragraphs is None:
        overlap_paragraphs = settings.SEMANTIC_CHUNK_OVERLAP_PARAGRAPHS

    blocks = split_text_into_semantic_blocks(text)

    if not blocks:
        return []

    chunks = []
    current_blocks = []
    chunk_index = 1

    def current_text_length(blocks_list):
        return len("\n\n".join(blocks_list).strip())

    def make_chunk(blocks_list, index):
        chunk_text = "\n\n".join(blocks_list).strip()

        return {
            "chunk_id": f"{document_name}_page_{page_number}_semantic_chunk_{index}",
            "document_name": document_name,
            "page_number": page_number,
            "chunk_index": index,
            "text": chunk_text,
            "character_count": len(chunk_text),
            "chunking_method": "paragraph_section_semantic",
            "block_count": len(blocks_list)
        }

    for block in blocks:
        block = block.strip()

        if not block:
            continue

        # If block itself is too large, split it by fixed size as fallback
        if len(block) > max_chunk_size:
            if current_blocks:
                chunks.append(make_chunk(current_blocks, chunk_index))
                chunk_index += 1
                current_blocks = []

            start = 0
            while start < len(block):
                sub_block = block[start:start + max_chunk_size].strip()

                if sub_block:
                    chunks.append(make_chunk([sub_block], chunk_index))
                    chunk_index += 1

                start += max_chunk_size

            continue

        candidate_blocks = current_blocks + [block]

        if current_text_length(candidate_blocks) <= max_chunk_size:
            current_blocks.append(block)
        else:
            if current_blocks:
                chunks.append(make_chunk(current_blocks, chunk_index))
                chunk_index += 1

                # Add overlap only if it does not exceed max size
                overlap_blocks = current_blocks[-overlap_paragraphs:] if overlap_paragraphs > 0 else []

                if current_text_length(overlap_blocks + [block]) <= max_chunk_size:
                    current_blocks = overlap_blocks + [block]
                else:
                    current_blocks = [block]
            else:
                current_blocks = [block]

    if current_blocks:
        final_chunk_text = "\n\n".join(current_blocks).strip()

        if chunks and len(final_chunk_text) < min_chunk_size:
            merged_text = chunks[-1]["text"] + "\n\n" + final_chunk_text

            if len(merged_text) <= max_chunk_size:
                chunks[-1]["text"] = merged_text.strip()
                chunks[-1]["character_count"] = len(chunks[-1]["text"])
                chunks[-1]["block_count"] += len(current_blocks)
            else:
                chunks.append(make_chunk(current_blocks, chunk_index))
        else:
            chunks.append(make_chunk(current_blocks, chunk_index))

    return chunks


def create_semantic_chunks_from_pages(
    pages: list,
    document_name: str,
    max_chunk_size: int = None,
    min_chunk_size: int = None,
    overlap_paragraphs: int = None
):
    all_chunks = []

    for page in pages:
        page_number = page.get("page_number")
        page_text = page.get("text", "")

        page_chunks = create_semantic_chunks_from_text(
            text=page_text,
            page_number=page_number,
            document_name=document_name,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            overlap_paragraphs=overlap_paragraphs
        )

        all_chunks.extend(page_chunks)

    for global_index, chunk in enumerate(all_chunks, start=1):
        chunk["global_chunk_index"] = global_index

    return all_chunks