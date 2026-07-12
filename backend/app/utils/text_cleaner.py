import re


def clean_text(text: str) -> str:
    if not text:
        return ""

    # Normalize line breaks
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove too many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces around newlines
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


def clean_chunk_text(text: str) -> str:
    if not text:
        return ""

    text = clean_text(text)

    # Convert internal newlines into spaces for cleaner chunks
    text = re.sub(r"\s+", " ", text)

    return text.strip()