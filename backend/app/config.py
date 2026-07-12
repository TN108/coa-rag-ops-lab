import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    APP_NAME = os.getenv("APP_NAME", "COA RAG Ops Lab")
    API_VERSION = os.getenv("API_VERSION", "0.1.0")

    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "pdf_chunks")

    # Embedding settings
    EMBEDDING_MODEL_NAME = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

    # OCR settings
    ENABLE_OCR = os.getenv("ENABLE_OCR", "false").lower() == "true"
    TESSERACT_CMD = os.getenv(
        "TESSERACT_CMD",
        "C:/Program Files/Tesseract-OCR/tesseract.exe"
    )
    OCR_DPI = int(os.getenv("OCR_DPI", "400"))

    # Fixed-size chunking settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

    # Lightweight semantic chunking settings
    SEMANTIC_CHUNK_MAX_SIZE = int(os.getenv("SEMANTIC_CHUNK_MAX_SIZE", "1200"))
    SEMANTIC_CHUNK_MIN_SIZE = int(os.getenv("SEMANTIC_CHUNK_MIN_SIZE", "300"))
    SEMANTIC_CHUNK_OVERLAP_PARAGRAPHS = int(
        os.getenv("SEMANTIC_CHUNK_OVERLAP_PARAGRAPHS", "1")
    )


settings = Settings()