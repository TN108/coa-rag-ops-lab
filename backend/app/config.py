import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    APP_NAME = os.getenv("APP_NAME", "COA RAG Ops Lab")
    API_VERSION = os.getenv("API_VERSION", "0.1.0")

    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "pdf_chunks")

    EMBEDDING_MODEL_NAME = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    ENABLE_OCR = os.getenv("ENABLE_OCR", "false").lower() == "true"
    TESSERACT_CMD = os.getenv(
        "TESSERACT_CMD",
        "C:/Program Files/Tesseract-OCR/tesseract.exe"
    )
    OCR_DPI = int(os.getenv("OCR_DPI", "200"))


settings = Settings()