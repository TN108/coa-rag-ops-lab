import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    APP_NAME = os.getenv(
        "APP_NAME",
        "COA RAG Ops Lab",
    )
    API_VERSION = os.getenv(
        "API_VERSION",
        "0.1.0",
    )

    # Qdrant settings
    QDRANT_HOST = os.getenv(
        "QDRANT_HOST",
        "localhost",
    )
    QDRANT_PORT = int(
        os.getenv(
            "QDRANT_PORT",
            "6333",
        )
    )

    QDRANT_FIXED_COLLECTION = os.getenv(
        "QDRANT_FIXED_COLLECTION",
        "pdf_chunks_fixed",
    )

    QDRANT_SEMANTIC_COLLECTION = os.getenv(
        "QDRANT_SEMANTIC_COLLECTION",
        "pdf_chunks_semantic",
    )

    # Default chunking/index strategy:
    # fixed or semantic
    CHUNKING_METHOD = os.getenv(
        "CHUNKING_METHOD",
        "semantic",
    ).strip().lower()

    # Embedding settings
    EMBEDDING_MODEL_NAME = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    EMBEDDING_BATCH_SIZE = int(
        os.getenv(
            "EMBEDDING_BATCH_SIZE",
            "16",
        )
    )

    EMBEDDING_DIMENSION = int(
        os.getenv(
            "EMBEDDING_DIMENSION",
            "384",
        )
    )

    # OCR settings
    ENABLE_OCR = (
        os.getenv(
            "ENABLE_OCR",
            "false",
        ).lower()
        == "true"
    )

    TESSERACT_CMD = os.getenv(
        "TESSERACT_CMD",
        (
            "C:/Program Files/"
            "Tesseract-OCR/"
            "tesseract.exe"
        ),
    )

    OCR_DPI = int(
        os.getenv(
            "OCR_DPI",
            "300",
        )
    )

    # Fixed-size chunking settings
    CHUNK_SIZE = int(
        os.getenv(
            "CHUNK_SIZE",
            "1000",
        )
    )

    CHUNK_OVERLAP = int(
        os.getenv(
            "CHUNK_OVERLAP",
            "200",
        )
    )

    # Lightweight semantic chunking settings
    SEMANTIC_CHUNK_MAX_SIZE = int(
        os.getenv(
            "SEMANTIC_CHUNK_MAX_SIZE",
            "1200",
        )
    )

    SEMANTIC_CHUNK_MIN_SIZE = int(
        os.getenv(
            "SEMANTIC_CHUNK_MIN_SIZE",
            "300",
        )
    )

    SEMANTIC_CHUNK_OVERLAP_PARAGRAPHS = int(
        os.getenv(
            "SEMANTIC_CHUNK_OVERLAP_PARAGRAPHS",
            "1",
        )
    )

    # LLM settings
    OLLAMA_BASE_URL = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )

    OLLAMA_MODEL = os.getenv(
        "OLLAMA_MODEL",
        "llama3.2:3b",
    )
    
    PLANNER_MODEL = os.getenv(
        "PLANNER_MODEL",
        "llama3.2:1b",
    )
     # Reranker settings
    RERANKER_MODEL_NAME = os.getenv(
        "RERANKER_MODEL_NAME",
        "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    )

    # Evaluation settings
    RETRIEVAL_TOP_K = int(
        os.getenv(
            "RETRIEVAL_TOP_K",
            "5",
        )
    )

    def get_collection_name(
        self,
        chunking_method: str | None = None,
    ) -> str:
        """
        Return the Qdrant collection corresponding to the
        selected chunking strategy.

        When chunking_method is not provided, the default
        CHUNKING_METHOD value from the environment is used.
        """

        method = (
            chunking_method
            or self.CHUNKING_METHOD
        ).strip().lower()

        if method == "fixed":
            return self.QDRANT_FIXED_COLLECTION

        if method == "semantic":
            return self.QDRANT_SEMANTIC_COLLECTION

        raise ValueError(
            "chunking_method must be either "
            "'fixed' or 'semantic'."
        )

    @property
    def active_qdrant_collection(self) -> str:
        """
        Return the collection for the default chunking
        strategy configured in the environment.
        """

        return self.get_collection_name(
            self.CHUNKING_METHOD
        )


settings = Settings()