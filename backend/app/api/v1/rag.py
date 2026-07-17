"""
RAG API router.

The complete RAG endpoints are currently implemented in
app.services.rag_service. This module exposes that router to FastAPI.
"""

from app.services.rag_service import router


__all__ = ["router"]