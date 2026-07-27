# backend/app/api/v1/rag_router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from app.services.rag_service import generate_rag_answer
from app.services.qdrant_service import search_similar_chunks  # <-- use existing function

router = APIRouter()

class RAGQuestionRequest(BaseModel):
    question: str
    top_k: int = 5
    min_retrieval_score: float = 0.1
    chunking_method: str = "semantic"

@router.post("/rag/ask")
def ask_rag(request: RAGQuestionRequest):
    try:
        # Use your existing Qdrant service to get top-k chunks
        # Here we assume you have a function to convert question to embedding
        from app.services.embedding_service import get_embedding_model
        embed_model = get_embedding_model()
        query_vector = embed_model.encode(request.question).tolist()

        retrieved_chunks: List[Dict] = search_similar_chunks(
            query_embedding=query_vector,
            top_k=request.top_k,
            chunking_method=request.chunking_method
        )

        answer = generate_rag_answer(
            question=request.question,
            retrieved_chunks=retrieved_chunks,
            min_score=request.min_retrieval_score
        )

        return {
            "answer": answer,
            "retrieved_chunks_count": len(retrieved_chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))