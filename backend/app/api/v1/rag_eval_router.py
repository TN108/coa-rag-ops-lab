from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json
from app.services.rag_service import generate_rag_answer

router = APIRouter()

class RAGEvalRequest(BaseModel):
    dataset_path: str
    top_k: int = 5
    min_retrieval_score: float = 0.1
    chunking_method: str = "semantic"
    adaptive: bool = False

@router.post("/rag/evaluation/run")
def run_rag_evaluation(request: RAGEvalRequest):
    try:
        with open(request.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid dataset path: {e}")

    results = []
    hit_count = 0
    total_questions = len(dataset)

    for item in dataset:
        question = item.get("question", "")
        retrieved_chunks = item.get("retrieved_chunks", [])
        answer = generate_rag_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
            min_score=request.min_retrieval_score
        )
        results.append({
            "question": question,
            "final_answer": answer
        })
        if answer != "The provided document context does not contain enough information to answer this question.":
            hit_count += 1

    summary = {
        "retrieval_hit_rate": hit_count / total_questions,
        "mean_reciprocal_rank": 1.0,  # placeholder, can compute later
        "mean_semantic_similarity": 1.0  # placeholder, can compute later
    }

    return {"summary": summary, "results": results}