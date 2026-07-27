# backend/app/services/rag_service.py
import re
from fastapi import HTTPException
from app.config import settings
from app.services.embedding_service import get_embedding_model
from app.services.reranker_service import get_reranker
from app.services.llm_service import get_llm  # only for Llama 3.2:3b if needed

FALLBACK_ANSWER = "The provided document context does not contain enough information to answer this question."

def check_unanswerable(retrieved_chunks, min_score=0.1):
    if not retrieved_chunks or max(c["score"] for c in retrieved_chunks) < min_score:
        return True
    return False

def validate_rag_answer(answer: str, retrieved_chunks: list[dict]) -> str:
    supported = any(chunk["text"].strip() in answer for chunk in retrieved_chunks)
    return answer if supported else FALLBACK_ANSWER

def build_rag_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context_parts = []
    for idx, chunk in enumerate(retrieved_chunks, 1):
        text = chunk.get("text", "").strip()
        if text:
            context_parts.append(
                f"Document: {chunk.get('document_name', 'Unknown')}\n"
                f"Page: {chunk.get('page_number', 'Unknown')}\n"
                f"Chunk ID: {chunk.get('chunk_id', 'Unknown')}\n"
                f"Text: {text}"
            )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are a document-grounded question-answering assistant.
Use only the retrieved context below. Do not hallucinate.
If the context does not provide enough information, respond exactly:
"{FALLBACK_ANSWER}"

Retrieved context:
{context}

Question:
{question}

Answer concisely, only based on the retrieved evidence.
""".strip()
    return prompt

def generate_rag_answer(question: str, retrieved_chunks: list[dict], min_score: float = 0.1) -> str:
    cleaned_question = question.strip()
    if not cleaned_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    valid_chunks = [c for c in retrieved_chunks if c.get("text", "").strip()]

    if check_unanswerable(valid_chunks, min_score):
        return FALLBACK_ANSWER

    prompt = build_rag_prompt(cleaned_question, valid_chunks)

    # Use Llama 3.2:3b via get_llm if needed
    llm = get_llm(model_name="llama3.2:3b")
    response = llm.generate(prompt)  # replaces generate_answer_with_ollama

    validated_answer = validate_rag_answer(response, valid_chunks)
    return validated_answer