# backend/app/services/llm_service.py
import re
import json
import requests
from functools import lru_cache
from fastapi import HTTPException
from app.config import settings

FALLBACK_ANSWER = (
    "The provided document context does not contain enough "
    "information to answer this question."
)

# Cached LLM client for COA
@lru_cache(maxsize=1)
def get_llm():
    """
    Returns a cached Llama 3.2:3b client (Ollama API wrapper).
    """
    return Llama3Client(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL
    )

class Llama3Client:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {
                        "temperature": 0,
                        "top_p": 0.8,
                        "repeat_penalty": 1.1,
                        "num_predict": 80,
                    },
                },
                timeout=180,
            )
        except requests.exceptions.Timeout as error:
            raise HTTPException(status_code=504, detail=f"Ollama request timed out: {error}")
        except requests.exceptions.ConnectionError as error:
            raise HTTPException(
                status_code=503,
                detail="Could not connect to Ollama. Confirm that Ollama is running and OLLAMA_BASE_URL is correct."
            )
        except requests.exceptions.RequestException as error:
            raise HTTPException(status_code=500, detail=f"Ollama request failed: {error}")

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Ollama returned status {response.status_code}: {response.text}")

        try:
            data = response.json()
        except ValueError as error:
            raise HTTPException(status_code=500, detail="Ollama returned an invalid JSON response.") from error

        return str(data.get("response") or "").strip()


def clean_generated_answer(answer: str) -> str:
    cleaned = answer.strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"(?i)^according to source \d+[,:]?\s*", "", cleaned)
    cleaned = re.sub(r"(?i)\baccording to source \d+[,:]?\s*", "", cleaned)
    cleaned = re.sub(r"(?i)\bsource \d+\s+(states|says|explains|indicates) that\s*", "", cleaned)
    cleaned = re.sub(r"(?i)^yes\.\s+(?!(?:it|the context|langgraph)\b)", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def generate_rag_answer(question: str, retrieved_chunks: list[dict]) -> str:
    cleaned_question = question.strip()
    if not cleaned_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    valid_chunks = [c for c in retrieved_chunks if str(c.get("text") or "").strip()]
    if not valid_chunks:
        return FALLBACK_ANSWER

    # Build prompt for LLM
    context_parts = []
    for idx, chunk in enumerate(valid_chunks, 1):
        text = str(chunk.get("text") or "").strip()
        if text:
            context_parts.append(
                "\n".join([
                    f"Document: {chunk.get('document_name','Unknown')}",
                    f"Page: {chunk.get('page_number','Unknown')}",
                    f"Chunk ID: {chunk.get('chunk_id','Unknown')}",
                    "Text:",
                    text
                ])
            )
    context = "\n\n---\n\n".join(context_parts)
    prompt = f"""
You are a document-grounded question-answering assistant.
Use only the retrieved context below.
Return a JSON array of facts with each claim and source chunk IDs.
Fallback answer if context is insufficient:
"{FALLBACK_ANSWER}"
Context:
{context}
Question:
{cleaned_question}
""".strip()

    llm = get_llm()
    answer = llm.generate(prompt)
    cleaned_answer = clean_generated_answer(answer)

    return cleaned_answer or FALLBACK_ANSWER