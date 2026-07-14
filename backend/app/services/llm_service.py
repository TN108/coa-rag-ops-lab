import requests
from fastapi import HTTPException

from app.config import settings


def build_rag_prompt(question: str, retrieved_chunks: list) -> str:
    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"Source {index}:\n"
            f"Document: {chunk.get('document_name')}\n"
            f"Page: {chunk.get('page_number')}\n"
            f"Chunk ID: {chunk.get('chunk_id')}\n"
            f"Text:\n{chunk.get('text')}\n"
        )

    context = "\n\n".join(context_parts)

    return f"""
You are a strict document-grounded RAG assistant.

Answer the question using only the provided context.
Do not use outside knowledge.
Do not infer facts that are not explicitly stated in the context.
Do not guess.
Do not add explanations that are not supported by the context.
Answer directly and briefly.

Do not say "according to Source 1", "according to Source 2", or similar source-number phrases.
The API already returns sources separately.

If the context only partially answers the question, state only what the context says and mention that no further detail is provided.

If the answer is not present in the context, say exactly:
"The provided document context does not contain enough information to answer this question."

Context:
{context}

Question:
{question}

Answer:
""".strip()


def generate_answer_with_ollama(prompt: str) -> str:
    try:
        response = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": settings.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "top_p": 0.8,
                    "repeat_penalty": 1.1
                }
            },
            timeout=120
        )
    except requests.exceptions.RequestException as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not connect to Ollama: {str(error)}"
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Ollama error: {response.text}"
        )

    data = response.json()
    return data.get("response", "").strip()


def generate_rag_answer(question: str, retrieved_chunks: list) -> str:
    if not retrieved_chunks:
        return "No relevant document chunks were retrieved."

    prompt = build_rag_prompt(
        question=question,
        retrieved_chunks=retrieved_chunks
    )

    answer = generate_answer_with_ollama(prompt)

    if not answer:
        return "The model did not generate an answer."

    return answer