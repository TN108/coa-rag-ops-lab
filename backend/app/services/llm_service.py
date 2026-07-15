import re
from typing import Any

import requests
from fastapi import HTTPException

from app.config import settings


FALLBACK_ANSWER = (
    "The provided document context does not contain enough "
    "information to answer this question."
)


def build_rag_prompt(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    context_parts: list[str] = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        text = str(
            chunk.get("text") or ""
        ).strip()

        if not text:
            continue

        context_parts.append(
            "\n".join(
                [
                    f"Source {index}",
                    (
                        "Document: "
                        f"{chunk.get('document_name', 'Unknown')}"
                    ),
                    (
                        "Page: "
                        f"{chunk.get('page_number', 'Unknown')}"
                    ),
                    (
                        "Chunk ID: "
                        f"{chunk.get('chunk_id', 'Unknown')}"
                    ),
                    "Text:",
                    text,
                ]
            )
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    return f"""
You are a document-grounded question-answering assistant.

Use only the retrieved context below.

Instructions:
1. Read all retrieved sources before answering.
2. Combine information from multiple sources when necessary.
3. Answer the question directly and concisely.
4. Do not use outside knowledge.
5. Do not invent facts or unsupported details.
6. Do not refuse only because the answer is not written in one exact sentence.
7. If the context supports a useful partial answer, provide only the supported information.
8. If the context does not support the answer, respond with the exact fallback sentence below and nothing else.
9. Never mention source numbers such as "Source 1", "Source 2", or "according to Source".
10. Do not mention document names, page numbers, or chunk IDs.
11. Do not begin a non-yes-or-no answer with "Yes" or "No".
12. Keep the answer between one and four sentences.
13. For yes-or-no questions, begin with "Yes" or "No" only when the context clearly supports that conclusion.

Exact fallback sentence:
"{FALLBACK_ANSWER}"

Retrieved context:
{context}

Question:
{question}

Answer:
""".strip()


def generate_answer_with_ollama(
    prompt: str,
) -> str:
    try:
        response = requests.post(
            (
                f"{settings.OLLAMA_BASE_URL}"
                "/api/generate"
            ),
            json={
                "model": settings.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "top_p": 0.8,
                    "repeat_penalty": 1.1,
                    "num_predict": 256,
                },
            },
            timeout=180,
        )

    except requests.exceptions.Timeout as error:
        raise HTTPException(
            status_code=504,
            detail=(
                "Ollama request timed out: "
                f"{error}"
            ),
        ) from error

    except requests.exceptions.ConnectionError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not connect to Ollama. "
                "Confirm that Ollama is running and "
                "OLLAMA_BASE_URL is correct."
            ),
        ) from error

    except requests.exceptions.RequestException as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Ollama request failed: "
                f"{error}"
            ),
        ) from error

    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Ollama returned status "
                f"{response.status_code}: "
                f"{response.text}"
            ),
        )

    try:
        data = response.json()
    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Ollama returned an invalid JSON response."
            ),
        ) from error

    return str(
        data.get("response") or ""
    ).strip()


def clean_generated_answer(
    answer: str,
) -> str:
    cleaned = answer.strip()

    if not cleaned:
        return ""

    cleaned = re.sub(
        r"(?i)^according to source \d+[,:]?\s*",
        "",
        cleaned,
    )

    cleaned = re.sub(
        r"(?i)\baccording to source \d+[,:]?\s*",
        "",
        cleaned,
    )

    cleaned = re.sub(
        (
            r"(?i)\bsource \d+\s+"
            r"(states|says|explains|indicates) that\s*"
        ),
        "",
        cleaned,
    )

    cleaned = re.sub(
        r"(?i)^yes\.\s+(?!(?:it|the context|langgraph)\b)",
        "",
        cleaned,
    )

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    )

    return cleaned.strip()


def generate_rag_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    cleaned_question = question.strip()

    if not cleaned_question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    valid_chunks = [
        chunk
        for chunk in retrieved_chunks
        if str(
            chunk.get("text") or ""
        ).strip()
    ]

    if not valid_chunks:
        return FALLBACK_ANSWER

    prompt = build_rag_prompt(
        question=cleaned_question,
        retrieved_chunks=valid_chunks,
    )

    answer = generate_answer_with_ollama(
        prompt
    )

    if not answer:
        return FALLBACK_ANSWER

    cleaned_answer = clean_generated_answer(
        answer
    )

    if not cleaned_answer:
        return FALLBACK_ANSWER

    return cleaned_answer