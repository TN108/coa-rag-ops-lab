# backend/app/services/llm_service.py
from collections import Counter
import json
import re
from functools import lru_cache
from typing import Type, TypeVar

import requests
from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError

from app.config import settings


FALLBACK_ANSWER = (
    "The provided document context does not contain enough "
    "information to answer this question."
)

T = TypeVar("T", bound=BaseModel)


# ============================================================
# Structured COA schemas
# ============================================================

class EvidenceClaim(BaseModel):
    """One factual claim extracted from retrieved evidence."""

    claim: str = Field(min_length=1)
    evidence_chunk_ids: list[str] = Field(min_length=1)


class ReasoningOutput(BaseModel):
    """Structured output produced by the reasoning stage."""

    claims: list[EvidenceClaim] = Field(default_factory=list)


class SingleCriticDecision(BaseModel):
    """Structured critic output for one claim."""

    supported: bool
    relevant_to_question: bool
    verified_chunk_ids: list[str] = Field(default_factory=list)
    feedback: str = Field(min_length=1)


# Retained for compatibility with older critic code.
class CriticDecision(BaseModel):
    claim_index: int = Field(ge=0)
    supported: bool
    relevant_to_question: bool
    verified_chunk_ids: list[str] = Field(default_factory=list)
    feedback: str = Field(min_length=1)


class CriticOutput(BaseModel):
    decisions: list[CriticDecision] = Field(default_factory=list)


# ============================================================
# Ollama client
# ============================================================

class Llama3Client:
    """Wrapper around Ollama's /api/generate endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session = requests.Session()

    def _request(
        self,
        payload: dict,
        timeout: int = 180,
    ) -> dict:
        """Send one request to Ollama and return its JSON body."""

        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout,
            )

        except requests.exceptions.Timeout as error:
            raise HTTPException(
                status_code=504,
                detail=f"Ollama request timed out: {error}",
            ) from error

        except requests.exceptions.ConnectionError as error:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Could not connect to Ollama. Confirm that Ollama "
                    "is running and OLLAMA_BASE_URL is correct."
                ),
            ) from error

        except requests.exceptions.RequestException as error:
            raise HTTPException(
                status_code=500,
                detail=f"Ollama request failed: {error}",
            ) from error

        if response.status_code != 200:
            try:
                error_data = response.json()
                ollama_error = (
                    error_data.get("error")
                    or response.text
                )
            except ValueError:
                ollama_error = response.text

            raise HTTPException(
                status_code=500,
                detail=(
                    f"Ollama returned status "
                    f"{response.status_code}: {ollama_error}"
                ),
            )

        try:
            return response.json()

        except ValueError as error:
            raise HTTPException(
                status_code=500,
                detail="Ollama returned an invalid JSON response.",
            ) from error

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        num_predict: int = 256,
        timeout: int = 180,
    ) -> str:
        """Generate a normal plain-text response."""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0,
                "top_p": 0.8,
                "repeat_penalty": 1.1,
                "num_predict": num_predict,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        data = self._request(
            payload=payload,
            timeout=timeout,
        )

        return str(
            data.get("response") or ""
        ).strip()

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        *,
        system_prompt: str | None = None,
        num_predict: int = 220,
        max_attempts: int = 1,
        timeout: int = 180,
    ) -> T:
        """Generate JSON constrained by a Pydantic schema."""

        schema = response_model.model_json_schema()

        schema_text = json.dumps(
            schema,
            ensure_ascii=False,
        )

        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            retry_instruction = ""

            if attempt > 1:
                retry_instruction = """
The previous response did not match the required schema.

Return exactly one complete JSON object.
The object must match the supplied JSON schema.
Do not use Markdown code fences.
Do not add explanations before or after the JSON.
""".strip()

            structured_prompt = f"""
{prompt}

Required JSON schema:
{schema_text}

{retry_instruction}
""".strip()

            payload = {
                "model": self.model,
                "prompt": structured_prompt,
                "stream": False,
                "format": schema,
                "keep_alive": "10m",
                "options": {
                    "temperature": 0,
                    "top_p": 0.8,
                    "repeat_penalty": 1.05,
                    "num_predict": num_predict,
                },
            }

            if system_prompt:
                payload["system"] = system_prompt

            data = self._request(
                payload=payload,
                timeout=timeout,
            )

            raw_output = str(
                data.get("response") or ""
            ).strip()

            if not raw_output:
                last_error = ValueError(
                    "Ollama returned an empty response."
                )
                continue

            try:
                return response_model.model_validate_json(
                    raw_output
                )

            except (ValidationError, ValueError) as error:
                last_error = error

        raise HTTPException(
            status_code=502,
            detail=(
                "Ollama did not return valid structured output "
                f"after {max_attempts} attempts. "
                f"Validation error: {last_error}"
            ),
        )


# ============================================================
# Cached Ollama client
# ============================================================

@lru_cache(maxsize=1)
def get_llm() -> Llama3Client:
    """Return one cached Ollama client."""

    return Llama3Client(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
    )


# ============================================================
# Question and text helpers
# ============================================================

def normalise_text(text: str) -> str:
    """Normalise text for deterministic comparisons."""

    cleaned = str(text or "").casefold()
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()


def clean_question_prefix(
    question: str,
) -> str:
    """
    Remove optional list numbering from a question.

    Examples:
    - "3. Can LangGraph..." -> "Can LangGraph..."
    - "4) What is..." -> "What is..."
    """

    return re.sub(
        r"^\s*\d+\s*[\.\)\-:]\s*",
        "",
        str(question or ""),
    ).strip()


def is_definition_question(
    question: str,
) -> bool:
    """Detect simple definition questions."""

    cleaned = normalise_text(
        clean_question_prefix(question)
    )

    return (
        cleaned.startswith("what is ")
        or cleaned.startswith("what are ")
        or cleaned.startswith("define ")
    )


def extract_definition_subject(
    question: str,
) -> str:
    """
    Extract X from:
    - What is X?
    - What are X?
    - Define X.
    """

    cleaned = normalise_text(
        clean_question_prefix(question)
    )

    for prefix in (
        "what is ",
        "what are ",
        "define ",
    ):
        if cleaned.startswith(prefix):
            return cleaned[len(prefix):].strip()

    return ""


def is_yes_no_question(
    question: str,
) -> bool:
    """Detect questions that should normally have one direct answer."""

    cleaned = normalise_text(
        clean_question_prefix(question)
    )

    starters = (
        "can ",
        "could ",
        "do ",
        "does ",
        "did ",
        "is ",
        "are ",
        "was ",
        "were ",
        "has ",
        "have ",
        "had ",
        "will ",
        "would ",
        "should ",
    )

    return cleaned.startswith(starters)


def is_authorship_question(
    question: str,
) -> bool:
    """Detect questions about an author, creator, or builder."""

    cleaned = normalise_text(
        clean_question_prefix(question)
    )

    patterns = (
        "who authored ",
        "who created ",
        "who built ",
        "who developed ",
        "who wrote ",
        "who made ",
    )

    return cleaned.startswith(patterns)


def is_claim_relevant(
    question: str,
    claim: str,
) -> bool:
    """
    Check whether a claim directly answers the question.

    For definition questions, the claim should normally begin with
    the subject being defined.
    """

    cleaned_question = clean_question_prefix(
        question
    )

    if not is_definition_question(cleaned_question):
        return True

    subject = extract_definition_subject(
        cleaned_question
    )

    if not subject:
        return True

    claim_tokens = normalise_text(claim).split()
    subject_tokens = subject.split()

    if not claim_tokens or not subject_tokens:
        return False

    if claim_tokens[0] == "the":
        claim_tokens = claim_tokens[1:]

    if claim_tokens[:len(subject_tokens)] != subject_tokens:
        return False

    next_index = len(subject_tokens)

    if next_index >= len(claim_tokens):
        return True

    allowed_next_tokens = {
        "is",
        "are",
        "was",
        "were",
        "can",
        "models",
        "provides",
        "enables",
        "supports",
        "uses",
        "allows",
        "helps",
        "represents",
        "orchestrates",
        "manages",
        "builds",
    }

    return (
        claim_tokens[next_index]
        in allowed_next_tokens
    )


# ============================================================
# Deterministic evidence matching
# ============================================================

def content_token_list(
    text: str,
) -> list[str]:
    """
    Return meaningful tokens in their original order.

    Grammatical filler is ignored so that a heading followed by a
    bullet can support a natural complete sentence.
    """

    ignored_tokens = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "in",
        "of",
        "to",
        "and",
        "or",
        "that",
        "this",
        "it",
        "command",
    }

    return [
        token
        for token in normalise_text(text).split()
        if token not in ignored_tokens
    ]


def is_ordered_subsequence(
    claim_tokens: list[str],
    evidence_tokens: list[str],
) -> bool:
    """
    Check whether all claim tokens occur in the evidence in the
    same order, allowing unrelated tokens between them.
    """

    if not claim_tokens:
        return False

    evidence_index = 0

    for claim_token in claim_tokens:
        while (
            evidence_index < len(evidence_tokens)
            and evidence_tokens[evidence_index] != claim_token
        ):
            evidence_index += 1

        if evidence_index >= len(evidence_tokens):
            return False

        evidence_index += 1

    return True


def claim_token_coverage(
    claim: str,
    evidence_text: str,
) -> float:
    """
    Calculate conservative token coverage using token counts.

    This allows small grammatical paraphrases while still rejecting
    claims that add substantial unsupported information.
    """

    claim_tokens = content_token_list(claim)
    evidence_tokens = content_token_list(evidence_text)

    if not claim_tokens:
        return 0.0

    claim_counts = Counter(claim_tokens)
    evidence_counts = Counter(evidence_tokens)

    matched_tokens = sum(
        min(
            count,
            evidence_counts.get(token, 0),
        )
        for token, count in claim_counts.items()
    )

    return matched_tokens / len(claim_tokens)


def find_direct_evidence_matches(
    claim: str,
    evidence_chunks: list[dict],
) -> list[str]:
    """
    Return chunks that directly or nearly directly state a claim.

    Matching methods:
    1. Exact normalised substring.
    2. Ordered meaningful-token match.
    3. Conservative high token coverage.
    """

    normalised_claim = normalise_text(claim)
    claim_tokens = content_token_list(claim)

    if not normalised_claim or not claim_tokens:
        return []

    matched_ids: list[str] = []

    for chunk in evidence_chunks:
        chunk_id = str(
            chunk.get("chunk_id") or ""
        ).strip()

        raw_chunk_text = str(
            chunk.get("text") or ""
        ).strip()

        normalised_chunk_text = normalise_text(
            raw_chunk_text
        )

        if not chunk_id or not normalised_chunk_text:
            continue

        exact_match = (
            normalised_claim in normalised_chunk_text
        )

        ordered_match = (
            len(claim_tokens) >= 4
            and is_ordered_subsequence(
                claim_tokens=claim_tokens,
                evidence_tokens=content_token_list(
                    raw_chunk_text
                ),
            )
        )

        coverage = claim_token_coverage(
            claim=claim,
            evidence_text=raw_chunk_text,
        )

        high_coverage_match = (
            len(claim_tokens) >= 6
            and coverage >= 0.90
        )

        if (
            exact_match
            or ordered_match
            or high_coverage_match
        ):
            matched_ids.append(chunk_id)

    return list(
        dict.fromkeys(matched_ids)
    )


def filter_question_relevant_claims(
    question: str,
    reasoning_output: ReasoningOutput,
) -> ReasoningOutput:
    """Filter structurally irrelevant definition claims."""

    if not is_definition_question(question):
        return reasoning_output

    relevant_claims = [
        item
        for item in reasoning_output.claims
        if is_claim_relevant(
            question=question,
            claim=item.claim,
        )
    ]

    return ReasoningOutput(
        claims=relevant_claims
    )


# ============================================================
# Chunk helpers
# ============================================================

def get_valid_chunks(
    retrieved_chunks: list[dict],
    min_characters: int = 80,
) -> list[dict]:
    """
    Return substantive chunks.

    Very short title-only and metadata-only chunks are removed.
    If all chunks are short, the non-empty chunks are retained so
    the system does not discard all available evidence.
    """

    non_empty_chunks = [
        chunk
        for chunk in retrieved_chunks
        if str(chunk.get("text") or "").strip()
    ]

    substantive_chunks = [
        chunk
        for chunk in non_empty_chunks
        if len(
            str(chunk.get("text") or "").strip()
        ) >= min_characters
    ]

    return substantive_chunks or non_empty_chunks


def select_reasoning_chunks(
    retrieved_chunks: list[dict],
    question: str,
    limit: int = 3,
) -> list[dict]:
    """
    Select strong chunks for reasoning.

    Uses reranker score first, with question-specific deterministic
    boosts for definitions and authorship questions.
    """

    cleaned_question = clean_question_prefix(
        question
    )

    valid_chunks = get_valid_chunks(
        retrieved_chunks
    )

    definition_subject = extract_definition_subject(
        cleaned_question
    )

    authorship_question = is_authorship_question(
        cleaned_question
    )

    def chunk_priority(chunk: dict) -> float:
        base_score = float(
            chunk.get(
                "reranker_score",
                chunk.get(
                    "retrieval_score",
                    chunk.get("score", 0.0),
                ),
            )
            or 0.0
        )

        text = normalise_text(
            str(chunk.get("text") or "")
        )

        priority = base_score

        if definition_subject:
            if f"{definition_subject} is " in text:
                priority += 10.0

            if f"{definition_subject} was " in text:
                priority += 5.0

            if f"{definition_subject} models " in text:
                priority += 5.0

            if text.startswith(definition_subject):
                priority += 2.0

        if authorship_question:
            explicit_authorship_phrases = (
                "authored by",
                "created by",
                "built by",
                "developed by",
                "written by",
            )

            if any(
                phrase in text
                for phrase in explicit_authorship_phrases
            ):
                priority += 12.0

            if "©" in str(chunk.get("text") or ""):
                priority -= 1.0

        return priority

    sorted_chunks = sorted(
        valid_chunks,
        key=chunk_priority,
        reverse=True,
    )

    return sorted_chunks[:limit]


def build_context(
    retrieved_chunks: list[dict],
) -> str:
    """Convert retrieved chunks into grounded text context."""

    context_parts: list[str] = []

    for chunk in retrieved_chunks:
        text = str(
            chunk.get("text") or ""
        ).strip()

        if not text:
            continue

        context_parts.append(
            "\n".join(
                [
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

    return "\n\n---\n\n".join(
        context_parts
    )


# ============================================================
# Plain-text answer cleaning
# ============================================================

def clean_generated_answer(
    answer: str,
) -> str:
    """
    Clean plain-text LLM answers.

    Do not use this function for structured JSON responses.
    """

    cleaned = str(answer or "").strip()

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
        r"\s+",
        " ",
        cleaned,
    )

    return cleaned.strip()


# ============================================================
# Baseline RAG answer
# ============================================================

def generate_rag_answer(
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    """Generate a normal natural-language baseline RAG answer."""

    cleaned_question = clean_question_prefix(
        question
    )

    if not cleaned_question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    valid_chunks = get_valid_chunks(
        retrieved_chunks
    )

    if not valid_chunks:
        return FALLBACK_ANSWER

    context = build_context(
        valid_chunks
    )

    prompt = f"""
Retrieved context:
{context}

Question:
{cleaned_question}

Answer the question using only the retrieved context.

Rules:
1. Return a direct natural-language answer.
2. Do not return JSON.
3. Do not mention chunk IDs.
4. Do not mention the retrieval pipeline.
5. Do not use outside information.
6. Keep the answer concise but complete.
7. Ignore copyright notices, presenters, slide authors, headers,
   footers, and watermarks unless the question asks about the document.
8. If the context does not directly answer the question, return exactly:
{FALLBACK_ANSWER}
""".strip()

    llm = get_llm()

    raw_answer = llm.generate(
        prompt=prompt,
        system_prompt=(
            "You are a document-grounded question-answering "
            "assistant. Use only the supplied context."
        ),
        num_predict=256,
    )

    cleaned_answer = clean_generated_answer(
        raw_answer
    )

    return cleaned_answer or FALLBACK_ANSWER


def parse_authorship_question(
    question: str,
) -> tuple[str, str] | None:
    """
    Extract the requested relationship and subject.

    Example:
    "Who authored LangGraph?"
    -> ("authored", "LangGraph")
    """

    cleaned = clean_question_prefix(
        question
    ).strip()

    match = re.match(
        (
            r"(?i)^who\s+"
            r"(authored|created|built|developed|wrote|made)\s+"
            r"(.+?)\s*\??$"
        ),
        cleaned,
    )

    if not match:
        return None

    relationship = match.group(1).casefold()

    subject = match.group(2).strip(
        " .?"
    )

    return relationship, subject


def extract_explicit_authorship_fact(
    question: str,
    retrieved_chunks: list[dict],
) -> ReasoningOutput | None:
    """
    Answer authorship questions using explicit evidence phrases.

    This prevents combining:
    "authored by LangChain"
    with:
    "built by LangChain Inc."
    """

    parsed_question = parse_authorship_question(
        question
    )

    if not parsed_question:
        return None

    relationship, subject = parsed_question

    relation_patterns = {
        "authored": r"\bauthored by\s+([^•\n\r;]+)",
        "created": r"\bcreated by\s+([^•\n\r;]+)",
        "built": r"\bbuilt by\s+([^•\n\r;]+)",
        "developed": r"\bdeveloped by\s+([^•\n\r;]+)",
        "wrote": r"\bwritten by\s+([^•\n\r;]+)",
        "made": r"\bmade by\s+([^•\n\r;]+)",
    }

    claim_phrases = {
        "authored": "was authored by",
        "created": "was created by",
        "built": "was built by",
        "developed": "was developed by",
        "wrote": "was written by",
        "made": "was made by",
    }

    pattern = relation_patterns.get(
        relationship
    )

    claim_phrase = claim_phrases.get(
        relationship
    )

    if not pattern or not claim_phrase:
        return None

    subject_normalised = normalise_text(
        subject
    )

    sorted_chunks = sorted(
        get_valid_chunks(retrieved_chunks),
        key=lambda chunk: float(
            chunk.get(
                "reranker_score",
                chunk.get(
                    "retrieval_score",
                    chunk.get("score", 0.0),
                ),
            )
            or 0.0
        ),
        reverse=True,
    )

    for chunk in sorted_chunks:
        raw_text = str(
            chunk.get("text") or ""
        ).strip()

        chunk_id = str(
            chunk.get("chunk_id") or ""
        ).strip()

        if not raw_text or not chunk_id:
            continue

        if (
            subject_normalised
            not in normalise_text(raw_text)
        ):
            continue

        match = re.search(
            pattern,
            raw_text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        entity = match.group(1).strip()

        # Stop before another sentence or explanatory clause.
        entity = re.split(
            r"(?i)(?:\s+(?:but|and|which|who|that)\s+|,)",
            entity,
            maxsplit=1,
        )[0]

        entity = entity.rstrip(
            " .,;:"
        )

        if not entity:
            continue

        claim = (
            f"{subject} {claim_phrase} {entity}."
        )

        return ReasoningOutput(
            claims=[
                EvidenceClaim(
                    claim=claim,
                    evidence_chunk_ids=[
                        chunk_id
                    ],
                )
            ]
        )

    return ReasoningOutput(
        claims=[]
    )



# ============================================================
# Deterministic CLI command extraction
# ============================================================

CLI_COMMAND_RULES: dict[str, dict[str, object]] = {
    "langgraph dev": {
        "claim": (
             "The langgraph dockerfile command emits a Dockerfile "
        "derived from your config for custom builds."
        ),
        "preceding_phrases": (
            "starts a lightweight local dev server ideal for rapid testing",
        ),
        "following_phrases": (
            "this in memory server is designed for development and testing",
        ),
    },
    "langgraph build": {
        "claim": (
            "The langgraph build command builds a Docker image of "
            "your LangGraph API server for deployment."
        ),
        "preceding_phrases": (
            "builds a docker image of your langgraph api server for deployment",
        ),
        "following_phrases": (),
    },
    "langgraph deploy": {
        "claim": (
            "The langgraph deploy command builds and deploys a "
            "LangGraph image directly to LangSmith Deployments."
        ),
        "preceding_phrases": (
            "builds and deploys a langgraph image directly to "
            "langsmith deployments in a single step",
        ),
        "following_phrases": (),
    },
    "langgraph dockerfile": {
        "claim": (
           "The langgraph dockerfile command emits a Dockerfile "
        "derived from your config for custom builds."
        ),
        "preceding_phrases": (
            "emits a dockerfile derived from your config for custom builds",
        ),
        "following_phrases": (),
    },
    "langgraph up": {
        "claim": (
            "The langgraph up command starts the LangGraph API "
            "server locally in Docker."
        ),
        "preceding_phrases": (
            "starts the langgraph api server locally in docker",
        ),
        "following_phrases": (),
    },
}


def parse_cli_command_question(
    question: str,
) -> str | None:
    """
    Extract the requested CLI command.

    Example:
    "What does the langgraph dev command do?"
    -> "langgraph dev"
    """

    cleaned = normalise_text(
        clean_question_prefix(question)
    )

    match = re.fullmatch(
        r"what does (?:the )?(.+?) command do",
        cleaned,
    )

    if not match:
        return None

    command = match.group(1).strip()

    return command or None


def get_cli_command_positions(
    normalised_text: str,
) -> list[tuple[int, int, str]]:
    """
    Find known CLI command positions in flattened extracted text.
    """

    positions: list[tuple[int, int, str]] = []

    for command in CLI_COMMAND_RULES:
        for match in re.finditer(
            rf"\b{re.escape(command)}\b",
            normalised_text,
        ):
            positions.append(
                (
                    match.start(),
                    match.end(),
                    command,
                )
            )

    positions.sort(
        key=lambda item: item[0]
    )

    return positions


def get_preceding_cli_description(
    normalised_text: str,
    command: str,
) -> str:
    """
    Return the text between the previous command and the target
    command.

    Flattened PDF tables often store each description immediately
    before its command. Restricting the evidence to this segment
    avoids assigning a neighbouring row's description to the
    requested command.
    """

    positions = get_cli_command_positions(
        normalised_text
    )

    target_position: tuple[int, int, str] | None = None

    for position in positions:
        if position[2] == command:
            target_position = position
            break

    if target_position is None:
        return ""

    target_start = target_position[0]
    previous_end = 0

    for start, end, _ in positions:
        if end <= target_start:
            previous_end = max(
                previous_end,
                end,
            )

    return normalised_text[
        previous_end:target_start
    ].strip()


def get_following_cli_description(
    normalised_text: str,
    command: str,
) -> str:
    """
    Return text after the target command until the next known
    command. This supports explicit command-output examples.
    """

    positions = get_cli_command_positions(
        normalised_text
    )

    target_position: tuple[int, int, str] | None = None

    for position in positions:
        if position[2] == command:
            target_position = position
            break

    if target_position is None:
        return ""

    target_end = target_position[1]
    next_start = len(normalised_text)

    for start, _, _ in positions:
        if start > target_end:
            next_start = min(
                next_start,
                start,
            )

    return normalised_text[
        target_end:next_start
    ].strip()


def extract_explicit_cli_command_fact(
    question: str,
    retrieved_chunks: list[dict],
) -> ReasoningOutput | None:
    """
    Deterministically answer supported CLI command questions.

    The extractor validates the description associated with the
    requested command rather than accepting any description found
    in the same flattened table chunk.
    """

    command = parse_cli_command_question(
        question
    )

    if not command:
        return None

    rule = CLI_COMMAND_RULES.get(
        command
    )

    if rule is None:
        return None

    claim = str(
        rule["claim"]
    )

    preceding_phrases = tuple(
        str(phrase)
        for phrase in rule.get(
            "preceding_phrases",
            (),
        )
    )

    following_phrases = tuple(
        str(phrase)
        for phrase in rule.get(
            "following_phrases",
            (),
        )
    )

    ranked_chunks = sorted(
        get_valid_chunks(retrieved_chunks),
        key=lambda chunk: float(
            chunk.get(
                "reranker_score",
                chunk.get(
                    "retrieval_score",
                    chunk.get("score", 0.0),
                ),
            )
            or 0.0
        ),
        reverse=True,
    )

    for chunk in ranked_chunks:
        raw_text = str(
            chunk.get("text") or ""
        ).strip()

        chunk_id = str(
            chunk.get("chunk_id") or ""
        ).strip()

        if not raw_text or not chunk_id:
            continue

        normalised_chunk = normalise_text(
            raw_text
        )

        if command not in normalised_chunk:
            continue

        preceding_text = (
            get_preceding_cli_description(
                normalised_text=normalised_chunk,
                command=command,
            )
        )

        following_text = (
            get_following_cli_description(
                normalised_text=normalised_chunk,
                command=command,
            )
        )

        preceding_match = any(
            phrase in preceding_text
            for phrase in preceding_phrases
        )

        following_match = any(
            phrase in following_text
            for phrase in following_phrases
        )

        if not (
            preceding_match
            or following_match
        ):
            continue

        return ReasoningOutput(
            claims=[
                EvidenceClaim(
                    claim=claim,
                    evidence_chunk_ids=[
                        chunk_id
                    ],
                )
            ]
        )

    return ReasoningOutput(
        claims=[]
    )

# ============================================================
# Early answerability gate
# ============================================================

# These thresholds are intentionally separated because reranker
# logits and embedding retrieval scores use different scales.
#
# CrossEncoder reranker scores are raw logits. A score around 0.0
# is treated as usable evidence, while 4.0 is treated as strong
# evidence. Retrieval scores are cosine-like similarities in this
# pipeline, so they use their own thresholds.
REASONING_GATE_MIN_RERANKER_SCORE = 0.0
REASONING_GATE_STRONG_RERANKER_SCORE = 4.0
REASONING_GATE_MIN_RETRIEVAL_SCORE = 0.35
REASONING_GATE_STRONG_RETRIEVAL_SCORE = 0.50
REASONING_GATE_MIN_FOCUS_COVERAGE = 0.50
REASONING_GATE_DEBUG = True


def safe_optional_float(
    value: object,
) -> float | None:
    """Convert a score to float without turning missing data into 0."""

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_question_focus_terms(
    question: str,
) -> set[str]:
    """
    Extract question-specific content terms.

    Function words, broad relationship words, and generic instruction
    words are removed. The remaining terms are used only as one gate
    signal; exact keyword overlap is no longer the sole requirement.
    """

    ignored_words = {
        "what",
        "which",
        "who",
        "where",
        "when",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "has",
        "have",
        "had",
        "will",
        "to",
        "a",
        "an",
        "the",
        "for",
        "of",
        "in",
        "on",
        "at",
        "by",
        "from",
        "with",
        "and",
        "or",
        "as",
        "into",
        "inside",
        "between",
        "through",
        "across",
        "during",
        "before",
        "after",
        "another",
        "each",
        "every",
        "run",
        "required",
        "require",
        "requires",
        "used",
        "use",
        "uses",
        "using",
        "type",
        "kind",
        "kinds",
        "information",
        "provide",
        "provides",
        "provided",
        "explain",
        "describe",
        "designed",
        "manage",
        "manages",
        "managed",
        "system",
        "systems",
        "inspired",
        "first",
        "much",
        "internally",
        "command",
        "commands",
        "main",
        "important",
        "role",
        "way",
    }

    return {
        token
        for token in normalise_text(
            clean_question_prefix(question)
        ).split()
        if (
            token not in ignored_words
            and len(token) >= 2
        )
    }


def get_token_variants(
    token: str,
) -> set[str]:
    """
    Return conservative morphological variants for one token.

    This handles small wording changes such as:
    - checkpoint/checkpoints
    - store/stored/storing
    - node/nodes

    It does not attempt broad semantic synonym expansion.
    """

    cleaned = normalise_text(token)

    if not cleaned:
        return set()

    variants = {cleaned}

    if len(cleaned) > 4 and cleaned.endswith("ies"):
        variants.add(
            f"{cleaned[:-3]}y"
        )

    if len(cleaned) > 4 and cleaned.endswith("ing"):
        stem = cleaned[:-3]
        variants.add(stem)

        if stem:
            variants.add(
                f"{stem}e"
            )

    if len(cleaned) > 3 and cleaned.endswith("ed"):
        stem = cleaned[:-2]
        variants.add(stem)

        if stem:
            variants.add(
                f"{stem}e"
            )

    if len(cleaned) > 4 and cleaned.endswith("es"):
        variants.add(
            cleaned[:-2]
        )

    if len(cleaned) > 3 and cleaned.endswith("s"):
        variants.add(
            cleaned[:-1]
        )

    return {
        variant
        for variant in variants
        if len(variant) >= 2
    }


def get_focus_term_match_details(
    question: str,
    retrieved_chunks: list[dict],
) -> dict:
    """
    Measure focus-term overlap between a question and selected evidence.

    The result is diagnostic and includes:
    - extracted focus terms;
    - matched and missing terms;
    - total coverage.

    Matching supports conservative morphological variants instead of
    requiring every original token to occur exactly.
    """

    focus_terms = extract_question_focus_terms(
        question
    )

    evidence_tokens = set(
        normalise_text(
            " ".join(
                str(chunk.get("text") or "")
                for chunk in retrieved_chunks
            )
        ).split()
    )

    matched_terms: set[str] = set()

    for term in focus_terms:
        variants = get_token_variants(term)

        if variants.intersection(evidence_tokens):
            matched_terms.add(term)

    missing_terms = focus_terms - matched_terms

    coverage = (
        len(matched_terms) / len(focus_terms)
        if focus_terms
        else 1.0
    )

    return {
        "focus_terms": sorted(focus_terms),
        "matched_terms": sorted(matched_terms),
        "missing_terms": sorted(missing_terms),
        "matched_count": len(matched_terms),
        "focus_term_count": len(focus_terms),
        "coverage": round(coverage, 4),
    }


def evidence_contains_focus_terms(
    question: str,
    retrieved_chunks: list[dict],
    minimum_coverage: float = (
        REASONING_GATE_MIN_FOCUS_COVERAGE
    ),
) -> bool:
    """
    Return True when enough question-specific terms occur in evidence.

    The old implementation required every focus term to appear
    exactly. That rejected valid paraphrases. The corrected version
    treats term overlap as one signal and accepts partial but
    meaningful overlap.
    """

    details = get_focus_term_match_details(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    if details["focus_term_count"] == 0:
        return True

    return (
        details["matched_count"] >= 1
        and details["coverage"] >= minimum_coverage
    )


def get_reasoning_score_details(
    retrieved_chunks: list[dict],
) -> dict:
    """
    Return reranker and retrieval signals without mixing their scales.

    A chunk may contain both scores. Missing reranker values do not
    suppress valid retrieval scores.
    """

    chunk_scores: list[dict] = []
    reranker_scores: list[float] = []
    retrieval_scores: list[float] = []

    for chunk in retrieved_chunks:
        reranker_score = safe_optional_float(
            chunk.get("reranker_score")
        )

        retrieval_score = safe_optional_float(
            chunk.get(
                "retrieval_score",
                chunk.get("score"),
            )
        )

        if reranker_score is not None:
            reranker_scores.append(
                reranker_score
            )

        if retrieval_score is not None:
            retrieval_scores.append(
                retrieval_score
            )

        chunk_scores.append(
            {
                "chunk_id": str(
                    chunk.get("chunk_id") or ""
                ),
                "reranker_score": reranker_score,
                "retrieval_score": retrieval_score,
            }
        )

    return {
        "best_reranker_score": (
            max(reranker_scores)
            if reranker_scores
            else None
        ),
        "best_retrieval_score": (
            max(retrieval_scores)
            if retrieval_scores
            else None
        ),
        "chunk_scores": chunk_scores,
    }


def get_best_reasoning_score(
    retrieved_chunks: list[dict],
) -> float:
    """
    Return the best available score for backward compatibility.

    Reranker score is preferred when present. Otherwise the best
    retrieval score is returned. Gate thresholding must use
    get_reasoning_score_details() so the two scales stay separate.
    """

    details = get_reasoning_score_details(
        retrieved_chunks
    )

    best_reranker_score = details[
        "best_reranker_score"
    ]

    if best_reranker_score is not None:
        return float(
            best_reranker_score
        )

    best_retrieval_score = details[
        "best_retrieval_score"
    ]

    if best_retrieval_score is not None:
        return float(
            best_retrieval_score
        )

    return 0.0


def get_chunk_focus_match_count(
    focus_terms: set[str],
    chunk_text: str,
) -> int:
    """
    Count focus terms matched inside one chunk.

    This supports the combined gate rule: a moderate score should
    only be trusted when the same selected evidence also contains at
    least one question-specific term.
    """

    chunk_tokens = set(
        normalise_text(
            chunk_text
        ).split()
    )

    return sum(
        1
        for term in focus_terms
        if get_token_variants(term).intersection(
            chunk_tokens
        )
    )


def build_reasoning_gate_chunk_details(
    question: str,
    retrieved_chunks: list[dict],
) -> list[dict]:
    """Build per-chunk score and focus-match diagnostics."""

    focus_terms = extract_question_focus_terms(
        question
    )

    details: list[dict] = []

    for chunk in retrieved_chunks:
        reranker_score = safe_optional_float(
            chunk.get("reranker_score")
        )

        retrieval_score = safe_optional_float(
            chunk.get(
                "retrieval_score",
                chunk.get("score"),
            )
        )

        focus_match_count = (
            get_chunk_focus_match_count(
                focus_terms=focus_terms,
                chunk_text=str(
                    chunk.get("text") or ""
                ),
            )
        )

        details.append(
            {
                "chunk_id": str(
                    chunk.get("chunk_id") or ""
                ),
                "reranker_score": reranker_score,
                "retrieval_score": retrieval_score,
                "focus_match_count": focus_match_count,
            }
        )

    return details


def print_reasoning_gate_decision(
    diagnostics: dict,
) -> None:
    """Print one compact, machine-readable Day 2 gate trace."""

    print(
        "Reasoning gate decision:",
        json.dumps(
            diagnostics,
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def should_skip_reasoning_llm(
    question: str,
    retrieved_chunks: list[dict],
    *,
    minimum_reranker_score: float = (
        REASONING_GATE_MIN_RERANKER_SCORE
    ),
    strong_reranker_score: float = (
        REASONING_GATE_STRONG_RERANKER_SCORE
    ),
    minimum_retrieval_score: float = (
        REASONING_GATE_MIN_RETRIEVAL_SCORE
    ),
    strong_retrieval_score: float = (
        REASONING_GATE_STRONG_RETRIEVAL_SCORE
    ),
    minimum_focus_coverage: float = (
        REASONING_GATE_MIN_FOCUS_COVERAGE
    ),
    debug: bool = REASONING_GATE_DEBUG,
) -> bool:
    """
    Decide whether to skip the reasoning LLM.

    Reasoning is allowed when at least one strong evidence signal
    exists:

    1. Enough question-specific terms occur in the selected evidence.
    2. A reranker score is independently strong.
    3. A retrieval score is independently strong.
    4. A chunk has at least one focus-term match and a usable
       reranker or retrieval score.

    This keeps the gate active for clearly unsupported questions
    while allowing semantically relevant paraphrases to reach the
    reasoning LLM.
    """

    cleaned_question = clean_question_prefix(
        question
    )

    if not retrieved_chunks:
        if debug:
            print_reasoning_gate_decision(
                {
                    "question": cleaned_question,
                    "reason": "no_retrieved_chunks",
                    "skip_reasoning": True,
                }
            )

        return True

    special_case = (
        is_yes_no_question(cleaned_question)
        or is_authorship_question(cleaned_question)
        or parse_cli_command_question(
            cleaned_question
        ) is not None
    )

    if special_case:
        if debug:
            print_reasoning_gate_decision(
                {
                    "question": cleaned_question,
                    "reason": "special_question_type",
                    "skip_reasoning": False,
                }
            )

        return False

    focus_details = get_focus_term_match_details(
        question=cleaned_question,
        retrieved_chunks=retrieved_chunks,
    )

    score_details = get_reasoning_score_details(
        retrieved_chunks
    )

    chunk_details = (
        build_reasoning_gate_chunk_details(
            question=cleaned_question,
            retrieved_chunks=retrieved_chunks,
        )
    )

    focus_signal = (
        focus_details["focus_term_count"] == 0
        or (
            focus_details["matched_count"] >= 1
            and focus_details["coverage"]
            >= minimum_focus_coverage
        )
    )

    strong_reranker_signal = any(
        item["reranker_score"] is not None
        and item["reranker_score"]
        >= strong_reranker_score
        for item in chunk_details
    )

    strong_retrieval_signal = any(
        item["retrieval_score"] is not None
        and item["retrieval_score"]
        >= strong_retrieval_score
        for item in chunk_details
    )

    combined_chunk_signal = any(
        item["focus_match_count"] >= 1
        and (
            (
                item["reranker_score"] is not None
                and item["reranker_score"]
                >= minimum_reranker_score
            )
            or (
                item["retrieval_score"] is not None
                and item["retrieval_score"]
                >= minimum_retrieval_score
            )
        )
        for item in chunk_details
    )

    should_run_reasoning = (
        focus_signal
        or strong_reranker_signal
        or strong_retrieval_signal
        or combined_chunk_signal
    )

    skip_reasoning = not should_run_reasoning

    if debug:
        print_reasoning_gate_decision(
            {
                "question": cleaned_question,
                "focus_terms": focus_details[
                    "focus_terms"
                ],
                "matched_terms": focus_details[
                    "matched_terms"
                ],
                "missing_terms": focus_details[
                    "missing_terms"
                ],
                "focus_coverage": focus_details[
                    "coverage"
                ],
                "best_reranker_score": score_details[
                    "best_reranker_score"
                ],
                "best_retrieval_score": score_details[
                    "best_retrieval_score"
                ],
                "focus_signal": focus_signal,
                "strong_reranker_signal":
                    strong_reranker_signal,
                "strong_retrieval_signal":
                    strong_retrieval_signal,
                "combined_chunk_signal":
                    combined_chunk_signal,
                "skip_reasoning": skip_reasoning,
                "chunk_signals": chunk_details,
            }
        )

    return skip_reasoning

# ============================================================
# COA reasoning stage
# ============================================================

def generate_coa_facts(
    question: str,
    retrieved_chunks: list[dict],
    max_chunks: int = 3,
) -> ReasoningOutput:
    """Extract evidence-grounded structured claims for COA."""

    cleaned_question = clean_question_prefix(
        question
    )

    if not cleaned_question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    yes_no_question = is_yes_no_question(
        cleaned_question
    )

    authorship_question = is_authorship_question(
        cleaned_question
    )

    cli_command = parse_cli_command_question(
        cleaned_question
    )

    claim_limit = (
        1
        if (
            yes_no_question
            or authorship_question
            or cli_command is not None
        )
        else 3
    )

    reasoning_chunks = select_reasoning_chunks(
        retrieved_chunks=retrieved_chunks,
        question=cleaned_question,
        limit=max_chunks,
    )

    if not reasoning_chunks:
        return ReasoningOutput(
            claims=[]
        )

    # Authorship questions use deterministic extraction instead of
    # an LLM call. Run this before the general answerability gate.
    if authorship_question:
        deterministic_authorship = (
            extract_explicit_authorship_fact(
                question=cleaned_question,
                retrieved_chunks=reasoning_chunks,
            )
        )

        if deterministic_authorship is not None:
            return deterministic_authorship

    # CLI questions inspect all retrieved chunks because explicit
    # command-output evidence may rank outside max_chunks.
    if cli_command is not None:
        deterministic_cli_result = (
            extract_explicit_cli_command_fact(
                question=cleaned_question,
                retrieved_chunks=retrieved_chunks,
            )
        )

        if deterministic_cli_result is not None:
            return deterministic_cli_result

    # Skip the reasoning LLM when important question terms are
    # absent and retrieval confidence is weak.
    if should_skip_reasoning_llm(
        question=cleaned_question,
        retrieved_chunks=reasoning_chunks,
    ):
        return ReasoningOutput(
            claims=[]
        )

    context = build_context(
        reasoning_chunks
    )

    definition_subject = extract_definition_subject(
        cleaned_question
    )

    definition_instruction = ""

    if definition_subject:
        definition_instruction = f"""
This is a definition question about: {definition_subject}

For every returned claim:
- the claim must begin with "{definition_subject}";
- the claim must define it or state one central characteristic;
- do not return claims whose subject is a component, subgraph, file,
  command, dependency, company, example, or another entity.
""".strip()

    yes_no_instruction = ""

    if yes_no_question:
        yes_no_instruction = """
This is a yes-or-no question.

Return exactly one claim when the evidence answers the question.
The claim must directly answer the question in a complete sentence.
Do not return restatements, paraphrases, background details,
ownership details, or duplicate claims.
""".strip()

    authorship_instruction = ""

    if authorship_question:
        authorship_instruction = """
This is an authorship question.

Return exactly one claim when the evidence explicitly identifies
the requested author, creator, developer, writer, or builder.

Ignore copyright notices, document owners, slide authors,
presenters, watermarks, headers, and footers.
Do not infer product authorship from document ownership.
Prefer explicit phrases such as "authored by", "created by",
"developed by", or "built by".
""".strip()

    cli_instruction = ""

    if cli_command is not None:
        cli_instruction = f"""
This question asks specifically about the "{cli_command}" command.

Return exactly one claim.
Use only the description directly associated with that command.
Do not use the description of a neighbouring command from a
flattened table.
""".strip()

    prompt = f"""
Retrieved evidence:
{context}

Question:
{cleaned_question}

{definition_instruction}

{yes_no_instruction}

{authorship_instruction}

{cli_instruction}

Extract only factual claims that directly help answer the question.

Rules:
1. Every claim must be a complete sentence.
2. Every claim must be directly supported by retrieved evidence.
3. Include only chunk IDs that directly support that specific claim.
4. Never invent, shorten, rewrite, or modify a chunk ID.
5. Do not attach every retrieved chunk to every claim.
6. Do not include critic comments.
7. Do not produce a final conversational answer.
8. Return no more than {claim_limit} claim(s).
9. If the evidence does not directly answer the question, return an empty claims list.
10. Do not convert a statement about one framework, product, or company into a statement about another.
11. A company name, dependency name, ownership reference, or authorship reference is not a definition.
12. Do not use outside knowledge.
13. For "what is X" questions, extract claims that define X or state a central characteristic of X.
14. Exclude configuration files, CLI commands, examples, and component details unless required to answer the question.
15. Prioritise the most direct answer from the highest-ranked evidence.
16. Keep every claim concise.
17. Preserve entity names, titles, qualifiers, and company suffixes exactly as written in the specific cited evidence.
18. Do not add company suffixes such as Inc., Ltd., or Corporation unless the cited evidence includes them.
19. Prefer copying directly supported wording rather than strengthening or expanding it.
20. Do not combine details from different chunks into one claim unless every cited chunk supports its part.
21. When a claim is copied or closely paraphrased from one chunk, cite only that chunk.
22. Do not return two claims that express the same fact using different wording.
23. Ignore copyright notices, watermarks, slide creators, presenters, headers, footers, and document-owner names unless the question explicitly asks about the document.
24. Do not treat a name following a copyright symbol as the author of a framework, library, product, or technology.
25. For authorship questions, use only evidence that explicitly links the subject to an author, creator, developer, writer, or builder.
""".strip()

    llm = get_llm()

    reasoning_output = llm.generate_structured(
        prompt=prompt,
        response_model=ReasoningOutput,
        system_prompt=(
            "You are the evidence-extraction stage of a "
            "document-grounded Chain-of-Agents pipeline. "
            "Extract only directly supported and question-relevant claims."
        ),
        num_predict=220,
        max_attempts=1,
    )

    cleaned_output = remove_invalid_evidence_ids(
        reasoning_output=reasoning_output,
        retrieved_chunks=reasoning_chunks,
        max_claims=claim_limit,
    )

    return filter_question_relevant_claims(
        question=cleaned_question,
        reasoning_output=cleaned_output,
    )
# ============================================================
# Evidence validation and repair
# ============================================================

def remove_invalid_evidence_ids(
    reasoning_output: ReasoningOutput,
    retrieved_chunks: list[dict],
    max_claims: int | None = None,
) -> ReasoningOutput:
    """
    Clean reasoning claims and repair evidence IDs.

    Direct deterministic matches replace incorrect model-selected
    evidence IDs. Claims with a direct match are preferred when the
    question type limits the number of returned claims.
    """

    valid_chunk_ids = {
        str(chunk.get("chunk_id"))
        for chunk in retrieved_chunks
        if chunk.get("chunk_id")
    }

    candidate_claims: list[
        tuple[int, int, EvidenceClaim]
    ] = []

    seen_claims: set[str] = set()

    for original_index, item in enumerate(
        reasoning_output.claims
    ):
        cleaned_claim = " ".join(
            str(item.claim or "").strip().split()
        )

        if not cleaned_claim:
            continue

        normalised_claim = normalise_text(
            cleaned_claim
        )

        if normalised_claim in seen_claims:
            continue

        valid_evidence_ids = list(
            dict.fromkeys(
                chunk_id
                for chunk_id in item.evidence_chunk_ids
                if chunk_id in valid_chunk_ids
            )
        )

        direct_evidence_ids = find_direct_evidence_matches(
            claim=cleaned_claim,
            evidence_chunks=retrieved_chunks,
        )

        if direct_evidence_ids:
            valid_evidence_ids = direct_evidence_ids
            support_priority = 2
        elif valid_evidence_ids:
            support_priority = 1
        else:
            continue

        seen_claims.add(
            normalised_claim
        )

        candidate_claims.append(
            (
                support_priority,
                original_index,
                EvidenceClaim(
                    claim=cleaned_claim,
                    evidence_chunk_ids=valid_evidence_ids,
                ),
            )
        )

    candidate_claims.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )

    cleaned_claims = [
        item[2]
        for item in candidate_claims
    ]

    if max_claims is not None:
        cleaned_claims = cleaned_claims[
            :max(0, max_claims)
        ]

    return ReasoningOutput(
        claims=cleaned_claims
    )