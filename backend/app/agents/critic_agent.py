# backend/app/agents/critic_agent.py

import json
import time

from app.agents.state import COAState
from app.services.llm_service import (
    SingleCriticDecision,
    find_direct_evidence_matches,
    get_llm,
    is_claim_relevant,
)


def build_evidence_items(
    cited_chunk_ids: list[str],
    chunk_map: dict[str, dict],
) -> list[dict]:
    """
    Build evidence objects for one claim using only valid cited chunks.
    """

    evidence_items: list[dict] = []

    for chunk_id in cited_chunk_ids:
        chunk = chunk_map.get(chunk_id)

        if not chunk:
            continue

        evidence_items.append(
            {
                "chunk_id": chunk_id,
                "document_name": chunk.get(
                    "document_name",
                    "Unknown",
                ),
                "page_number": chunk.get(
                    "page_number",
                    "Unknown",
                ),
                "text": str(
                    chunk.get("text") or ""
                ).strip(),
            }
        )

    return evidence_items


def normalise_critic_feedback(
    *,
    supported: bool,
    feedback: str,
) -> str:
    """
    Prevent contradictions such as:
    supported=False with feedback saying the claim is supported.
    """

    cleaned_feedback = " ".join(
        str(feedback or "").strip().split()
    )

    if not cleaned_feedback:
        return (
            "The claim was verified against its cited evidence."
            if supported
            else "The cited evidence did not verify the claim."
        )

    feedback_lower = cleaned_feedback.casefold()

    if not supported:
        contradictory_phrases = (
            "is supported",
            "directly supports",
            "clearly supports",
            "supports the claim",
        )

        if any(
            phrase in feedback_lower
            for phrase in contradictory_phrases
        ):
            return (
                "The cited evidence was not accepted as direct "
                "support for this claim."
            )

    return cleaned_feedback


def criticise_single_claim(
    *,
    question: str,
    claim: str,
    cited_chunk_ids: list[str],
    chunk_map: dict[str, dict],
) -> dict:
    """
    Verify one claim against only its cited evidence.

    Verification order:
    1. Validate the claim and cited chunk IDs.
    2. Check for a direct deterministic evidence match.
    3. Use the LLM only when the claim is a paraphrase or uncertain.
    """

    cleaned_claim = " ".join(
        str(claim or "").strip().split()
    )

    valid_cited_ids = list(
        dict.fromkeys(
            chunk_id
            for chunk_id in cited_chunk_ids
            if chunk_id in chunk_map
        )
    )

    relevance = is_claim_relevant(
        question=question,
        claim=cleaned_claim,
    )

    if not cleaned_claim:
        return {
            "supported": False,
            "relevant_to_question": False,
            "verified_chunk_ids": [],
            "feedback": "The claim is empty.",
            "verification_method": "validation",
        }

    if not valid_cited_ids:
        return {
            "supported": False,
            "relevant_to_question": relevance,
            "verified_chunk_ids": [],
            "feedback": (
                "No valid cited evidence was available "
                "for this claim."
            ),
            "verification_method": "validation",
        }

    evidence_items = build_evidence_items(
        cited_chunk_ids=valid_cited_ids,
        chunk_map=chunk_map,
    )

    if not evidence_items:
        return {
            "supported": False,
            "relevant_to_question": relevance,
            "verified_chunk_ids": [],
            "feedback": (
                "The cited chunks did not contain usable text."
            ),
            "verification_method": "validation",
        }

    # Fast path: approve exact or near-exact claims without an LLM call.
    direct_matches = find_direct_evidence_matches(
        claim=cleaned_claim,
        evidence_chunks=evidence_items,
    )

    if direct_matches:
        return {
            "supported": True,
            "relevant_to_question": relevance,
            "verified_chunk_ids": direct_matches,
            "feedback": (
                "The claim is directly stated in the cited evidence."
            ),
            "verification_method": "direct_match",
        }

    # Slow path: use the LLM only for paraphrased or uncertain claims.
    prompt = f"""
Question:
{question}

Claim:
{cleaned_claim}

Cited evidence:
{json.dumps(evidence_items, ensure_ascii=False, indent=2)}

Evaluate only this single claim.

Rules:
1. Compare only the supplied claim with the supplied evidence.
2. Do not consider any other claims.
3. supported=true only when the supplied evidence directly states or clearly entails the claim.
4. Do not use outside knowledge.
5. Do not rescue the claim using uncited evidence.
6. Do not reject a claim merely because the evidence contains additional information.
7. relevant_to_question=true only if the claim directly helps answer the question.
8. For "what is X" questions, definitions and central characteristics of X are relevant.
9. Components, configuration files, commands, implementation details, and examples are normally irrelevant to a definition question.
10. If supported=true, verified_chunk_ids must contain at least one supplied chunk ID copied exactly.
11. If supported=false, verified_chunk_ids must be empty.
12. verified_chunk_ids may contain only IDs present in the supplied evidence.
13. Feedback must explain the exact agreement or mismatch in one concise sentence.
14. Every material part of the claim must be supported.
15. Reject added names, titles, qualifiers, versions, dates, numbers, or company suffixes that are absent from the evidence.
16. Do not combine separate bullets or sentences into a stronger claim.
17. Preserve entity names exactly as written in the cited evidence.
18. If any material part is unsupported, set supported=false.
""".strip()

    llm = get_llm()

    decision = llm.generate_structured(
        prompt=prompt,
        response_model=SingleCriticDecision,
        system_prompt=(
            "Verify one claim against only its cited evidence. "
    "Every material part of the claim must be supported. "
    "Reject added specificity, entity qualifiers, and company "
    "suffixes that are absent from the evidence. "
    "When supported is true, copy at least one supplied "
    "chunk ID exactly into verified_chunk_ids."
        ),
        num_predict=128,
        max_attempts=1,
    )

    verified_chunk_ids = list(
        dict.fromkeys(
            chunk_id
            for chunk_id in decision.verified_chunk_ids
            if chunk_id in valid_cited_ids
        )
    )

    supported = bool(
        decision.supported
        and verified_chunk_ids
    )

    if not supported:
        verified_chunk_ids = []

    feedback = normalise_critic_feedback(
        supported=supported,
        feedback=decision.feedback,
    )

    # Prefer deterministic relevance for definition questions.
    # For non-definition questions, use the critic's decision.
    final_relevance = (
        relevance
        if question.strip().casefold().startswith(
            ("what is ", "what are ", "define ")
        )
        else bool(decision.relevant_to_question)
    )

    return {
        "supported": supported,
        "relevant_to_question": final_relevance,
        "verified_chunk_ids": verified_chunk_ids,
        "feedback": feedback,
        "verification_method": "llm",
    }


def critic_agent(
    state: COAState,
) -> COAState:
    """
    Verify every structured reasoning claim independently.

    Direct evidence matches are handled without an LLM call.
    The LLM is used only for paraphrased or uncertain claims.
    """

    print("Running Critic Agent")

    question = str(
        state.get("question") or ""
    ).strip()

    structured_facts = (
        state.get("structured_facts") or []
    )

    retrieved_chunks = (
        state.get("retrieved_chunks") or []
    )

    state.setdefault("latency", {})

    if not question:
        state["critic_results"] = []
        state["critic_feedback"] = (
            "The critic received an empty question."
        )
        state["latency"]["critic_ms"] = 0.0
        return state

    if not structured_facts:
        state["critic_results"] = []
        state["critic_feedback"] = (
            "No structured facts were available "
            "for verification."
        )
        state["latency"]["critic_ms"] = 0.0
        return state

    chunk_map = {
        str(chunk.get("chunk_id")): chunk
        for chunk in retrieved_chunks
        if chunk.get("chunk_id")
    }

    start = time.perf_counter()

    critic_results: list[dict] = []

    for claim_index, fact in enumerate(
        structured_facts
    ):
        claim = str(
            fact.get("claim") or ""
        ).strip()

        cited_chunk_ids = [
            str(chunk_id)
            for chunk_id in (
                fact.get("evidence_chunk_ids") or []
            )
        ]

        try:
            decision = criticise_single_claim(
                question=question,
                claim=claim,
                cited_chunk_ids=cited_chunk_ids,
                chunk_map=chunk_map,
            )

        except Exception as error:
            decision = {
                "supported": False,
                "relevant_to_question": False,
                "verified_chunk_ids": [],
                "feedback": (
                    "The critic could not verify this claim: "
                    f"{error}"
                ),
                "verification_method": "error",
            }

        critic_results.append(
            {
                "claim_index": claim_index,
                "claim": claim,
                "supported": bool(
                    decision["supported"]
                ),
                "relevant_to_question": bool(
                    decision["relevant_to_question"]
                ),
                "verified_chunk_ids": decision[
                    "verified_chunk_ids"
                ],
                "feedback": decision["feedback"],
                "verification_method": decision[
                    "verification_method"
                ],
            }
        )

    critic_ms = (
        time.perf_counter() - start
    ) * 1000

    feedback_lines: list[str] = []

    for result in critic_results:
        supported = result["supported"]
        relevant = result["relevant_to_question"]

        if supported and relevant:
            verdict = "approved"
        elif supported and not relevant:
            verdict = "supported but irrelevant"
        else:
            verdict = "rejected"

        feedback_lines.append(
            f"Claim {result['claim_index'] + 1} "
            f"{verdict}: {result['feedback']}"
        )

    state["critic_results"] = critic_results

    state["critic_feedback"] = "\n".join(
        feedback_lines
    )

    state["latency"]["critic_ms"] = round(
        critic_ms,
        2,
    )

    return state
def has_unsupported_qualifiers(
    claim: str,
    evidence_items: list[dict],
) -> bool:
    """
    Detect qualifiers that appear in the claim but not in its evidence.

    This prevents changes such as:
    LangChain -> LangChain Inc.
    """

    qualifier_tokens = {
        "inc",
        "incorporated",
        "ltd",
        "limited",
        "llc",
        "corporation",
        "corp",
        "plc",
    }

    claim_tokens = set(
        re.findall(
            r"\b\w+\b",
            claim.casefold(),
        )
    )

    evidence_text = " ".join(
        str(item.get("text") or "")
        for item in evidence_items
    ).casefold()

    evidence_tokens = set(
        re.findall(
            r"\b\w+\b",
            evidence_text,
        )
    )

    added_qualifiers = (
        claim_tokens
        & qualifier_tokens
        - evidence_tokens
    )

    return bool(added_qualifiers)