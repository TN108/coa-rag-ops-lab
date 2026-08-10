import json
import re
import time
from app.services.llm_service import (
    SingleCriticDecision,
    claim_token_coverage,
    evidence_match_token_list,
    extract_question_focus_terms,
    find_direct_evidence_matches,
    get_llm,
    get_token_variants,
    is_claim_relevant,
    normalise_text,
    shared_canonical_bigram_count,
)

from app.agents.state import COAState
from app.services.llm_service import (
    SingleCriticDecision,
    claim_token_coverage,
    comparison_claim_is_relevant,
    detect_question_type,
    evidence_match_token_list,
    extract_question_focus_terms,
    find_direct_evidence_matches,
    get_llm,
    get_token_variants,
    is_claim_relevant,
    normalise_text,
    shared_canonical_bigram_count,
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
    Prevent contradictions such as supported=False while the
    feedback says that the claim is supported.
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
                "The cited evidence did not verify every "
                "material part of the claim."
            )

    return cleaned_feedback


def has_unsupported_qualifiers(
    claim: str,
    evidence_items: list[dict],
) -> bool:
    """
    Detect corporate or organizational qualifiers that appear
    in the claim but not in the cited evidence.

    Example:
    Evidence: LangChain
    Claim: LangChain Inc.
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
            str(claim or "").casefold(),
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


def requires_specific_answer(
    question: str,
) -> bool:
    """
    Return True when the question requests a specific identity,
    value, category, or named item instead of a general explanation.

    Examples:
    - Who created the system?
    - Which database is required?
    - What is the price?
    - How many components are used?
    """

    cleaned_question = " ".join(
        str(question or "")
        .strip()
        .casefold()
        .split()
    )

    if not cleaned_question:
        return False

    specific_prefixes = (
        "who ",
        "when ",
        "where ",
        "which ",
        "how many ",
        "how much ",
    )

    if cleaned_question.startswith(
        specific_prefixes
    ):
        return True

    requested_value_terms = {
        "price",
        "cost",
        "amount",
        "number",
        "percentage",
        "date",
        "year",
        "version",
        "model",
        "database",
        "language",
        "location",
        "name",
        "person",
        "organization",
        "company",
    }

    question_tokens = set(
        re.findall(
            r"\b\w+\b",
            cleaned_question,
        )
    )

    return bool(
        question_tokens
        & requested_value_terms
    )



def claim_matches_question_focus(
    question: str,
    claim: str,
    minimum_coverage: float = 0.50,
) -> bool:
    """
    Check whether a claim addresses the meaningful focus of a question.

    Generic context terms such as framework names and words like
    application or workflow are excluded because they do not identify
    the actual information requested.
    """

    focus_terms = extract_question_focus_terms(
        question
    )

    generic_context_terms = {
        "langgraph",
        "application",
        "applications",
        "workflow",
        "workflows",
        "framework",
        "frameworks",
        "graph",
        "graphs",
        "system",
        "systems",
    }

    meaningful_focus_terms = {
        term
        for term in focus_terms
        if term not in generic_context_terms
    }

    if not meaningful_focus_terms:
        meaningful_focus_terms = focus_terms

    if not meaningful_focus_terms:
        return True

    claim_tokens = set(
        normalise_text(claim).split()
    )

    matched_terms: set[str] = set()

    for term in meaningful_focus_terms:
        variants = get_token_variants(term)

        if variants.intersection(claim_tokens):
            matched_terms.add(term)

    coverage = (
        len(matched_terms)
        / len(meaningful_focus_terms)
    )

    minimum_matches = (
        1
        if len(meaningful_focus_terms) <= 2
        else 2
    )

    return (
        len(matched_terms) >= minimum_matches
        and coverage >= minimum_coverage
    )


def find_joint_evidence_support(
    claim: str,
    evidence_items: list[dict],
) -> list[str]:
    """
    Return cited chunks that jointly support one claim.

    This path is used only when no individual chunk directly
    supports the complete claim. The combined evidence must satisfy
    conservative coverage and phrase-overlap requirements, and at
    least two chunks must independently contribute meaningful terms.
    """

    if len(evidence_items) < 2:
        return []

    combined_text = "\n".join(
        str(item.get("text") or "")
        for item in evidence_items
    )

    claim_tokens = evidence_match_token_list(
        claim
    )

    if len(claim_tokens) < 6:
        return []

    coverage = claim_token_coverage(
        claim=claim,
        evidence_text=combined_text,
    )

    shared_bigram_count = (
        shared_canonical_bigram_count(
            claim=claim,
            evidence_text=combined_text,
        )
    )

    strict_joint_match = (
        coverage >= 0.90
    )

    guarded_joint_match = (
        len(claim_tokens) >= 8
        and coverage >= 0.85
        and shared_bigram_count >= 3
    )

    if not (
        strict_joint_match
        or guarded_joint_match
    ):
        print(
            "Joint evidence check:",
            {
                "claim": claim,
                "coverage": round(
                    coverage,
                    3,
                ),
                "shared_bigram_count":
                    shared_bigram_count,
                "strict_joint_match":
                    strict_joint_match,
                "guarded_joint_match":
                    guarded_joint_match,
                "selected": False,
            },
        )
        return []

    generic_tokens = {
        "langgraph",
        "graph",
        "workflow",
        "application",
        "system",
        "node",
        "function",
    }

    meaningful_claim_tokens = (
        set(claim_tokens)
        - generic_tokens
    )

    contributing_ids: list[str] = []
    contribution_details: dict[str, list[str]] = {}

    for item in evidence_items:
        chunk_id = str(
            item.get("chunk_id") or ""
        ).strip()

        if not chunk_id:
            continue

        chunk_tokens = set(
            evidence_match_token_list(
                str(item.get("text") or "")
            )
        )

        meaningful_matches = sorted(
            meaningful_claim_tokens
            & chunk_tokens
        )

        contribution_details[
            chunk_id
        ] = meaningful_matches

        if len(meaningful_matches) >= 2:
            contributing_ids.append(
                chunk_id
            )

    selected = (
        len(contributing_ids) >= 2
    )

    print(
        "Joint evidence check:",
        {
            "claim": claim,
            "coverage": round(
                coverage,
                3,
            ),
            "shared_bigram_count":
                shared_bigram_count,
            "strict_joint_match":
                strict_joint_match,
            "guarded_joint_match":
                guarded_joint_match,
            "contributions":
                contribution_details,
            "contributing_ids":
                contributing_ids,
            "selected": selected,
        },
    )

    if not selected:
        return []

    return list(
        dict.fromkeys(
            contributing_ids
        )
    )

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
    1. Validate the claim and cited evidence IDs.
    2. Determine question-aware relevance.
    3. Check for deterministic evidence support.
    4. Use the LLM for paraphrased or uncertain support.
    5. Keep evidence support and question relevance separate.
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

    # Existing definition-aware relevance check.
    deterministic_relevance = is_claim_relevant(
        question=question,
        claim=cleaned_claim,
    )

    # Existing general focus check.
    question_focus_match = (
        claim_matches_question_focus(
            question=question,
            claim=cleaned_claim,
        )
    )

    # Day 5: identify the general type of question.
    question_type = detect_question_type(
        question
    )

    # Day 5: comparison questions require comparison meaning,
    # not merely a related fact about the same subject.
    if question_type == "comparison":
        question_aware_relevance = (
            comparison_claim_is_relevant(
                question=question,
                claim=cleaned_claim,
            )
        )
    else:
        question_aware_relevance = (
            question_focus_match
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
            "relevant_to_question": (
                question_aware_relevance
            ),
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
            "relevant_to_question": (
                question_aware_relevance
            ),
            "verified_chunk_ids": [],
            "feedback": (
                "The cited chunks did not contain usable text."
            ),
            "verification_method": "validation",
        }

    specific_question = requires_specific_answer(
        question
    )

    unsupported_qualifiers = (
        has_unsupported_qualifiers(
            claim=cleaned_claim,
            evidence_items=evidence_items,
        )
    )

    definition_question = (
        question.strip()
        .casefold()
        .startswith(
            (
                "what is ",
                "what are ",
                "define ",
            )
        )
    )

    # Definition questions retain the stricter Day 4 relevance rule.
    # All other ordinary questions use question-type-aware relevance.
    if definition_question:
        direct_match_relevance = (
            deterministic_relevance
        )
    else:
        direct_match_relevance = (
            question_aware_relevance
        )

    direct_matches = find_direct_evidence_matches(
        claim=cleaned_claim,
        evidence_chunks=evidence_items,
    )

    joint_matches: list[str] = []

    if not direct_matches:
        joint_matches = find_joint_evidence_support(
            claim=cleaned_claim,
            evidence_items=evidence_items,
        )

    deterministic_support_ids = (
        direct_matches
        or joint_matches
    )

    # A deterministic evidence match establishes support.
    # Relevance remains a separate decision.
    #
    # Specific-answer questions still use the LLM for relevance
    # because a supported statement may fail to supply the requested
    # identity, value, date, number, system, or named item.
    if (
        deterministic_support_ids
        and not specific_question
        and not unsupported_qualifiers
    ):
        if direct_matches:
            support_description = (
                "directly stated in"
            )
            method_prefix = "direct_match"

        else:
            support_description = (
                "jointly supported by"
            )
            method_prefix = (
                "joint_direct_match"
            )

        if direct_match_relevance:
            feedback = (
                f"The claim is {support_description} "
                "the cited evidence."
            )
            verification_method = (
                method_prefix
            )

        else:
            feedback = (
                f"The claim is {support_description} the cited "
                "evidence, but it does not address the main focus "
                "of the question."
            )
            verification_method = (
                f"{method_prefix}_irrelevant"
            )

        return {
            "supported": True,
            "relevant_to_question": (
                direct_match_relevance
            ),
            "verified_chunk_ids": (
                deterministic_support_ids
            ),
            "feedback": feedback,
            "verification_method": verification_method,
        }

    if specific_question:
        relevance_rules = """
12. Evidence support and question relevance are separate decisions.

13. Judge supported using only the evidence. Do not set
    supported=false merely because the claim is irrelevant.

14. Set relevant_to_question=true only when the claim supplies the
    specific identity, value, item, or category requested.

15. A general related statement is not sufficient when the question
    asks for a specific person, organization, date, number, price,
    model, database, language, location, or named item.

16. Example:
    Question: "Which storage system is required?"
    Claim: "Several software dependencies are required."
    The claim may be supported, but relevant_to_question=false
    because it does not identify a storage system.
""".strip()

    else:
        relevance_rules = """
12. This is not a specific-value or specific-identity question.

13. Judge only whether the cited evidence supports the claim.

14. Do not set supported=false because of relevance concerns.

15. Set relevant_to_question=true. Question relevance is handled
    separately by deterministic application logic.
""".strip()

    prompt = f"""
Question:
{question}

Claim:
{cleaned_claim}

Cited evidence set:
{json.dumps(evidence_items, ensure_ascii=False, indent=2)}

Evaluate only this single claim against the complete cited evidence set.

Rules:

1. Set supported=true only when every material part of the claim is
   directly stated or clearly entailed by the cited evidence.

2. Exact wording is not required. Accept conservative paraphrases
   that preserve the same meaning.

3. Accept harmless differences such as:
   - singular versus plural
   - active versus passive voice
   - small grammatical changes
   - shortened wording
   - equivalent verbs
   - equivalent technical terminology

4. Evaluate all cited chunks together. Different chunks may support
   different material parts of the claim.

5. Do not require every cited chunk to independently contain the
   complete claim.

6. Accept clear logical entailments from the supplied evidence.

7. Examples of valid meaning-preserving entailment:
   - Evidence: "The process terminates after validation."
     Claim: "The process ends after validation."

   - Evidence: "Execution returns to an earlier processing stage."
     Claim: "The process can repeat an earlier stage."

   - Evidence: "Links determine movement between stages and may be
     fixed or conditional."
     Claim: "Stages are connected using fixed or conditional links."

   - Evidence: "The terminal state stops further execution."
     Claim: "The process ends when it reaches the terminal state."

8. Reject the claim when it adds unsupported information, including:
   - new names or entities
   - new numbers, dates, prices, or versions
   - new causes or effects
   - new implementation details
   - new systems, products, or technologies
   - stronger guarantees
   - outside knowledge

9. A valid paraphrase changes wording but does not add meaning.

10. Use only the supplied cited evidence set.

11. Do not use question relevance to change the evidence-support
    decision.

{relevance_rules}

17. If supported=true, verified_chunk_ids must contain the exact IDs
    of all supplied chunks that contribute support.

18. If multiple chunks jointly support the claim, include all
    contributing chunk IDs.

19. If supported=false, verified_chunk_ids must be empty.

20. verified_chunk_ids may contain only IDs from the supplied
    evidence set.

21. Feedback must explain the evidence agreement or unsupported
    addition in one concise sentence.
""".strip()

    llm = get_llm()

    decision = llm.generate_structured(
        prompt=prompt,
        response_model=SingleCriticDecision,
        system_prompt=(
            "You are a conservative semantic evidence verifier. "
            "Judge whether the supplied evidence supports every "
            "material part of the claim. "
            "Never use question relevance to change the evidence "
            "support decision. "
            "Judge meaning rather than exact word overlap. "
            "Accept faithful paraphrases, clear entailments, and "
            "evidence distributed across multiple cited chunks. "
            "Reject unsupported additions and outside knowledge. "
            "Follow the supplied relevance instructions exactly. "
            "Copy contributing evidence IDs exactly."
        ),
        num_predict=160,
        max_attempts=1,
    )

    llm_verified_ids = list(
        dict.fromkeys(
            chunk_id
            for chunk_id in (
                decision.verified_chunk_ids or []
            )
            if chunk_id in valid_cited_ids
        )
    )

    # When a specific-answer question has deterministic evidence
    # support, preserve that support and use the LLM only for its
    # relevance decision.
    if (
        deterministic_support_ids
        and specific_question
        and not unsupported_qualifiers
    ):
        supported = True
        verified_chunk_ids = (
            deterministic_support_ids
        )

    else:
        supported = bool(
            decision.supported
            and llm_verified_ids
            and not unsupported_qualifiers
        )

        verified_chunk_ids = (
            llm_verified_ids
            if supported
            else []
        )

    # Specific-answer checking takes priority over ordinary
    # definition checking.
    #
    # Example:
    # "What is the price?" begins with "What is", but requests a
    # particular value rather than a definition.
    if specific_question:
        final_relevance = bool(
            decision.relevant_to_question
        )

    elif definition_question:
        final_relevance = (
            deterministic_relevance
        )

    else:
        # Day 5: comparison questions now use comparison-aware
        # relevance in both deterministic and LLM support paths.
        final_relevance = (
            question_aware_relevance
        )

    feedback = normalise_critic_feedback(
        supported=supported,
        feedback=decision.feedback,
    )

    feedback_lower = feedback.casefold()

    relevance_only_phrases = (
        "not relevant",
        "does not answer",
        "does not provide the type",
        "does not provide information requested",
    )

    # Correct feedback when deterministic relevance logic overrides
    # an unreliable LLM relevance result.
    if (
        supported
        and final_relevance
        and not decision.relevant_to_question
    ):
        feedback = (
            "The claim is supported by the cited evidence "
            "and addresses the question."
        )

    elif (
        supported
        and not final_relevance
    ):
        feedback = (
            "The claim is supported by the cited evidence, "
            "but it does not provide the information requested "
            "by the question."
        )

    elif (
        not supported
        and final_relevance
        and any(
            phrase in feedback_lower
            for phrase in relevance_only_phrases
        )
    ):
        feedback = (
            "The cited evidence did not verify every "
            "material part of the claim."
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
    The LLM is used for paraphrased, uncertain, multi-chunk,
    or specific-answer claims.
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

            if not isinstance(decision, dict):
                raise ValueError(
                    "criticise_single_claim returned no decision."
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
        relevant = result[
            "relevant_to_question"
        ]

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