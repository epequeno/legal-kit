"""Stage 8: Citation verification & confidence scoring.

Four independent checks on every claim-citation pair:
1. Citation existence — does the cited section exist?
2. Quote verification — does quoted text match the source?
3. Entailment verification — does the source support the claim?
4. Completeness check — are relevant sections missing?
"""

from __future__ import annotations

import re

from legalkit.llm.client import LLMClient
from legalkit.llm.prompts import ENTAILMENT_SYSTEM, ENTAILMENT_USER
from legalkit.models.citation import (
    Claim,
    Citation,
    VerifiedCitation,
    VerificationStatus,
    EntailmentClass,
)
from legalkit.models.document import DocumentNode
from legalkit.models.report import ConfidenceLevel, ConfidenceScore, VerifiedClaim
from legalkit.models.retrieval import RankedChunk
from legalkit.utils.text import fuzzy_match_score, find_best_match, normalize_section_id
from legalkit.pipeline.nli_model import LocalNLIModel


def verify_claims(
    claims: list[Claim],
    doc_tree: DocumentNode,
    ranked_chunks: list[RankedChunk],
    llm: LLMClient,
    use_llm_entailment: bool = True,
    doc_id: str = "",
    query: str = "",
    local_nli_model: "LocalNLIModel | None" = None,
) -> list[VerifiedClaim]:
    """Run the full verification pipeline on parsed claims.

    Args:
        claims: Parsed claims from the generation stage.
        doc_tree: The full document structure for existence checks.
        ranked_chunks: Retrieved chunks for source text lookup.
        llm: LLM client for entailment checking.
        use_llm_entailment: If True, use LLM for entailment. If False, skip.
        doc_id: Document identifier for logging.
        query: Original user query for logging.

    Returns:
        List of VerifiedClaim with all checks completed.
    """
    # Build a lookup from section ID to chunk/node for quick access
    chunk_map: dict[str, RankedChunk] = {}
    for rc in ranked_chunks:
        if rc.chunk.node_id not in chunk_map:
            chunk_map[rc.chunk.node_id] = rc

    verified_claims: list[VerifiedClaim] = []

    for claim in claims:
        if claim.is_gap:
            # Gap acknowledgments don't need verification
            verified_claims.append(VerifiedClaim(
                text=claim.text,
                citations=[],
                confidence=ConfidenceScore(
                    level=ConfidenceLevel.HIGH,
                    score=1.0,
                    flags=["gap_acknowledgment"],
                ),
                has_hedging=claim.has_hedging,
                is_gap=True,
            ))
            continue

        verified_citations: list[VerifiedCitation] = []

        for citation in claim.citations:
            vc = _verify_single_citation(
                citation=citation,
                claim_text=claim.text,
                doc_tree=doc_tree,
                chunk_map=chunk_map,
                llm=llm,
                use_llm_entailment=use_llm_entailment,
                doc_id=doc_id,
                query=query,
                local_nli_model=local_nli_model,
            )
            verified_citations.append(vc)

        # Compute claim-level confidence
        confidence = _compute_claim_confidence(
            verified_citations,
            claim.has_hedging,
        )

        verified_claims.append(VerifiedClaim(
            text=claim.text,
            citations=verified_citations,
            confidence=confidence,
            has_hedging=claim.has_hedging,
            is_gap=False,
        ))

    return verified_claims


def _verify_single_citation(
    citation: Citation,
    claim_text: str,
    doc_tree: DocumentNode,
    chunk_map: dict[str, RankedChunk],
    llm: LLMClient,
    use_llm_entailment: bool,
    doc_id: str = "",
    query: str = "",
    local_nli_model: "LocalNLIModel | None" = None,
) -> VerifiedCitation:
    """Run all four verification checks on a single citation."""

    section_id = citation.section_id
    normalized_id = normalize_section_id(section_id)

    # ── Check 1: Citation Existence ──────────────────────────────────
    # Strategy: try to resolve the citation against (in order):
    # 1. The retrieved chunks (what the LLM actually saw)
    # 2. The document tree (structural nodes)
    # 3. Parent section (§8.3(b) → §8.3)
    # 4. Fuzzy match against all chunk node_ids

    source_text = ""
    char_span = (0, 0)
    section_path: list[str] = []
    retrieval_method = ""
    retrieval_score = 0.0
    exists = False

    # Try 1: Direct chunk match
    rc = chunk_map.get(section_id) or chunk_map.get(normalized_id)
    if rc:
        exists = True
        source_text = rc.chunk.text
        char_span = (rc.chunk.char_start, rc.chunk.char_end)
        section_path = rc.chunk.node_path
        retrieval_method = rc.method.value
        retrieval_score = rc.score

    # Try 2: Document tree
    if not exists:
        node = doc_tree.find_by_id(section_id)
        if node is None:
            node = doc_tree.find_by_id(normalized_id)
        if node is not None:
            exists = True
            source_text = node.text or node.all_text
            char_span = (node.char_start, node.char_end)
            section_path = node.path if hasattr(node, "_path") else [node.id]

    # Try 3: Parent section (§8.3(b) → §8.3)
    if not exists:
        parent_id = _get_parent_section_id(section_id)
        if parent_id:
            rc = chunk_map.get(parent_id)
            if rc:
                exists = True
                source_text = rc.chunk.text
                char_span = (rc.chunk.char_start, rc.chunk.char_end)
                section_path = rc.chunk.node_path
                retrieval_method = rc.method.value
                retrieval_score = rc.score
            else:
                node = doc_tree.find_by_id(parent_id)
                if node is not None:
                    exists = True
                    source_text = node.text or node.all_text
                    char_span = (node.char_start, node.char_end)
                    section_path = node.path if hasattr(node, "_path") else [node.id]

    # Try 4: Fuzzy match — the LLM might use a slightly different ID format
    if not exists:
        for chunk_node_id, rc in chunk_map.items():
            # Check if the citation is a substring match or close variant
            if (section_id.lower() in chunk_node_id.lower() or
                chunk_node_id.lower() in section_id.lower() or
                _ids_match_fuzzy(section_id, chunk_node_id)):
                exists = True
                source_text = rc.chunk.text
                char_span = (rc.chunk.char_start, rc.chunk.char_end)
                section_path = rc.chunk.node_path
                retrieval_method = rc.method.value
                retrieval_score = rc.score
                break

    # ── Check 2: Quote Verification ──────────────────────────────────
    quote_similarity = 0.0
    if citation.quoted_text and source_text:
        match = find_best_match(citation.quoted_text, source_text, min_score=0.5)
        if match:
            quote_similarity = match[0]
        else:
            quote_similarity = fuzzy_match_score(citation.quoted_text, source_text)

    # ── Check 3: Entailment Verification ─────────────────────────────
    entailment = EntailmentClass.NEUTRAL
    entailment_score = 0.0
    entailment_explanation = ""

    if exists and source_text and use_llm_entailment:
        # Choose entailment backend
        if local_nli_model is not None:
            entailment, entailment_score, entailment_explanation = _check_entailment_local(
                section_id=section_id,
                passage=source_text[:2000],  # cap passage length
                claim=claim_text,
                nli_model=local_nli_model,
            )
        else:
            entailment, entailment_score, entailment_explanation = (
                _check_entailment_llm(
                    section_id=section_id,
                    passage=source_text[:2000],  # cap passage length
                    claim=claim_text,
                    llm=llm,
                )
            )

        # ── Log entailment sample for training data ──────────────────
        try:
            from legalkit.eval.entailment_log import get_logger
            get_logger().log(
                passage=source_text[:2000],
                claim=claim_text,
                entailment_class=entailment.value,
                confidence=entailment_score,
                explanation=entailment_explanation,
                section_id=section_id,
                doc_id=doc_id,
                query=query,
                judge_model=llm.config.generation_model,
            )
        except Exception:
            pass  # never let logging break the pipeline

    # ── Determine overall status ─────────────────────────────────────
    if not exists:
        status = VerificationStatus.UNVERIFIED
    elif entailment == EntailmentClass.ENTAILED and entailment_score >= 0.8:
        status = VerificationStatus.VERIFIED
    elif entailment in (EntailmentClass.ENTAILED, EntailmentClass.PARTIALLY_ENTAILED):
        status = VerificationStatus.PARTIALLY_VERIFIED
    elif entailment == EntailmentClass.CONTRADICTED:
        status = VerificationStatus.UNVERIFIED
    else:
        status = VerificationStatus.PARTIALLY_VERIFIED

    return VerifiedCitation(
        section_id=section_id,
        section_path=section_path,
        source_text=source_text[:500],  # truncate for report
        char_span=char_span,
        retrieval_method=retrieval_method,
        retrieval_score=retrieval_score,
        exists=exists,
        quote_similarity=quote_similarity,
        entailment=entailment,
        entailment_score=entailment_score,
        entailment_explanation=entailment_explanation,
        status=status,
    )


def _ids_match_fuzzy(citation_id: str, chunk_id: str) -> bool:
    """Check if a citation ID and chunk ID refer to the same section.

    Handles format mismatches like:
    - '§1.01' vs '§1.1'
    - 'Section 5' vs '§5'
    - 'Fiverr-para' vs 'fiverr-para'
    """
    a = citation_id.lower().strip("§").strip()
    b = chunk_id.lower().strip("§").strip()
    if a == b:
        return True
    # Normalize leading zeros: 1.01 → 1.1
    a_norm = re.sub(r"\.0+(\d)", r".\1", a)
    b_norm = re.sub(r"\.0+(\d)", r".\1", b)
    if a_norm == b_norm:
        return True
    return False


def _get_parent_section_id(section_id: str) -> str | None:
    """Get the parent section ID for sub-clause references.

    Examples:
        '§8.3(b)' -> '§8.3'
        '§8.3(a)' -> '§8.3'
        '§8.3' -> None (already a section)
    """
    # Remove sub-clause marker like (a), (b), (i)
    m = re.match(r"(§[\d.]+)\([a-zA-Z0-9]+\)$", section_id)
    if m:
        return m.group(1)
    return None


def _check_entailment_llm(
    section_id: str,
    passage: str,
    claim: str,
    llm: LLMClient,
) -> tuple[EntailmentClass, float, str]:
    """Use LLM to check if passage entails the claim."""
    prompt = ENTAILMENT_USER.format(
        section_id=section_id,
        passage=passage,
        claim=claim,
    )

    try:
        response = llm.generate(
            system=ENTAILMENT_SYSTEM,
            user=prompt,
            max_tokens=200,
        )

        return _parse_entailment_response(response)

    except Exception as e:
        return (EntailmentClass.NEUTRAL, 0.0, f"Entailment check failed: {e}")


def _check_entailment_local(
    section_id: str,
    passage: str,
    claim: str,
    nli_model: LocalNLIModel,
) -> tuple[EntailmentClass, float, str]:
    """Check entailment using the local fine-tuned model.

    Same signature as _check_entailment_llm() — callers can swap them.

    Args:
        section_id: Document section identifier (used in logging only).
        passage: The source text passage (premise).
        claim: The claim to verify (hypothesis).
        nli_model: Loaded LocalNLIModel instance.

    Returns:
        (EntailmentClass, confidence_float, explanation_str)
    """
    try:
        label, confidence, explanation = nli_model.predict(passage, claim)
        entailment_class = EntailmentClass(label)
        return entailment_class, confidence, explanation
    except Exception as exc:
        return EntailmentClass.NEUTRAL, 0.0, f"Local NLI check failed: {exc}"


def _parse_entailment_response(
    response: str,
) -> tuple[EntailmentClass, float, str]:
    """Parse the LLM entailment response."""
    classification = EntailmentClass.NEUTRAL
    confidence = 0.0
    explanation = ""

    # Parse CLASSIFICATION line
    class_match = re.search(
        r"CLASSIFICATION:\s*(ENTAILED|PARTIALLY_ENTAILED|CONTRADICTED|NEUTRAL)",
        response,
        re.IGNORECASE,
    )
    if class_match:
        class_str = class_match.group(1).upper()
        try:
            classification = EntailmentClass(class_str.lower())
        except ValueError:
            # Handle case mapping
            mapping = {
                "ENTAILED": EntailmentClass.ENTAILED,
                "PARTIALLY_ENTAILED": EntailmentClass.PARTIALLY_ENTAILED,
                "CONTRADICTED": EntailmentClass.CONTRADICTED,
                "NEUTRAL": EntailmentClass.NEUTRAL,
            }
            classification = mapping.get(class_str, EntailmentClass.NEUTRAL)

    # Parse CONFIDENCE line
    conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", response)
    if conf_match:
        confidence = min(1.0, max(0.0, float(conf_match.group(1))))

    # Parse EXPLANATION line
    expl_match = re.search(r"EXPLANATION:\s*(.+?)(?:\n|$)", response)
    if expl_match:
        explanation = expl_match.group(1).strip()

    return classification, confidence, explanation


# ---------------------------------------------------------------------------
# Check 4: Completeness check
# ---------------------------------------------------------------------------


def check_completeness(
    question: str,
    answer: str,
    cited_section_ids: set[str],
    doc_tree: DocumentNode,
    llm: LLMClient,
) -> tuple[list[str], list[str]]:
    """Check for gaps and potentially relevant uncited sections.

    Returns:
        (gaps, related_unresolved) — lists of strings describing
        missing information and potentially relevant sections.
    """
    from legalkit.llm.prompts import COMPLETENESS_SYSTEM, COMPLETENESS_USER

    # Find sections not cited in the answer
    all_nodes = doc_tree.all_nodes()
    uncited = []
    for node in all_nodes:
        if node.id not in cited_section_ids and node.text and len(node.text) > 50:
            uncited.append(f"[{node.id}] {node.title}: {node.text[:100]}...")

    if not uncited:
        return [], []

    uncited_text = "\n".join(uncited[:20])  # cap at 20 sections

    prompt = COMPLETENESS_USER.format(
        question=question,
        answer=answer,
        uncited_sections=uncited_text,
    )

    try:
        response = llm.generate(
            system=COMPLETENESS_SYSTEM,
            user=prompt,
            max_tokens=500,
        )

        if "NO_GAPS" in response:
            return [], []

        gaps = [
            line.strip().lstrip("- •")
            for line in response.strip().split("\n")
            if line.strip() and not line.strip().startswith("NO_GAPS")
        ]
        return gaps, []

    except Exception:
        return [], []


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


def _compute_claim_confidence(
    verified_citations: list[VerifiedCitation],
    has_hedging: bool,
) -> ConfidenceScore:
    """Compute composite confidence score for a claim."""
    if not verified_citations:
        return ConfidenceScore(
            level=ConfidenceLevel.LOW,
            score=0.0,
            corroboration_count=0,
            flags=["no_citations"],
        )

    flags: list[str] = []
    scores: list[float] = []

    for vc in verified_citations:
        scores.append(vc.composite_score)

        if not vc.exists:
            flags.append("hallucinated_reference")
        if vc.entailment == EntailmentClass.CONTRADICTED:
            flags.append("contradicted")
        if vc.entailment == EntailmentClass.PARTIALLY_ENTAILED:
            flags.append("ambiguous_entailment")

    best_score = max(scores)
    corroboration = sum(1 for s in scores if s > 0.6)

    if len(verified_citations) == 1:
        flags.append("single_source")

    if has_hedging:
        flags.append("hedging_language")

    # Determine level
    if best_score >= 0.85 and "hallucinated_reference" not in flags:
        level = ConfidenceLevel.HIGH
    elif best_score >= 0.55:
        level = ConfidenceLevel.MEDIUM
    else:
        level = ConfidenceLevel.LOW

    return ConfidenceScore(
        level=level,
        score=round(best_score, 3),
        corroboration_count=corroboration,
        flags=list(dict.fromkeys(flags)),  # dedupe
    )
