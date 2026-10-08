"""Stage 9: Evidence report assembly.

Combines generation, verification, and completeness checking
into a structured EvidenceReport.
"""

from __future__ import annotations

import time

from legalkit.llm.client import LLMClient
from legalkit.models.document import DocumentNode
from legalkit.models.report import (
    ConfidenceLevel,
    ConfidenceScore,
    EvidenceReport,
    RetrievalSummary,
    VerifiedClaim,
)
from legalkit.models.retrieval import RetrievalMethod, RetrievalResult
from legalkit.pipeline.generate import generate_answer
from legalkit.pipeline.verify import verify_claims, check_completeness


def build_evidence_report(
    question: str,
    retrieval_result: RetrievalResult,
    doc_tree: DocumentNode,
    llm: LLMClient,
    doc_id: str = "",
    local_nli_model=None,
) -> EvidenceReport:
    """Run the full pipeline from retrieval results to evidence report.

    Stages 7-9:
    7. Generate grounded answer with citations
    8. Verify each citation
    9. Check completeness and assemble report

    Args:
        question: The user's question.
        retrieval_result: Output from the retrieval pipeline.
        doc_tree: Parsed document tree.
        llm: LLM client.
        doc_id: Document identifier.

    Returns:
        Complete EvidenceReport.
    """
    t0 = time.time()
    all_chunks = retrieval_result.all_chunks

    # ── Stage 7: Generate ────────────────────────────────────────────
    raw_answer, claims = generate_answer(
        question=question,
        ranked_chunks=all_chunks,
        llm=llm,
    )

    # ── Stage 8: Verify ──────────────────────────────────────────────
    verified_claims = verify_claims(
        claims=claims,
        doc_tree=doc_tree,
        ranked_chunks=all_chunks,
        llm=llm,
        doc_id=doc_id,
        query=question,
    )

    # ── Stage 8b: Completeness ───────────────────────────────────────
    cited_ids = set()
    for vc in verified_claims:
        for cit in vc.citations:
            cited_ids.add(cit.section_id)

    gaps, related = check_completeness(
        question=question,
        answer=raw_answer,
        cited_section_ids=cited_ids,
        doc_tree=doc_tree,
        llm=llm,
    )

    # ── Stage 9: Assemble ────────────────────────────────────────────
    retrieval_summary = _build_retrieval_summary(retrieval_result)
    overall_confidence = _compute_overall_confidence(verified_claims)

    elapsed = time.time() - t0

    return EvidenceReport(
        question=question,
        answer=raw_answer,
        claims=verified_claims,
        overall_confidence=overall_confidence,
        retrieval_summary=retrieval_summary,
        gaps=gaps,
        related_unresolved=related,
        doc_id=doc_id,
        model=llm.config.generation_model,
        elapsed_seconds=round(elapsed, 2),
    )


def _build_retrieval_summary(result: RetrievalResult) -> RetrievalSummary:
    """Summarize retrieval metadata for transparency."""
    semantic_hits = sum(
        1 for rc in result.ranked_chunks
        if rc.method in (RetrievalMethod.SEMANTIC, RetrievalMethod.HYBRID)
    )
    bm25_hits = sum(
        1 for rc in result.ranked_chunks
        if rc.method in (RetrievalMethod.BM25, RetrievalMethod.HYBRID)
    )
    scores = [rc.score for rc in result.ranked_chunks if rc.score > 0]

    return RetrievalSummary(
        chunks_retrieved=len(result.ranked_chunks),
        cross_ref_chunks_added=len(result.cross_ref_chunks),
        semantic_hits=semantic_hits,
        bm25_hits=bm25_hits,
        query_expansions=result.expanded_queries,
        avg_retrieval_score=round(sum(scores) / len(scores), 4) if scores else 0.0,
    )


def _compute_overall_confidence(
    verified_claims: list[VerifiedClaim],
) -> ConfidenceScore:
    """Compute overall answer confidence from individual claims."""
    if not verified_claims:
        return ConfidenceScore(
            level=ConfidenceLevel.LOW,
            score=0.0,
            flags=["no_claims"],
        )

    # Filter out gap claims for scoring
    substantive = [vc for vc in verified_claims if not vc.is_gap]
    if not substantive:
        return ConfidenceScore(
            level=ConfidenceLevel.MEDIUM,
            score=0.5,
            flags=["all_gaps"],
        )

    claim_scores = [vc.confidence.score for vc in substantive]
    avg_score = sum(claim_scores) / len(claim_scores)

    # Collect all flags
    all_flags: list[str] = []
    for vc in substantive:
        all_flags.extend(vc.confidence.flags)

    has_hallucination = "hallucinated_reference" in all_flags
    has_contradiction = "contradicted" in all_flags
    all_single_source = all(
        "single_source" in vc.confidence.flags for vc in substantive if vc.citations
    )

    if has_hallucination:
        all_flags.append("contains_hallucinated_citations")
    if has_contradiction:
        all_flags.append("contains_contradictions")
    if all_single_source:
        all_flags.append("all_claims_single_source")

    # Penalize for problems
    if has_hallucination:
        avg_score *= 0.6
    if has_contradiction:
        avg_score *= 0.5

    if avg_score >= 0.80:
        level = ConfidenceLevel.HIGH
    elif avg_score >= 0.55:
        level = ConfidenceLevel.MEDIUM
    else:
        level = ConfidenceLevel.LOW

    total_corroboration = sum(
        vc.confidence.corroboration_count for vc in substantive
    )

    return ConfidenceScore(
        level=level,
        score=round(avg_score, 3),
        corroboration_count=total_corroboration,
        flags=list(dict.fromkeys(all_flags)),
    )
