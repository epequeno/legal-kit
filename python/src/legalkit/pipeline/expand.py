"""Stage 4.5: Query expansion.

Generates LLM-produced alternative phrasings of the user's query before
retrieval. Each phrasing is embedded and searched independently; results
from all phrasings are merged via Reciprocal Rank Fusion (RRF) alongside
the original query results.

The goal is to bridge the vocabulary gap between plain-English questions and
the formal legal language found in contracts — e.g. a user asking "can they
fire me without notice?" is more likely to match chunks that say "terminate
employment without cause upon written notice" when expanded phrasings cover
both registers.

Usage::

    from legalkit.pipeline.expand import expand_query_with_retrieval

    result = expand_query_with_retrieval(
        query="What happens if either party wants to end the agreement early?",
        index=index,
        doc_tree=root,
        llm=llm,
        top_k=20,
    )
    # result.expanded_queries contains the generated phrasings
    # result.ranked_chunks is already RRF-merged across all phrasings
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from legalkit.models.document import DocumentNode
from legalkit.models.retrieval import RankedChunk, RetrievalMethod, RetrievalResult
from legalkit.pipeline.index import DocumentIndex
from legalkit.pipeline.retrieve import retrieve

logger = logging.getLogger(__name__)

# Default number of alternative phrasings to generate
DEFAULT_N_EXPANSIONS = 3

# RRF constant — same value used in hybrid retrieval
_RRF_K = 60


def expand_query(
    query: str,
    llm: object,  # LLMClient — avoid circular import
    n: int = DEFAULT_N_EXPANSIONS,
) -> list[str]:
    """Generate alternative phrasings of a query using the LLM.

    Returns a list of up to ``n`` alternative phrasings. Returns an empty
    list (with a warning) if the LLM call fails, so callers can gracefully
    fall back to the original query.

    Args:
        query: The original user question.
        llm: :class:`~legalkit.llm.client.LLMClient` instance.
        n: Number of alternative phrasings to request.

    Returns:
        List of alternative query strings (may be empty on failure).
    """
    try:
        return llm.expand_query(query=query, n=n)
    except Exception as exc:
        logger.warning("Query expansion failed, falling back to original: %s", exc)
        return []


def expand_query_with_retrieval(
    query: str,
    query_embedding: np.ndarray,
    index: DocumentIndex,
    doc_tree: DocumentNode,
    llm: object,
    *,
    top_k: int = 20,
    n_expansions: int = DEFAULT_N_EXPANSIONS,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    max_cross_ref: int = 10,
    definition_index=None,  # DefinitionIndex | None — avoid circular import
    max_definitions: int = 5,
) -> RetrievalResult:
    """Run retrieval with query expansion and multi-phrasing RRF fusion.

    Steps:
    1. Generate ``n_expansions`` alternative phrasings via LLM.
    2. Embed each phrasing (plus the original).
    3. Run hybrid retrieval for every phrasing independently.
    4. Merge all per-phrasing ranked lists via RRF.
    5. Expand cross-references from the final merged set.

    Args:
        query: The original user question.
        query_embedding: Pre-computed embedding for the original query.
        index: The document index.
        doc_tree: Parsed document tree for cross-reference lookup.
        llm: :class:`~legalkit.llm.client.LLMClient` instance.
        top_k: Chunks to retrieve per phrasing (pre-merge).
        n_expansions: How many alternative phrasings to generate.
        embed_fn: Callable ``(texts) -> np.ndarray`` for embedding expansions.
            Defaults to :func:`~legalkit.pipeline.embed.embed_texts`.
        max_cross_ref: Maximum cross-reference chunks to add.

    Returns:
        :class:`~legalkit.models.retrieval.RetrievalResult` with
        ``expanded_queries`` populated and ``ranked_chunks`` merged across
        all phrasings.
    """
    if embed_fn is None:
        from legalkit.pipeline.embed import embed_texts
        embed_fn = embed_texts

    # Step 1: Generate expansions
    expansions = expand_query(query, llm, n=n_expansions)
    logger.debug("Query expansions for %r: %s", query, expansions)

    # Step 2: Embed expansions (batch call for efficiency)
    expansion_embeddings: list[np.ndarray] = []
    if expansions:
        try:
            emb_matrix = embed_fn(expansions)
            expansion_embeddings = [emb_matrix[i] for i in range(len(expansions))]
        except Exception as exc:
            logger.warning("Failed to embed query expansions: %s", exc)
            expansions = []

    # Step 3: Retrieve for each phrasing (original + expansions)
    all_phrasings: list[tuple[str, np.ndarray]] = [(query, query_embedding)]
    for phrase, emb in zip(expansions, expansion_embeddings):
        all_phrasings.append((phrase, emb))

    # Collect ranked lists for RRF: list of list[RankedChunk]
    per_phrasing_results: list[list[RankedChunk]] = []
    for phrase, emb in all_phrasings:
        result = retrieve(
            query=phrase,
            query_embedding=emb,
            index=index,
            doc_tree=doc_tree,
            top_k=top_k,
            max_cross_ref=0,  # cross-ref expansion done once at the end
        )
        per_phrasing_results.append(result.ranked_chunks)

    # Step 4: Multi-phrasing RRF fusion
    merged = _rrf_merge(per_phrasing_results, top_k=top_k, rrf_k=_RRF_K)

    # Step 5: Cross-reference expansion on merged results
    from legalkit.pipeline.retrieve import _expand_cross_references, _expand_definitions
    cross_ref_chunks = _expand_cross_references(
        ranked_chunks=merged,
        index=index,
        doc_tree=doc_tree,
        max_chunks=max_cross_ref,
    )

    # Step 6: Definition expansion
    definition_chunks: list = []
    if definition_index is not None:
        definition_chunks = _expand_definitions(
            ranked_chunks=merged,
            cross_ref_chunks=cross_ref_chunks,
            index=index,
            definition_index=definition_index,
            max_chunks=max_definitions,
        )

    return RetrievalResult(
        query=query,
        expanded_queries=expansions,
        ranked_chunks=merged,
        cross_ref_chunks=cross_ref_chunks,
        definition_chunks=definition_chunks,
    )


def _rrf_merge(
    ranked_lists: list[list[RankedChunk]],
    top_k: int,
    rrf_k: int = _RRF_K,
) -> list[RankedChunk]:
    """Merge multiple ranked lists of RankedChunk via Reciprocal Rank Fusion.

    Each unique chunk (by ``chunk.id``) receives a score of
    ``sum(1 / (rrf_k + rank))`` across all lists it appears in.
    The best ``RankedChunk`` instance for each chunk (highest original score)
    is used as the representative, with its ``score`` replaced by the RRF score.

    Args:
        ranked_lists: One list of :class:`RankedChunk` per query phrasing.
        top_k: Maximum number of results to return.
        rrf_k: RRF smoothing constant (default 60).

    Returns:
        Merged and re-ranked list of :class:`RankedChunk`.
    """
    rrf_scores: dict[str, float] = {}
    # Keep the best (highest original score) RankedChunk per chunk id
    best_rc: dict[str, RankedChunk] = {}

    for ranked in ranked_lists:
        for rank, rc in enumerate(ranked):
            cid = rc.chunk.id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            if cid not in best_rc or rc.score > best_rc[cid].score:
                best_rc[cid] = rc

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    merged: list[RankedChunk] = []
    for cid in sorted_ids[:top_k]:
        rc = best_rc[cid]
        # Replace score with RRF score; mark method as HYBRID
        merged.append(RankedChunk(
            chunk=rc.chunk,
            score=rrf_scores[cid],
            method=RetrievalMethod.HYBRID,
            semantic_rank=rc.semantic_rank,
            bm25_rank=rc.bm25_rank,
            semantic_score=rc.semantic_score,
            bm25_score=rc.bm25_score,
        ))

    return merged
