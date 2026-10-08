"""Stage 5-6: Hybrid retrieval + cross-reference expansion."""

from __future__ import annotations

from legalkit.models.document import DocumentNode
from legalkit.models.retrieval import (
    Chunk,
    RankedChunk,
    RetrievalMethod,
    RetrievalResult,
)
from legalkit.pipeline.index import DocumentIndex
from legalkit.utils.text import normalize_section_id

import numpy as np


def retrieve(
    query: str,
    query_embedding: np.ndarray,
    index: DocumentIndex,
    doc_tree: DocumentNode,
    top_k: int = 20,
    max_cross_ref: int = 10,
    definition_index=None,   # DefinitionIndex | None — avoid circular import
    max_definitions: int = 5,
) -> RetrievalResult:
    """Run hybrid retrieval with cross-reference expansion.

    1. Hybrid search (semantic + BM25 via RRF)
    2. Expand cross-references from retrieved chunks
    3. Return ranked results with full provenance

    Args:
        query: The user's question text.
        query_embedding: Embedding vector for the query.
        index: The document index.
        doc_tree: Parsed document tree for cross-reference lookup.
        top_k: Number of chunks to retrieve.
        max_cross_ref: Maximum cross-reference chunks to add.
        definition_index: Optional DefinitionIndex for definition expansion.
        max_definitions: Maximum definition chunks to add.

    Returns:
        RetrievalResult with ranked chunks, cross-reference chunks, and
        definition chunks.
    """
    # Stage 5: Hybrid retrieval
    hybrid_results = index.search_hybrid(
        query_embedding=query_embedding,
        query_text=query,
        top_k=top_k,
    )

    ranked_chunks: list[RankedChunk] = []
    for chunk_idx, rrf_score, sem_rank, bm25_rank in hybrid_results:
        chunk = index.chunks[chunk_idx]

        # Determine primary retrieval method
        if sem_rank is not None and bm25_rank is not None:
            method = RetrievalMethod.HYBRID
        elif sem_rank is not None:
            method = RetrievalMethod.SEMANTIC
        else:
            method = RetrievalMethod.BM25

        # Get raw scores
        sem_score = None
        bm25_score = None
        if sem_rank is not None:
            sem_results = index.search_semantic(query_embedding, top_k=top_k * 2)
            for idx, score in sem_results:
                if idx == chunk_idx:
                    sem_score = score
                    break
        if bm25_rank is not None:
            bm25_results = index.search_bm25(query, top_k=top_k * 2)
            for idx, score in bm25_results:
                if idx == chunk_idx:
                    bm25_score = score
                    break

        ranked_chunks.append(RankedChunk(
            chunk=chunk,
            score=rrf_score,
            method=method,
            semantic_rank=sem_rank,
            bm25_rank=bm25_rank,
            semantic_score=sem_score,
            bm25_score=bm25_score,
        ))

    # Stage 6: Cross-reference expansion
    cross_ref_chunks = _expand_cross_references(
        ranked_chunks=ranked_chunks,
        index=index,
        doc_tree=doc_tree,
        max_chunks=max_cross_ref,
    )

    # Stage 6b: Definition expansion
    definition_chunks: list[RankedChunk] = []
    if definition_index is not None:
        definition_chunks = _expand_definitions(
            ranked_chunks=ranked_chunks,
            cross_ref_chunks=cross_ref_chunks,
            index=index,
            definition_index=definition_index,
            max_chunks=max_definitions,
        )

    return RetrievalResult(
        query=query,
        ranked_chunks=ranked_chunks,
        cross_ref_chunks=cross_ref_chunks,
        definition_chunks=definition_chunks,
    )


def _expand_definitions(
    ranked_chunks: list[RankedChunk],
    cross_ref_chunks: list[RankedChunk],
    index: DocumentIndex,
    definition_index,  # DefinitionIndex
    max_chunks: int = 5,
) -> list[RankedChunk]:
    """Expand definitions referenced in retrieved chunks.

    Scans the text of all already-retrieved chunks for defined terms.
    For each term found whose definition chunk is not yet in the result
    set, adds the definition chunk.

    Args:
        ranked_chunks: Primary retrieved chunks.
        cross_ref_chunks: Cross-reference expanded chunks.
        index: Document index (used to find definition chunks by node_id).
        definition_index: DefinitionIndex built from the document.
        max_chunks: Maximum number of definition chunks to add.

    Returns:
        List of definition RankedChunks (may be empty).
    """
    # Build set of already-retrieved node IDs to avoid duplicates
    already_retrieved: set[str] = {
        rc.chunk.node_id for rc in ranked_chunks + cross_ref_chunks
    }

    # Scan all retrieved text for defined terms
    all_retrieved_text = " ".join(
        rc.chunk.text for rc in ranked_chunks + cross_ref_chunks
    )
    referenced_defs = definition_index.find_references(all_retrieved_text)

    definition_chunks: list[RankedChunk] = []
    for entry in referenced_defs:
        if len(definition_chunks) >= max_chunks:
            break
        if entry.node_id in already_retrieved:
            continue
        # Find the chunk in the index that corresponds to this definition node
        for chunk in index.chunks:
            if chunk.node_id == entry.node_id and chunk.node_id not in already_retrieved:
                definition_chunks.append(RankedChunk(
                    chunk=chunk,
                    score=0.0,
                    method=RetrievalMethod.DEFINITION,
                ))
                already_retrieved.add(chunk.node_id)
                break

    return definition_chunks


def _expand_cross_references(
    ranked_chunks: list[RankedChunk],
    index: DocumentIndex,
    doc_tree: DocumentNode,
    max_chunks: int = 10,
) -> list[RankedChunk]:
    """Expand cross-references found in retrieved chunks.

    For each cross-reference in the retrieved chunks, find the
    corresponding chunk in the index and add it to the results.
    """
    # Collect all cross-references from retrieved chunks
    retrieved_node_ids = {rc.chunk.node_id for rc in ranked_chunks}
    xref_section_ids: list[str] = []

    for rc in ranked_chunks:
        for ref in rc.chunk.cross_references:
            normalized = normalize_section_id(ref)
            if normalized not in retrieved_node_ids:
                xref_section_ids.append(normalized)

    # Deduplicate
    xref_section_ids = list(dict.fromkeys(xref_section_ids))

    # Find chunks that match the cross-referenced sections
    cross_ref_chunks: list[RankedChunk] = []
    for section_id in xref_section_ids:
        if len(cross_ref_chunks) >= max_chunks:
            break

        for chunk in index.chunks:
            if chunk.node_id == section_id and chunk.node_id not in retrieved_node_ids:
                cross_ref_chunks.append(RankedChunk(
                    chunk=chunk,
                    score=0.0,  # not scored by retrieval
                    method=RetrievalMethod.CROSS_REFERENCE,
                ))
                retrieved_node_ids.add(chunk.node_id)
                break

    return cross_ref_chunks
