"""Retrieval models — chunks and search results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RetrievalMethod(Enum):
    """How a chunk was retrieved."""

    SEMANTIC = "semantic"
    BM25 = "bm25"
    HYBRID = "hybrid"
    CROSS_REFERENCE = "cross_reference"
    DEFINITION = "definition"


@dataclass
class Chunk:
    """A chunk of document text ready for indexing and retrieval.

    Chunks are aligned to structural boundaries (sections, clauses)
    and carry their structural lineage for citation purposes.
    """

    id: str
    """Unique chunk identifier."""

    text: str
    """Raw chunk text content."""

    context_text: str | None = None
    """LLM-generated context prepended before embedding (Stage 3)."""

    table_repr: str | None = None
    """Serialized table representation (headers + rows) when chunk contains a table.
    Used in place of raw text for embedding when present."""

    doc_id: str = ""
    """Which document this chunk belongs to."""

    node_id: str = ""
    """Which structural node this chunk belongs to."""

    node_path: list[str] = field(default_factory=list)
    """Ancestry path, e.g. ['Article IV', '§4.2', '§4.2(a)']."""

    char_start: int = 0
    """Start offset in original document."""

    char_end: int = 0
    """End offset in original document."""

    cross_references: list[str] = field(default_factory=list)
    """Section IDs referenced within this chunk."""

    @property
    def embedding_text(self) -> str:
        """Text used for embedding.

        Priority: context_text > table_repr > raw text.
        context_text is prepended to whichever base text is used.
        """
        base = self.table_repr if self.table_repr else self.text
        if self.context_text:
            return f"{self.context_text}\n\n{base}"
        return base

    @property
    def token_estimate(self) -> int:
        """Rough token count estimate (1 token ≈ 4 chars)."""
        return len(self.embedding_text) // 4

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ") + ("..." if len(self.text) > 60 else "")
        return f"Chunk(id={self.id!r}, node={self.node_id!r}, text={preview!r})"


@dataclass
class RankedChunk:
    """A chunk with retrieval scoring information."""

    chunk: Chunk
    """The underlying chunk."""

    score: float
    """Combined retrieval score (after RRF or reranking)."""

    method: RetrievalMethod
    """How this chunk was retrieved."""

    semantic_rank: int | None = None
    """Rank in semantic search results (None if not retrieved by semantic)."""

    bm25_rank: int | None = None
    """Rank in BM25 search results (None if not retrieved by BM25)."""

    semantic_score: float | None = None
    """Raw cosine similarity score."""

    bm25_score: float | None = None
    """Raw BM25 score."""


@dataclass
class RetrievalResult:
    """Complete result of a retrieval operation."""

    query: str
    """The original query."""

    expanded_queries: list[str] = field(default_factory=list)
    """Alternative query phrasings used for retrieval."""

    ranked_chunks: list[RankedChunk] = field(default_factory=list)
    """Chunks ranked by combined relevance score."""

    cross_ref_chunks: list[RankedChunk] = field(default_factory=list)
    """Additional chunks pulled in via cross-reference expansion."""

    definition_chunks: list[RankedChunk] = field(default_factory=list)
    """Definition chunks for defined terms referenced in retrieved chunks."""

    @property
    def all_chunks(self) -> list[RankedChunk]:
        """All chunks (direct + cross-reference + definitions), direct first."""
        return self.ranked_chunks + self.cross_ref_chunks + self.definition_chunks

    @property
    def top_k(self) -> int:
        return len(self.ranked_chunks)
