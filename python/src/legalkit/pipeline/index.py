"""Stage 3: Embedding & BM25 indexing.

Creates the vector and keyword search indices from contextualized chunks.
"""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from legalkit.models.retrieval import Chunk


class DocumentIndex:
    """In-memory document index with vector and BM25 search.

    Stores chunks and provides both semantic (cosine similarity)
    and keyword (BM25) retrieval.
    """

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

    def build(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> None:
        """Build the index from chunks and their embeddings.

        Args:
            chunks: List of chunks (should have context_text set if available).
            embeddings: numpy array of shape (n_chunks, embedding_dim).
        """
        self.chunks = chunks
        self.embeddings = embeddings

        # Build BM25 index from embedding_text (context + chunk text)
        self._tokenized_corpus = [
            _tokenize(chunk.embedding_text) for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self._tokenized_corpus)

    def search_semantic(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        """Semantic search by cosine similarity.

        Returns list of (chunk_index, score) sorted by score desc.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        doc_norms = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        )
        scores = doc_norms @ query_norm

        # Top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def search_bm25(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        """BM25 keyword search.

        Returns list of (chunk_index, score) sorted by score desc.
        """
        if self.bm25 is None or len(self.chunks) == 0:
            return []

        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]

    def search_hybrid(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 20,
        rrf_k: int = 60,
    ) -> list[tuple[int, float, int | None, int | None]]:
        """Hybrid search combining semantic and BM25 via Reciprocal Rank Fusion.

        Returns list of (chunk_index, rrf_score, semantic_rank, bm25_rank)
        sorted by rrf_score desc.
        """
        semantic_results = self.search_semantic(query_embedding, top_k=top_k * 2)
        bm25_results = self.search_bm25(query_text, top_k=top_k * 2)

        # Build rank maps
        semantic_ranks: dict[int, int] = {
            idx: rank for rank, (idx, _) in enumerate(semantic_results)
        }
        bm25_ranks: dict[int, int] = {
            idx: rank for rank, (idx, _) in enumerate(bm25_results)
        }

        # Compute RRF scores
        all_indices = set(semantic_ranks.keys()) | set(bm25_ranks.keys())
        rrf_scores: list[tuple[int, float, int | None, int | None]] = []

        for idx in all_indices:
            score = 0.0
            sem_rank = semantic_ranks.get(idx)
            bm25_rank = bm25_ranks.get(idx)

            if sem_rank is not None:
                score += 1.0 / (rrf_k + sem_rank)
            if bm25_rank is not None:
                score += 1.0 / (rrf_k + bm25_rank)

            rrf_scores.append((idx, score, sem_rank, bm25_rank))

        # Sort by RRF score descending
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        return rrf_scores[:top_k]

    @property
    def size(self) -> int:
        return len(self.chunks)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer with lowercasing and punctuation removal."""
    import re
    text = text.lower()
    # Keep alphanumeric and basic punctuation relevant to legal text
    tokens = re.findall(r"[a-z0-9§]+(?:[.\-'][a-z0-9]+)*", text)
    return tokens
