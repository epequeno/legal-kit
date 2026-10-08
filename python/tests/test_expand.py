"""Tests for query expansion (Stage 4.5)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from legalkit.models.retrieval import RankedChunk, RetrievalMethod
from legalkit.pipeline.expand import (
    DEFAULT_N_EXPANSIONS,
    _rrf_merge,
    expand_query,
    expand_query_with_retrieval,
)
from legalkit.models.retrieval import Chunk

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _make_chunk(chunk_id: str, text: str = "clause text") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        doc_id="doc",
        node_id=chunk_id,
        node_path=["Article I", chunk_id],
        char_start=0,
        char_end=len(text),
    )


def _make_rc(chunk_id: str, score: float = 1.0, method=RetrievalMethod.HYBRID) -> RankedChunk:
    return RankedChunk(
        chunk=_make_chunk(chunk_id),
        score=score,
        method=method,
    )


_DEFAULT_EXPANSIONS = [
    "What are the termination provisions?",
    "Under what conditions may the agreement be dissolved?",
    "Circumstances permitting early contract termination",
]


def _make_llm(expansions: list[str] | None = None) -> MagicMock:
    llm = MagicMock()
    llm.expand_query.return_value = _DEFAULT_EXPANSIONS if expansions is None else expansions
    return llm


# ---------------------------------------------------------------------------
# expand_query
# ---------------------------------------------------------------------------


class TestExpandQuery:
    def test_returns_llm_expansions(self):
        llm = _make_llm(["Alt 1", "Alt 2", "Alt 3"])
        result = expand_query("original question", llm, n=3)
        assert result == ["Alt 1", "Alt 2", "Alt 3"]

    def test_passes_correct_args_to_llm(self):
        llm = _make_llm()
        expand_query("What is the notice period?", llm, n=4)
        llm.expand_query.assert_called_once_with(query="What is the notice period?", n=4)

    def test_returns_empty_list_on_llm_failure(self):
        llm = MagicMock()
        llm.expand_query.side_effect = RuntimeError("API error")
        result = expand_query("question", llm)
        assert result == []

    def test_returns_empty_list_gracefully(self):
        llm = _make_llm([])
        result = expand_query("question", llm)
        assert result == []

    def test_default_n_is_used(self):
        llm = _make_llm()
        expand_query("question", llm)
        _, kwargs = llm.expand_query.call_args
        assert kwargs["n"] == DEFAULT_N_EXPANSIONS


# ---------------------------------------------------------------------------
# _rrf_merge
# ---------------------------------------------------------------------------


class TestRRFMerge:
    def test_single_list_passthrough(self):
        ranked = [_make_rc(f"chunk-{i}", score=1.0 / (i + 1)) for i in range(5)]
        merged = _rrf_merge([ranked], top_k=5)
        ids = [rc.chunk.id for rc in merged]
        assert ids == [rc.chunk.id for rc in ranked]

    def test_top_k_respected(self):
        ranked = [_make_rc(f"chunk-{i}") for i in range(10)]
        merged = _rrf_merge([ranked], top_k=3)
        assert len(merged) == 3

    def test_consistent_chunk_wins_when_in_all_lists(self):
        """A chunk appearing at rank 1 in all lists should win."""
        winner = "winner-chunk"
        list1 = [_make_rc(winner)] + [_make_rc(f"other-{i}") for i in range(5)]
        list2 = [_make_rc(winner)] + [_make_rc(f"other-{i+10}") for i in range(5)]
        list3 = [_make_rc(winner)] + [_make_rc(f"other-{i+20}") for i in range(5)]

        merged = _rrf_merge([list1, list2, list3], top_k=10)
        assert merged[0].chunk.id == winner

    def test_chunk_appearing_in_multiple_lists_scores_higher(self):
        shared = "shared-chunk"
        unique = "unique-chunk"

        # shared appears in both lists; unique only in list2 (at rank 0)
        list1 = [_make_rc(shared, score=0.5)] + [_make_rc(f"a-{i}") for i in range(4)]
        list2 = [_make_rc(unique, score=0.9), _make_rc(shared, score=0.3)] + [
            _make_rc(f"b-{i}") for i in range(3)
        ]

        merged = _rrf_merge([list1, list2], top_k=10)
        ids = [rc.chunk.id for rc in merged]

        # shared appears in both lists so should beat unique which only appears once
        assert ids.index(shared) < ids.index(unique)

    def test_deduplicates_chunks(self):
        chunk_id = "dup-chunk"
        list1 = [_make_rc(chunk_id)]
        list2 = [_make_rc(chunk_id)]

        merged = _rrf_merge([list1, list2], top_k=10)
        ids = [rc.chunk.id for rc in merged]
        assert ids.count(chunk_id) == 1

    def test_empty_lists_returns_empty(self):
        assert _rrf_merge([], top_k=10) == []

    def test_all_empty_sublists(self):
        assert _rrf_merge([[], [], []], top_k=10) == []

    def test_method_set_to_hybrid(self):
        ranked = [_make_rc("chunk-0", method=RetrievalMethod.SEMANTIC)]
        merged = _rrf_merge([ranked], top_k=5)
        assert all(rc.method == RetrievalMethod.HYBRID for rc in merged)

    def test_rrf_scores_are_positive(self):
        ranked = [_make_rc(f"c-{i}") for i in range(5)]
        merged = _rrf_merge([ranked], top_k=5)
        assert all(rc.score > 0 for rc in merged)

    def test_scores_decrease_monotonically(self):
        ranked = [_make_rc(f"c-{i}") for i in range(5)]
        merged = _rrf_merge([ranked], top_k=5)
        scores = [rc.score for rc in merged]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# expand_query_with_retrieval — integration with mock index + LLM
# ---------------------------------------------------------------------------


def _make_index_and_tree(n_chunks: int = 10):
    """Build a minimal mock index and document tree."""
    from legalkit.pipeline.index import DocumentIndex
    from legalkit.pipeline.parse import parse_document

    sample_text = (FIXTURE_DIR / "sample_contract.txt").read_text()
    from legalkit.pipeline.chunk import chunk_document

    root = parse_document(sample_text, doc_id="test")
    chunks = chunk_document(root, doc_id="test")[:n_chunks]

    # Random embeddings
    rng = np.random.default_rng(42)
    emb_dim = 16
    embeddings = rng.random((len(chunks), emb_dim), dtype=np.float32)

    # Normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)

    # Patch BM25 to avoid tokenization issues with tiny embeddings
    index = DocumentIndex()
    index.build(chunks, embeddings)

    return index, root, embeddings, emb_dim


class TestExpandQueryWithRetrieval:
    def test_returns_retrieval_result(self):
        index, root, embeddings, emb_dim = _make_index_and_tree()
        llm = _make_llm()

        rng = np.random.default_rng(0)
        query_emb = rng.random(emb_dim, dtype=np.float32)
        query_emb /= np.linalg.norm(query_emb)

        # Mock embed_fn to return random embeddings for expansions
        def mock_embed(texts):
            return rng.random((len(texts), emb_dim), dtype=np.float32)

        result = expand_query_with_retrieval(
            query="What are the termination conditions?",
            query_embedding=query_emb,
            index=index,
            doc_tree=root,
            llm=llm,
            top_k=5,
            n_expansions=3,
            embed_fn=mock_embed,
        )

        assert result is not None
        assert len(result.ranked_chunks) > 0

    def test_expanded_queries_populated(self):
        index, root, embeddings, emb_dim = _make_index_and_tree()
        llm = _make_llm(["Alt 1", "Alt 2"])

        rng = np.random.default_rng(1)
        query_emb = rng.random(emb_dim, dtype=np.float32)

        def mock_embed(texts):
            return rng.random((len(texts), emb_dim), dtype=np.float32)

        result = expand_query_with_retrieval(
            query="What is the termination clause?",
            query_embedding=query_emb,
            index=index,
            doc_tree=root,
            llm=llm,
            top_k=5,
            n_expansions=2,
            embed_fn=mock_embed,
        )

        assert result.expanded_queries == ["Alt 1", "Alt 2"]

    def test_falls_back_gracefully_when_expansion_fails(self):
        index, root, embeddings, emb_dim = _make_index_and_tree()
        llm = MagicMock()
        llm.expand_query.side_effect = RuntimeError("LLM down")

        rng = np.random.default_rng(2)
        query_emb = rng.random(emb_dim, dtype=np.float32)

        result = expand_query_with_retrieval(
            query="What is the notice period?",
            query_embedding=query_emb,
            index=index,
            doc_tree=root,
            llm=llm,
            top_k=5,
        )

        # Should still return results from original query alone
        assert len(result.ranked_chunks) > 0
        assert result.expanded_queries == []

    def test_original_query_always_included(self):
        """Even with expansions, the original query results are merged in."""
        index, root, embeddings, emb_dim = _make_index_and_tree()
        llm = _make_llm(["Expansion 1", "Expansion 2"])

        rng = np.random.default_rng(3)
        query_emb = rng.random(emb_dim, dtype=np.float32)
        embed_call_count = {"n": 0}

        def mock_embed(texts):
            embed_call_count["n"] += 1
            return rng.random((len(texts), emb_dim), dtype=np.float32)

        expand_query_with_retrieval(
            query="original",
            query_embedding=query_emb,
            index=index,
            doc_tree=root,
            llm=llm,
            top_k=5,
            n_expansions=2,
            embed_fn=mock_embed,
        )

        # embed_fn called once for the 2 expansions (original uses pre-computed emb)
        assert embed_call_count["n"] == 1

    def test_no_duplicate_chunks_in_result(self):
        index, root, embeddings, emb_dim = _make_index_and_tree()
        llm = _make_llm(["Alt 1", "Alt 2", "Alt 3"])

        rng = np.random.default_rng(4)
        query_emb = rng.random(emb_dim, dtype=np.float32)

        def mock_embed(texts):
            return rng.random((len(texts), emb_dim), dtype=np.float32)

        result = expand_query_with_retrieval(
            query="question",
            query_embedding=query_emb,
            index=index,
            doc_tree=root,
            llm=llm,
            top_k=10,
            n_expansions=3,
            embed_fn=mock_embed,
        )

        chunk_ids = [rc.chunk.id for rc in result.ranked_chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunks found in result"

    def test_embed_fn_called_with_expansion_texts(self):
        index, root, embeddings, emb_dim = _make_index_and_tree()
        llm = _make_llm(["Phrasing A", "Phrasing B"])

        rng = np.random.default_rng(5)
        query_emb = rng.random(emb_dim, dtype=np.float32)
        captured_texts = []

        def mock_embed(texts):
            captured_texts.extend(texts)
            return rng.random((len(texts), emb_dim), dtype=np.float32)

        expand_query_with_retrieval(
            query="original",
            query_embedding=query_emb,
            index=index,
            doc_tree=root,
            llm=llm,
            top_k=5,
            n_expansions=2,
            embed_fn=mock_embed,
        )

        assert "Phrasing A" in captured_texts
        assert "Phrasing B" in captured_texts


# ---------------------------------------------------------------------------
# LLMClient.expand_query unit test
# ---------------------------------------------------------------------------


class TestLLMClientExpandQuery:
    def test_parses_one_phrasing_per_line(self):
        from legalkit.llm.client import LLMClient

        client = LLMClient.__new__(LLMClient)
        client.config = MagicMock()
        client.config.context_model = "test-model"

        with patch.object(client, "generate", return_value="Line one\nLine two\nLine three"):
            result = client.expand_query("question", n=3)

        assert result == ["Line one", "Line two", "Line three"]

    def test_strips_blank_lines(self):
        from legalkit.llm.client import LLMClient

        client = LLMClient.__new__(LLMClient)
        client.config = MagicMock()
        client.config.context_model = "test-model"

        with patch.object(client, "generate", return_value="\nLine one\n\nLine two\n\n"):
            result = client.expand_query("question", n=3)

        assert result == ["Line one", "Line two"]

    def test_caps_at_n(self):
        from legalkit.llm.client import LLMClient

        client = LLMClient.__new__(LLMClient)
        client.config = MagicMock()
        client.config.context_model = "test-model"

        with patch.object(
            client, "generate", return_value="L1\nL2\nL3\nL4\nL5"
        ):
            result = client.expand_query("question", n=3)

        assert len(result) == 3
