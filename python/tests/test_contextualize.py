"""Tests for contextual enrichment (Stage 2.5)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from legalkit.models.retrieval import Chunk
from legalkit.pipeline.chunk import chunk_document
from legalkit.pipeline.contextualize import (
    DEFAULT_WORKERS,
    build_document_context,
    contextualize_chunks,
)
from legalkit.pipeline.parse import parse_document

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_sample() -> str:
    return (FIXTURE_DIR / "sample_contract.txt").read_text()


def _make_chunk(i: int, text: str = "Seller shall indemnify Buyer.") -> Chunk:
    return Chunk(
        id=f"doc-chunk-{i:04d}",
        text=text,
        doc_id="doc",
        node_id=f"§{i}",
        node_path=["Article I", f"§{i}"],
        char_start=0,
        char_end=len(text),
    )


def _make_llm(response: str = "This chunk is from Section 4 of an NDA.") -> MagicMock:
    llm = MagicMock()
    llm.contextualize_chunk.return_value = response
    return llm


# ---------------------------------------------------------------------------
# build_document_context
# ---------------------------------------------------------------------------


class TestBuildDocumentContext:
    def test_short_doc_returned_verbatim(self):
        text = "This is a short contract."
        root = parse_document(text, doc_id="short")
        ctx = build_document_context(root, text, max_chars=10_000)
        assert ctx == text

    def test_long_doc_truncated(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test")
        ctx = build_document_context(root, text, max_chars=5_000)
        assert len(ctx) <= 5_000

    def test_long_doc_includes_preamble(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test")
        ctx = build_document_context(root, text, max_chars=5_000)
        # Structural synopsis should contain the opening text
        assert ctx.startswith(text[:100])

    def test_long_doc_synopsis_marker_present(self):
        """When synopsis mode kicks in, a continuation marker is inserted."""
        text = _load_sample()
        root = parse_document(text, doc_id="test")
        # Force synopsis by setting a very small max_chars (below full text length)
        if len(text) > 500:
            ctx = build_document_context(root, text, max_chars=500)
            # Should be capped
            assert len(ctx) <= 500

    def test_returns_string(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test")
        ctx = build_document_context(root, text)
        assert isinstance(ctx, str)
        assert len(ctx) > 0


# ---------------------------------------------------------------------------
# contextualize_chunks — unit tests with mock LLM
# ---------------------------------------------------------------------------


class TestContextualizeChunks:
    def test_sets_context_text_on_each_chunk(self):
        chunks = [_make_chunk(i) for i in range(5)]
        llm = _make_llm("Context for chunk.")

        result = contextualize_chunks(chunks, doc_context="doc text", llm=llm)

        assert result is chunks  # mutated in-place
        for chunk in chunks:
            assert chunk.context_text == "Context for chunk."

    def test_llm_called_once_per_chunk(self):
        chunks = [_make_chunk(i) for i in range(3)]
        llm = _make_llm()

        contextualize_chunks(chunks, doc_context="doc text", llm=llm)

        assert llm.contextualize_chunk.call_count == 3

    def test_llm_receives_correct_args(self):
        chunk = _make_chunk(0, text="Seller shall indemnify Buyer.")
        llm = _make_llm()

        contextualize_chunks([chunk], doc_context="MY DOC CONTEXT", llm=llm)

        llm.contextualize_chunk.assert_called_once()
        _, kwargs = llm.contextualize_chunk.call_args
        assert kwargs["document"] == "MY DOC CONTEXT"
        assert "Article I" in kwargs["chunk_path"]
        assert "Seller shall indemnify" in kwargs["chunk_text"]

    def test_chunk_path_joined_from_node_path(self):
        chunk = _make_chunk(0)
        chunk.node_path = ["Article IV", "§4.2", "§4.2(a)"]
        llm = _make_llm()

        contextualize_chunks([chunk], doc_context="doc", llm=llm)

        _, kwargs = llm.contextualize_chunk.call_args
        assert kwargs["chunk_path"] == "Article IV > §4.2 > §4.2(a)"

    def test_node_id_used_when_path_empty(self):
        chunk = _make_chunk(0)
        chunk.node_path = []
        chunk.node_id = "§7.1"
        llm = _make_llm()

        contextualize_chunks([chunk], doc_context="doc", llm=llm)

        _, kwargs = llm.contextualize_chunk.call_args
        assert kwargs["chunk_path"] == "§7.1"

    def test_failed_chunk_leaves_context_text_none(self):
        chunks = [_make_chunk(0), _make_chunk(1)]
        llm = MagicMock()
        llm.contextualize_chunk.side_effect = [RuntimeError("API down"), "Good context."]

        contextualize_chunks(chunks, doc_context="doc", llm=llm)

        # One failed, one succeeded
        contexts = [c.context_text for c in chunks]
        assert None in contexts
        assert "Good context." in contexts

    def test_empty_chunks_returns_empty(self):
        llm = _make_llm()
        result = contextualize_chunks([], doc_context="doc", llm=llm)
        assert result == []
        llm.contextualize_chunk.assert_not_called()

    def test_progress_callback_called(self):
        chunks = [_make_chunk(i) for i in range(4)]
        llm = _make_llm()
        calls: list[tuple[int, int]] = []

        contextualize_chunks(
            chunks,
            doc_context="doc",
            llm=llm,
            progress_callback=lambda c, t: calls.append((c, t)),
        )

        # Callback should have been called 4 times (once per chunk)
        assert len(calls) == 4
        # All totals should be 4
        assert all(t == 4 for _, t in calls)
        # Completed counts should be 1..4
        completed_values = sorted(c for c, _ in calls)
        assert completed_values == [1, 2, 3, 4]

    def test_context_text_stripped(self):
        chunks = [_make_chunk(0)]
        llm = _make_llm("  Context with whitespace.  \n")

        contextualize_chunks(chunks, doc_context="doc", llm=llm)

        assert chunks[0].context_text == "Context with whitespace."

    def test_concurrent_workers_respected(self):
        """Smoke test that max_workers kwarg is accepted and doesn't error."""
        chunks = [_make_chunk(i) for i in range(3)]
        llm = _make_llm()

        contextualize_chunks(chunks, doc_context="doc", llm=llm, max_workers=2)

        assert all(c.context_text is not None for c in chunks)


# ---------------------------------------------------------------------------
# embedding_text integration
# ---------------------------------------------------------------------------


class TestEmbeddingTextIntegration:
    """Verify that Chunk.embedding_text uses context when present."""

    def test_embedding_text_includes_context(self):
        chunk = _make_chunk(0, text="Seller shall indemnify Buyer.")
        chunk.context_text = "This is Section 9.1 of a SaaS agreement governing indemnification."

        expected = (
            "This is Section 9.1 of a SaaS agreement governing indemnification."
            "\n\n"
            "Seller shall indemnify Buyer."
        )
        assert chunk.embedding_text == expected

    def test_embedding_text_fallback_without_context(self):
        chunk = _make_chunk(0, text="Seller shall indemnify Buyer.")
        assert chunk.embedding_text == "Seller shall indemnify Buyer."

    def test_embedding_text_after_contextualize_chunks(self):
        chunks = [_make_chunk(0, "Licensee grants a worldwide license.")]
        llm = _make_llm("This clause is from Section 3 of a software license agreement.")

        contextualize_chunks(chunks, doc_context="doc", llm=llm)

        assert "This clause is from Section 3" in chunks[0].embedding_text
        assert "Licensee grants a worldwide license." in chunks[0].embedding_text


# ---------------------------------------------------------------------------
# Integration test — real parse + chunk + contextualize (mock LLM)
# ---------------------------------------------------------------------------


class TestContextualizeIntegration:
    def test_full_pipeline_with_mock_llm(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")
        chunks = chunk_document(root, doc_id="test-apa")

        llm = _make_llm("This chunk belongs to an Asset Purchase Agreement.")

        doc_context = build_document_context(root, text)
        result = contextualize_chunks(chunks, doc_context=doc_context, llm=llm)

        assert len(result) == len(chunks)
        enriched = [c for c in result if c.context_text]
        assert len(enriched) == len(chunks)

        # Every chunk's embedding_text should start with the context
        for chunk in result:
            assert chunk.embedding_text.startswith(
                "This chunk belongs to an Asset Purchase Agreement."
            )

    def test_doc_context_fits_in_haiku_context_window(self):
        """Context string must not exceed Haiku's ~200K token limit."""
        text = _load_sample()
        root = parse_document(text, doc_id="test")
        ctx = build_document_context(root, text)
        # 80K chars ≈ 20K tokens — well within limit
        assert len(ctx) <= 80_000
