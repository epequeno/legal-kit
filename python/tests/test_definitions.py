"""Tests for definition extraction and linking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from legalkit.models.document import NodeType
from legalkit.models.retrieval import RetrievalMethod
from legalkit.pipeline.definitions import DefinitionEntry, DefinitionIndex
from legalkit.pipeline.parse import find_definitions, parse_document

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_sample() -> str:
    return (FIXTURE_DIR / "sample_contract.txt").read_text()


def _load_nda() -> str:
    return (FIXTURE_DIR / "nda_bare_sections.txt").read_text()


# ---------------------------------------------------------------------------
# find_definitions (parse.py)
# ---------------------------------------------------------------------------


class TestFindDefinitions:
    def test_finds_simple_term(self):
        text = '"Affiliate" means any Person that controls another Person.'
        results = find_definitions(text)
        assert len(results) == 1
        term, offset, body = results[0]
        assert term == "Affiliate"

    def test_captures_definition_body(self):
        text = '"Affiliate" means any Person that controls another Person.'
        _, _, body = find_definitions(text)[0]
        assert "Person" in body
        assert "controls" in body

    def test_shall_mean_variant(self):
        text = '"Business Day" shall mean any day other than Saturday.'
        results = find_definitions(text)
        assert results[0][0] == "Business Day"

    def test_multiple_terms(self):
        text = (
            '"Affiliate" means any Person that controls another.\n\n'
            '"Business Day" means any day other than Saturday.'
        )
        results = find_definitions(text)
        terms = [r[0] for r in results]
        assert "Affiliate" in terms
        assert "Business Day" in terms

    def test_returns_char_offset(self):
        text = 'Some preamble text.\n\n"Term" means something.'
        results = find_definitions(text)
        assert len(results) == 1
        _, offset, _ = results[0]
        assert offset > 0
        assert text[offset:offset + 1] == '"'

    def test_no_definitions_returns_empty(self):
        text = "The parties hereby agree to the following terms and conditions."
        assert find_definitions(text) == []

    def test_sample_contract_finds_definitions(self):
        text = _load_sample()
        results = find_definitions(text)
        terms = [r[0] for r in results]
        assert "Affiliate" in terms
        assert "Business Day" in terms
        assert "Person" in terms


# ---------------------------------------------------------------------------
# DefinitionEntry
# ---------------------------------------------------------------------------


class TestDefinitionEntry:
    def _make_entry(self) -> DefinitionEntry:
        return DefinitionEntry(
            term="Material Adverse Effect",
            definition_text="any change that has a material adverse effect on the business",
            node_id="§1.1",
            node_text='"Material Adverse Effect" means any change...',
            char_start=0,
            char_end=100,
        )

    def test_normalized_term(self):
        entry = self._make_entry()
        assert entry.normalized_term == "material adverse effect"

    def test_full_text_format(self):
        entry = self._make_entry()
        assert '"Material Adverse Effect" means' in entry.full_text
        assert "material adverse effect" in entry.full_text.lower()

    def test_repr(self):
        entry = self._make_entry()
        assert "Material Adverse Effect" in repr(entry)


# ---------------------------------------------------------------------------
# DefinitionIndex.build
# ---------------------------------------------------------------------------


class TestDefinitionIndexBuild:
    def test_builds_from_sample_contract(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test")
        idx = DefinitionIndex()
        idx.build(root)
        assert len(idx) > 0

    def test_finds_affiliate(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test")
        idx = DefinitionIndex()
        idx.build(root)
        assert idx.lookup("Affiliate") is not None

    def test_finds_business_day(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test")
        idx = DefinitionIndex()
        idx.build(root)
        assert idx.lookup("Business Day") is not None

    def test_finds_person(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test")
        idx = DefinitionIndex()
        idx.build(root)
        assert idx.lookup("Person") is not None

    def test_finds_purchase_price(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test")
        idx = DefinitionIndex()
        idx.build(root)
        # "Purchase Price" has the meaning set forth in Section 2.3
        entry = idx.lookup("Purchase Price")
        assert entry is not None

    def test_terms_list(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test")
        idx = DefinitionIndex()
        idx.build(root)
        assert "Affiliate" in idx.terms

    def test_empty_document(self):
        root = parse_document("No definitions here.", doc_id="empty")
        idx = DefinitionIndex()
        idx.build(root)
        assert len(idx) == 0

    def test_definition_body_captured(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test")
        idx = DefinitionIndex()
        idx.build(root)
        entry = idx.lookup("Affiliate")
        assert entry is not None
        assert len(entry.definition_text) > 0

    def test_node_id_recorded(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test")
        idx = DefinitionIndex()
        idx.build(root)
        entry = idx.lookup("Affiliate")
        assert entry is not None
        assert entry.node_id  # non-empty node ID


# ---------------------------------------------------------------------------
# DefinitionIndex.lookup
# ---------------------------------------------------------------------------


class TestDefinitionIndexLookup:
    def _make_index(self) -> DefinitionIndex:
        text = _load_sample()
        root = parse_document(text, doc_id="test")
        idx = DefinitionIndex()
        idx.build(root)
        return idx

    def test_lookup_case_insensitive(self):
        idx = self._make_index()
        assert idx.lookup("affiliate") is not None
        assert idx.lookup("AFFILIATE") is not None
        assert idx.lookup("Affiliate") is not None

    def test_lookup_unknown_term_returns_none(self):
        idx = self._make_index()
        assert idx.lookup("NonExistentTerm") is None

    def test_lookup_returns_entry(self):
        idx = self._make_index()
        entry = idx.lookup("Affiliate")
        assert isinstance(entry, DefinitionEntry)
        assert entry.term == "Affiliate"


# ---------------------------------------------------------------------------
# DefinitionIndex.find_references
# ---------------------------------------------------------------------------


class TestDefinitionIndexFindReferences:
    def _make_index(self) -> DefinitionIndex:
        text = _load_sample()
        root = parse_document(text, doc_id="test")
        idx = DefinitionIndex()
        idx.build(root)
        return idx

    def test_finds_term_in_text(self):
        idx = self._make_index()
        text = "The Affiliate of Seller shall have the right to terminate."
        refs = idx.find_references(text)
        terms = [r.term for r in refs]
        assert "Affiliate" in terms

    def test_finds_multiple_terms(self):
        idx = self._make_index()
        text = "The Business Day following the Closing Date."
        refs = idx.find_references(text)
        terms = [r.term for r in refs]
        assert "Business Day" in terms

    def test_no_duplicates_in_results(self):
        idx = self._make_index()
        text = "Affiliate means a related Affiliate of the company."
        refs = idx.find_references(text)
        terms = [r.term for r in refs]
        assert terms.count("Affiliate") <= 1

    def test_empty_text_returns_empty(self):
        idx = self._make_index()
        assert idx.find_references("") == []

    def test_no_match_returns_empty(self):
        idx = self._make_index()
        assert idx.find_references("completely unrelated text about nothing") == []

    def test_returns_definition_entries(self):
        idx = self._make_index()
        refs = idx.find_references("The Affiliate shall comply.")
        assert all(isinstance(r, DefinitionEntry) for r in refs)

    def test_no_pattern_when_empty_index(self):
        idx = DefinitionIndex()
        idx.build(parse_document("No definitions.", doc_id="x"))
        assert idx.find_references("some text") == []


# ---------------------------------------------------------------------------
# Node annotation (_annotate_definitions in parse.py)
# ---------------------------------------------------------------------------


class TestNodeAnnotation:
    def test_definition_nodes_have_defined_terms(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test")
        nodes_with_terms = [
            n for n in root.all_nodes()
            if n.defined_terms
        ]
        assert len(nodes_with_terms) > 0

    def test_affiliate_term_on_node(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test")
        nodes_with_affiliate = [
            n for n in root.all_nodes()
            if "Affiliate" in n.defined_terms
        ]
        assert len(nodes_with_affiliate) >= 1

    def test_definition_node_type_promoted(self):
        """Nodes whose text is primarily definitions get NodeType.DEFINITION."""
        text = _load_sample()
        root = parse_document(text, doc_id="test")
        def_nodes = [n for n in root.all_nodes() if n.type == NodeType.DEFINITION]
        assert len(def_nodes) > 0


# ---------------------------------------------------------------------------
# _expand_definitions in retrieve.py
# ---------------------------------------------------------------------------


class TestExpandDefinitions:
    def _build_setup(self):
        """Build a real index + definition index from the sample contract."""
        from legalkit.pipeline.chunk import chunk_document
        from legalkit.pipeline.index import DocumentIndex

        text = _load_sample()
        root = parse_document(text, doc_id="test")
        chunks = chunk_document(root, doc_id="test")

        rng = np.random.default_rng(42)
        dim = 16
        emb = rng.random((len(chunks), dim), dtype=np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        index = DocumentIndex()
        index.build(chunks, emb)

        def_index = DefinitionIndex()
        def_index.build(root)

        return index, def_index, chunks

    def test_expand_definitions_returns_list(self):
        from legalkit.pipeline.retrieve import _expand_definitions
        from legalkit.models.retrieval import RankedChunk

        index, def_index, chunks = self._build_setup()
        # Simulate retrieved chunks that mention "Affiliate"
        rc = RankedChunk(
            chunk=next(c for c in chunks if "Affiliate" in c.text or "Encumbrance" in c.text),
            score=0.9,
            method=RetrievalMethod.HYBRID,
        )
        result = _expand_definitions(
            ranked_chunks=[rc],
            cross_ref_chunks=[],
            index=index,
            definition_index=def_index,
        )
        assert isinstance(result, list)

    def test_expand_definitions_method_is_definition(self):
        from legalkit.pipeline.retrieve import _expand_definitions
        from legalkit.models.retrieval import RankedChunk

        index, def_index, chunks = self._build_setup()
        # Use a chunk that mentions many defined terms
        text_chunk = next(
            (c for c in chunks if "Material Adverse Effect" in c.text
             or "Business Day" in c.text),
            chunks[0],
        )
        rc = RankedChunk(chunk=text_chunk, score=0.9, method=RetrievalMethod.HYBRID)
        result = _expand_definitions(
            ranked_chunks=[rc],
            cross_ref_chunks=[],
            index=index,
            definition_index=def_index,
        )
        assert all(r.method == RetrievalMethod.DEFINITION for r in result)

    def test_no_duplicate_nodes_added(self):
        from legalkit.pipeline.retrieve import _expand_definitions
        from legalkit.models.retrieval import RankedChunk

        index, def_index, chunks = self._build_setup()
        # Mark ALL chunks as already retrieved
        ranked = [
            RankedChunk(chunk=c, score=0.5, method=RetrievalMethod.HYBRID)
            for c in chunks
        ]
        result = _expand_definitions(
            ranked_chunks=ranked,
            cross_ref_chunks=[],
            index=index,
            definition_index=def_index,
        )
        # Nothing new should be added since everything is already retrieved
        assert result == []

    def test_respects_max_chunks(self):
        from legalkit.pipeline.retrieve import _expand_definitions
        from legalkit.models.retrieval import RankedChunk

        index, def_index, chunks = self._build_setup()
        # Chunk mentioning many defined terms
        heavy_text_chunk = max(
            chunks,
            key=lambda c: len(def_index.find_references(c.text)),
        )
        rc = RankedChunk(chunk=heavy_text_chunk, score=0.9, method=RetrievalMethod.HYBRID)
        result = _expand_definitions(
            ranked_chunks=[rc],
            cross_ref_chunks=[],
            index=index,
            definition_index=def_index,
            max_chunks=2,
        )
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# Full retrieve() integration with definition_index
# ---------------------------------------------------------------------------


class TestRetrieveWithDefinitions:
    def test_retrieve_populates_definition_chunks(self):
        from legalkit.pipeline.retrieve import retrieve
        from legalkit.pipeline.chunk import chunk_document
        from legalkit.pipeline.index import DocumentIndex

        text = _load_sample()
        root = parse_document(text, doc_id="test")
        chunks = chunk_document(root, doc_id="test")

        rng = np.random.default_rng(0)
        dim = 16
        emb = rng.random((len(chunks), dim), dtype=np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        index = DocumentIndex()
        index.build(chunks, emb)

        def_index = DefinitionIndex()
        def_index.build(root)

        q_emb = rng.random(dim, dtype=np.float32)

        result = retrieve(
            query="What are the indemnification obligations?",
            query_embedding=q_emb,
            index=index,
            doc_tree=root,
            top_k=5,
            definition_index=def_index,
        )

        # definition_chunks is a list (may be empty if no terms referenced)
        assert isinstance(result.definition_chunks, list)

    def test_retrieve_without_definition_index_has_empty_def_chunks(self):
        from legalkit.pipeline.retrieve import retrieve
        from legalkit.pipeline.chunk import chunk_document
        from legalkit.pipeline.index import DocumentIndex

        text = _load_sample()
        root = parse_document(text, doc_id="test")
        chunks = chunk_document(root, doc_id="test")

        rng = np.random.default_rng(1)
        dim = 16
        emb = rng.random((len(chunks), dim), dtype=np.float32)
        index = DocumentIndex()
        index.build(chunks, emb)

        q_emb = rng.random(dim, dtype=np.float32)
        result = retrieve(
            query="purchase price",
            query_embedding=q_emb,
            index=index,
            doc_tree=root,
            top_k=5,
        )
        assert result.definition_chunks == []

    def test_all_chunks_includes_definitions(self):
        from legalkit.pipeline.retrieve import retrieve
        from legalkit.pipeline.chunk import chunk_document
        from legalkit.pipeline.index import DocumentIndex

        text = _load_sample()
        root = parse_document(text, doc_id="test")
        chunks = chunk_document(root, doc_id="test")

        rng = np.random.default_rng(2)
        dim = 16
        emb = rng.random((len(chunks), dim), dtype=np.float32)
        index = DocumentIndex()
        index.build(chunks, emb)

        def_index = DefinitionIndex()
        def_index.build(root)

        q_emb = rng.random(dim, dtype=np.float32)
        result = retrieve(
            query="indemnification",
            query_embedding=q_emb,
            index=index,
            doc_tree=root,
            top_k=5,
            definition_index=def_index,
        )

        # all_chunks = ranked + cross_ref + definitions
        expected = (
            len(result.ranked_chunks)
            + len(result.cross_ref_chunks)
            + len(result.definition_chunks)
        )
        assert len(result.all_chunks) == expected
