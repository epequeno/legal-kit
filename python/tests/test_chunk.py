"""Tests for document chunking."""

from pathlib import Path

from legalkit.pipeline.parse import parse_document
from legalkit.pipeline.chunk import chunk_document

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_sample() -> str:
    return (FIXTURE_DIR / "sample_contract.txt").read_text()


class TestChunkDocument:
    def test_produces_chunks(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")
        chunks = chunk_document(root, doc_id="test-apa")

        assert len(chunks) > 0
        print(f"\nProduced {len(chunks)} chunks")

    def test_chunks_have_ids(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")
        chunks = chunk_document(root, doc_id="test-apa")

        for chunk in chunks:
            assert chunk.id
            assert chunk.id.startswith("test-apa-chunk-")

    def test_chunks_have_node_ids(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")
        chunks = chunk_document(root, doc_id="test-apa")

        for chunk in chunks:
            assert chunk.node_id

    def test_chunks_respect_max_size(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")
        chunks = chunk_document(root, doc_id="test-apa", max_chars=3000)

        for chunk in chunks:
            assert len(chunk.text) <= 3500  # allow small overshoot from merging

    def test_chunks_cover_key_content(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")
        chunks = chunk_document(root, doc_id="test-apa")

        all_text = " ".join(c.text for c in chunks)

        # Key content should appear in at least one chunk
        assert "Purchase Price" in all_text
        assert "indemnif" in all_text.lower()
        assert "Fifty Million" in all_text
        assert "$5,000,000" in all_text

    def test_chunks_have_cross_references(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")
        chunks = chunk_document(root, doc_id="test-apa")

        chunks_with_xrefs = [c for c in chunks if c.cross_references]
        assert len(chunks_with_xrefs) > 0

    def test_chunks_have_valid_offsets(self):
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")
        chunks = chunk_document(root, doc_id="test-apa")

        for chunk in chunks:
            assert chunk.char_start >= 0
            assert chunk.char_end >= chunk.char_start

    def test_print_chunks(self, capsys):
        """Visual test — run with pytest -s."""
        text = _load_sample()
        root = parse_document(text, doc_id="test-apa")
        chunks = chunk_document(root, doc_id="test-apa")

        print(f"\n{'='*80}")
        print(f"Total chunks: {len(chunks)}")
        print(f"{'='*80}")
        for chunk in chunks:
            xrefs = f" xrefs={chunk.cross_references}" if chunk.cross_references else ""
            print(
                f"\n[{chunk.id}] node={chunk.node_id} "
                f"({len(chunk.text)} chars){xrefs}"
            )
            preview = chunk.text[:120].replace("\n", " ")
            print(f"  {preview}...")
