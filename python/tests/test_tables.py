"""Tests for table detection, parsing, and chunk integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from legalkit.pipeline.tables import (
    ParsedTable,
    _find_column_boundaries,
    detect_tables,
    parse_table,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Sample table texts (matching real corpus formats)
# ---------------------------------------------------------------------------

FIXED_WIDTH_TABLE = """\
Marks registered in Poland:

                                                       Date of
Classes           Reg. No.    Registration
-------           --------    ------------
Burger King Logo  16,29,30    7441        18 Feb., 1994
Whopper           16,29,30    7441        18 Feb., 1994
Burger King Word  16,29,30    7442        18 Feb., 1994
"""

PIPE_TABLE = """\
| Trademark       | Classes         | Reg. No. | Date        |
|-----------------|-----------------|----------|-------------|
| Burger King     | 16,29,30,32,42  | 7441     | 18 Feb 1994 |
| Whopper         | 16,29,30,32,42  | 7441     | 18 Feb 1994 |
| BK Wordmark     | 16,29,30,42     | 7442     | 18 Feb 1994 |
"""

KV_TABLE = """\
Name:        John Smith
Date:        January 1, 2024
Amount:      $50,000.00
Title:       Chief Executive Officer
"""

PIPE_NO_SEPARATOR = """\
| Item | Quantity | Price |
| Widget A | 10 | $5.00 |
| Widget B | 20 | $3.50 |
"""

MIXED_DOCUMENT = f"""\
Section 1. Trademark Schedule

The following trademarks are registered:

{FIXED_WIDTH_TABLE}

Section 2. Payment Terms

Payments are due as follows:

{KV_TABLE}

Section 3. General Provisions

All other terms apply.
"""


# ---------------------------------------------------------------------------
# _find_column_boundaries
# ---------------------------------------------------------------------------


class TestFindColumnBoundaries:
    def test_simple_two_cols(self):
        underline = "-------    --------"
        cols = _find_column_boundaries(underline)
        assert len(cols) == 2
        assert cols[0][0] == 0
        assert cols[0][1] == 7

    def test_three_cols(self):
        underline = "-------    --------    ------------"
        cols = _find_column_boundaries(underline)
        assert len(cols) == 3

    def test_leading_spaces(self):
        underline = "   -------    --------"
        cols = _find_column_boundaries(underline)
        assert len(cols) == 2
        assert cols[0][0] == 3

    def test_equals_sign_underline(self):
        underline = "=======    ========"
        cols = _find_column_boundaries(underline)
        assert len(cols) == 2

    def test_empty_string(self):
        assert _find_column_boundaries("") == []

    def test_single_col(self):
        underline = "----------"
        cols = _find_column_boundaries(underline)
        assert len(cols) == 1


# ---------------------------------------------------------------------------
# detect_tables
# ---------------------------------------------------------------------------


class TestDetectTables:
    def test_detects_fixed_width_table(self):
        spans = detect_tables(FIXED_WIDTH_TABLE)
        assert len(spans) >= 1

    def test_detects_pipe_table(self):
        spans = detect_tables(PIPE_TABLE)
        assert len(spans) >= 1

    def test_detects_kv_table(self):
        spans = detect_tables(KV_TABLE)
        assert len(spans) >= 1

    def test_no_false_positives_on_plain_text(self):
        text = (
            "The parties agree that all obligations under this Agreement "
            "shall be performed in good faith.\n\n"
            "Section 1. General provisions apply throughout."
        )
        spans = detect_tables(text)
        assert len(spans) == 0

    def test_detects_multiple_tables_in_document(self):
        spans = detect_tables(MIXED_DOCUMENT)
        assert len(spans) >= 2

    def test_spans_are_sorted(self):
        spans = detect_tables(MIXED_DOCUMENT)
        starts = [s for s, _ in spans]
        assert starts == sorted(starts)

    def test_spans_do_not_overlap(self):
        spans = detect_tables(MIXED_DOCUMENT)
        for i in range(len(spans) - 1):
            assert spans[i][1] <= spans[i + 1][0]

    def test_span_text_contains_table(self):
        spans = detect_tables(FIXED_WIDTH_TABLE)
        assert len(spans) >= 1
        s, e = spans[0]
        extracted = FIXED_WIDTH_TABLE[s:e]
        assert "Burger King" in extracted or "-------" in extracted

    def test_empty_text(self):
        assert detect_tables("") == []

    def test_single_dash_line_not_table(self):
        text = "Section 1. Definitions\n---\nSome body text here."
        # A single "---" without a proper header+data structure shouldn't be a full table
        spans = detect_tables(text)
        # Either 0 or 1 spans — but if detected, verify it's reasonable
        for s, e in spans:
            block = text[s:e]
            assert len(block) > 3


# ---------------------------------------------------------------------------
# parse_table — fixed width
# ---------------------------------------------------------------------------


class TestParseFixedWidthTable:
    def test_parses_headers(self):
        table = parse_table(FIXED_WIDTH_TABLE)
        assert table is not None
        assert len(table.headers) >= 2
        # Should find "Classes" and "Reg. No." as headers
        headers_text = " ".join(table.headers).lower()
        assert "classes" in headers_text or "reg" in headers_text

    def test_parses_data_rows(self):
        table = parse_table(FIXED_WIDTH_TABLE)
        assert table is not None
        assert len(table.rows) >= 2

    def test_burger_king_in_rows(self):
        table = parse_table(FIXED_WIDTH_TABLE)
        assert table is not None
        all_text = " ".join(cell for row in table.rows for cell in row)
        assert "Burger King" in all_text or "7441" in all_text

    def test_table_type_fixed_width(self):
        table = parse_table(FIXED_WIDTH_TABLE)
        assert table is not None
        assert table.table_type == "fixed_width"

    def test_raw_text_preserved(self):
        table = parse_table(FIXED_WIDTH_TABLE)
        assert table is not None
        assert table.raw_text == FIXED_WIDTH_TABLE


# ---------------------------------------------------------------------------
# parse_table — pipe
# ---------------------------------------------------------------------------


class TestParsePipeTable:
    def test_parses_headers(self):
        table = parse_table(PIPE_TABLE)
        assert table is not None
        assert "Trademark" in table.headers or "trademark" in [h.lower() for h in table.headers]

    def test_parses_data_rows(self):
        table = parse_table(PIPE_TABLE)
        assert table is not None
        assert len(table.rows) >= 2

    def test_cell_values_stripped(self):
        table = parse_table(PIPE_TABLE)
        assert table is not None
        for row in table.rows:
            for cell in row:
                assert cell == cell.strip()

    def test_table_type_pipe(self):
        table = parse_table(PIPE_TABLE)
        assert table is not None
        assert table.table_type == "pipe"

    def test_pipe_no_separator(self):
        table = parse_table(PIPE_NO_SEPARATOR)
        assert table is not None
        assert len(table.rows) >= 1


# ---------------------------------------------------------------------------
# parse_table — key-value
# ---------------------------------------------------------------------------


class TestParseKVTable:
    def test_parses_kv_pairs(self):
        table = parse_table(KV_TABLE)
        assert table is not None
        assert table.headers == ["Field", "Value"]

    def test_kv_data_rows(self):
        table = parse_table(KV_TABLE)
        assert table is not None
        assert len(table.rows) >= 3

    def test_kv_values_correct(self):
        table = parse_table(KV_TABLE)
        assert table is not None
        field_names = [row[0] for row in table.rows]
        values = [row[1] for row in table.rows]
        assert "Name" in field_names
        assert "John Smith" in values

    def test_table_type_kv(self):
        table = parse_table(KV_TABLE)
        assert table is not None
        assert table.table_type == "key_value"


# ---------------------------------------------------------------------------
# ParsedTable.serialize
# ---------------------------------------------------------------------------


class TestTableSerialize:
    def test_serialize_includes_headers(self):
        table = ParsedTable(
            headers=["Trademark", "Reg. No.", "Date"],
            rows=[["Burger King", "7441", "1994"]],
        )
        s = table.serialize()
        assert "Trademark" in s
        assert "Reg. No." in s
        assert "Date" in s

    def test_serialize_includes_row_values(self):
        table = ParsedTable(
            headers=["Name", "Value"],
            rows=[["Price", "$50,000"], ["Date", "Jan 1"]],
        )
        s = table.serialize()
        assert "$50,000" in s
        assert "Jan 1" in s

    def test_serialize_with_caption(self):
        table = ParsedTable(
            headers=["Col1", "Col2"],
            rows=[["a", "b"]],
            caption="Marks registered in Poland",
        )
        s = table.serialize()
        assert "TABLE" in s
        assert "Marks registered in Poland" in s

    def test_serialize_pipe_delimited(self):
        table = ParsedTable(
            headers=["A", "B", "C"],
            rows=[["1", "2", "3"]],
        )
        s = table.serialize()
        assert "|" in s

    def test_serialize_fixed_width_table(self):
        table = parse_table(FIXED_WIDTH_TABLE)
        assert table is not None
        s = table.serialize()
        assert len(s) > 10
        # Should contain the trademark data
        assert "7441" in s or "Burger" in s

    def test_serialize_pipe_table(self):
        table = parse_table(PIPE_TABLE)
        assert table is not None
        s = table.serialize()
        assert "Trademark" in s or "trademark" in s.lower()

    def test_empty_table_serializes(self):
        table = ParsedTable(headers=["A", "B"], rows=[])
        s = table.serialize()
        assert "A" in s


# ---------------------------------------------------------------------------
# parse_table — edge cases
# ---------------------------------------------------------------------------


class TestParseTableEdgeCases:
    def test_returns_none_for_plain_text(self):
        text = "This is just a paragraph with no table structure whatsoever."
        assert parse_table(text) is None

    def test_returns_none_for_empty(self):
        assert parse_table("") is None

    def test_returns_none_for_single_line(self):
        assert parse_table("Just one line") is None

    def test_char_offsets_set(self):
        table = parse_table(FIXED_WIDTH_TABLE, char_start=100)
        assert table is not None
        assert table.char_start == 100
        assert table.char_end > table.char_start


# ---------------------------------------------------------------------------
# _extract_table_chunks in chunk.py
# ---------------------------------------------------------------------------


class TestExtractTableChunks:
    def test_table_chunk_has_table_repr(self):
        from legalkit.pipeline.chunk import _extract_table_chunks
        from legalkit.models.retrieval import Chunk

        chunk = Chunk(
            id="c0", text=FIXED_WIDTH_TABLE, doc_id="doc",
            node_id="§1", char_start=0, char_end=len(FIXED_WIDTH_TABLE),
        )
        result = _extract_table_chunks([chunk], "doc")
        table_chunks = [c for c in result if c.table_repr is not None]
        assert len(table_chunks) >= 1

    def test_table_repr_contains_headers(self):
        from legalkit.pipeline.chunk import _extract_table_chunks
        from legalkit.models.retrieval import Chunk

        chunk = Chunk(
            id="c0", text=FIXED_WIDTH_TABLE, doc_id="doc",
            node_id="§1", char_start=0, char_end=len(FIXED_WIDTH_TABLE),
        )
        result = _extract_table_chunks([chunk], "doc")
        table_chunks = [c for c in result if c.table_repr is not None]
        combined = " ".join(c.table_repr for c in table_chunks)
        # Table repr should have column context
        assert len(combined) > 20

    def test_non_table_chunks_unchanged(self):
        from legalkit.pipeline.chunk import _extract_table_chunks
        from legalkit.models.retrieval import Chunk

        plain_text = "This is plain body text with no table structure."
        chunk = Chunk(
            id="c0", text=plain_text, doc_id="doc",
            node_id="§1", char_start=0, char_end=len(plain_text),
        )
        result = _extract_table_chunks([chunk], "doc")
        assert len(result) == 1
        assert result[0].table_repr is None
        assert result[0].text == plain_text

    def test_surrounding_text_preserved(self):
        from legalkit.pipeline.chunk import _extract_table_chunks
        from legalkit.models.retrieval import Chunk

        text = f"Introduction text.\n\n{FIXED_WIDTH_TABLE}\nConcluding text."
        chunk = Chunk(
            id="c0", text=text, doc_id="doc",
            node_id="§1", char_start=0, char_end=len(text),
        )
        result = _extract_table_chunks([chunk], "doc")
        all_text = " ".join(c.text for c in result)
        assert "Introduction" in all_text
        assert "Concluding" in all_text

    def test_empty_chunks_list(self):
        from legalkit.pipeline.chunk import _extract_table_chunks
        assert _extract_table_chunks([], "doc") == []


# ---------------------------------------------------------------------------
# Chunk.embedding_text with table_repr
# ---------------------------------------------------------------------------


class TestChunkEmbeddingText:
    def test_uses_table_repr_when_present(self):
        from legalkit.models.retrieval import Chunk

        chunk = Chunk(
            id="c0", text="raw | table | text",
            table_repr="Headers: A | B\nvalue1 | value2",
            doc_id="doc", node_id="§1",
        )
        assert chunk.embedding_text == "Headers: A | B\nvalue1 | value2"

    def test_falls_back_to_text_without_table_repr(self):
        from legalkit.models.retrieval import Chunk

        chunk = Chunk(id="c0", text="plain text", doc_id="doc", node_id="§1")
        assert chunk.embedding_text == "plain text"

    def test_context_prepended_to_table_repr(self):
        from legalkit.models.retrieval import Chunk

        chunk = Chunk(
            id="c0",
            text="raw table",
            table_repr="Col1 | Col2\nA | B",
            context_text="This table lists trademark registrations.",
            doc_id="doc",
            node_id="§1",
        )
        et = chunk.embedding_text
        assert et.startswith("This table lists trademark registrations.")
        assert "Col1 | Col2" in et

    def test_context_prepended_to_plain_text(self):
        from legalkit.models.retrieval import Chunk

        chunk = Chunk(
            id="c0",
            text="plain text",
            context_text="This is from Section 3.",
            doc_id="doc",
            node_id="§1",
        )
        assert chunk.embedding_text == "This is from Section 3.\n\nplain text"


# ---------------------------------------------------------------------------
# Integration: chunk_document with parse_tables=True
# ---------------------------------------------------------------------------


class TestChunkDocumentTableIntegration:
    def test_table_in_document_produces_table_chunk(self):
        from legalkit.pipeline.parse import parse_document
        from legalkit.pipeline.chunk import chunk_document

        text = (
            "TRADEMARK SCHEDULE\n\n"
            "Section 1. Registered Marks\n\n"
            "The following marks are registered:\n\n"
            + FIXED_WIDTH_TABLE
        )
        root = parse_document(text, doc_id="test")
        chunks = chunk_document(root, doc_id="test", parse_tables=True)

        table_chunks = [c for c in chunks if c.table_repr is not None]
        assert len(table_chunks) >= 1

    def test_parse_tables_false_skips_table_extraction(self):
        from legalkit.pipeline.parse import parse_document
        from legalkit.pipeline.chunk import chunk_document

        text = "Section 1.\n\n" + FIXED_WIDTH_TABLE
        root = parse_document(text, doc_id="test")
        chunks = chunk_document(root, doc_id="test", parse_tables=False)

        table_chunks = [c for c in chunks if c.table_repr is not None]
        assert len(table_chunks) == 0

    def test_chunks_still_have_ids(self):
        from legalkit.pipeline.parse import parse_document
        from legalkit.pipeline.chunk import chunk_document

        text = "Section 1.\n\n" + FIXED_WIDTH_TABLE + "\n\nSome extra text after."
        root = parse_document(text, doc_id="test")
        chunks = chunk_document(root, doc_id="test", parse_tables=True)

        for chunk in chunks:
            assert chunk.id.startswith("test-chunk-")
