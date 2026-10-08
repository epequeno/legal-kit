"""Tests for multi-format text extraction."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from legalkit.pipeline.extract import (
    SUPPORTED_EXTENSIONS,
    _extract_txt,
    _table_to_text,
    extract_text,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"

SAMPLE_LEGAL_TEXT = """\
SOFTWARE LICENSE AGREEMENT

This Agreement is entered into as of January 1, 2024.

1. Definitions

"Licensed Software" means the proprietary software product.

2. License Grant

Licensor grants a non-exclusive license to Licensee.
"""


# ---------------------------------------------------------------------------
# Helpers — create minimal in-memory DOCX and PDF files
# ---------------------------------------------------------------------------


def _make_docx(text: str | None = None, with_table: bool = False) -> Path:
    """Create a minimal DOCX file in a temp directory."""
    import docx as _docx

    doc = _docx.Document()
    doc.add_heading("Test Agreement", level=1)
    doc.add_paragraph(text or "This is a test legal document.")
    doc.add_paragraph("Section 1. Definitions")
    doc.add_paragraph('"Term" means something important.')

    if with_table:
        table = doc.add_table(rows=3, cols=3)
        headers = ["Trademark", "Reg. No.", "Date"]
        for i, h in enumerate(headers):
            table.rows[0].cells[i].text = h
        table.rows[1].cells[0].text = "Brand A"
        table.rows[1].cells[1].text = "12345"
        table.rows[1].cells[2].text = "2024-01-01"
        table.rows[2].cells[0].text = "Brand B"
        table.rows[2].cells[1].text = "67890"
        table.rows[2].cells[2].text = "2024-06-01"

    tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
    doc.save(tmp.name)
    return Path(tmp.name)


def _make_pdf(text: str = "Test legal document content.") -> Path:
    """Create a minimal PDF file in a temp directory."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text, fontsize=11)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc.save(tmp.name)
    doc.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# SUPPORTED_EXTENSIONS
# ---------------------------------------------------------------------------


class TestSupportedExtensions:
    def test_txt_supported(self):
        assert ".txt" in SUPPORTED_EXTENSIONS

    def test_pdf_supported(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS

    def test_docx_supported(self):
        assert ".docx" in SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# extract_text dispatch
# ---------------------------------------------------------------------------


class TestExtractTextDispatch:
    def test_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "doc.xyz"
        f.write_text("content")
        with pytest.raises(ValueError, match="Unsupported"):
            extract_text(f)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_text(tmp_path / "nonexistent.txt")

    def test_txt_dispatches_correctly(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Hello legal world")
        result = extract_text(f)
        assert result == "Hello legal world"

    def test_returns_string(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("test")
        assert isinstance(extract_text(f), str)


# ---------------------------------------------------------------------------
# Plain text extraction
# ---------------------------------------------------------------------------


class TestExtractTxt:
    def test_reads_utf8(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Café agreement — €50,000", encoding="utf-8")
        assert "Café" in _extract_txt(f)

    def test_latin1_fallback(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_bytes("Caf\xe9 agreement".encode("latin-1"))
        result = _extract_txt(f)
        assert len(result) > 0

    def test_reads_sample_fixture(self):
        text = _extract_txt(FIXTURE_DIR / "sample_contract.txt")
        assert len(text) > 1000
        assert "Purchase Price" in text

    def test_extract_text_txt_file(self, tmp_path):
        f = tmp_path / "agreement.txt"
        f.write_text(SAMPLE_LEGAL_TEXT)
        result = extract_text(f)
        assert "LICENSE AGREEMENT" in result


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


class TestExtractPDF:
    def test_extracts_text_from_simple_pdf(self):
        pdf_path = _make_pdf("This is a test legal agreement.\nSection 1. Definitions.")
        try:
            result = extract_text(pdf_path)
            assert isinstance(result, str)
            assert len(result) > 0
        finally:
            pdf_path.unlink(missing_ok=True)

    def test_extracted_text_contains_content(self):
        pdf_path = _make_pdf("ASSET PURCHASE AGREEMENT\n\nPurchase Price: $50,000,000")
        try:
            result = extract_text(pdf_path)
            assert "ASSET PURCHASE" in result or "Purchase" in result
        finally:
            pdf_path.unlink(missing_ok=True)

    def test_multipage_pdf_joined(self):
        import fitz

        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((50, 50), f"Page {i + 1} content.", fontsize=11)

        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        doc.save(tmp.name)
        doc.close()
        pdf_path = Path(tmp.name)

        try:
            result = extract_text(pdf_path)
            assert "Page 1" in result
            assert "Page 2" in result
            assert "Page 3" in result
        finally:
            pdf_path.unlink(missing_ok=True)

    def test_scanned_pdf_raises(self, tmp_path):
        """A PDF with no text layer (image-only) should raise RuntimeError."""
        import fitz

        # Create a PDF with a page but no text inserted
        doc = fitz.open()
        doc.new_page()  # blank page — no text
        pdf_path = tmp_path / "blank.pdf"
        doc.save(str(pdf_path))
        doc.close()

        with pytest.raises(RuntimeError, match="No text extracted"):
            extract_text(pdf_path)

    def test_pdf_metadata(self):
        from legalkit.pipeline.extract import extract_pdf_metadata

        pdf_path = _make_pdf("Test content")
        try:
            meta = extract_pdf_metadata(pdf_path)
            assert "pages" in meta
            assert meta["pages"] >= 1
            assert "title" in meta
            assert "encrypted" in meta
        finally:
            pdf_path.unlink(missing_ok=True)

    def test_nonexistent_pdf_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_text(tmp_path / "missing.pdf")


# ---------------------------------------------------------------------------
# DOCX extraction
# ---------------------------------------------------------------------------


class TestExtractDocx:
    def test_extracts_text_from_docx(self):
        docx_path = _make_docx()
        try:
            result = extract_text(docx_path)
            assert isinstance(result, str)
            assert len(result) > 0
        finally:
            docx_path.unlink(missing_ok=True)

    def test_extracts_paragraph_content(self):
        docx_path = _make_docx("The parties hereby agree to the following terms.")
        try:
            result = extract_text(docx_path)
            assert "parties hereby agree" in result
        finally:
            docx_path.unlink(missing_ok=True)

    def test_heading_preserved(self):
        docx_path = _make_docx()
        try:
            result = extract_text(docx_path)
            assert "Test Agreement" in result
        finally:
            docx_path.unlink(missing_ok=True)

    def test_table_serialized_as_pipe(self):
        docx_path = _make_docx(with_table=True)
        try:
            result = extract_text(docx_path)
            # Table should produce pipe-delimited rows
            assert "|" in result
            assert "Trademark" in result
            assert "Brand A" in result
        finally:
            docx_path.unlink(missing_ok=True)

    def test_definition_extractable(self):
        """Definitions in DOCX should survive extraction and be findable."""
        from legalkit.pipeline.parse import find_definitions

        docx_path = _make_docx()
        try:
            text = extract_text(docx_path)
            defs = find_definitions(text)
            terms = [d[0] for d in defs]
            assert "Term" in terms
        finally:
            docx_path.unlink(missing_ok=True)

    def test_docx_feeds_parse_document(self):
        """Extracted DOCX text should parse into a document tree."""
        from legalkit.pipeline.parse import parse_document

        docx_path = _make_docx()
        try:
            text = extract_text(docx_path)
            root = parse_document(text, doc_id="test-docx")
            assert root is not None
            assert len(root.all_nodes()) >= 1
        finally:
            docx_path.unlink(missing_ok=True)

    def test_nonexistent_docx_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_text(tmp_path / "missing.docx")


# ---------------------------------------------------------------------------
# _table_to_text (DOCX table serialization)
# ---------------------------------------------------------------------------


class TestTableToText:
    def _make_mock_table(self, rows: list[list[str]]):
        """Build a minimal mock that mimics python-docx Table structure."""
        mock_table = MagicMock()
        mock_rows = []
        for row_data in rows:
            mock_row = MagicMock()
            mock_cells = []
            for cell_text in row_data:
                mock_cell = MagicMock()
                mock_cell.text = cell_text
                mock_cells.append(mock_cell)
            mock_row.cells = mock_cells
            mock_rows.append(mock_row)
        mock_table.rows = mock_rows
        return mock_table

    def test_produces_pipe_rows(self):
        table = self._make_mock_table([
            ["Name", "Value"],
            ["Price", "$50,000"],
        ])
        result = _table_to_text(table)
        assert "|" in result
        assert "Name" in result
        assert "$50,000" in result

    def test_inserts_separator_after_header(self):
        table = self._make_mock_table([
            ["Col1", "Col2"],
            ["val1", "val2"],
        ])
        result = _table_to_text(table)
        lines = result.splitlines()
        # Line 0: header, Line 1: separator (---), Line 2: data
        assert "---" in lines[1]

    def test_empty_table_returns_empty(self):
        table = self._make_mock_table([])
        assert _table_to_text(table) == ""

    def test_merged_cells_deduplicated(self):
        """python-docx repeats cell text for merged cells — dedup them."""
        table = self._make_mock_table([
            ["Header", "Header", "Other"],  # first two cells merged
        ])
        result = _table_to_text(table)
        # "Header" should appear only once
        assert result.count("Header") == 1


# ---------------------------------------------------------------------------
# Round-trip: extract → parse → chunk
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_pdf_round_trip(self):
        """PDF text should survive parse→chunk without errors."""
        from legalkit.pipeline.parse import parse_document
        from legalkit.pipeline.chunk import chunk_document

        pdf_path = _make_pdf(
            "ASSET PURCHASE AGREEMENT\n\n"
            "Section 1.1 Definitions\n\n"
            '"Affiliate" means any Person that controls another.\n\n'
            "Section 2.1 Purchase Price\n\n"
            "The purchase price shall be $50,000,000."
        )
        try:
            text = extract_text(pdf_path)
            root = parse_document(text, doc_id="test-pdf")
            chunks = chunk_document(root, doc_id="test-pdf")
            assert len(chunks) > 0
            all_text = " ".join(c.text for c in chunks)
            assert "50,000,000" in all_text
        finally:
            pdf_path.unlink(missing_ok=True)

    def test_docx_round_trip(self):
        """DOCX text should survive parse→chunk without errors."""
        from legalkit.pipeline.parse import parse_document
        from legalkit.pipeline.chunk import chunk_document

        docx_path = _make_docx(
            '"Term" means something important in this agreement.'
        )
        try:
            text = extract_text(docx_path)
            root = parse_document(text, doc_id="test-docx")
            chunks = chunk_document(root, doc_id="test-docx")
            assert len(chunks) > 0
        finally:
            docx_path.unlink(missing_ok=True)
