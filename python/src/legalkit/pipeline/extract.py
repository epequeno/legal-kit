"""Text extraction from multiple document formats.

Provides a single ``extract_text(path)`` entry point that dispatches to
the appropriate extractor based on file extension:

- ``.txt`` — read directly (UTF-8 with latin-1 fallback)
- ``.pdf`` — PyMuPDF text layer extraction (machine-readable PDFs only)
- ``.docx`` — python-docx structured paragraph extraction

All extractors return a plain ``str`` that feeds directly into
``parse_document()``.  No OCR is performed — scanned PDFs will return
empty or near-empty text.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".doc"}


def extract_text(path: Path) -> str:
    """Extract plain text from a document file.

    Args:
        path: Path to a ``.txt``, ``.pdf``, or ``.docx`` file.

    Returns:
        Extracted text as a unicode string.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
        RuntimeError: If extraction fails for format-specific reasons.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return _extract_txt(path)
    elif suffix == ".pdf":
        return _extract_pdf(path)
    elif suffix in (".docx", ".doc"):
        return _extract_docx(path)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix!r}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------


def _extract_txt(path: Path) -> str:
    """Read a plain-text file with UTF-8 / latin-1 fallback."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


# ---------------------------------------------------------------------------
# PDF via PyMuPDF
# ---------------------------------------------------------------------------


def _extract_pdf(path: Path) -> str:
    """Extract text from a machine-readable PDF using PyMuPDF.

    Preserves page structure with form-feed separators so the parser can
    treat page breaks as paragraph boundaries if needed.

    Args:
        path: Path to the PDF file.

    Returns:
        Extracted text.  May be empty or near-empty for scanned PDFs.

    Raises:
        RuntimeError: If PyMuPDF fails to open or read the file.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for PDF extraction. "
            "Install with: uv add pymupdf"
        ) from exc

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF {path.name}: {exc}") from exc

    pages: list[str] = []
    for page_num, page in enumerate(doc, start=1):
        # "text" mode preserves layout; "blocks" can be useful too but
        # plain text is more compatible with our regex-based parser
        page_text = page.get_text("text")
        if page_text.strip():
            pages.append(page_text)

    doc.close()

    if not pages:
        raise RuntimeError(
            f"No text extracted from {path.name}. "
            "The PDF may be scanned or image-only. OCR is not supported."
        )

    return "\n\n".join(pages)


def extract_pdf_metadata(path: Path) -> dict:
    """Extract PDF metadata (title, author, creation date, page count).

    Args:
        path: Path to the PDF file.

    Returns:
        Dict with keys: title, author, subject, creator, pages, encrypted.
    """
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF required") from exc

    doc = fitz.open(str(path))
    meta = doc.metadata or {}
    result = {
        "title": meta.get("title", ""),
        "author": meta.get("author", ""),
        "subject": meta.get("subject", ""),
        "creator": meta.get("creator", ""),
        "pages": len(doc),
        "encrypted": doc.is_encrypted,
    }
    doc.close()
    return result


# ---------------------------------------------------------------------------
# DOCX via python-docx
# ---------------------------------------------------------------------------


def _extract_docx(path: Path) -> str:
    """Extract text from a DOCX file preserving document structure.

    Strategy:
    - Paragraphs with heading styles become section-like headings
    - Table cells are serialized row-by-row with pipe separators
    - Normal paragraphs are joined with double newlines

    This produces output that the existing parser (``parse_document``) and
    table detector (``detect_tables``) can process correctly.

    Args:
        path: Path to the ``.docx`` file.

    Returns:
        Extracted text with structure hints preserved.

    Raises:
        RuntimeError: If python-docx fails to open the file.
    """
    try:
        import docx
    except ImportError as exc:
        raise RuntimeError(
            "python-docx is required for DOCX extraction. "
            "Install with: uv add python-docx"
        ) from exc

    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open DOCX {path.name}: {exc}") from exc

    blocks: list[str] = []

    # Process document body elements in order (paragraphs and tables
    # interleaved as they appear in the document)
    for element in _iter_body_elements(document):
        if hasattr(element, "text"):
            # Paragraph
            para = element
            text = para.text.strip()
            if not text:
                continue

            style_name = (para.style.name or "").lower() if para.style else ""

            if "heading" in style_name:
                # Preserve heading level as prefix for the parser to detect
                level = _heading_level(style_name)
                prefix = "#" * level + " " if level else ""
                blocks.append(f"{prefix}{text}")
            else:
                blocks.append(text)

        elif hasattr(element, "rows"):
            # Table
            table_block = _table_to_text(element)
            if table_block:
                blocks.append(table_block)

    if not blocks:
        raise RuntimeError(
            f"No text extracted from {path.name}. "
            "The DOCX may be empty or use an unsupported structure."
        )

    return "\n\n".join(blocks)


def _iter_body_elements(document):
    """Iterate paragraphs and tables in document body order.

    python-docx exposes ``document.paragraphs`` and ``document.tables``
    separately, but iterating ``document.element.body`` preserves order.
    """
    from docx.oxml.ns import qn
    import docx

    body = document.element.body
    for child in body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "p":
            yield docx.text.paragraph.Paragraph(child, document)
        elif tag == "tbl":
            yield docx.table.Table(child, document)


def _heading_level(style_name: str) -> int:
    """Parse heading level from a style name like 'heading 1' → 1."""
    import re
    m = re.search(r"(\d+)", style_name)
    return int(m.group(1)) if m else 1


def _table_to_text(table) -> str:
    """Serialise a DOCX table to pipe-delimited text for the table parser."""
    rows: list[str] = []
    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        # Deduplicate merged cells (python-docx repeats them)
        deduped: list[str] = []
        for cell in cells:
            if not deduped or cell != deduped[-1]:
                deduped.append(cell)
        rows.append("| " + " | ".join(deduped) + " |")

    if not rows:
        return ""

    # Insert a markdown separator after the first row (header)
    if len(rows) >= 2:
        col_count = rows[0].count("|") - 1
        separator = "| " + " | ".join(["---"] * col_count) + " |"
        rows.insert(1, separator)

    return "\n".join(rows)
