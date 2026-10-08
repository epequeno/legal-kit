"""Stage 2: Structural chunking.

Splits a parsed document tree into chunks aligned to structural
boundaries. Each chunk carries its structural lineage for citation.
"""

from __future__ import annotations

from legalkit.models.document import DocumentNode, NodeType
from legalkit.models.retrieval import Chunk


# Chunk size bounds (in characters)
# Optimised via chunk-sweep against LegalBench-RAG (40 queries, 4 datasets):
# 1500 chars outperforms 3000 at R@16+ and is the best config at R@64 (99.65%).
# 3000 is stronger only below k=8; the pipeline retrieves at k=20 by default,
# putting us firmly in 1500-territory. ContractNLI in particular improves
# from 91.95% → 100.00% R@64 with this change.
MIN_CHUNK_CHARS = 100
MAX_CHUNK_CHARS = 1500  # ~375 tokens — sweet spot confirmed by chunk-sweep
OVERLAP_SENTENCES = 1


def chunk_document(
    root: DocumentNode,
    doc_id: str = "",
    min_chars: int = MIN_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
    parse_tables: bool = True,
) -> list[Chunk]:
    """Chunk a parsed document tree into retrieval-ready chunks.

    Strategy:
    - Walk the document tree depth-first
    - Leaf nodes with text become chunks (if above min_chars)
    - Nodes exceeding max_chars are split at paragraph/sentence boundaries
    - Tiny adjacent nodes are merged
    - Every chunk carries its structural path for citation
    - Table blocks within node text are extracted as atomic chunks with
      column-header-aware serialization

    Args:
        root: Parsed document tree root.
        doc_id: Document identifier.
        min_chars: Minimum chunk size in characters.
        max_chars: Maximum chunk size in characters.
        parse_tables: If True, detect and serialize table blocks.

    Returns:
        List of Chunks ready for indexing.
    """
    doc_id = doc_id or root.id
    raw_chunks: list[Chunk] = []

    _walk_and_chunk(root, doc_id, raw_chunks, max_chars)

    # If the parser found no structure (e.g., privacy policies, plain prose),
    # fall back to paragraph-based chunking of the raw document text.
    if not raw_chunks and root.text:
        raw_chunks = _fallback_chunk(root.text, doc_id, max_chars)
    elif not raw_chunks and root.all_text:
        raw_chunks = _fallback_chunk(root.all_text, doc_id, max_chars)

    # Extract and serialize table blocks within chunks
    if parse_tables:
        raw_chunks = _extract_table_chunks(raw_chunks, doc_id)

    # Merge tiny chunks with their neighbors
    merged = _merge_small_chunks(raw_chunks, min_chars, max_chars)

    # Assign sequential IDs
    for i, chunk in enumerate(merged):
        chunk.id = f"{doc_id}-chunk-{i:04d}"

    return merged


def _walk_and_chunk(
    node: DocumentNode,
    doc_id: str,
    chunks: list[Chunk],
    max_chars: int,
) -> None:
    """Recursively walk the tree and produce chunks."""
    # If node has children, we chunk the node's own text (if any)
    # and recurse into children. This way each structural level
    # gets its own chunk(s).
    if node.type == NodeType.DOCUMENT:
        # Don't chunk the document root itself, just its children
        for child in node.children:
            _walk_and_chunk(child, doc_id, chunks, max_chars)
        return

    text = node.text.strip()

    if not text and not node.children:
        return

    if text:
        # This node has its own text content
        if len(text) <= max_chars:
            chunks.append(_make_chunk(node, text, doc_id))
        else:
            # Split large text at paragraph boundaries
            for sub_text, offset in _split_text(text, max_chars):
                chunk = _make_chunk(node, sub_text, doc_id)
                chunk.char_start = node.char_start + offset
                chunk.char_end = node.char_start + offset + len(sub_text)
                chunks.append(chunk)

    # Recurse into children
    for child in node.children:
        _walk_and_chunk(child, doc_id, chunks, max_chars)


def _make_chunk(node: DocumentNode, text: str, doc_id: str) -> Chunk:
    """Create a Chunk from a DocumentNode."""
    return Chunk(
        id="",  # assigned later
        text=text,
        doc_id=doc_id,
        node_id=node.id,
        node_path=node.path if hasattr(node, "_path") else [node.id],
        char_start=node.char_start,
        char_end=node.char_end,
        cross_references=node.cross_references.copy(),
    )


def _split_text(text: str, max_chars: int) -> list[tuple[str, int]]:
    """Split text into pieces at paragraph boundaries, respecting max_chars.

    Returns list of (text, offset_from_start) tuples.
    """
    # First try splitting on double newlines (paragraphs)
    paragraphs = re.split(r"\n\s*\n", text)

    if len(paragraphs) <= 1:
        # No paragraph breaks — split on single newlines
        paragraphs = text.split("\n")

    if len(paragraphs) <= 1:
        # No line breaks at all — split on sentences
        paragraphs = _split_sentences(text)

    pieces: list[tuple[str, int]] = []
    current_text = ""
    current_offset = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if current_text and len(current_text) + len(para) + 2 > max_chars:
            # Flush current
            pieces.append((current_text.strip(), current_offset))
            current_offset = text.find(para, current_offset + len(current_text))
            if current_offset == -1:
                current_offset = 0
            current_text = para
        else:
            if not current_text:
                current_offset = text.find(para)
                if current_offset == -1:
                    current_offset = 0
            current_text = current_text + "\n\n" + para if current_text else para

    if current_text.strip():
        pieces.append((current_text.strip(), current_offset))

    return pieces


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (simple heuristic)."""
    # Split on period/question/exclamation followed by space and capital
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return parts


def _merge_small_chunks(
    chunks: list[Chunk],
    min_chars: int,
    max_chars: int,
) -> list[Chunk]:
    """Merge chunks smaller than min_chars with their neighbors."""
    if not chunks:
        return []

    merged: list[Chunk] = []

    for chunk in chunks:
        # Table chunks are atomic — never merge them with neighbours
        if chunk.table_repr is not None:
            merged.append(chunk)
            continue

        if (
            merged
            and merged[-1].table_repr is None   # don't merge into a table chunk
            and len(merged[-1].text) < min_chars
            and merged[-1].node_id == chunk.node_id
            and len(merged[-1].text) + len(chunk.text) + 2 <= max_chars
        ):
            # Merge with previous
            prev = merged[-1]
            prev.text = prev.text + "\n\n" + chunk.text
            prev.char_end = chunk.char_end
            prev.cross_references = list(
                dict.fromkeys(prev.cross_references + chunk.cross_references)
            )
        elif (
            merged
            and merged[-1].table_repr is None   # don't merge into a table chunk
            and len(chunk.text) < min_chars
            and merged[-1].node_id == chunk.node_id
            and len(merged[-1].text) + len(chunk.text) + 2 <= max_chars
        ):
            # Current is too small, merge into previous
            prev = merged[-1]
            prev.text = prev.text + "\n\n" + chunk.text
            prev.char_end = chunk.char_end
            prev.cross_references = list(
                dict.fromkeys(prev.cross_references + chunk.cross_references)
            )
        else:
            merged.append(chunk)

    return merged


def _fallback_chunk(
    text: str,
    doc_id: str,
    max_chars: int,
) -> list[Chunk]:
    """Fallback chunking for unstructured documents.

    Splits on paragraph boundaries (double newlines), preserving
    character offsets. Used when no structural markers are found.
    """
    from legalkit.pipeline.parse import extract_cross_references

    paragraphs: list[tuple[str, int]] = []
    current_pos = 0

    # Split on double newlines, or newline + whitespace (common in privacy policies)
    for part in re.split(r"(\n\s*\n|\n\s{2,})", text):
        stripped = part.strip()
        if stripped and len(stripped) > 20:
            # Find the actual position in original text
            idx = text.find(stripped, current_pos)
            if idx >= 0:
                paragraphs.append((stripped, idx))
                current_pos = idx + len(stripped)
            else:
                current_pos += len(part)
        else:
            current_pos += len(part)

    # Group paragraphs into chunks
    chunks: list[Chunk] = []
    current_text = ""
    current_start = 0

    for para_text, para_start in paragraphs:
        if current_text and len(current_text) + len(para_text) + 2 > max_chars:
            # Flush current
            chunks.append(Chunk(
                id="",
                text=current_text,
                doc_id=doc_id,
                node_id=f"{doc_id}-para",
                node_path=[doc_id],
                char_start=current_start,
                char_end=current_start + len(current_text),
                cross_references=extract_cross_references(current_text),
            ))
            current_text = para_text
            current_start = para_start
        else:
            if not current_text:
                current_start = para_start
            current_text = current_text + "\n\n" + para_text if current_text else para_text

    if current_text.strip():
        chunks.append(Chunk(
            id="",
            text=current_text,
            doc_id=doc_id,
            node_id=f"{doc_id}-para",
            node_path=[doc_id],
            char_start=current_start,
            char_end=current_start + len(current_text),
            cross_references=extract_cross_references(current_text),
        ))

    return chunks


def _extract_table_chunks(chunks: list[Chunk], doc_id: str) -> list[Chunk]:
    """Replace table blocks within chunks with serialized table chunks.

    For each chunk whose text contains a detected table block:
    - Extract the table block as a separate atomic chunk with ``table_repr``
      set to the serialized column-header-aware representation.
    - Keep surrounding non-table text as sibling chunks.

    This ensures:
    1. Tables are never split mid-row by the merge/split logic.
    2. The embedding uses column-header context rather than raw pipe/dash text.

    Args:
        chunks: Input chunks (may contain table text).
        doc_id: Document ID for new chunk IDs.

    Returns:
        New list of chunks where table blocks are separated and annotated.
    """
    from legalkit.pipeline.tables import detect_tables, parse_table

    result: list[Chunk] = []

    for chunk in chunks:
        text = chunk.text
        spans = detect_tables(text)

        if not spans:
            result.append(chunk)
            continue

        # Split chunk around table blocks
        prev_end = 0
        for t_start, t_end in spans:
            # Text before the table
            before = text[prev_end:t_start]
            if before.strip():
                pre_chunk = Chunk(
                    id="",
                    text=before.strip(),
                    doc_id=chunk.doc_id,
                    node_id=chunk.node_id,
                    node_path=chunk.node_path,
                    char_start=chunk.char_start + prev_end,
                    char_end=chunk.char_start + t_start,
                    cross_references=chunk.cross_references,
                )
                result.append(pre_chunk)

            # The table block itself
            raw_table = text[t_start:t_end]
            table = parse_table(raw_table, char_start=chunk.char_start + t_start)
            table_repr = table.serialize() if table else raw_table

            tbl_chunk = Chunk(
                id="",
                text=raw_table,
                table_repr=table_repr,
                doc_id=chunk.doc_id,
                node_id=chunk.node_id,
                node_path=chunk.node_path,
                char_start=chunk.char_start + t_start,
                char_end=chunk.char_start + t_end,
                cross_references=chunk.cross_references,
            )
            result.append(tbl_chunk)
            prev_end = t_end

        # Text after the last table
        after = text[prev_end:]
        if after.strip():
            post_chunk = Chunk(
                id="",
                text=after.strip(),
                doc_id=chunk.doc_id,
                node_id=chunk.node_id,
                node_path=chunk.node_path,
                char_start=chunk.char_start + prev_end,
                char_end=chunk.char_end,
                cross_references=chunk.cross_references,
            )
            result.append(post_chunk)

    return result


import re  # noqa: E402 — needed by _split_text, placed here to avoid circular
