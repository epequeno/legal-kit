"""Stage 1: Document parsing & structuring.

Parses legal document text into a structural tree of DocumentNodes.
Each node carries stable identifiers and char offsets for citation.
"""

from __future__ import annotations

import re
from dataclasses import field
from pathlib import Path

from legalkit.models.document import DocumentNode, NodeType


# ---------------------------------------------------------------------------
# Cross-reference detection
# ---------------------------------------------------------------------------

_XREF_PATTERNS = [
    # "Section 4.2(a)", "Sections 3.1 and 3.2"
    re.compile(r"[Ss]ections?\s+([\d]+(?:\.[\d]+)*(?:\([a-zA-Z0-9]+\))?(?:\s*(?:and|,|or)\s*[\d]+(?:\.[\d]+)*(?:\([a-zA-Z0-9]+\))?)*)"),
    # "Article IV", "Articles II and III"
    re.compile(r"[Aa]rticles?\s+([IVXLCDM]+(?:\s*(?:and|,|or)\s*[IVXLCDM]+)*)"),
    # "Exhibit A", "Schedule 1"
    re.compile(r"(?:Exhibit|Schedule|Annex|Appendix)\s+([A-Z0-9]+)", re.IGNORECASE),
    # "as defined in Section ...", "pursuant to Section ..."
    re.compile(r"(?:as\s+defined\s+in|pursuant\s+to|subject\s+to|in\s+accordance\s+with)\s+[Ss]ection\s+([\d]+(?:\.[\d]+)*(?:\([a-zA-Z0-9]+\))?)"),
]


def extract_cross_references(text: str) -> list[str]:
    """Extract section/article/exhibit references from text."""
    refs: list[str] = []
    for pattern in _XREF_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(1)
            # Split on 'and', 'or', ','
            parts = re.split(r"\s*(?:and|or|,)\s*", raw)
            for part in parts:
                part = part.strip()
                if part:
                    refs.append(part)
    return list(dict.fromkeys(refs))  # dedupe, preserve order


# ---------------------------------------------------------------------------
# Section heading detection
# ---------------------------------------------------------------------------

# Matches patterns like:
#   "ARTICLE IV", "Article 4", "ARTICLE IV - INDEMNIFICATION"
_ARTICLE_RE = re.compile(
    r"^(?:ARTICLE|Article)\s+([IVXLCDM]+|\d+)(?:\s*[-:.]\s*(.+))?$",
    re.MULTILINE,
)

# Matches patterns like:
#   "Section 4.2", "SECTION 4.2(a)", "Section 4.2 Indemnification"
#   "Section 4.2. Indemnification", "4.2(a) Cap"
_SECTION_RE = re.compile(
    r"^(?:(?:SECTION|Section)\s+)"
    r"((\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\(([a-zA-Z0-9]+)\))?)"
    r"(?:[\s.:-]+([A-Z].+))?$",
    re.MULTILINE,
)

# Lettered/roman sub-clauses at line start: "(a) text...", "(i) text..."
# Only matched at the start of a line (re.MULTILINE) to avoid inline references.
_SUBCLAUSE_RE = re.compile(
    r"^\(([a-zA-Z0-9]+)\)\s+(.+)",
    re.MULTILINE,
)

# Bare numeric sections without the "Section" keyword:
# "1.1 Definitions", "1.1. Definitions", "2.3.1 Sub-term"
# Requires at least X. to start; title must be short, non-list-like (no comma/semicolon).
_BARE_SECTION_RE = re.compile(
    r"^((\d+)\.(\d+)?(?:\.(\d+))?(?:\(([a-zA-Z0-9]+)\))?)"
    r"\.?\s{1,4}([A-Z][^\n,;]{1,70})$",
    re.MULTILINE,
)

# Roman numeral article/section headers without the "ARTICLE" keyword:
# "I. DEFINITIONS", "IV. Representations and Warranties"
# Validated further with _is_roman_numeral() to reject false positives.
_ROMAN_HEADER_RE = re.compile(
    r"^([IVXLCDM]+)\.?\s{1,4}([A-Z][^\n]{3,70})$",
    re.MULTILINE,
)

# Valid Roman numeral pattern (I through MMMCMXCIX)
_ROMAN_NUMERAL_RE = re.compile(
    r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$",
    re.IGNORECASE,
)


def _is_roman_numeral(s: str) -> bool:
    """Return True if s is a valid Roman numeral (I–MMMCMXCIX)."""
    return bool(s) and bool(_ROMAN_NUMERAL_RE.match(s))

# Definition patterns: "Affiliate" means ..., "Closing Date" shall mean ...
# Group 1: term name. Group 2: definition body (rest of the sentence/paragraph).
_DEFINITION_RE = re.compile(
    r'["\u201c]([^"\u201d]+)["\u201d]\s+(?:shall\s+)?'
    r'(?:means?|refers?\s+to|has\s+the\s+meaning(?:\s+set\s+forth\s+in\s+\S+)?)'
    r'([^"\u201c]*?)(?=\n\s*\n|["\u201c]|\Z)',
    re.IGNORECASE | re.DOTALL,
)

# Recitals: "WHEREAS, ..."
_RECITAL_RE = re.compile(
    r"^WHEREAS[,:]?\s+(.+)",
    re.MULTILINE,
)

# Exhibit/Schedule headers
_EXHIBIT_RE = re.compile(
    r"^(EXHIBIT|SCHEDULE|ANNEX|APPENDIX)\s+([A-Z0-9]+)(?:\s*[-:.]\s*(.+))?$",
    re.MULTILINE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_document(text: str, doc_id: str = "doc") -> DocumentNode:
    """Parse a legal document into a structural tree.

    Args:
        text: The full document text.
        doc_id: Identifier for this document.

    Returns:
        Root DocumentNode with children representing the document structure.
    """
    root = DocumentNode(
        id=doc_id,
        type=NodeType.DOCUMENT,
        title=doc_id,
        text="",
        char_start=0,
        char_end=len(text),
        level=0,
    )
    root.path = [doc_id]

    # Find all structural markers with their positions
    markers = _find_all_markers(text)

    if not markers:
        # No structure found — treat entire text as a single paragraph
        root.text = text
        root.cross_references = extract_cross_references(text)
        return root

    # Sort by position
    markers.sort(key=lambda m: m["start"])

    # Handle preamble text before first marker
    if markers[0]["start"] > 0:
        preamble_text = text[: markers[0]["start"]].strip()
        if preamble_text:
            preamble = DocumentNode(
                id=f"{doc_id}-preamble",
                type=NodeType.PREAMBLE,
                title="Preamble",
                text=preamble_text,
                char_start=0,
                char_end=markers[0]["start"],
                parent_id=doc_id,
                level=1,
                cross_references=extract_cross_references(preamble_text),
            )
            preamble.path = [doc_id, preamble.id]
            root.children.append(preamble)

    # Assign text spans: each marker's text runs until the next marker
    for i, marker in enumerate(markers):
        end = markers[i + 1]["start"] if i + 1 < len(markers) else len(text)
        # Text is everything after the heading line until the next marker
        heading_end = text.find("\n", marker["start"])
        if heading_end == -1 or heading_end >= end:
            heading_end = end
        marker["body_start"] = heading_end
        marker["body_end"] = end
        marker["body"] = text[heading_end:end].strip()

    # Build the tree
    _build_tree(root, markers, text, doc_id)

    # Annotate nodes that contain definitions
    _annotate_definitions(root)

    return root


def _find_all_markers(text: str) -> list[dict]:
    """Find all structural markers in the text."""
    markers = []
    # Track matched positions to prevent double-matching the same text span
    matched_starts: set[int] = set()

    def _add(marker: dict) -> None:
        if marker["start"] not in matched_starts:
            markers.append(marker)
            matched_starts.add(marker["start"])

    # ── Known-keyword patterns (highest priority) ─────────────────────────

    for match in _ARTICLE_RE.finditer(text):
        _add({
            "type": NodeType.ARTICLE,
            "start": match.start(),
            "id_raw": match.group(1),
            "title": (match.group(2) or "").strip(),
            "level": 1,
        })

    for match in _SECTION_RE.finditer(text):
        full_id = match.group(1)
        title = (match.group(6) or "").strip()
        parts = [match.group(2), match.group(3), match.group(4), match.group(5)]
        depth = sum(1 for p in parts if p is not None)
        level = depth + 1

        node_type = NodeType.SECTION
        if depth >= 3:
            node_type = NodeType.SUBCLAUSE
        elif depth >= 2:
            node_type = NodeType.SUBSECTION

        _add({
            "type": node_type,
            "start": match.start(),
            "id_raw": full_id,
            "title": title,
            "level": level,
        })

    for match in _EXHIBIT_RE.finditer(text):
        _add({
            "type": NodeType.EXHIBIT,
            "start": match.start(),
            "id_raw": match.group(2),
            "title": (match.group(3) or "").strip(),
            "level": 1,
        })

    for match in _RECITAL_RE.finditer(text):
        _add({
            "type": NodeType.RECITAL,
            "start": match.start(),
            "id_raw": f"recital-{sum(1 for m in markers if m['type'] == NodeType.RECITAL) + 1}",
            "title": "",
            "level": 1,
        })

    # ── Lettered/roman subclauses: (a), (b), (i), (ii) ───────────────────
    # Only at line start (re.MULTILINE). Levels: single letter = 4, roman/digit = 5.

    for match in _SUBCLAUSE_RE.finditer(text):
        letter = match.group(1)
        inline_text = match.group(2).strip()
        # Single lowercase letter → direct child of a section (level 4)
        # Roman numeral or digit → nested subclause (level 5)
        if re.match(r"^[a-z]$", letter):
            level = 4
        else:
            level = 5
        _add({
            "type": NodeType.SUBCLAUSE,
            "start": match.start(),
            "id_raw": letter,
            "title": inline_text,   # inline text on the (a) line
            "level": level,
        })

    # ── Bare numeric sections: "1.1 Title", "2.3. Title" ─────────────────

    for match in _BARE_SECTION_RE.finditer(text):
        # Strip any trailing period that got pulled into the id group
        full_id = match.group(1).rstrip(".")
        title = match.group(6).strip()
        # Parse depth from captured groups (same logic as _SECTION_RE)
        parts = [match.group(2), match.group(3), match.group(4), match.group(5)]
        depth = sum(1 for p in parts if p is not None)
        level = depth + 1

        node_type = NodeType.SECTION
        if depth >= 3:
            node_type = NodeType.SUBCLAUSE
        elif depth >= 2:
            node_type = NodeType.SUBSECTION

        _add({
            "type": node_type,
            "start": match.start(),
            "id_raw": full_id,
            "title": title,
            "level": level,
        })

    # ── Roman numeral headers: "IV. REPRESENTATIONS..." ───────────────────
    # Only accept validated roman numerals to prevent false positives.

    for match in _ROMAN_HEADER_RE.finditer(text):
        numeral = match.group(1)
        title = match.group(2).strip()
        if not _is_roman_numeral(numeral):
            continue
        _add({
            "type": NodeType.ARTICLE,
            "start": match.start(),
            "id_raw": numeral,
            "title": title,
            "level": 1,
        })

    return markers


def _make_section_id(marker: dict, doc_id: str) -> str:
    """Generate a stable section ID from a marker."""
    node_type = marker["type"]
    raw_id = marker["id_raw"]

    if node_type == NodeType.ARTICLE:
        return f"Article-{raw_id}"
    elif node_type == NodeType.EXHIBIT:
        return f"Exhibit-{raw_id}"
    elif node_type == NodeType.RECITAL:
        return raw_id
    elif node_type in (NodeType.SECTION, NodeType.SUBSECTION, NodeType.SUBCLAUSE):
        return f"§{raw_id}"
    else:
        return f"{doc_id}-{raw_id}"


def _build_tree(
    root: DocumentNode,
    markers: list[dict],
    text: str,
    doc_id: str,
) -> None:
    """Build the document tree from sorted markers using level-based nesting."""
    stack: list[tuple[int, DocumentNode]] = [(0, root)]

    for marker in markers:
        body_text = marker.get("body", "")
        title = marker.get("title", "")

        # ── Determine parent first (needed for compound subclause IDs) ──
        while len(stack) > 1 and stack[-1][0] >= marker["level"]:
            stack.pop()
        parent = stack[-1][1]

        # ── Generate node ID ─────────────────────────────────────────────
        if marker["type"] == NodeType.SUBCLAUSE and parent.type in (
            NodeType.SECTION, NodeType.SUBSECTION, NodeType.SUBCLAUSE,
        ):
            # Compound ID: parent section + (letter) → e.g. §2.1(a)
            parent_base = parent.id.lstrip("§")
            node_id = f"§{parent_base}({marker['id_raw']})"
        else:
            node_id = _make_section_id(marker, doc_id)

        # ── Resolve text and title ────────────────────────────────────────
        if marker["type"] == NodeType.SUBCLAUSE:
            # For subclauses the regex captures inline text in "title".
            # Combine it with any multi-line continuation in "body".
            parts = [p for p in (title, body_text) if p]
            body_text = "\n".join(parts)
            title = ""
        elif not body_text and title and len(title) > 80:
            # Single-line section where all content landed in the title field.
            # Split at the first sentence boundary if possible.
            dot_pos = title.find(". ")
            if 0 < dot_pos < 60:
                body_text = title[dot_pos + 2:]
                title = title[:dot_pos + 1]
            else:
                body_text = title
                title = ""

        node = DocumentNode(
            id=node_id,
            type=marker["type"],
            title=title,
            text=body_text,
            char_start=marker["start"],
            char_end=marker.get("body_end", marker["start"]),
            level=marker["level"],
            cross_references=extract_cross_references(body_text),
        )

        node.parent_id = parent.id
        node.path = (
            parent.path + [node.id]
            if hasattr(parent, "_path") or parent.path
            else [parent.id, node.id]
        )
        parent.children.append(node)
        stack.append((marker["level"], node))


# ---------------------------------------------------------------------------
# Definition annotation
# ---------------------------------------------------------------------------


def _annotate_definitions(root: DocumentNode) -> None:
    """Walk the tree and annotate nodes that contain defined terms.

    Sets ``node.defined_terms`` and upgrades ``node.type`` to
    ``NodeType.DEFINITION`` for leaf nodes whose entire text is a
    definition block.
    """
    for node in root.all_nodes():
        if node.type == NodeType.DOCUMENT:
            continue
        text = node.text or ""
        if not text:
            continue
        defs = find_definitions(text)
        if not defs:
            continue
        terms = [term for term, _, _ in defs]
        node.defined_terms = list(dict.fromkeys(node.defined_terms + terms))
        # If the node contains ONLY definitions (no children, short text),
        # promote its type to DEFINITION for clearer tree semantics.
        if not node.children and len(terms) >= 1:
            # Heuristic: >60% of text matches definition patterns
            def_text_len = sum(len(body) for _, _, body in defs)
            if def_text_len > 0 and def_text_len / max(len(text), 1) > 0.4:
                node.type = NodeType.DEFINITION


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def find_definitions(text: str) -> list[tuple[str, int, str]]:
    """Find defined terms and their positions in the text.

    Returns list of (term, char_offset, definition_text) tuples.
    ``definition_text`` is the text of the definition body (after "means").
    """
    results = []
    for match in _DEFINITION_RE.finditer(text):
        term = match.group(1).strip()
        body = (match.group(2) or "").strip().rstrip(".")
        results.append((term, match.start(), body))
    return results


def parse_document_with_ml(
    text: str,
    doc_id: str = "doc",
    model_path: str | None = None,
    min_nodes_threshold: int = 5,
) -> DocumentNode:
    """Parse document using hybrid regex+ML approach.
    
    Convenience wrapper around parse_document_hybrid() that returns
    just the DocumentNode (discards the method string).
    Uses the model at model_path, or the default location
    models/structure-parser/model-dir/ if model_path is not specified.
    
    Args:
        text: Document text to parse
        doc_id: Document identifier
        model_path: Path to ML model directory (None = use default)
        min_nodes_threshold: Minimum structural nodes before considering regex successful
        
    Returns:
        Root DocumentNode with children representing document structure
    """
    from legalkit.pipeline.parse_ml import parse_document_hybrid
    
    default_path = Path(__file__).parent.parent.parent.parent / "models/structure-parser/model-dir"
    path = model_path or (str(default_path) if default_path.exists() else None)
    
    root, _ = parse_document_hybrid(
        text, doc_id=doc_id, model_path=path,
        min_nodes_threshold=min_nodes_threshold
    )
    return root


def document_summary(root: DocumentNode, indent: int = 0) -> str:
    """Pretty-print the document tree structure."""
    lines = []
    prefix = "  " * indent
    type_str = root.type.value.upper()
    title = f" — {root.title}" if root.title else ""
    text_len = len(root.text)
    lines.append(f"{prefix}{type_str} [{root.id}]{title} ({text_len} chars)")

    for child in root.children:
        lines.append(document_summary(child, indent + 1))

    return "\n".join(lines)
