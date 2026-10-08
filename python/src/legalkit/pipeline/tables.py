"""Table detection and parsing for plain-text legal documents.

Legal documents contain several tabular formats:

1. **Fixed-width with dash underlines** (most common in EDGAR/corpus docs):
       Classes           Reg. No.    Date
       -------           --------    ----
       Burger King Logo  7441        18 Feb 1994

2. **Pipe-delimited** (markdown-style, less common):
       | Classes | Reg. No. | Date |
       |---------|----------|------|
       | BK Logo | 7441     | 1994 |

3. **Key-value two-column** (schedules, signature blocks):
       Name:    John Smith
       Date:    January 1, 2024
       Amount:  $50,000

Detected tables are serialized into a clean, column-header-aware text
representation for embedding, ensuring the model understands what each
value means (critical for queries like "what is the registration number
for Burger King Logo?").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Minimum rows (excluding header) to consider something a table
_MIN_DATA_ROWS = 1

# Minimum columns to consider something a table
_MIN_COLS = 2

# Regex: a line consisting only of dashes, equals, and spaces (underline row)
_UNDERLINE_RE = re.compile(r"^[\s\-=]+$")

# Regex: a pipe-delimited row — at least two pipe characters
_PIPE_ROW_RE = re.compile(r"^\s*\|.*\|.*$")

# Regex: separator row in a pipe table — |---|---| or |===|===|
_PIPE_SEP_RE = re.compile(r"^\s*\|[\s\-=|]+\|\s*$")

# Regex: key-value line — "Key:    Value" or "Key      Value"
_KV_ROW_RE = re.compile(r"^(\S[^:]{0,40}):[ \t]{2,}(.+)$")


@dataclass
class ParsedTable:
    """A parsed tabular structure from a legal document."""

    headers: list[str]
    """Column header names. Empty if no headers detected."""

    rows: list[list[str]]
    """Data rows as list of cell value lists."""

    caption: str = ""
    """Optional caption from the line above the table."""

    raw_text: str = ""
    """Original raw text of the table block."""

    char_start: int = 0
    char_end: int = 0

    table_type: str = "fixed_width"
    """One of 'fixed_width', 'pipe', 'key_value'."""

    @property
    def is_empty(self) -> bool:
        return not self.rows

    @property
    def num_cols(self) -> int:
        if self.headers:
            return len(self.headers)
        return max((len(r) for r in self.rows), default=0)

    def serialize(self) -> str:
        """Convert to a clean text representation for embedding.

        Format:
            [TABLE caption]
            Col1 | Col2 | Col3
            value1 | value2 | value3
            ...
        """
        lines: list[str] = []

        if self.caption:
            lines.append(f"[TABLE: {self.caption}]")

        if self.headers:
            lines.append(" | ".join(self.headers))

        for row in self.rows:
            # Pad short rows to header width
            padded = row + [""] * max(0, len(self.headers) - len(row))
            lines.append(" | ".join(cell for cell in padded))

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ParsedTable(type={self.table_type!r}, "
            f"headers={self.headers}, rows={len(self.rows)})"
        )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_tables(text: str) -> list[tuple[int, int]]:
    """Find table-like spans in ``text``.

    Returns a list of ``(start, end)`` character offsets for each detected
    table block (including any caption line immediately above).

    Args:
        text: Document text to scan.

    Returns:
        List of ``(start, end)`` spans, sorted by start position.
    """
    spans: list[tuple[int, int]] = []
    spans.extend(_detect_pipe_tables(text))
    spans.extend(_detect_fixed_width_tables(text))
    spans.extend(_detect_kv_tables(text))
    # Merge overlapping spans
    return _merge_spans(sorted(spans))


def _detect_pipe_tables(text: str) -> list[tuple[int, int]]:
    """Find pipe-delimited table blocks."""
    lines = text.splitlines(keepends=True)
    offsets = _line_offsets(lines)
    spans: list[tuple[int, int]] = []

    i = 0
    while i < len(lines):
        if _PIPE_ROW_RE.match(lines[i]):
            start_line = i
            # Walk forward while lines look like pipe rows or separators
            while i < len(lines) and (
                _PIPE_ROW_RE.match(lines[i]) or _PIPE_SEP_RE.match(lines[i])
            ):
                i += 1
            end_line = i - 1
            if end_line > start_line:  # at least 2 rows
                start = offsets[start_line]
                end = offsets[end_line] + len(lines[end_line])
                spans.append((start, end))
        else:
            i += 1

    return spans


def _detect_fixed_width_tables(text: str) -> list[tuple[int, int]]:
    """Find fixed-width tables identified by a dash-underline row."""
    lines = text.splitlines(keepends=True)
    offsets = _line_offsets(lines)
    spans: list[tuple[int, int]] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Underline row: only dashes/equals/spaces, at least 6 chars, has actual dashes
        if (
            len(stripped) >= 6
            and _UNDERLINE_RE.match(stripped)
            and "-" in stripped
            and i > 0
        ):
            # Header is the line above the underline
            header_line = lines[i - 1]
            if not header_line.strip():
                continue  # blank line above — skip

            # Collect data rows below the underline
            j = i + 1
            data_lines: list[int] = []
            while j < len(lines):
                row = lines[j].rstrip("\n")
                if not row.strip():
                    # One blank line is allowed as a row separator in some tables
                    if data_lines and j + 1 < len(lines) and lines[j + 1].strip():
                        j += 1
                        continue
                    break
                if _UNDERLINE_RE.match(row.strip()) and "-" in row:
                    # Another underline — end of table
                    break
                data_lines.append(j)
                j += 1

            if len(data_lines) >= _MIN_DATA_ROWS:
                # Include caption line (line before header) if it looks like one
                caption_line = i - 2
                if (
                    caption_line >= 0
                    and lines[caption_line].strip()
                    and not _UNDERLINE_RE.match(lines[caption_line].strip())
                ):
                    start_idx = caption_line
                else:
                    start_idx = i - 1  # header line

                end_idx = data_lines[-1]
                spans.append((offsets[start_idx], offsets[end_idx] + len(lines[end_idx])))

    return spans


def _detect_kv_tables(text: str) -> list[tuple[int, int]]:
    """Find key-value blocks (3+ consecutive 'Key:  Value' lines)."""
    lines = text.splitlines(keepends=True)
    offsets = _line_offsets(lines)
    spans: list[tuple[int, int]] = []

    i = 0
    while i < len(lines):
        if _KV_ROW_RE.match(lines[i].rstrip()):
            start = i
            while i < len(lines) and _KV_ROW_RE.match(lines[i].rstrip()):
                i += 1
            if i - start >= 3:  # at least 3 KV pairs to be a "table"
                spans.append((offsets[start], offsets[i - 1] + len(lines[i - 1])))
        else:
            i += 1

    return spans


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_table(raw_text: str, char_start: int = 0) -> ParsedTable | None:
    """Parse a raw table block into a ``ParsedTable``.

    Tries pipe, fixed-width, and key-value formats in order.

    Args:
        raw_text: The raw text of the detected table block.
        char_start: Offset of this block in the original document.

    Returns:
        ``ParsedTable`` or ``None`` if parsing fails.
    """
    text = raw_text.strip()
    char_end = char_start + len(raw_text)

    result = (
        _parse_pipe_table(text)
        or _parse_fixed_width_table(text)
        or _parse_kv_table(text)
    )

    if result is None:
        return None

    result.raw_text = raw_text
    result.char_start = char_start
    result.char_end = char_end
    return result


def _parse_pipe_table(text: str) -> ParsedTable | None:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    rows_raw = [l for l in lines if _PIPE_ROW_RE.match(l)]
    if len(rows_raw) < 2:
        return None

    def split_pipe_row(line: str) -> list[str]:
        # Strip leading/trailing pipes then split
        line = line.strip().strip("|")
        return [cell.strip() for cell in line.split("|")]

    # Identify header and separator rows
    headers: list[str] = []
    data_rows: list[list[str]] = []

    for i, line in enumerate(rows_raw):
        if _PIPE_SEP_RE.match(line):
            # The row before this is the header
            if i > 0:
                headers = split_pipe_row(rows_raw[i - 1])
        elif not headers:
            pass  # will become header on next separator or first row
        else:
            row = split_pipe_row(line)
            if row != headers:
                data_rows.append(row)

    if not data_rows:
        # No separator found — treat first row as header
        if len(rows_raw) >= 2:
            headers = split_pipe_row(rows_raw[0])
            data_rows = [split_pipe_row(r) for r in rows_raw[1:]]

    if not data_rows or len(headers) < _MIN_COLS:
        return None

    return ParsedTable(headers=headers, rows=data_rows, table_type="pipe")


def _parse_fixed_width_table(text: str) -> ParsedTable | None:
    lines = text.splitlines()

    # Find the underline row
    underline_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) >= 4 and _UNDERLINE_RE.match(stripped) and "-" in stripped:
            underline_idx = i
            break

    if underline_idx is None or underline_idx == 0:
        return None

    underline = lines[underline_idx]
    header_line = lines[underline_idx - 1]

    # Parse column boundaries from underline
    cols = _find_column_boundaries(underline)
    if len(cols) < _MIN_COLS:
        return None

    headers = [_extract_cell(header_line, s, e) for s, e in cols]
    # Filter out empty headers
    headers = [h for h in headers if h]

    if len(headers) < _MIN_COLS:
        return None

    # Caption: line before header if present and non-empty
    caption = ""
    if underline_idx >= 2:
        cap = lines[underline_idx - 2].strip()
        if cap and not _UNDERLINE_RE.match(cap):
            caption = cap

    # Parse data rows
    data_rows: list[list[str]] = []
    for line in lines[underline_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if _UNDERLINE_RE.match(stripped) and "-" in stripped:
            break
        row = [_extract_cell(line, s, e) for s, e in cols]
        row = [c for c in row if c]  # drop trailing empty cells
        if row:
            data_rows.append(row)

    if not data_rows:
        return None

    return ParsedTable(
        headers=headers,
        rows=data_rows,
        caption=caption,
        table_type="fixed_width",
    )


def _parse_kv_table(text: str) -> ParsedTable | None:
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        m = _KV_ROW_RE.match(line.rstrip())
        if m:
            pairs.append((m.group(1).strip(), m.group(2).strip()))

    if len(pairs) < 3:
        return None

    return ParsedTable(
        headers=["Field", "Value"],
        rows=[[k, v] for k, v in pairs],
        table_type="key_value",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_column_boundaries(underline: str) -> list[tuple[int, int]]:
    """Parse column start/end positions from a dash-underline row."""
    boundaries: list[tuple[int, int]] = []
    in_dash = False
    start = 0
    for i, ch in enumerate(underline):
        if ch in "-=":
            if not in_dash:
                start = i
                in_dash = True
        else:
            if in_dash:
                boundaries.append((start, i))
                in_dash = False
    if in_dash:
        boundaries.append((start, len(underline)))
    return boundaries


def _extract_cell(line: str, start: int, end: int) -> str:
    """Extract and strip a cell from a fixed-width line at column [start:end]."""
    if len(line) <= start:
        return ""
    # Extend slightly beyond end to catch values that spill into next column
    cell = line[start : end + 2] if len(line) > end else line[start:]
    return cell.strip()


def _line_offsets(lines: list[str]) -> list[int]:
    """Return the character offset of the start of each line."""
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)
    return offsets


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent spans."""
    if not spans:
        return []
    merged = [spans[0]]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
