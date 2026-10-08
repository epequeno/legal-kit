"""Text utilities — fuzzy matching, normalization, etc."""

from __future__ import annotations

from rapidfuzz import fuzz


def fuzzy_match_score(text_a: str, text_b: str) -> float:
    """Compute normalized similarity between two texts. Returns 0.0-1.0."""
    if not text_a or not text_b:
        return 0.0
    return fuzz.ratio(text_a.lower().strip(), text_b.lower().strip()) / 100.0


def find_best_match(needle: str, haystack: str, min_score: float = 0.7) -> tuple[float, int, int] | None:
    """Find the best fuzzy match of needle within haystack.

    Returns (score, start, end) or None if no match above min_score.
    """
    if not needle or not haystack:
        return None

    needle_lower = needle.lower().strip()
    haystack_lower = haystack.lower()

    # Try exact match first
    idx = haystack_lower.find(needle_lower)
    if idx >= 0:
        return (1.0, idx, idx + len(needle_lower))

    # Sliding window fuzzy match
    needle_len = len(needle_lower)
    best_score = 0.0
    best_start = 0

    # Use windows of varying sizes around needle length
    for window_ratio in [1.0, 1.1, 0.9, 1.2, 0.8]:
        window = max(10, int(needle_len * window_ratio))
        for start in range(0, len(haystack_lower) - window + 1, max(1, window // 4)):
            end = min(start + window, len(haystack_lower))
            candidate = haystack_lower[start:end]
            score = fuzz.ratio(needle_lower, candidate) / 100.0
            if score > best_score:
                best_score = score
                best_start = start

    if best_score >= min_score:
        best_end = min(best_start + needle_len, len(haystack))
        return (best_score, best_start, best_end)

    return None


def normalize_section_id(raw: str) -> str:
    """Normalize a section reference for matching.

    Examples:
        'Section 4.2(a)' -> '§4.2(a)'
        '§ 4.2' -> '§4.2'
        'Article IV' -> 'Article-IV'
    """
    import re

    raw = raw.strip()

    # "Section X.Y" -> "§X.Y"
    m = re.match(r"(?:Section|SECTION)\s+([\d.]+(?:\([a-zA-Z0-9]+\))?)", raw, re.IGNORECASE)
    if m:
        return f"§{m.group(1)}"

    # "§ X.Y" -> "§X.Y"
    m = re.match(r"§\s*([\d.]+(?:\([a-zA-Z0-9]+\))?)", raw)
    if m:
        return f"§{m.group(1)}"

    # "Article IV" -> "Article-IV"
    m = re.match(r"(?:Article|ARTICLE)\s+([IVXLCDM]+|\d+)", raw, re.IGNORECASE)
    if m:
        return f"Article-{m.group(1)}"

    return raw
