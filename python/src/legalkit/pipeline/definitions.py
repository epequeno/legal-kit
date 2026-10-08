"""Definition extraction and linking.

Builds a DefinitionIndex from a parsed document tree, mapping defined
terms to their definition text. At retrieval time, chunks referencing
defined terms are enriched with the corresponding definition chunks so
the generation stage has the full picture.

Typical usage::

    from legalkit.pipeline.definitions import DefinitionIndex

    def_index = DefinitionIndex()
    def_index.build(root)

    # Look up a specific term
    entry = def_index.lookup("Material Adverse Effect")

    # Find all defined terms mentioned in a chunk's text
    refs = def_index.find_references(chunk.text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from legalkit.models.document import DocumentNode


@dataclass
class DefinitionEntry:
    """A single defined term and its definition text."""

    term: str
    """The defined term exactly as it appears in quotes, e.g. 'Material Adverse Effect'."""

    definition_text: str
    """The body of the definition — text after 'means' or 'shall mean'."""

    node_id: str
    """ID of the document node containing this definition."""

    node_text: str
    """Full text of the containing node (useful for embedding enrichment)."""

    char_start: int
    """Start offset in the original document."""

    char_end: int
    """End offset in the original document."""

    @property
    def normalized_term(self) -> str:
        return self.term.lower().strip()

    @property
    def full_text(self) -> str:
        """'Term' means definition_text — compact one-liner for injection."""
        return f'"{self.term}" means {self.definition_text}'.strip(".")

    def __repr__(self) -> str:
        preview = self.definition_text[:60].replace("\n", " ")
        return f"DefinitionEntry(term={self.term!r}, text={preview!r}...)"


class DefinitionIndex:
    """Maps defined terms to their definition entries.

    Built from a parsed ``DocumentNode`` tree. Supports exact lookup and
    full-text scanning to find all defined terms mentioned in a passage.
    """

    def __init__(self) -> None:
        self._by_term: dict[str, DefinitionEntry] = {}
        # Compiled regex for efficient multi-term scanning
        self._pattern: re.Pattern | None = None

    def build(self, root: DocumentNode) -> None:
        """Build the index from all nodes in the document tree.

        Walks every node, calls ``find_definitions()`` on its text, and
        populates the term → ``DefinitionEntry`` mapping. As a side effect,
        nodes that define terms have their ``defined_terms`` field confirmed
        (it may already be set by ``_annotate_definitions`` in parse.py).

        Args:
            root: The parsed document tree root.
        """
        from legalkit.pipeline.parse import find_definitions

        for node in root.all_nodes():
            text = node.text or ""
            if not text:
                continue
            for term, offset, body in find_definitions(text):
                if not term or not body:
                    continue
                normalized = term.lower().strip()
                # Prefer the first (most authoritative) definition if seen twice
                if normalized not in self._by_term:
                    self._by_term[normalized] = DefinitionEntry(
                        term=term,
                        definition_text=body,
                        node_id=node.id,
                        node_text=text,
                        char_start=node.char_start,
                        char_end=node.char_end,
                    )

        self._compile_pattern()

    def _compile_pattern(self) -> None:
        """Compile a single regex that matches any known defined term."""
        if not self._by_term:
            self._pattern = None
            return

        # Sort longest-first so "Material Adverse Effect" matches before
        # a hypothetical shorter "Material Adverse" term
        terms_sorted = sorted(self._by_term, key=len, reverse=True)
        escaped = [re.escape(t) for t in terms_sorted]
        self._pattern = re.compile(
            r"(?<!\w)(" + "|".join(escaped) + r")(?!\w)",
            re.IGNORECASE,
        )

    def lookup(self, term: str) -> DefinitionEntry | None:
        """Return the definition for ``term``, or None if not found.

        Args:
            term: Term to look up (case-insensitive).

        Returns:
            ``DefinitionEntry`` or ``None``.
        """
        return self._by_term.get(term.lower().strip())

    def find_references(self, text: str) -> list[DefinitionEntry]:
        """Scan ``text`` for any defined terms and return their entries.

        Each term is returned at most once, in order of first appearance.

        Args:
            text: Text to scan (e.g. a chunk's text).

        Returns:
            List of ``DefinitionEntry`` for each distinct term found.
        """
        if not self._pattern or not text:
            return []

        seen: set[str] = set()
        results: list[DefinitionEntry] = []
        for match in self._pattern.finditer(text):
            key = match.group(1).lower()
            if key not in seen:
                entry = self._by_term.get(key)
                if entry:
                    seen.add(key)
                    results.append(entry)
        return results

    def __len__(self) -> int:
        return len(self._by_term)

    @property
    def terms(self) -> list[str]:
        """All defined terms (original casing)."""
        return [e.term for e in self._by_term.values()]

    def __repr__(self) -> str:
        return f"DefinitionIndex({len(self)} terms)"
