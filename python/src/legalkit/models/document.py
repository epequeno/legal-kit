"""Document structure models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class NodeType(Enum):
    """Types of structural nodes in a legal document."""

    DOCUMENT = "document"
    ARTICLE = "article"
    SECTION = "section"
    SUBSECTION = "subsection"
    CLAUSE = "clause"
    SUBCLAUSE = "subclause"
    DEFINITION = "definition"
    RECITAL = "recital"
    EXHIBIT = "exhibit"
    SCHEDULE = "schedule"
    PARAGRAPH = "paragraph"
    PREAMBLE = "preamble"


@dataclass
class DocumentNode:
    """A structural node in a parsed legal document.

    The document is represented as a tree of nodes, where each node
    corresponds to a structural element (section, clause, definition, etc.).
    Every node carries char offsets back to the original document text,
    enabling precise citation.
    """

    id: str
    """Stable identifier, e.g. '§4.2(a)', 'Article-IV', 'Def-Affiliate'."""

    type: NodeType
    """What kind of structural element this is."""

    title: str
    """Section title or heading, e.g. 'Indemnification'. Empty string if none."""

    text: str
    """Full text content of this node (not including children)."""

    char_start: int
    """Start offset in the original document text."""

    char_end: int
    """End offset in the original document text."""

    children: list[DocumentNode] = field(default_factory=list)
    """Child nodes in the structural hierarchy."""

    parent_id: str | None = None
    """ID of the parent node, or None for the root."""

    cross_references: list[str] = field(default_factory=list)
    """Section IDs referenced within this node's text."""

    defined_terms: list[str] = field(default_factory=list)
    """Terms formally defined within this node (e.g. 'Material Adverse Effect')."""

    level: int = 0
    """Nesting depth (0 = root/document, 1 = article, 2 = section, etc.)."""

    @property
    def path(self) -> list[str]:
        """Ancestry path from root to this node (built during tree construction)."""
        # This is populated by the parser after tree construction
        return getattr(self, "_path", [self.id])

    @path.setter
    def path(self, value: list[str]) -> None:
        self._path = value

    @property
    def all_text(self) -> str:
        """Full text of this node and all descendants."""
        parts = [self.text]
        for child in self.children:
            parts.append(child.all_text)
        return "\n".join(parts)

    def find_by_id(self, node_id: str) -> DocumentNode | None:
        """Find a descendant node by ID (case-insensitive)."""
        if self.id.lower() == node_id.lower():
            return self
        for child in self.children:
            result = child.find_by_id(node_id)
            if result is not None:
                return result
        return None

    def all_nodes(self) -> list[DocumentNode]:
        """Flatten the tree into a list of all nodes (depth-first)."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.all_nodes())
        return nodes

    def __repr__(self) -> str:
        child_count = len(self.children)
        text_preview = self.text[:60].replace("\n", " ") + ("..." if len(self.text) > 60 else "")
        return (
            f"DocumentNode(id={self.id!r}, type={self.type.value}, "
            f"children={child_count}, text={text_preview!r})"
        )
