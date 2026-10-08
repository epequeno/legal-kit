"""Evidence report models — the final output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

from legalkit.models.citation import VerifiedCitation, VerificationStatus, EntailmentClass


class ConfidenceLevel(Enum):
    """Human-readable confidence level."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ConfidenceScore:
    """Confidence assessment for a claim or overall answer."""

    level: ConfidenceLevel
    """High / medium / low."""

    score: float
    """Numeric score 0.0-1.0."""

    corroboration_count: int = 0
    """How many independent chunks support this."""

    flags: list[str] = field(default_factory=list)
    """Risk flags: 'single_source', 'ambiguous_entailment', 'cross_ref_unresolved', etc."""


@dataclass
class VerifiedClaim:
    """A claim with full verification results."""

    text: str
    """The claim text."""

    citations: list[VerifiedCitation] = field(default_factory=list)
    """Verified citations supporting this claim."""

    confidence: ConfidenceScore = field(
        default_factory=lambda: ConfidenceScore(ConfidenceLevel.LOW, 0.0)
    )
    """Confidence assessment."""

    has_hedging: bool = False
    """LLM expressed uncertainty."""

    is_gap: bool = False
    """Claim acknowledges missing information."""


@dataclass
class RetrievalSummary:
    """Summary of the retrieval process for transparency."""

    total_chunks_searched: int = 0
    """How many chunks were in the index."""

    chunks_retrieved: int = 0
    """How many chunks made it past retrieval."""

    cross_ref_chunks_added: int = 0
    """Additional chunks pulled via cross-reference."""

    semantic_hits: int = 0
    """Chunks found via semantic search."""

    bm25_hits: int = 0
    """Chunks found via BM25 search."""

    query_expansions: list[str] = field(default_factory=list)
    """Alternative queries generated."""

    avg_retrieval_score: float = 0.0
    """Average relevance score of retrieved chunks."""


@dataclass
class EvidenceReport:
    """The complete output of the citation engine.

    This is the primary artifact — a structured evidence report
    with verified claims, confidence scores, and gap analysis.
    """

    question: str
    """The user's question."""

    answer: str
    """Full text answer with inline citations."""

    claims: list[VerifiedClaim] = field(default_factory=list)
    """Individual claims extracted and verified."""

    overall_confidence: ConfidenceScore = field(
        default_factory=lambda: ConfidenceScore(ConfidenceLevel.LOW, 0.0)
    )
    """Overall confidence in the answer."""

    retrieval_summary: RetrievalSummary = field(default_factory=RetrievalSummary)
    """Transparency: how retrieval was performed."""

    gaps: list[str] = field(default_factory=list)
    """What the system could not find information about."""

    related_unresolved: list[str] = field(default_factory=list)
    """Potentially relevant sections that weren't cited."""

    doc_id: str = ""
    """Which document was queried."""

    model: str = ""
    """Which LLM was used for generation."""

    elapsed_seconds: float = 0.0
    """Total pipeline time."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""

        def _convert(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, tuple):
                return list(obj)
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _convert(v) for k, v in asdict(obj).items()}
            if isinstance(obj, list):
                return [_convert(item) for item in obj]
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            return obj

        return _convert(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
