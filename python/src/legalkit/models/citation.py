"""Citation and verification models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class VerificationStatus(Enum):
    """Result of citation verification."""

    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    NOT_CHECKED = "not_checked"


class EntailmentClass(Enum):
    """NLI entailment classification."""

    ENTAILED = "entailed"
    PARTIALLY_ENTAILED = "partially_entailed"
    CONTRADICTED = "contradicted"
    NEUTRAL = "neutral"


@dataclass
class Citation:
    """A citation extracted from LLM-generated text (pre-verification)."""

    section_id: str
    """The section identifier cited, e.g. '§4.2(a)'."""

    quoted_text: str | None = None
    """Exact text quoted from the source, if any."""

    claim_text: str = ""
    """The claim this citation supports."""


@dataclass
class VerifiedCitation:
    """A citation after running through the verification pipeline."""

    section_id: str
    """The section identifier cited."""

    section_path: list[str] = field(default_factory=list)
    """Full path, e.g. ['Article IV', 'Section 4.2', '(a)']."""

    source_text: str = ""
    """The actual text from the document at this citation location."""

    char_span: tuple[int, int] = (0, 0)
    """Character offsets in the original document."""

    # Retrieval info
    retrieval_method: str = ""
    """How the source chunk was retrieved: 'semantic', 'bm25', 'cross_reference'."""

    retrieval_score: float = 0.0
    """Relevance score from retrieval."""

    # Verification results
    exists: bool = False
    """Check 1: Does this section ID exist in the document?"""

    quote_similarity: float = 0.0
    """Check 2: How closely does quoted text match source? 0.0-1.0."""

    entailment: EntailmentClass = EntailmentClass.NEUTRAL
    """Check 3: Does the source text entail the claim?"""

    entailment_score: float = 0.0
    """Check 3: Confidence of entailment classification. 0.0-1.0."""

    entailment_explanation: str = ""
    """Check 3: Brief explanation of the entailment judgment."""

    # Overall
    status: VerificationStatus = VerificationStatus.NOT_CHECKED
    """Overall verification status after all checks."""

    @property
    def composite_score(self) -> float:
        """Weighted composite verification score."""
        existence = 1.0 if self.exists else 0.0
        return (
            0.3 * existence
            + 0.2 * self.quote_similarity
            + 0.4 * self.entailment_score
            + 0.1 * self.retrieval_score
        )


@dataclass
class Claim:
    """A single factual claim extracted from the LLM's response."""

    text: str
    """The claim text."""

    citations: list[Citation] = field(default_factory=list)
    """Citations the LLM provided for this claim."""

    has_hedging: bool = False
    """Whether the LLM expressed uncertainty about this claim."""

    is_gap: bool = False
    """Whether this claim acknowledges missing information."""
