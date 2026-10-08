"""Data models for LegalKit."""

from legalkit.models.document import DocumentNode, NodeType
from legalkit.models.retrieval import Chunk, RankedChunk, RetrievalResult
from legalkit.models.citation import (
    Claim,
    Citation,
    VerifiedCitation,
    VerificationStatus,
    EntailmentClass,
)
from legalkit.models.report import (
    EvidenceReport,
    ConfidenceLevel,
    ConfidenceScore,
    VerifiedClaim,
    RetrievalSummary,
)

__all__ = [
    "DocumentNode",
    "NodeType",
    "Chunk",
    "RankedChunk",
    "RetrievalResult",
    "Claim",
    "Citation",
    "VerifiedCitation",
    "VerificationStatus",
    "EntailmentClass",
    "EvidenceReport",
    "ConfidenceLevel",
    "ConfidenceScore",
    "VerifiedClaim",
    "RetrievalSummary",
]
