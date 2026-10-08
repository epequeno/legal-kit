"""Internal test suite — 50 hand-labelled query/answer/verification triples.

This is the most authoritative evaluation dataset in the project: ground
truth we own, not borrowed from public benchmarks. It serves three purposes:

1. **Calibration anchor** — verifies that the LLM-generated entailment labels
   in the training data are correct. Even 50 reviewed samples is enough to
   measure systematic label bias.

2. **Regression test suite** — run after every pipeline change. A drop in
   performance here means we broke something; an improvement here means we
   actually made the pipeline better.

3. **Training data seed** — these approved triples are high-confidence examples
   for the fine-tuning dataset.

Schema
------
Each triple contains:

- ``query``       — the question posed to the pipeline
- ``ground_truth`` — correct answer + supporting citations
- ``verification`` — per-claim labels (is_correct, entailment, citation)
- ``metadata``    — difficulty, clause_type, query_type, review status

Triple status lifecycle:
    draft → needs_review → reviewed → approved

Usage::

    from legalkit.eval.test_suite import load_suite, SuiteStats

    triples = load_suite()
    stats = SuiteStats.from_triples(triples)
    print(stats.summary())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator


# Default path — can be overridden
DEFAULT_PATH = Path("data/test_suite/triples.json")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TripleStatus(Enum):
    DRAFT = "draft"
    NEEDS_REVIEW = "needs_review"
    REVIEWED = "reviewed"
    APPROVED = "approved"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QueryType(Enum):
    FACTUAL = "factual"        # What is X?
    EXISTENCE = "existence"    # Is there a clause about X?
    CONDITIONAL = "conditional"  # What happens if X?
    DEFINITION = "definition"  # What does X mean?
    SYNTHESIS = "synthesis"    # Multi-section reasoning
    NUANCE = "nuance"          # Knowledge qualifiers, carve-outs, edge cases


class EntailmentLabel(Enum):
    ENTAILED = "entailed"
    PARTIALLY_ENTAILED = "partially_entailed"
    CONTRADICTED = "contradicted"
    NEUTRAL = "neutral"
    NOT_FOUND = "not_found"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CitationRef:
    """A single citation in a triple."""
    section_id: str
    quote: str = ""            # exact or near-exact quote from the source
    entailment: str = "entailed"  # EntailmentLabel value


@dataclass
class ClaimLabel:
    """Verification label for a single claim within an answer."""
    text: str                  # the claim text
    is_correct: bool = True    # is this claim factually correct?
    citations: list[str] = field(default_factory=list)  # section IDs
    entailment: str = "entailed"
    notes: str = ""


@dataclass
class GroundTruth:
    """Ground-truth answer and citations for a query."""
    answer: str                # the correct answer
    citations: list[CitationRef] = field(default_factory=list)


@dataclass
class Verification:
    """Per-claim verification labels."""
    claims: list[ClaimLabel] = field(default_factory=list)


@dataclass
class TripleMetadata:
    """Metadata about a triple."""
    difficulty: str = "medium"
    query_type: str = "factual"
    clause_type: str = ""
    requires_inference: bool = False
    status: str = "needs_review"
    reviewer: str | None = None
    reviewed_at: str | None = None
    notes: str = ""


@dataclass
class Triple:
    """A single query/answer/verification triple."""

    id: str
    query: str
    document: dict            # {file_path, doc_id, contract_type}
    ground_truth: GroundTruth
    verification: Verification
    metadata: TripleMetadata

    @property
    def status(self) -> TripleStatus:
        return TripleStatus(self.metadata.status)

    @property
    def difficulty(self) -> Difficulty:
        return Difficulty(self.metadata.difficulty)

    @property
    def query_type(self) -> QueryType:
        return QueryType(self.metadata.query_type)

    @property
    def is_approved(self) -> bool:
        return self.metadata.status == TripleStatus.APPROVED.value

    @property
    def needs_review(self) -> bool:
        return self.metadata.status in (
            TripleStatus.NEEDS_REVIEW.value, TripleStatus.DRAFT.value
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "document": self.document,
            "ground_truth": {
                "answer": self.ground_truth.answer,
                "citations": [
                    {"section_id": c.section_id, "quote": c.quote,
                     "entailment": c.entailment}
                    for c in self.ground_truth.citations
                ],
            },
            "verification": {
                "claims": [
                    {"text": cl.text, "is_correct": cl.is_correct,
                     "citations": cl.citations, "entailment": cl.entailment,
                     "notes": cl.notes}
                    for cl in self.verification.claims
                ],
            },
            "metadata": {
                "difficulty": self.metadata.difficulty,
                "query_type": self.metadata.query_type,
                "clause_type": self.metadata.clause_type,
                "requires_inference": self.metadata.requires_inference,
                "status": self.metadata.status,
                "reviewer": self.metadata.reviewer,
                "reviewed_at": self.metadata.reviewed_at,
                "notes": self.metadata.notes,
            },
        }


# ---------------------------------------------------------------------------
# Loading / saving
# ---------------------------------------------------------------------------


def load_suite(path: Path | None = None) -> list[Triple]:
    """Load the test suite from disk.

    Args:
        path: Override the default path. Creates the file with an empty suite
            if it doesn't exist.

    Returns:
        List of :class:`Triple`.
    """
    p = path or DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Test suite not found: {p}\n"
            "Run 'legalkit suite-init' to create it."
        )

    with open(p) as f:
        data = json.load(f)

    triples = []
    for t in data.get("triples", []):
        gt_data = t.get("ground_truth", {})
        v_data = t.get("verification", {})
        m_data = t.get("metadata", {})

        ground_truth = GroundTruth(
            answer=gt_data.get("answer", ""),
            citations=[
                CitationRef(
                    section_id=c["section_id"],
                    quote=c.get("quote", ""),
                    entailment=c.get("entailment", "entailed"),
                )
                for c in gt_data.get("citations", [])
            ],
        )

        verification = Verification(
            claims=[
                ClaimLabel(
                    text=cl["text"],
                    is_correct=cl.get("is_correct", True),
                    citations=cl.get("citations", []),
                    entailment=cl.get("entailment", "entailed"),
                    notes=cl.get("notes", ""),
                )
                for cl in v_data.get("claims", [])
            ],
        )

        metadata = TripleMetadata(
            difficulty=m_data.get("difficulty", "medium"),
            query_type=m_data.get("query_type", "factual"),
            clause_type=m_data.get("clause_type", ""),
            requires_inference=m_data.get("requires_inference", False),
            status=m_data.get("status", "needs_review"),
            reviewer=m_data.get("reviewer"),
            reviewed_at=m_data.get("reviewed_at"),
            notes=m_data.get("notes", ""),
        )

        triples.append(Triple(
            id=t["id"],
            query=t["query"],
            document=t.get("document", {}),
            ground_truth=ground_truth,
            verification=verification,
            metadata=metadata,
        ))

    return triples


def save_suite(triples: list[Triple], path: Path | None = None) -> None:
    """Save the test suite to disk.

    Args:
        triples: Triples to save.
        path: Override the default path.
    """
    p = path or DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "version": "1.0",
        "description": "Hand-labelled query/answer/verification triples for internal evaluation.",
        "document": "tests/fixtures/sample_contract.txt",
        "triples": [t.to_dict() for t in triples],
    }
    with open(p, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class SuiteStats:
    """Summary statistics of the test suite."""

    total: int
    by_status: dict[str, int]
    by_difficulty: dict[str, int]
    by_query_type: dict[str, int]
    by_clause_type: dict[str, int]
    approved_count: int
    needs_review_count: int

    @classmethod
    def from_triples(cls, triples: list[Triple]) -> "SuiteStats":
        from collections import Counter

        status_c = Counter(t.metadata.status for t in triples)
        diff_c = Counter(t.metadata.difficulty for t in triples)
        qtype_c = Counter(t.metadata.query_type for t in triples)
        clause_c = Counter(t.metadata.clause_type for t in triples if t.metadata.clause_type)

        return cls(
            total=len(triples),
            by_status=dict(status_c),
            by_difficulty=dict(diff_c),
            by_query_type=dict(qtype_c),
            by_clause_type=dict(clause_c),
            approved_count=status_c.get("approved", 0),
            needs_review_count=(
                status_c.get("needs_review", 0) + status_c.get("draft", 0)
            ),
        )

    def summary(self) -> str:
        lines = [
            f"Test Suite Statistics",
            f"{'─' * 40}",
            f"  Total triples:     {self.total}",
            f"  Approved:          {self.approved_count}  ({self.approved_count/max(self.total,1):.0%})",
            f"  Needs review:      {self.needs_review_count}",
            f"  Reviewed (pending): {self.by_status.get('reviewed', 0)}",
            "",
            "  By difficulty:",
        ]
        for d in ("easy", "medium", "hard"):
            n = self.by_difficulty.get(d, 0)
            lines.append(f"    {d:<10} {n}")
        lines.append("")
        lines.append("  By query type:")
        for qt, n in sorted(self.by_query_type.items(), key=lambda x: -x[1]):
            lines.append(f"    {qt:<16} {n}")
        lines.append("")
        lines.append("  By clause type (top 10):")
        for ct, n in sorted(self.by_clause_type.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"    {ct:<24} {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of validating the test suite."""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            f"Validation: {'✓ PASS' if self.is_valid else '✗ FAIL'}",
            f"  Errors:   {len(self.errors)}",
            f"  Warnings: {len(self.warnings)}",
        ]
        for e in self.errors:
            lines.append(f"  [ERROR] {e}")
        for w in self.warnings:
            lines.append(f"  [WARN]  {w}")
        return "\n".join(lines)


def validate_suite(triples: list[Triple]) -> ValidationResult:
    """Validate the test suite for schema compliance and completeness.

    Args:
        triples: Triples to validate.

    Returns:
        :class:`ValidationResult`.
    """
    result = ValidationResult()

    ids = [t.id for t in triples]
    if len(ids) != len(set(ids)):
        dupes = [id for id in ids if ids.count(id) > 1]
        result.errors.append(f"Duplicate IDs: {set(dupes)}")

    valid_statuses = {s.value for s in TripleStatus}
    valid_difficulties = {d.value for d in Difficulty}
    valid_query_types = {qt.value for qt in QueryType}
    valid_entailments = {e.value for e in EntailmentLabel}

    for t in triples:
        prefix = f"[{t.id}]"

        if not t.query.strip():
            result.errors.append(f"{prefix} Empty query")

        if t.metadata.status not in valid_statuses:
            result.errors.append(
                f"{prefix} Invalid status: {t.metadata.status!r}"
            )

        if t.metadata.difficulty not in valid_difficulties:
            result.errors.append(
                f"{prefix} Invalid difficulty: {t.metadata.difficulty!r}"
            )

        if t.metadata.query_type not in valid_query_types:
            result.errors.append(
                f"{prefix} Invalid query_type: {t.metadata.query_type!r}"
            )

        if t.metadata.status in ("reviewed", "approved"):
            if not t.ground_truth.answer.strip():
                result.errors.append(f"{prefix} Status={t.metadata.status} but empty answer")
            if not t.ground_truth.citations:
                result.warnings.append(f"{prefix} No citations (may be a gap query)")
            for cl in t.verification.claims:
                if cl.entailment not in valid_entailments:
                    result.errors.append(
                        f"{prefix} Invalid entailment label: {cl.entailment!r}"
                    )

        if t.metadata.status == "needs_review" and not t.ground_truth.answer.strip():
            result.warnings.append(f"{prefix} Needs review and has no proposed answer")

    if len(triples) < 50:
        result.warnings.append(
            f"Suite has {len(triples)} triples; target is 50+"
        )

    approved = sum(1 for t in triples if t.is_approved)
    if approved < 10:
        result.warnings.append(
            f"Only {approved} approved triples; calibration needs ≥10"
        )

    return result
