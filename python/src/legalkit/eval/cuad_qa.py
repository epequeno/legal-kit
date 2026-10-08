"""CUAD clause QA evaluation.

Evaluates retrieval accuracy on the CUAD (Contract Understanding Atticus
Dataset) benchmark, broken down by clause type. CUAD covers 35 clause
categories — governing law, expiration date, non-compete, IP assignment,
termination for convenience, liability cap, audit rights, etc.

Unlike the aggregate LegalBench-RAG metrics, the per-clause breakdown
answers a concrete question: *which kinds of clause can the pipeline
reliably find?* This maps directly to how lawyers use the system.

The evaluation uses the existing :func:`~legalkit.eval.benchmark.run_benchmark`
infrastructure and adds clause-type grouping and per-category recall on top.

Usage::

    legalkit cuad-eval --max-per-clause 20 --top-k 64
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from legalkit.eval.benchmark import load_benchmark, run_benchmark
from legalkit.eval.metrics import BenchmarkResults, QueryResult


# ---------------------------------------------------------------------------
# Clause type extraction
# ---------------------------------------------------------------------------

# CUAD queries follow "Consider the X agreement...; <clause question>"
# We extract the question part and normalise to a short slug.

_CLAUSE_NORMALIZATIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"governing law", re.I),                    "governing_law"),
    (re.compile(r"expiration date", re.I),                  "expiration_date"),
    (re.compile(r"anti.?assignment", re.I),                 "anti_assignment"),
    (re.compile(r"cap on liability", re.I),                 "liability_cap"),
    (re.compile(r"licenses? (?:are )?granted", re.I),       "license_grant"),
    (re.compile(r"audit rights?", re.I),                    "audit_rights"),
    (re.compile(r"renewal term", re.I),                     "renewal_term"),
    (re.compile(r"terminat(?:e|ion|ed) for convenience", re.I), "termination_convenience"),
    (re.compile(r"exclusivity", re.I),                      "exclusivity"),
    (re.compile(r"(?:post.?termination|after the termination)", re.I), "post_termination_services"),
    (re.compile(r"revenue.?(?:or profit).?shar", re.I),     "revenue_sharing"),
    (re.compile(r"insurance", re.I),                        "insurance"),
    (re.compile(r"minimum commitment", re.I),               "minimum_commitment"),
    (re.compile(r"non.?transferable", re.I),                "non_transferable"),
    (re.compile(r"notice period.{0,30}terminat", re.I),     "notice_period_termination"),
    (re.compile(r"intellectual property.{0,20}(?:own|assign)", re.I), "ip_ownership"),
    (re.compile(r"uncapped liability", re.I),               "uncapped_liability"),
    (re.compile(r"non.?compete", re.I),                     "non_compete"),
    (re.compile(r"change of control", re.I),                "change_of_control"),
    (re.compile(r"covenant not to sue", re.I),              "covenant_not_to_sue"),
    (re.compile(r"most.?favou?red nation", re.I),           "most_favored_nation"),
    (re.compile(r"unlimited.{0,20}licen|all.you.can.eat", re.I), "unlimited_license"),
    (re.compile(r"right of first (?:refusal|offer|negotiation)", re.I), "right_of_first_refusal"),
    (re.compile(r"volume restriction", re.I),               "volume_restrictions"),
    (re.compile(r"exception.{0,30}(?:competi|restrict)", re.I), "competitive_restriction_exceptions"),
    (re.compile(r"duration.{0,20}warrant", re.I),           "warranty_duration"),
    (re.compile(r"solicit.{0,20}customer", re.I),           "non_solicitation"),
    (re.compile(r"non.?disparagement", re.I),               "non_disparagement"),
    (re.compile(r"affiliate.{0,30}licens", re.I),           "affiliate_license"),
    (re.compile(r"price restriction|price control", re.I),  "price_restrictions"),
    (re.compile(r"solicit.{0,20}employ", re.I),             "employee_non_solicitation"),
    (re.compile(r"indemnif", re.I),                         "indemnification"),
    (re.compile(r"warranty", re.I),                         "warranty"),
    (re.compile(r"confidential", re.I),                     "confidentiality"),
    (re.compile(r"arbitrat|dispute resolution", re.I),      "dispute_resolution"),
    (re.compile(r"limitation.{0,10}liabilit", re.I),        "limitation_of_liability"),
    (re.compile(r"effective date", re.I),                   "effective_date"),
    (re.compile(r"payment term", re.I),                     "payment_terms"),
    (re.compile(r"liquidated damage", re.I),                "liquidated_damages"),
    (re.compile(r"survival", re.I),                         "survival"),
    (re.compile(r"force majeure", re.I),                    "force_majeure"),
    (re.compile(r"third.?party beneficiar", re.I),          "third_party_beneficiary"),
    (re.compile(r"entire agreement", re.I),                 "entire_agreement"),
    (re.compile(r"amendment", re.I),                        "amendment"),
]


def extract_clause_type(query: str) -> str:
    """Extract a normalised clause type slug from a CUAD query string.

    CUAD queries follow the pattern::

        Consider the [Agreement Name]; [clause question]?

    The question part is matched against known clause patterns. Returns
    ``"other"`` if no known pattern matches.

    Args:
        query: Full CUAD query string.

    Returns:
        Normalised clause type slug, e.g. ``"governing_law"``.
    """
    # Split on semicolon to get the question part
    parts = query.split(";", 1)
    question = parts[1].strip() if len(parts) > 1 else query

    for pattern, slug in _CLAUSE_NORMALIZATIONS:
        if pattern.search(question):
            return slug

    # Fall back to first few words of question, lowercased and slugified
    words = re.sub(r"[^a-z0-9 ]", "", question.lower()).split()[:4]
    return "_".join(words) if words else "other"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ClauseTypeResult:
    """Retrieval results for one clause type across all queries of that type."""

    clause_type: str
    """Normalised clause type slug."""

    query_results: list[QueryResult] = field(default_factory=list)
    """Individual query results for this clause type."""

    def recall_at_k(self, k: int) -> float:
        from legalkit.eval.metrics import BenchmarkResults
        if not self.query_results:
            return 0.0
        results = BenchmarkResults(results=self.query_results)
        return results.recall_at_k(k)

    def precision_at_k(self, k: int) -> float:
        from legalkit.eval.metrics import BenchmarkResults
        if not self.query_results:
            return 0.0
        results = BenchmarkResults(results=self.query_results)
        return results.precision_at_k(k)

    @property
    def n_queries(self) -> int:
        return len(self.query_results)


@dataclass
class CUADReport:
    """Full CUAD clause QA evaluation report."""

    clause_results: dict[str, ClauseTypeResult]
    """Per-clause-type results, keyed by clause_type slug."""

    overall: BenchmarkResults
    """Aggregate results across all CUAD queries."""

    top_k: int = 64
    """Retrieval depth used."""

    @property
    def sorted_by_recall(self) -> list[ClauseTypeResult]:
        """Clause types sorted by recall@top_k (best first)."""
        return sorted(
            self.clause_results.values(),
            key=lambda r: r.recall_at_k(self.top_k),
            reverse=True,
        )

    @property
    def sorted_by_difficulty(self) -> list[ClauseTypeResult]:
        """Clause types sorted by recall@top_k (hardest first)."""
        return sorted(
            self.clause_results.values(),
            key=lambda r: r.recall_at_k(self.top_k),
        )

    def recall_at_k(self, k: int) -> float:
        return self.overall.recall_at_k(k)

    def summary_table(self, ks: list[int] | None = None) -> str:
        """Plain-text per-clause-type recall table."""
        ks = ks or [1, 4, 16, 64]
        ks = [k for k in ks if k <= self.top_k]

        col_w = 28
        header = f"{'Clause Type':<{col_w}}" + "".join(f"  R@{k:>2}" for k in ks) + "    N"
        sep = "─" * (col_w + len(ks) * 8 + 6)
        lines = [header, sep]

        for cr in self.sorted_by_recall:
            row = f"{cr.clause_type:<{col_w}}"
            for k in ks:
                row += f"  {cr.recall_at_k(k)*100:>5.1f}"
            row += f"  {cr.n_queries:>4}"
            lines.append(row)

        lines.append(sep)
        overall_row = f"{'OVERALL':<{col_w}}"
        for k in ks:
            overall_row += f"  {self.recall_at_k(k)*100:>5.1f}"
        overall_row += f"  {sum(cr.n_queries for cr in self.clause_results.values()):>4}"
        lines.append(overall_row)

        return "\n".join(lines)

    def to_dict(self, ks: list[int] | None = None) -> dict:
        ks = ks or [1, 4, 16, 64]
        return {
            "top_k": self.top_k,
            "overall": {f"recall@{k}": round(self.recall_at_k(k), 4) for k in ks},
            "per_clause_type": {
                ct: {
                    "n": cr.n_queries,
                    **{f"recall@{k}": round(cr.recall_at_k(k), 4) for k in ks},
                }
                for ct, cr in self.clause_results.items()
            },
        }


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def run_cuad_eval(
    max_per_clause: int | None = None,
    max_total: int | None = None,
    top_k: int = 64,
    min_chars: int = 100,
    max_chars: int = 1500,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> CUADReport:
    """Run the CUAD clause QA benchmark.

    Loads CUAD queries from the LegalBench-RAG benchmark file, groups them
    by clause type, runs retrieval via the existing benchmark runner, and
    returns a :class:`CUADReport` with per-clause-type recall breakdowns.

    Args:
        max_per_clause: Maximum queries per clause type (for fast evaluation).
        max_total: Cap total number of queries across all types.
        top_k: Retrieval depth.
        min_chars: Minimum chunk size.
        max_chars: Maximum chunk size.
        progress_callback: Optional ``(stage, current, total) -> None``.

    Returns:
        :class:`CUADReport`.
    """
    # Load CUAD queries only
    all_queries = load_benchmark(datasets=["cuad"], mini=False)

    # Group by clause type and optionally sample
    by_type: dict[str, list] = defaultdict(list)
    for q in all_queries:
        ct = extract_clause_type(q.query)
        by_type[ct].append(q)

    # Sample per clause type
    sampled = []
    for ct, qs in sorted(by_type.items()):
        if max_per_clause:
            qs = qs[:max_per_clause]
        sampled.extend(qs)

    if max_total:
        sampled = sampled[:max_total]

    if progress_callback:
        progress_callback(
            f"Loaded {len(sampled)} CUAD queries across {len(by_type)} clause types",
            0, len(sampled),
        )

    # Run the retrieval benchmark
    overall_results = run_benchmark(
        queries=sampled,
        max_chunks_per_query=top_k,
        min_chars=min_chars,
        max_chars=max_chars,
        progress_callback=progress_callback,
    )

    # Map query index → clause type
    query_to_type: dict[str, str] = {
        q.query: extract_clause_type(q.query) for q in sampled
    }

    # Group QueryResult objects by clause type
    clause_results: dict[str, ClauseTypeResult] = {}
    for qr in overall_results.results:
        ct = query_to_type.get(qr.query, "other")
        if ct not in clause_results:
            clause_results[ct] = ClauseTypeResult(clause_type=ct)
        clause_results[ct].query_results.append(qr)

    return CUADReport(
        clause_results=clause_results,
        overall=overall_results,
        top_k=top_k,
    )
