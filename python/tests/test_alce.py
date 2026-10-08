"""Tests for ALCE citation quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from legalkit.eval.alce import (
    ALCEMetrics,
    ALCEQueryResult,
    ALCEReport,
    _string_f1,
    _tokenize,
    compute_alce_metrics,
)
from legalkit.models.citation import (
    EntailmentClass,
    VerificationStatus,
    VerifiedCitation,
)
from legalkit.models.report import (
    ConfidenceLevel,
    ConfidenceScore,
    EvidenceReport,
    VerifiedClaim,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_verified_citation(status: VerificationStatus) -> VerifiedCitation:
    return VerifiedCitation(
        section_id="§1.1",
        exists=status != VerificationStatus.UNVERIFIED,
        entailment=EntailmentClass.ENTAILED
        if status == VerificationStatus.VERIFIED
        else EntailmentClass.NEUTRAL,
        entailment_score=0.9 if status == VerificationStatus.VERIFIED else 0.3,
        status=status,
    )


def _make_claim(
    text: str = "A claim.",
    citation_statuses: list[VerificationStatus] | None = None,
    is_gap: bool = False,
) -> VerifiedClaim:
    citations = [
        _make_verified_citation(s)
        for s in (citation_statuses or [VerificationStatus.VERIFIED])
    ]
    return VerifiedClaim(
        text=text,
        citations=citations,
        confidence=ConfidenceScore(ConfidenceLevel.HIGH, 0.9),
        is_gap=is_gap,
    )


def _make_report(
    claims: list[VerifiedClaim],
    answer: str = "This is the answer.",
) -> EvidenceReport:
    return EvidenceReport(
        question="What are the terms?",
        answer=answer,
        claims=claims,
        overall_confidence=ConfidenceScore(ConfidenceLevel.HIGH, 0.9),
        doc_id="test-doc",
    )


# ---------------------------------------------------------------------------
# _string_f1
# ---------------------------------------------------------------------------


class TestStringF1:
    def test_identical_strings(self):
        assert _string_f1("hello world", "hello world") == pytest.approx(1.0)

    def test_empty_prediction(self):
        assert _string_f1("", "hello world") == 0.0

    def test_empty_ground_truth(self):
        assert _string_f1("hello world", "") == 0.0

    def test_no_overlap(self):
        assert _string_f1("foo bar", "baz qux") == 0.0

    def test_partial_overlap(self):
        # "hello world" vs "hello earth" — 1 common token "hello"
        # pred_tokens = {hello:1, world:1}, gt_tokens = {hello:1, earth:1}
        # precision = 1/2, recall = 1/2, f1 = 0.5
        f1 = _string_f1("hello world", "hello earth")
        assert 0.4 < f1 < 0.6

    def test_case_insensitive(self):
        assert _string_f1("Hello World", "hello world") == pytest.approx(1.0)

    def test_order_invariant(self):
        # F1 measures token overlap, not sequence match
        f1_a = _string_f1("the quick brown fox", "fox brown quick the")
        assert f1_a == pytest.approx(1.0)

    def test_subset(self):
        # prediction is subset of ground truth
        f1 = _string_f1("hello", "hello world foo bar")
        assert 0 < f1 < 1.0


# ---------------------------------------------------------------------------
# compute_alce_metrics
# ---------------------------------------------------------------------------


class TestComputeALCEMetrics:
    def test_all_verified_perfect_recall(self):
        """All claims have verified citations → recall = 1.0."""
        claims = [
            _make_claim("Claim 1.", [VerificationStatus.VERIFIED]),
            _make_claim("Claim 2.", [VerificationStatus.VERIFIED]),
        ]
        report = _make_report(claims)
        m = compute_alce_metrics(report)
        assert m.citation_recall == pytest.approx(1.0)

    def test_no_verified_zero_recall(self):
        """No verified citations → recall = 0.0, hallucination = 1.0."""
        claims = [
            _make_claim("Claim 1.", [VerificationStatus.UNVERIFIED]),
            _make_claim("Claim 2.", [VerificationStatus.UNVERIFIED]),
        ]
        report = _make_report(claims)
        m = compute_alce_metrics(report)
        assert m.citation_recall == pytest.approx(0.0)
        assert m.hallucination_rate == pytest.approx(1.0)

    def test_partial_recall(self):
        """One verified, one unverified → recall = 0.5."""
        claims = [
            _make_claim("Claim 1.", [VerificationStatus.VERIFIED]),
            _make_claim("Claim 2.", [VerificationStatus.UNVERIFIED]),
        ]
        report = _make_report(claims)
        m = compute_alce_metrics(report)
        assert m.citation_recall == pytest.approx(0.5)

    def test_partially_verified_counts_as_verified(self):
        """PARTIALLY_VERIFIED citations count toward recall and precision."""
        claims = [_make_claim("A claim.", [VerificationStatus.PARTIALLY_VERIFIED])]
        report = _make_report(claims)
        m = compute_alce_metrics(report)
        assert m.citation_recall == pytest.approx(1.0)
        assert m.citation_precision == pytest.approx(1.0)

    def test_citation_precision(self):
        """2 verified, 1 unverified → precision = 2/3."""
        claim = VerifiedClaim(
            text="A claim.",
            citations=[
                _make_verified_citation(VerificationStatus.VERIFIED),
                _make_verified_citation(VerificationStatus.VERIFIED),
                _make_verified_citation(VerificationStatus.UNVERIFIED),
            ],
            confidence=ConfidenceScore(ConfidenceLevel.HIGH, 0.9),
        )
        report = _make_report([claim])
        m = compute_alce_metrics(report)
        assert m.citation_precision == pytest.approx(2 / 3, abs=1e-3)

    def test_gap_claims_excluded(self):
        """Gap acknowledgments don't affect citation metrics."""
        gap_claim = _make_claim("Missing info.", is_gap=True)
        real_claim = _make_claim("Real claim.", [VerificationStatus.VERIFIED])
        report = _make_report([gap_claim, real_claim])
        m = compute_alce_metrics(report)
        assert m.n_claims == 1  # only the real claim counts
        assert m.citation_recall == pytest.approx(1.0)

    def test_no_claims_zero_everything(self):
        """Empty claims → all zeros."""
        report = _make_report([])
        m = compute_alce_metrics(report)
        assert m.citation_recall == 0.0
        assert m.citation_precision == 0.0
        assert m.n_claims == 0

    def test_f1_is_harmonic_mean(self):
        """Citation F1 = 2 * recall * precision / (recall + precision)."""
        claims = [
            _make_claim("Verified.", [VerificationStatus.VERIFIED]),
            _make_claim("Unverified.", [VerificationStatus.UNVERIFIED]),
        ]
        report = _make_report(claims)
        m = compute_alce_metrics(report)
        expected_f1 = 2 * m.citation_recall * m.citation_precision / (
            m.citation_recall + m.citation_precision
        ) if (m.citation_recall + m.citation_precision) > 0 else 0
        assert m.citation_f1 == pytest.approx(expected_f1, abs=0.001)

    def test_hallucination_is_complement_of_recall(self):
        """Hallucination rate + recall should sum to 1 (for claims with citations)."""
        claims = [
            _make_claim("Verified.", [VerificationStatus.VERIFIED]),
            _make_claim("Unverified.", [VerificationStatus.UNVERIFIED]),
        ]
        report = _make_report(claims)
        m = compute_alce_metrics(report)
        assert m.citation_recall + m.hallucination_rate == pytest.approx(1.0)

    def test_string_f1_with_ground_truth(self):
        """String F1 is computed when ground truth is provided."""
        report = _make_report([], answer="The purchase price is fifty million dollars.")
        m = compute_alce_metrics(report, ground_truth="The purchase price is fifty million dollars.")
        assert m.answer_string_f1 == pytest.approx(1.0)

    def test_string_f1_none_without_ground_truth(self):
        """String F1 is None when no ground truth is provided."""
        report = _make_report([])
        m = compute_alce_metrics(report, ground_truth=None)
        assert m.answer_string_f1 is None

    def test_n_claims_counts_non_gap(self):
        claims = [
            _make_claim("A.", is_gap=False),
            _make_claim("B.", is_gap=False),
            _make_claim("C.", is_gap=True),
        ]
        report = _make_report(claims)
        m = compute_alce_metrics(report)
        assert m.n_claims == 2

    def test_n_citations_counts_all(self):
        claim = VerifiedClaim(
            text="A claim.",
            citations=[
                _make_verified_citation(VerificationStatus.VERIFIED),
                _make_verified_citation(VerificationStatus.VERIFIED),
                _make_verified_citation(VerificationStatus.UNVERIFIED),
            ],
            confidence=ConfidenceScore(ConfidenceLevel.HIGH, 0.9),
        )
        report = _make_report([claim])
        m = compute_alce_metrics(report)
        assert m.n_citations == 3
        assert m.n_verified_citations == 2


# ---------------------------------------------------------------------------
# ALCEReport
# ---------------------------------------------------------------------------


class TestALCEReport:
    def _make_result(
        self,
        recall: float,
        precision: float,
        hallucination: float,
        str_f1: float | None = 0.5,
        dataset: str = "test",
    ) -> ALCEQueryResult:
        f1 = (2 * recall * precision / (recall + precision)
              if (recall + precision) > 0 else 0.0)
        metrics = ALCEMetrics(
            citation_recall=recall,
            citation_precision=precision,
            citation_f1=f1,
            answer_string_f1=str_f1,
            hallucination_rate=hallucination,
            n_claims=5,
            n_citations=5,
            n_verified_citations=int(precision * 5),
        )
        return ALCEQueryResult(
            query="q", dataset=dataset, answer="a",
            ground_truth="gt", metrics=metrics,
        )

    def test_avg_citation_recall(self):
        report = ALCEReport(results=[
            self._make_result(recall=0.8, precision=0.9, hallucination=0.2),
            self._make_result(recall=0.6, precision=0.7, hallucination=0.4),
        ])
        assert report.avg_citation_recall == pytest.approx(0.7)

    def test_avg_citation_precision(self):
        report = ALCEReport(results=[
            self._make_result(recall=0.8, precision=0.9, hallucination=0.2),
            self._make_result(recall=0.6, precision=0.7, hallucination=0.4),
        ])
        assert report.avg_citation_precision == pytest.approx(0.8)

    def test_avg_hallucination_rate(self):
        report = ALCEReport(results=[
            self._make_result(recall=0.8, precision=0.9, hallucination=0.2),
            self._make_result(recall=0.6, precision=0.7, hallucination=0.4),
        ])
        assert report.avg_hallucination_rate == pytest.approx(0.3)

    def test_avg_answer_string_f1(self):
        report = ALCEReport(results=[
            self._make_result(0.8, 0.9, 0.2, str_f1=0.6),
            self._make_result(0.6, 0.7, 0.4, str_f1=0.4),
        ])
        assert report.avg_answer_string_f1 == pytest.approx(0.5)

    def test_avg_string_f1_none_when_absent(self):
        report = ALCEReport(results=[
            self._make_result(0.8, 0.9, 0.2, str_f1=None),
        ])
        assert report.avg_answer_string_f1 is None

    def test_per_dataset_split(self):
        report = ALCEReport(results=[
            self._make_result(0.8, 0.9, 0.2, dataset="cuad"),
            self._make_result(0.6, 0.7, 0.4, dataset="contractnli"),
            self._make_result(0.7, 0.8, 0.3, dataset="cuad"),
        ])
        per_ds = report.per_dataset()
        assert "cuad" in per_ds
        assert "contractnli" in per_ds
        assert len(per_ds["cuad"].results) == 2
        assert len(per_ds["contractnli"].results) == 1

    def test_empty_report(self):
        report = ALCEReport(results=[])
        assert report.avg_citation_recall == 0.0
        assert report.avg_citation_precision == 0.0
        assert report.avg_answer_string_f1 is None

    def test_summary_table_returns_string(self):
        report = ALCEReport(results=[
            self._make_result(0.85, 0.90, 0.15),
        ])
        table = report.summary_table()
        assert isinstance(table, str)
        assert "Citation Recall" in table
        assert "Citation Precision" in table
        assert "Hallucination" in table

    def test_to_dict_structure(self):
        report = ALCEReport(results=[
            self._make_result(0.8, 0.9, 0.2),
        ])
        d = report.to_dict()
        assert "avg_citation_recall" in d
        assert "avg_citation_precision" in d
        assert "avg_citation_f1" in d
        assert "avg_hallucination_rate" in d
        assert "per_query" in d

    def test_total_claims_and_citations(self):
        report = ALCEReport(results=[
            self._make_result(0.8, 0.9, 0.2),
            self._make_result(0.6, 0.7, 0.4),
        ])
        assert report.total_claims == 10
        assert report.total_citations == 10


# ---------------------------------------------------------------------------
# GroundTruthSnippet answer field
# ---------------------------------------------------------------------------


class TestGroundTruthSnippetAnswer:
    def test_snippet_has_answer_field(self):
        from legalkit.eval.benchmark import GroundTruthSnippet
        s = GroundTruthSnippet(file_path="doc.txt", span=(0, 100), answer="The answer.")
        assert s.answer == "The answer."

    def test_snippet_default_empty_answer(self):
        from legalkit.eval.benchmark import GroundTruthSnippet
        s = GroundTruthSnippet(file_path="doc.txt", span=(0, 100))
        assert s.answer == ""

    def test_load_benchmark_populates_answer(self):
        """Verify that load_benchmark fills in the answer field from JSON."""
        from legalkit.eval.benchmark import load_benchmark
        try:
            queries = load_benchmark(datasets=["cuad"], mini=True, max_per_dataset=5)
        except FileNotFoundError:
            pytest.skip("CUAD dataset not available")
        # At least some should have answers
        answers = [s.answer for q in queries for s in q.snippets]
        assert any(a for a in answers), "No answers found in CUAD benchmark"


# ---------------------------------------------------------------------------
# Integration: compute_alce_metrics from real benchmark data
# ---------------------------------------------------------------------------


class TestALCEIntegration:
    def test_perfect_pipeline_metrics(self):
        """Simulate a perfect pipeline: all claims verified."""
        verified_claims = [
            _make_claim(f"Claim {i}.", [VerificationStatus.VERIFIED])
            for i in range(5)
        ]
        report = _make_report(
            verified_claims,
            answer="The agreement has five verified provisions.",
        )
        m = compute_alce_metrics(
            report,
            ground_truth="The agreement has five verified provisions.",
        )
        assert m.citation_recall == pytest.approx(1.0)
        assert m.citation_precision == pytest.approx(1.0)
        assert m.citation_f1 == pytest.approx(1.0)
        assert m.hallucination_rate == pytest.approx(0.0)
        assert m.answer_string_f1 == pytest.approx(1.0)

    def test_all_hallucinated_metrics(self):
        """Simulate a fully hallucinating pipeline."""
        unverified_claims = [
            _make_claim(f"Claim {i}.", [VerificationStatus.UNVERIFIED])
            for i in range(3)
        ]
        report = _make_report(unverified_claims)
        m = compute_alce_metrics(report)
        assert m.citation_recall == pytest.approx(0.0)
        assert m.hallucination_rate == pytest.approx(1.0)
        assert m.citation_f1 == pytest.approx(0.0)
