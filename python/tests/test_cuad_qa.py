"""Tests for CUAD clause QA evaluation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from legalkit.eval.cuad_qa import (
    ClauseTypeResult,
    CUADReport,
    extract_clause_type,
    run_cuad_eval,
)
from legalkit.eval.metrics import BenchmarkResults, QueryResult, SpanMatch


# ---------------------------------------------------------------------------
# extract_clause_type
# ---------------------------------------------------------------------------


class TestExtractClauseType:
    def _q(self, question: str) -> str:
        return f"Consider the Agreement between A and B; {question}"

    def test_governing_law(self):
        assert extract_clause_type(self._q("What is the governing law for this contract?")) == "governing_law"

    def test_expiration_date(self):
        assert extract_clause_type(self._q("What is the expiration date of this contract?")) == "expiration_date"

    def test_anti_assignment(self):
        assert extract_clause_type(self._q("Is there an anti-assignment clause in this contract?")) == "anti_assignment"

    def test_liability_cap(self):
        assert extract_clause_type(self._q("Is there a cap on liability under this contract?")) == "liability_cap"

    def test_license_grant(self):
        assert extract_clause_type(self._q("What licenses are granted under this contract?")) == "license_grant"

    def test_audit_rights(self):
        assert extract_clause_type(self._q("What are the audit rights under this contract?")) == "audit_rights"

    def test_termination_convenience(self):
        assert extract_clause_type(
            self._q("Can this contract be terminated for convenience?")
        ) == "termination_convenience"

    def test_non_compete(self):
        assert extract_clause_type(self._q("Is there a non-compete clause in this contract?")) == "non_compete"

    def test_ip_ownership(self):
        assert extract_clause_type(
            self._q("How is intellectual property ownership assigned in this contract?")
        ) == "ip_ownership"

    def test_change_of_control(self):
        assert extract_clause_type(
            self._q("What happens in the event of a change of control?")
        ) == "change_of_control"

    def test_indemnification(self):
        assert extract_clause_type(self._q("What are the indemnification obligations?")) == "indemnification"

    def test_returns_string(self):
        result = extract_clause_type("Some query without semicolon")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unknown_returns_other_or_slug(self):
        result = extract_clause_type("Consider the contract; What is the frobnitz provision?")
        assert isinstance(result, str)

    def test_no_semicolon_still_works(self):
        result = extract_clause_type("What is the governing law?")
        assert result == "governing_law"

    def test_real_cuad_queries(self):
        """Spot-check against actual CUAD query strings."""
        queries = [
            ("Consider the X Agreement; What is the governing law for this contract?", "governing_law"),
            ("Consider the Y Agreement; Is there an anti-assignment clause in this contract?", "anti_assignment"),
            ("Consider the Z Agreement; What is the expiration date of this contract?", "expiration_date"),
            ("Consider the W Agreement; Is there a non-compete clause in this contract?", "non_compete"),
            ("Consider the V Agreement; What are the audit rights under this contract?", "audit_rights"),
        ]
        for query, expected in queries:
            assert extract_clause_type(query) == expected, f"Failed for: {query}"


# ---------------------------------------------------------------------------
# ClauseTypeResult
# ---------------------------------------------------------------------------


def _make_query_results(n: int, recall: float = 0.8) -> list[QueryResult]:
    results = []
    for i in range(n):
        gt = SpanMatch(file_path="doc.txt", start=0, end=100)
        if recall >= 1.0:
            retrieved = [SpanMatch(file_path="doc.txt", start=0, end=100)]
        elif recall <= 0.0:
            retrieved = [SpanMatch(file_path="doc.txt", start=500, end=600)]
        else:
            end = int(100 * recall)
            retrieved = [SpanMatch(file_path="doc.txt", start=0, end=end)]
        results.append(QueryResult(
            query=f"q{i}", tags=["cuad"],
            ground_truth=[gt], retrieved=retrieved,
        ))
    return results


class TestClauseTypeResult:
    def test_n_queries(self):
        cr = ClauseTypeResult("governing_law", _make_query_results(5))
        assert cr.n_queries == 5

    def test_recall_at_k_perfect(self):
        cr = ClauseTypeResult("governing_law", _make_query_results(5, recall=1.0))
        assert cr.recall_at_k(64) == pytest.approx(1.0)

    def test_recall_at_k_zero(self):
        cr = ClauseTypeResult("governing_law", _make_query_results(5, recall=0.0))
        assert cr.recall_at_k(64) == pytest.approx(0.0)

    def test_empty_has_zero_recall(self):
        cr = ClauseTypeResult("governing_law", [])
        assert cr.recall_at_k(64) == 0.0


# ---------------------------------------------------------------------------
# CUADReport
# ---------------------------------------------------------------------------


def _make_report(clause_data: dict[str, tuple[int, float]]) -> CUADReport:
    """Build a CUADReport from {clause_type: (n_queries, recall)} dict."""
    all_qrs = []
    clause_results = {}
    for ct, (n, recall) in clause_data.items():
        qrs = _make_query_results(n, recall)
        clause_results[ct] = ClauseTypeResult(ct, qrs)
        all_qrs.extend(qrs)

    overall = BenchmarkResults(results=all_qrs)
    return CUADReport(clause_results=clause_results, overall=overall, top_k=64)


class TestCUADReport:
    def _make(self) -> CUADReport:
        return _make_report({
            "governing_law":    (10, 0.95),
            "expiration_date":  (10, 0.90),
            "non_compete":      (5,  0.60),
            "change_of_control":(5,  0.40),
            "liability_cap":    (8,  0.80),
        })

    def test_sorted_by_recall_best_first(self):
        report = self._make()
        ordered = [cr.clause_type for cr in report.sorted_by_recall]
        assert ordered[0] == "governing_law"

    def test_sorted_by_difficulty_worst_first(self):
        report = self._make()
        ordered = [cr.clause_type for cr in report.sorted_by_difficulty]
        assert ordered[0] == "change_of_control"

    def test_overall_recall_between_0_and_1(self):
        report = self._make()
        assert 0.0 <= report.recall_at_k(64) <= 1.0

    def test_summary_table_returns_string(self):
        report = self._make()
        table = report.summary_table()
        assert isinstance(table, str)
        assert "governing_law" in table
        assert "OVERALL" in table

    def test_summary_table_shows_all_types(self):
        report = self._make()
        table = report.summary_table()
        for ct in ["governing_law", "non_compete", "change_of_control"]:
            assert ct in table

    def test_to_dict_structure(self):
        report = self._make()
        d = report.to_dict()
        assert "overall" in d
        assert "per_clause_type" in d
        assert "recall@64" in d["overall"]

    def test_to_dict_per_clause_type(self):
        report = self._make()
        d = report.to_dict()
        for ct in ["governing_law", "non_compete"]:
            assert ct in d["per_clause_type"]
            assert "n" in d["per_clause_type"][ct]
            assert "recall@64" in d["per_clause_type"][ct]

    def test_overall_recall_weighted_by_count(self):
        """More queries → more influence on overall recall."""
        report = _make_report({
            "high_recall": (100, 1.0),  # 100 queries, perfect recall
            "low_recall":  (1,   0.0),  # 1 query, zero recall
        })
        # Overall should be close to 1.0 given 100:1 weighting
        assert report.recall_at_k(64) > 0.9


# ---------------------------------------------------------------------------
# run_cuad_eval — mocked
# ---------------------------------------------------------------------------


def _make_mock_query(query: str, file_path: str = "cuad/doc.txt"):
    from legalkit.eval.benchmark import BenchmarkQuery, GroundTruthSnippet
    return BenchmarkQuery(
        query=query,
        snippets=[GroundTruthSnippet(file_path=file_path, span=(0, 100))],
        tags=["cuad"],
    )


class TestRunCUADEval:
    def _mock_queries(self) -> list:
        return [
            _make_mock_query("Consider the X; What is the governing law for this contract?"),
            _make_mock_query("Consider the X; What is the expiration date of this contract?"),
            _make_mock_query("Consider the X; Is there a non-compete clause in this contract?"),
            _make_mock_query("Consider the X; Is there an anti-assignment clause in this contract?"),
        ]

    def test_returns_cuad_report(self):
        mock_results = BenchmarkResults(results=[
            QueryResult(
                query=f"Consider the X; {q}",
                tags=["cuad"],
                ground_truth=[SpanMatch("cuad/doc.txt", 0, 100)],
                retrieved=[SpanMatch("cuad/doc.txt", 0, 100)],
            )
            for q in [
                "What is the governing law for this contract?",
                "What is the expiration date of this contract?",
                "Is there a non-compete clause in this contract?",
                "Is there an anti-assignment clause in this contract?",
            ]
        ])

        with (
            patch("legalkit.eval.cuad_qa.load_benchmark", return_value=self._mock_queries()),
            patch("legalkit.eval.cuad_qa.run_benchmark", return_value=mock_results),
        ):
            report = run_cuad_eval(max_per_clause=5, top_k=64)

        assert isinstance(report, CUADReport)

    def test_clause_types_populated(self):
        mock_results = BenchmarkResults(results=[
            QueryResult(
                query="Consider the X; What is the governing law for this contract?",
                tags=["cuad"],
                ground_truth=[SpanMatch("cuad/doc.txt", 0, 100)],
                retrieved=[SpanMatch("cuad/doc.txt", 0, 100)],
            )
        ])

        with (
            patch("legalkit.eval.cuad_qa.load_benchmark",
                  return_value=[_make_mock_query("Consider the X; What is the governing law for this contract?")]),
            patch("legalkit.eval.cuad_qa.run_benchmark", return_value=mock_results),
        ):
            report = run_cuad_eval(max_per_clause=5)

        assert "governing_law" in report.clause_results

    def test_max_per_clause_respected(self):
        """Each clause type should be capped at max_per_clause queries."""
        queries = [
            _make_mock_query("Consider the X; What is the governing law for this contract?")
            for _ in range(20)
        ]
        mock_results = BenchmarkResults(results=[
            QueryResult(
                query="Consider the X; What is the governing law for this contract?",
                tags=["cuad"],
                ground_truth=[SpanMatch("cuad/doc.txt", 0, 100)],
                retrieved=[SpanMatch("cuad/doc.txt", 0, 100)],
            )
            for _ in range(5)  # max_per_clause=5
        ])

        captured_queries = []

        def mock_run_benchmark(queries, **kwargs):
            captured_queries.extend(queries)
            return mock_results

        with (
            patch("legalkit.eval.cuad_qa.load_benchmark", return_value=queries),
            patch("legalkit.eval.cuad_qa.run_benchmark", side_effect=mock_run_benchmark),
        ):
            run_cuad_eval(max_per_clause=5)

        # Should only pass 5 queries to run_benchmark (cap applied)
        assert len(captured_queries) <= 5


# ---------------------------------------------------------------------------
# Integration: load real CUAD benchmark and check clause types
# ---------------------------------------------------------------------------


class TestRealCUADData:
    def test_load_cuad_queries(self):
        """Verify we can load CUAD queries and extract clause types."""
        from legalkit.eval.benchmark import load_benchmark

        try:
            queries = load_benchmark(datasets=["cuad"], mini=False, max_per_dataset=50)
        except FileNotFoundError:
            pytest.skip("CUAD dataset not available")

        assert len(queries) > 0

        clause_types = {extract_clause_type(q.query) for q in queries}
        assert len(clause_types) >= 5  # Should cover multiple clause types

    def test_known_clause_types_present(self):
        """Known clause types should appear in the full CUAD dataset."""
        from legalkit.eval.benchmark import load_benchmark

        try:
            queries = load_benchmark(datasets=["cuad"], mini=False, max_per_dataset=100)
        except FileNotFoundError:
            pytest.skip("CUAD dataset not available")

        clause_types = {extract_clause_type(q.query) for q in queries}
        expected = {"governing_law", "expiration_date", "anti_assignment"}
        found = expected & clause_types
        assert len(found) >= 2, f"Expected clause types {expected}, found {clause_types}"

    def test_total_clause_type_coverage(self):
        """At least 20 distinct clause types should be identifiable."""
        from legalkit.eval.benchmark import load_benchmark

        try:
            queries = load_benchmark(datasets=["cuad"], mini=False)
        except FileNotFoundError:
            pytest.skip("CUAD dataset not available")

        clause_types = {extract_clause_type(q.query) for q in queries}
        assert len(clause_types) >= 20, f"Only found {len(clause_types)} clause types"
