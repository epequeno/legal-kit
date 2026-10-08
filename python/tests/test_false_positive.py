"""Tests for false positive analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

from legalkit.eval.false_positive import (
    FailureCase,
    FailureCategory,
    FalsePositiveReport,
    analyze_nli_failures,
    categorize_failure,
    _CATEGORY_DESCRIPTIONS,
    _CATEGORY_RECOMMENDATIONS,
)
from legalkit.eval.nli_eval import (
    NLIEvalResult,
    NLIJudgment,
    NLITestCase,
    load_edge_cases,
)

EDGE_CASES_PATH = Path(__file__).parent.parent / "data/eval/legal_nli_edge_cases.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nli_result(
    predictions: list[str],
    ground_truths: list[str],
    passages: list[str] | None = None,
    claims: list[str] | None = None,
    confidences: list[float] | None = None,
    categories: list[str] | None = None,
) -> NLIEvalResult:
    n = len(predictions)
    passages = passages or ["passage text"] * n
    claims = claims or ["claim text"] * n
    confidences = confidences or [0.8] * n
    categories = categories or ["test"] * n

    cases = [
        NLITestCase(
            id=f"c{i}", category=categories[i],
            passage=passages[i], claim=claims[i], label=gt,
        )
        for i, gt in enumerate(ground_truths)
    ]
    judgments = [
        NLIJudgment(case_id=f"c{i}", predicted=p, confidence=c, latency_ms=50.0)
        for i, (p, c) in enumerate(zip(predictions, confidences))
    ]
    return NLIEvalResult(
        model_name="test-model",
        judgments=judgments,
        test_cases=cases,
        total_latency_ms=n * 50.0,
    )


# ---------------------------------------------------------------------------
# categorize_failure — heuristic rules
# ---------------------------------------------------------------------------


class TestCategorizeFailure:
    def test_notwithstanding_triggers_carveout(self):
        passage = "Notwithstanding Section 8.1, Buyer shall have no indemnification obligation."
        claim = "Buyer has no indemnification obligation for any losses."
        cat = categorize_failure(passage, claim, "entailed", "contradicted")
        assert cat == FailureCategory.CARVEOUT_MISSED

    def test_provided_that_triggers_carveout(self):
        passage = "MAE means any change; provided, however, that economic conditions shall not constitute a MAE."
        claim = "An economic downturn constitutes a MAE."
        cat = categorize_failure(passage, claim, "entailed", "contradicted")
        assert cat == FailureCategory.CARVEOUT_MISSED

    def test_except_that_triggers_carveout(self):
        passage = "Licensor grants a license except that Licensee may not sublicense."
        claim = "Licensee may sublicense the software."
        cat = categorize_failure(passage, claim, "entailed", "contradicted")
        assert cat == FailureCategory.CARVEOUT_MISSED

    def test_may_with_mandatory_claim_triggers_modal(self):
        passage = "The board may, in its sole discretion, accelerate vesting upon a Change of Control."
        claim = "Vesting automatically accelerates upon a Change of Control."
        cat = categorize_failure(passage, claim, "entailed", "contradicted")
        assert cat == FailureCategory.MODAL_CONFUSION

    def test_knowledge_qualifier_triggers_qualifier_omission(self):
        passage = "To the Knowledge of Seller, there are no pending legal proceedings."
        claim = "There are no pending legal proceedings."
        cat = categorize_failure(passage, claim, "entailed", "partially_entailed")
        assert cat == FailureCategory.QUALIFIER_OMISSION

    def test_material_qualifier_triggers_qualifier_omission(self):
        passage = "The Company has complied in all material respects with applicable law."
        claim = "The Company has not violated any applicable law."
        cat = categorize_failure(passage, claim, "entailed", "partially_entailed")
        assert cat == FailureCategory.QUALIFIER_OMISSION

    def test_up_to_triggers_quantitative(self):
        passage = "Buyer shall pay a termination fee of up to $5,000,000."
        claim = "The termination fee is $5,000,000."
        cat = categorize_failure(passage, claim, "entailed", "partially_entailed")
        assert cat == FailureCategory.QUANTITATIVE_IMPRECISION

    def test_exceeding_triggers_quantitative(self):
        passage = "Seller shall indemnify Buyer for losses exceeding $500,000 (the Deductible)."
        claim = "Seller must pay the first $500,000 of any losses."
        cat = categorize_failure(passage, claim, "entailed", "contradicted")
        assert cat == FailureCategory.QUANTITATIVE_IMPRECISION

    def test_subject_to_triggers_condition(self):
        passage = "The obligation to close is subject to receipt of regulatory approvals."
        claim = "Buyer must close the transaction regardless of regulatory status."
        cat = categorize_failure(passage, claim, "entailed", "contradicted")
        assert cat == FailureCategory.CONDITION_STRIPPED

    def test_unknown_for_plain_mismatch(self):
        passage = "The Restricted Period lasts three years from closing."
        claim = "The Restricted Period lasts five years from closing."
        cat = categorize_failure(passage, claim, "entailed", "contradicted")
        # No clear heuristic trigger
        assert cat == FailureCategory.UNKNOWN

    def test_returns_failure_category_enum(self):
        cat = categorize_failure("any passage", "any claim", "entailed", "contradicted")
        assert isinstance(cat, FailureCategory)

    def test_all_categories_reachable(self):
        """At least some categories are reachable via the heuristics."""
        reachable = set()
        cases = [
            ("Notwithstanding anything, Seller retains rights.", "Seller has no rights.", "entailed", "contradicted"),
            ("Board may accelerate vesting at discretion.", "Vesting automatically accelerates.", "entailed", "contradicted"),
            ("To Seller's Knowledge, no proceedings exist.", "No proceedings exist.", "entailed", "partially_entailed"),
            ("Losses up to $5M may be claimed.", "Losses of exactly $5M are recoverable.", "entailed", "partially_entailed"),
            ("Closing is subject to regulatory approval.", "Buyer must close unconditionally.", "entailed", "contradicted"),
        ]
        for passage, claim, pred, true in cases:
            reachable.add(categorize_failure(passage, claim, pred, true))
        assert len(reachable) >= 4


# ---------------------------------------------------------------------------
# FailureCase
# ---------------------------------------------------------------------------


class TestFailureCase:
    def _make_fp(self) -> FailureCase:
        return FailureCase(
            case_id="c1",
            passage="passage",
            claim="claim",
            predicted_label="entailed",
            true_label="contradicted",
            category=FailureCategory.CARVEOUT_MISSED,
        )

    def _make_fn(self) -> FailureCase:
        return FailureCase(
            case_id="c2",
            passage="passage",
            claim="claim",
            predicted_label="neutral",
            true_label="entailed",
            category=FailureCategory.MODAL_CONFUSION,
        )

    def _make_correct(self) -> FailureCase:
        return FailureCase(
            case_id="c3",
            passage="passage",
            claim="claim",
            predicted_label="entailed",
            true_label="entailed",
            category=FailureCategory.UNKNOWN,
        )

    def test_is_false_positive(self):
        assert self._make_fp().is_false_positive
        assert not self._make_fn().is_false_positive
        assert not self._make_correct().is_false_positive

    def test_is_false_negative(self):
        assert self._make_fn().is_false_negative
        assert not self._make_fp().is_false_negative
        assert not self._make_correct().is_false_negative

    def test_partially_entailed_pred_is_positive(self):
        fc = FailureCase(
            case_id="x", passage="p", claim="c",
            predicted_label="partially_entailed",
            true_label="contradicted",
            category=FailureCategory.UNKNOWN,
        )
        assert fc.is_false_positive

    def test_neutral_pred_vs_entailed_is_false_negative(self):
        fc = FailureCase(
            case_id="x", passage="p", claim="c",
            predicted_label="neutral",
            true_label="partially_entailed",
            category=FailureCategory.UNKNOWN,
        )
        assert fc.is_false_negative


# ---------------------------------------------------------------------------
# FalsePositiveReport
# ---------------------------------------------------------------------------


class TestFalsePositiveReport:
    def _make_report(self) -> FalsePositiveReport:
        cases = [
            FailureCase("c1", "p", "c", "entailed", "contradicted",
                        FailureCategory.CARVEOUT_MISSED),
            FailureCase("c2", "p", "c", "entailed", "contradicted",
                        FailureCategory.CARVEOUT_MISSED),
            FailureCase("c3", "p", "c", "entailed", "neutral",
                        FailureCategory.MODAL_CONFUSION),
            FailureCase("c4", "p", "c", "neutral", "entailed",
                        FailureCategory.MODAL_CONFUSION),  # false negative
        ]
        return FalsePositiveReport(cases=cases, source="test")

    def test_false_positives_count(self):
        r = self._make_report()
        assert len(r.false_positives) == 3

    def test_false_negatives_count(self):
        r = self._make_report()
        assert len(r.false_negatives) == 1

    def test_category_counts(self):
        r = self._make_report()
        counts = r.category_counts
        assert counts[FailureCategory.CARVEOUT_MISSED] == 2
        assert counts[FailureCategory.MODAL_CONFUSION] == 1

    def test_top_categories_sorted(self):
        r = self._make_report()
        cats = [cat for cat, _ in r.top_categories]
        assert cats[0] == FailureCategory.CARVEOUT_MISSED

    def test_examples_for(self):
        r = self._make_report()
        examples = r.examples_for(FailureCategory.CARVEOUT_MISSED)
        assert len(examples) >= 1
        assert all(e.category == FailureCategory.CARVEOUT_MISSED for e in examples)

    def test_examples_limited_by_n(self):
        r = self._make_report()
        examples = r.examples_for(FailureCategory.CARVEOUT_MISSED, n=1)
        assert len(examples) == 1

    def test_summary_table_returns_string(self):
        r = self._make_report()
        table = r.summary_table()
        assert isinstance(table, str)
        assert "carveout_missed" in table

    def test_summary_table_shows_percentages(self):
        r = self._make_report()
        table = r.summary_table()
        assert "%" in table

    def test_to_dict_structure(self):
        r = self._make_report()
        d = r.to_dict()
        assert "total_failures" in d
        assert "false_positives" in d
        assert "false_negatives" in d
        assert "category_distribution" in d
        assert "cases" in d

    def test_to_dict_case_fields(self):
        r = self._make_report()
        d = r.to_dict()
        for case in d["cases"]:
            assert "id" in case
            assert "category" in case
            assert "predicted" in case
            assert "true" in case

    def test_empty_report(self):
        r = FalsePositiveReport(cases=[], source="empty")
        assert r.false_positives == []
        assert r.category_counts == {}
        table = r.summary_table()
        assert isinstance(table, str)


# ---------------------------------------------------------------------------
# analyze_nli_failures
# ---------------------------------------------------------------------------


class TestAnalyzeNLIFailures:
    def test_returns_report(self):
        result = _make_nli_result(
            predictions=["entailed", "contradicted"],
            ground_truths=["contradicted", "entailed"],
        )
        report = analyze_nli_failures(result)
        assert isinstance(report, FalsePositiveReport)

    def test_finds_false_positives(self):
        result = _make_nli_result(
            predictions=["entailed", "entailed"],
            ground_truths=["contradicted", "entailed"],
        )
        report = analyze_nli_failures(result)
        assert len(report.false_positives) == 1

    def test_no_failures_for_perfect_result(self):
        result = _make_nli_result(
            predictions=["entailed", "contradicted"],
            ground_truths=["entailed", "contradicted"],
        )
        report = analyze_nli_failures(result)
        assert len(report.cases) == 0

    def test_source_contains_model_name(self):
        result = _make_nli_result(["entailed"], ["contradicted"])
        report = analyze_nli_failures(result)
        assert "test-model" in report.source

    def test_categories_assigned(self):
        result = _make_nli_result(
            predictions=["entailed"],
            ground_truths=["contradicted"],
            passages=["Notwithstanding Section 8, Buyer has no obligation."],
            claims=["Buyer must fulfill the obligation."],
        )
        report = analyze_nli_failures(result)
        assert len(report.cases) == 1
        assert report.cases[0].category == FailureCategory.CARVEOUT_MISSED

    def test_include_false_negatives_true(self):
        result = _make_nli_result(
            predictions=["neutral"],
            ground_truths=["entailed"],
        )
        report = analyze_nli_failures(result, include_false_negatives=True)
        assert len(report.false_negatives) == 1

    def test_include_false_negatives_false(self):
        result = _make_nli_result(
            predictions=["neutral"],
            ground_truths=["entailed"],
        )
        report = analyze_nli_failures(result, include_false_negatives=False)
        assert len(report.cases) == 0

    def test_confidence_recorded(self):
        result = _make_nli_result(
            predictions=["entailed"],
            ground_truths=["contradicted"],
            confidences=[0.92],
        )
        report = analyze_nli_failures(result)
        assert report.cases[0].confidence == pytest.approx(0.92)

    def test_explanation_from_case(self):
        from legalkit.eval.nli_eval import NLITestCase, NLIJudgment
        case = NLITestCase(
            id="x", category="test", passage="p", claim="c",
            label="contradicted",
            explanation="This is the explanation.",
        )
        judgment = NLIJudgment(
            case_id="x", predicted="entailed", confidence=0.8, latency_ms=50.0
        )
        result = NLIEvalResult(
            model_name="m", judgments=[judgment],
            test_cases=[case], total_latency_ms=50.0,
        )
        report = analyze_nli_failures(result)
        assert report.cases[0].explanation == "This is the explanation."


# ---------------------------------------------------------------------------
# Taxonomy completeness
# ---------------------------------------------------------------------------


class TestTaxonomy:
    def test_all_categories_have_descriptions(self):
        for cat in FailureCategory:
            assert cat in _CATEGORY_DESCRIPTIONS, f"{cat} missing description"
            assert len(_CATEGORY_DESCRIPTIONS[cat]) > 10

    def test_all_categories_have_recommendations(self):
        for cat in FailureCategory:
            assert cat in _CATEGORY_RECOMMENDATIONS, f"{cat} missing recommendation"
            assert len(_CATEGORY_RECOMMENDATIONS[cat]) > 10

    def test_category_values_are_strings(self):
        for cat in FailureCategory:
            assert isinstance(cat.value, str)

    def test_unknown_category_exists(self):
        assert FailureCategory.UNKNOWN in FailureCategory.__members__.values()


# ---------------------------------------------------------------------------
# Integration: analyze real NLI edge case failures
# ---------------------------------------------------------------------------


class TestAnalyzeRealEdgeCases:
    def test_analyze_from_simulated_deberta_failures(self):
        """Simulate DeBERTa-style failures on real edge cases."""
        cases = load_edge_cases(EDGE_CASES_PATH)

        # Simulate: DeBERTa predicts "entailed" for everything (bias toward positive)
        from legalkit.eval.nli_eval import NLIJudgment
        judgments = [
            NLIJudgment(case_id=c.id, predicted="entailed", confidence=0.75, latency_ms=100)
            for c in cases
        ]
        result = NLIEvalResult(
            model_name="simulated-biased",
            judgments=judgments,
            test_cases=cases,
            total_latency_ms=len(cases) * 100.0,
        )

        report = analyze_nli_failures(result)

        # All contradicted/neutral cases become false positives
        contradicted_cases = [c for c in cases if c.label in ("contradicted", "neutral")]
        assert len(report.false_positives) == len(contradicted_cases)

    def test_top_category_is_carveout_or_modal_for_legal_text(self):
        """Legal text false positives should cluster around carveouts and modals."""
        cases = load_edge_cases(EDGE_CASES_PATH)
        from legalkit.eval.nli_eval import NLIJudgment

        # Predict entailed for all contradicted cases (the known DeBERTa failure mode)
        judgments = [
            NLIJudgment(
                case_id=c.id,
                predicted="entailed" if c.label in ("contradicted",) else c.label,
                confidence=0.75,
                latency_ms=100,
            )
            for c in cases
        ]
        result = NLIEvalResult(
            model_name="simulated",
            judgments=judgments,
            test_cases=cases,
            total_latency_ms=len(cases) * 100.0,
        )

        report = analyze_nli_failures(result)

        # Top categories should include carveout or modal
        top_cats = {cat for cat, _ in report.top_categories[:3]}
        legal_categories = {
            FailureCategory.CARVEOUT_MISSED,
            FailureCategory.MODAL_CONFUSION,
            FailureCategory.QUALIFIER_OMISSION,
            FailureCategory.QUANTITATIVE_IMPRECISION,
        }
        assert len(top_cats & legal_categories) >= 1
