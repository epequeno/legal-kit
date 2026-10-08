"""Tests for confidence calibration."""

from __future__ import annotations

import pytest

from legalkit.eval.calibration import (
    CalibrationBucket,
    CalibrationResult,
    compute_calibration,
    calibrate_nli_result,
)


# ---------------------------------------------------------------------------
# compute_calibration — core mechanics
# ---------------------------------------------------------------------------


class TestComputeCalibration:
    def test_perfect_calibration(self):
        """Model that is always right with high confidence."""
        scores = [0.95] * 10
        labels = [True] * 10
        result = compute_calibration(scores, labels)
        # All in bucket 0.9-1.0, accuracy=1.0, avg_conf≈0.95, gap=0.05
        assert result.ece < 0.10
        assert result.brier_score < 0.05

    def test_random_baseline(self):
        """Model that always predicts 0.5 — uninformative."""
        scores = [0.5] * 20
        labels = [True] * 10 + [False] * 10
        result = compute_calibration(scores, labels)
        # Brier score for 50% correct with 0.5 confidence = 0.25
        assert abs(result.brier_score - 0.25) < 0.01

    def test_perfect_wrong(self):
        """Model that is always wrong with high confidence — worst case."""
        scores = [0.95] * 10
        labels = [False] * 10
        result = compute_calibration(scores, labels)
        # acc=0.0, conf≈0.95, gap≈0.95
        assert result.ece > 0.5
        assert result.brier_score > 0.5

    def test_ece_between_0_and_1(self):
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        labels = [False, True, True, True, True]
        result = compute_calibration(scores, labels)
        assert 0.0 <= result.ece <= 1.0

    def test_mce_gte_ece(self):
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        labels = [False, True, True, True, True]
        result = compute_calibration(scores, labels)
        assert result.mce >= result.ece

    def test_brier_between_0_and_1(self):
        scores = [0.1, 0.9, 0.5, 0.8, 0.2]
        labels = [False, True, True, False, False]
        result = compute_calibration(scores, labels)
        assert 0.0 <= result.brier_score <= 1.0

    def test_correct_bucket_count(self):
        scores = [0.05, 0.15, 0.55, 0.95]
        labels = [True, False, True, True]
        result = compute_calibration(scores, labels, n_buckets=10)
        assert len(result.buckets) == 10

    def test_non_empty_buckets(self):
        scores = [0.05, 0.15, 0.55, 0.95]
        labels = [True, False, True, True]
        result = compute_calibration(scores, labels, n_buckets=10)
        non_empty = [b for b in result.buckets if b.n > 0]
        assert len(non_empty) == 4

    def test_bucket_ns_sum_to_total(self):
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        labels = [True] * 5
        result = compute_calibration(scores, labels)
        assert sum(b.n for b in result.buckets) == 5

    def test_n_total_correct(self):
        scores = [0.5] * 7
        labels = [True] * 7
        result = compute_calibration(scores, labels)
        assert result.n_total == 7

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_calibration([], [])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_calibration([0.5, 0.6], [True])

    def test_score_of_1_goes_in_last_bucket(self):
        """Score exactly 1.0 should be included in the last bucket."""
        scores = [1.0]
        labels = [True]
        result = compute_calibration(scores, labels)
        last_bucket = result.buckets[-1]
        assert last_bucket.n == 1

    def test_custom_n_buckets(self):
        scores = [0.1, 0.5, 0.9]
        labels = [True, True, True]
        result = compute_calibration(scores, labels, n_buckets=5)
        assert len(result.buckets) == 5

    def test_single_sample(self):
        result = compute_calibration([0.8], [True])
        assert result.n_total == 1
        assert result.brier_score == pytest.approx((0.8 - 1.0) ** 2)

    def test_overconfident_detection(self):
        """Model always scores 0.9 but only right 50% of the time."""
        scores = [0.9] * 10
        labels = [True, False] * 5
        result = compute_calibration(scores, labels)
        assert result.is_overconfident

    def test_underconfident_detection(self):
        """Model scores 0.1 but is right 90% of the time."""
        scores = [0.1] * 10
        labels = [True] * 9 + [False]
        result = compute_calibration(scores, labels)
        assert result.is_underconfident

    def test_well_calibrated_not_overconfident(self):
        """Model with acc slightly above confidence is not overconfident."""
        # 5 in 0.7-0.8 bucket, 4/5 correct → acc=0.8, conf=0.75 → gap=0.05
        # acc > conf means slightly underconfident, NOT overconfident
        scores = [0.75] * 5
        labels = [True, True, True, True, False]
        result = compute_calibration(scores, labels)
        assert not result.is_overconfident


# ---------------------------------------------------------------------------
# CalibrationBucket
# ---------------------------------------------------------------------------


class TestCalibrationBucket:
    def test_gap_computation(self):
        b = CalibrationBucket(0.8, 0.9, n=10, avg_confidence=0.85, accuracy=0.70)
        assert b.gap == pytest.approx(0.15)

    def test_gap_is_absolute(self):
        b = CalibrationBucket(0.1, 0.2, n=5, avg_confidence=0.15, accuracy=0.40)
        assert b.gap == pytest.approx(0.25)

    def test_midpoint(self):
        b = CalibrationBucket(0.4, 0.5, n=3, avg_confidence=0.45, accuracy=0.5)
        assert b.midpoint == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# CalibrationResult
# ---------------------------------------------------------------------------


class TestCalibrationResult:
    def _make_result(self, ece=0.05, mce=0.10, brier=0.15, n=20) -> CalibrationResult:
        buckets = [
            CalibrationBucket(0.0, 0.5, n=10, avg_confidence=0.3, accuracy=0.4),
            CalibrationBucket(0.5, 1.0, n=10, avg_confidence=0.75, accuracy=0.9),
        ]
        return CalibrationResult(
            ece=ece, mce=mce, brier_score=brier,
            buckets=buckets, n_total=n, n_buckets_used=2,
        )

    def test_summary_contains_ece(self):
        result = self._make_result()
        assert "ECE" in result.summary()
        assert "0.050" in result.summary()

    def test_reliability_diagram_text_runs(self):
        result = compute_calibration(
            [0.1, 0.5, 0.9, 0.8, 0.3],
            [False, True, True, True, False],
        )
        diagram = result.reliability_diagram_text()
        assert isinstance(diagram, str)
        assert "ECE=" in diagram

    def test_reliability_diagram_shows_all_buckets(self):
        result = compute_calibration(
            [0.05, 0.15, 0.55, 0.95],
            [False, True, True, True],
        )
        diagram = result.reliability_diagram_text()
        # Should mention all 10 bucket boundaries
        assert "0.0" in diagram
        assert "0.9" in diagram


# ---------------------------------------------------------------------------
# calibrate_nli_result
# ---------------------------------------------------------------------------


class TestCalibrateNLIResult:
    def _make_nli_result(self, predictions, ground_truths, confidences):
        from legalkit.eval.nli_eval import NLIEvalResult, NLIJudgment, NLITestCase

        cases = [
            NLITestCase(id=f"c{i}", category="test", passage="p", claim="cl", label=gt)
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
            total_latency_ms=len(judgments) * 50.0,
        )

    def test_returns_calibration_result(self):
        result = self._make_nli_result(
            predictions=["entailed", "contradicted", "neutral"],
            ground_truths=["entailed", "contradicted", "neutral"],
            confidences=[0.9, 0.8, 0.7],
        )
        cal = calibrate_nli_result(result)
        assert isinstance(cal, CalibrationResult)

    def test_perfect_confident_model(self):
        """Model always right with high confidence."""
        result = self._make_nli_result(
            predictions=["entailed"] * 5,
            ground_truths=["entailed"] * 5,
            confidences=[0.9] * 5,
        )
        cal = calibrate_nli_result(result)
        assert cal.n_total == 5
        # All correct with 0.9 confidence → low ECE
        assert cal.ece < 0.20  # gap = |1.0 - 0.9| = 0.1

    def test_custom_score_fn(self):
        """Can use a custom function to extract score."""
        result = self._make_nli_result(
            predictions=["entailed"],
            ground_truths=["entailed"],
            confidences=[0.5],
        )
        # Use constant 0.8 score regardless of judgment
        cal = calibrate_nli_result(result, score_fn=lambda j: 0.8)
        bucket = next(b for b in cal.buckets if b.n > 0)
        assert abs(bucket.avg_confidence - 0.8) < 0.01

    def test_nli_eval_result_calibration_method(self):
        """NLIEvalResult.calibration() convenience method."""
        result = self._make_nli_result(
            predictions=["entailed", "contradicted"],
            ground_truths=["entailed", "neutral"],
            confidences=[0.85, 0.70],
        )
        cal = result.calibration()
        assert isinstance(cal, CalibrationResult)
        assert cal.n_total == 2

    def test_binary_correctness_used(self):
        """Partially entailed predicted for entailed should count as correct (both positive)."""
        result = self._make_nli_result(
            predictions=["partially_entailed"],
            ground_truths=["entailed"],
            confidences=[0.6],
        )
        cal = calibrate_nli_result(result)
        # 1 correct (both binary positive) → accuracy in bucket = 1.0
        bucket = next(b for b in cal.buckets if b.n > 0)
        assert bucket.accuracy == pytest.approx(1.0)

    def test_contradicted_vs_neutral_both_negative(self):
        """Contradicted predicted for neutral = correct (both binary negative)."""
        result = self._make_nli_result(
            predictions=["contradicted"],
            ground_truths=["neutral"],
            confidences=[0.8],
        )
        cal = calibrate_nli_result(result)
        bucket = next(b for b in cal.buckets if b.n > 0)
        assert bucket.accuracy == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integration: compute calibration on real NLI edge cases (structure only)
# ---------------------------------------------------------------------------


class TestCalibrationOnEdgeCases:
    def test_compute_calibration_from_edge_case_scores(self):
        """Smoke test: compute calibration from realistic legal NLI scores."""
        from legalkit.eval.nli_eval import load_edge_cases
        from pathlib import Path

        cases = load_edge_cases(
            Path(__file__).parent.parent / "data/eval/legal_nli_edge_cases.json"
        )

        # Simulate model outputs: entailment-biased scores (as DeBERTa tends to be)
        scores = [0.7 if c.label in ("entailed", "partially_entailed") else 0.4
                  for c in cases]
        labels = [c.label in ("entailed", "partially_entailed") for c in cases]

        result = compute_calibration(scores, labels)
        assert isinstance(result, CalibrationResult)
        assert result.n_total == len(cases)
        assert 0.0 <= result.ece <= 1.0

    def test_overconfident_pattern_detected(self):
        """When model always says 0.9 but is wrong half the time, detect overconfidence."""
        n = 20
        scores = [0.9] * n
        labels = [True] * (n // 2) + [False] * (n // 2)
        result = compute_calibration(scores, labels)
        assert result.is_overconfident
        assert result.ece > 0.30
