"""Confidence calibration measurement.

A well-calibrated model reports confidence=0.85 when it is actually
correct 85% of the time. Miscalibration means the confidence scores are
misleading — either overconfident (model says 0.85 but only 60% correct)
or underconfident (model says 0.5 but actually 80% correct).

This module computes:

- **Expected Calibration Error (ECE)** — weighted mean of |accuracy −
  confidence| across equal-width buckets. The primary calibration metric.
- **Maximum Calibration Error (MCE)** — worst single bucket gap.
- **Brier Score** — proper scoring rule: mean squared error of probability
  predictions. Lower is better; 0 is perfect; 1 is maximally wrong.
- **Reliability diagram data** — (confidence, accuracy) pairs per bucket
  for visualization.

Usage::

    from legalkit.eval.calibration import compute_calibration

    result = compute_calibration(
        scores=[0.9, 0.6, 0.3, 0.8, 0.7],
        labels=[True, True, False, True, False],
    )
    print(f"ECE: {result.ece:.3f}")
    print(result.reliability_diagram_text())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class CalibrationBucket:
    """One confidence bucket in a reliability diagram."""

    confidence_low: float
    """Lower bound of this bucket (inclusive)."""

    confidence_high: float
    """Upper bound of this bucket (exclusive, except the last)."""

    n: int
    """Number of predictions in this bucket."""

    avg_confidence: float
    """Mean predicted confidence of samples in this bucket."""

    accuracy: float
    """Actual fraction of correct predictions in this bucket."""

    @property
    def gap(self) -> float:
        """|accuracy − avg_confidence| — calibration gap for this bucket."""
        return abs(self.accuracy - self.avg_confidence)

    @property
    def midpoint(self) -> float:
        return (self.confidence_low + self.confidence_high) / 2

    def __repr__(self) -> str:
        return (
            f"Bucket[{self.confidence_low:.1f}-{self.confidence_high:.1f}] "
            f"n={self.n} conf={self.avg_confidence:.3f} "
            f"acc={self.accuracy:.3f} gap={self.gap:.3f}"
        )


@dataclass
class CalibrationResult:
    """Calibration metrics for a set of probability predictions."""

    ece: float
    """Expected Calibration Error — weighted mean |accuracy − confidence|.
    0 = perfect calibration. Typical good range: < 0.05."""

    mce: float
    """Maximum Calibration Error — worst single bucket gap."""

    brier_score: float
    """Brier score (mean squared error of probabilities).
    0 = perfect. 0.25 = uninformative (always predict 0.5)."""

    buckets: list[CalibrationBucket]
    """Per-bucket statistics for reliability diagram."""

    n_total: int
    """Total number of predictions."""

    n_buckets_used: int
    """Number of non-empty buckets."""

    @property
    def is_overconfident(self) -> bool:
        """True if the model tends to predict higher confidence than warranted."""
        non_empty = [b for b in self.buckets if b.n > 0]
        if not non_empty:
            return False
        avg_gap = sum(b.avg_confidence - b.accuracy for b in non_empty) / len(non_empty)
        return avg_gap > 0.05

    @property
    def is_underconfident(self) -> bool:
        """True if the model tends to predict lower confidence than warranted."""
        non_empty = [b for b in self.buckets if b.n > 0]
        if not non_empty:
            return False
        avg_gap = sum(b.accuracy - b.avg_confidence for b in non_empty) / len(non_empty)
        return avg_gap > 0.05

    def reliability_diagram_text(self, width: int = 50) -> str:
        """Render a text reliability diagram.

        Shows confidence buckets (x-axis) vs actual accuracy (y-axis),
        with the ideal diagonal marked as reference.

        Example output::

            Reliability Diagram (n=33)
            Confidence → Accuracy  [gap]
            0.0-0.1  ░░░░░░░░░░░░░░░░  n=0  (empty)
            0.1-0.2  ░░░░░░░░░░░░░░░░  n=0  (empty)
            0.5-0.6  █████░░░░░░░░░░░  n=5  acc=0.40 conf=0.52 ▲
            ...
        """
        lines = [
            f"Reliability Diagram (n={self.n_total}, ECE={self.ece:.3f})",
            f"{'Bucket':<12} {'Accuracy':>10} {'Confidence':>12} {'N':>5}  {'Gap':>6}",
            "─" * 60,
        ]
        for b in self.buckets:
            if b.n == 0:
                lines.append(
                    f"{b.confidence_low:.1f}–{b.confidence_high:.1f}"
                    f"{'':>38}  n=0"
                )
                continue
            # Bar showing accuracy
            bar_len = int(b.accuracy * width // 2)
            ideal_len = int(b.midpoint * width // 2)
            bar = "█" * bar_len + "░" * (width // 2 - bar_len)
            # Direction indicator
            direction = "▲" if b.accuracy > b.avg_confidence + 0.05 else (
                "▼" if b.accuracy < b.avg_confidence - 0.05 else "≈"
            )
            lines.append(
                f"{b.confidence_low:.1f}–{b.confidence_high:.1f}"
                f"  {b.accuracy:>6.1%}  {b.avg_confidence:>10.3f}"
                f"  {b.n:>5}  {b.gap:>5.3f}  {direction}"
            )
        lines.append("─" * 60)
        lines.append(
            f"ECE={self.ece:.4f}  MCE={self.mce:.4f}  Brier={self.brier_score:.4f}"
        )
        return "\n".join(lines)

    def summary(self) -> str:
        direction = (
            "overconfident" if self.is_overconfident
            else "underconfident" if self.is_underconfident
            else "well-calibrated"
        )
        return (
            f"ECE={self.ece:.3f}  MCE={self.mce:.3f}  Brier={self.brier_score:.3f}"
            f"  [{direction}]  n={self.n_total}"
        )


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_calibration(
    scores: list[float],
    labels: list[bool],
    n_buckets: int = 10,
) -> CalibrationResult:
    """Compute calibration metrics from predicted scores and ground-truth labels.

    Args:
        scores: Predicted confidence scores in [0, 1]. Higher = more confident
            the prediction is correct.
        labels: Ground-truth correctness (True = prediction was correct).
        n_buckets: Number of equal-width buckets for the reliability diagram.
            Default 10 gives 0.0–0.1, 0.1–0.2, ..., 0.9–1.0.

    Returns:
        :class:`CalibrationResult` with ECE, MCE, Brier score, and bucket data.

    Raises:
        ValueError: If ``scores`` and ``labels`` have different lengths or are empty.
    """
    if len(scores) != len(labels):
        raise ValueError(
            f"scores ({len(scores)}) and labels ({len(labels)}) must have the same length"
        )
    if not scores:
        raise ValueError("scores and labels must not be empty")

    n = len(scores)
    bucket_size = 1.0 / n_buckets

    # Build buckets
    buckets: list[CalibrationBucket] = []
    for i in range(n_buckets):
        low = i * bucket_size
        high = (i + 1) * bucket_size

        indices = [
            j for j, s in enumerate(scores)
            if low <= s < high or (i == n_buckets - 1 and s == 1.0)
        ]

        if not indices:
            buckets.append(CalibrationBucket(
                confidence_low=low,
                confidence_high=high,
                n=0,
                avg_confidence=low + bucket_size / 2,
                accuracy=0.0,
            ))
        else:
            avg_conf = sum(scores[j] for j in indices) / len(indices)
            acc = sum(1 for j in indices if labels[j]) / len(indices)
            buckets.append(CalibrationBucket(
                confidence_low=low,
                confidence_high=high,
                n=len(indices),
                avg_confidence=avg_conf,
                accuracy=acc,
            ))

    # ECE — weighted by bucket size
    ece = sum(
        (b.n / n) * b.gap
        for b in buckets if b.n > 0
    )

    # MCE — worst non-empty bucket
    non_empty_gaps = [b.gap for b in buckets if b.n > 0]
    mce = max(non_empty_gaps) if non_empty_gaps else 0.0

    # Brier score — mean squared error
    brier = sum((s - float(l)) ** 2 for s, l in zip(scores, labels)) / n

    n_used = sum(1 for b in buckets if b.n > 0)

    return CalibrationResult(
        ece=round(ece, 6),
        mce=round(mce, 6),
        brier_score=round(brier, 6),
        buckets=buckets,
        n_total=n,
        n_buckets_used=n_used,
    )


# ---------------------------------------------------------------------------
# Convenience: calibrate from NLI eval results
# ---------------------------------------------------------------------------


def calibrate_nli_result(
    eval_result,  # NLIEvalResult
    score_fn=None,
) -> CalibrationResult:
    """Compute calibration from an NLI evaluation result.

    By default uses the entailment confidence score and binary correctness
    (positive = entailed ∪ partially_entailed).

    Args:
        eval_result: :class:`~legalkit.eval.nli_eval.NLIEvalResult`.
        score_fn: Optional callable ``(judgment) -> float`` to extract a
            confidence score. Defaults to ``judgment.confidence``.

    Returns:
        :class:`CalibrationResult`.
    """
    from legalkit.eval.nli_eval import _BINARY_POSITIVE

    if score_fn is None:
        score_fn = lambda j: j.confidence

    scores: list[float] = []
    labels: list[bool] = []

    for judgment, case in zip(eval_result.judgments, eval_result.test_cases):
        score = float(score_fn(judgment))
        # Binary correct: model binary prediction matches ground truth binary
        predicted_positive = judgment.predicted in _BINARY_POSITIVE
        actual_positive = case.label in _BINARY_POSITIVE
        correct = predicted_positive == actual_positive
        scores.append(score)
        labels.append(correct)

    return compute_calibration(scores, labels)


def calibrate_entailment_log(
    log_dir=None,
    reviewed_only: bool = False,
) -> CalibrationResult | None:
    """Compute calibration from the entailment log.

    Uses the logged LLM confidence scores. If ``reviewed_only=True``, only
    uses samples with human-reviewed labels (``sample.reviewed == True``).

    Note: Without human review, labels come from the LLM judge itself, so
    calibration measures self-consistency rather than true calibration.
    Meaningful calibration requires at least some human-reviewed samples.

    Args:
        log_dir: Override log directory. Defaults to ``data/entailment_logs``.
        reviewed_only: If True, only use human-reviewed samples.

    Returns:
        :class:`CalibrationResult`, or ``None`` if no samples found.
    """
    from legalkit.eval.entailment_log import EntailmentLogger

    samples = EntailmentLogger.load_all(log_dir)
    if reviewed_only:
        samples = [s for s in samples if s.reviewed]

    if not samples:
        return None

    from legalkit.eval.nli_eval import _BINARY_POSITIVE

    scores: list[float] = []
    labels: list[bool] = []

    for sample in samples:
        scores.append(min(1.0, max(0.0, sample.confidence)))
        # Use corrected label if available, else logged label
        cls = sample.corrected_class or sample.entailment_class
        is_positive = cls in _BINARY_POSITIVE
        # For self-consistency: "correct" = confidence aligns with class
        # For reviewed samples: "correct" = matches human label
        labels.append(is_positive)

    return compute_calibration(scores, labels)
