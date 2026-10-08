"""Tests for evaluation metrics."""

from legalkit.eval.metrics import (
    SpanMatch,
    compute_precision,
    compute_recall,
    span_overlap,
    QueryResult,
    BenchmarkResults,
)


class TestSpanOverlap:
    def test_exact_match(self):
        a = SpanMatch("doc.txt", 0, 100)
        b = SpanMatch("doc.txt", 0, 100)
        assert span_overlap(a, b) == 100

    def test_partial_overlap(self):
        a = SpanMatch("doc.txt", 0, 100)
        b = SpanMatch("doc.txt", 50, 150)
        assert span_overlap(a, b) == 50

    def test_no_overlap(self):
        a = SpanMatch("doc.txt", 0, 50)
        b = SpanMatch("doc.txt", 100, 200)
        assert span_overlap(a, b) == 0

    def test_different_files(self):
        a = SpanMatch("doc1.txt", 0, 100)
        b = SpanMatch("doc2.txt", 0, 100)
        assert span_overlap(a, b) == 0

    def test_contained(self):
        a = SpanMatch("doc.txt", 25, 75)
        b = SpanMatch("doc.txt", 0, 100)
        assert span_overlap(a, b) == 50


class TestPrecision:
    def test_perfect_precision(self):
        retrieved = [SpanMatch("doc.txt", 0, 100)]
        gt = [SpanMatch("doc.txt", 0, 100)]
        assert compute_precision(retrieved, gt) == 1.0

    def test_half_precision(self):
        retrieved = [SpanMatch("doc.txt", 0, 100)]
        gt = [SpanMatch("doc.txt", 50, 150)]
        assert compute_precision(retrieved, gt) == 0.5

    def test_zero_precision(self):
        retrieved = [SpanMatch("doc.txt", 200, 300)]
        gt = [SpanMatch("doc.txt", 0, 100)]
        assert compute_precision(retrieved, gt) == 0.0

    def test_empty_retrieved(self):
        assert compute_precision([], [SpanMatch("doc.txt", 0, 100)]) == 0.0


class TestRecall:
    def test_perfect_recall(self):
        retrieved = [SpanMatch("doc.txt", 0, 100)]
        gt = [SpanMatch("doc.txt", 0, 100)]
        assert compute_recall(retrieved, gt) == 1.0

    def test_partial_recall(self):
        retrieved = [SpanMatch("doc.txt", 0, 50)]
        gt = [SpanMatch("doc.txt", 0, 100)]
        assert compute_recall(retrieved, gt) == 0.5

    def test_zero_recall(self):
        retrieved = [SpanMatch("doc.txt", 200, 300)]
        gt = [SpanMatch("doc.txt", 0, 100)]
        assert compute_recall(retrieved, gt) == 0.0

    def test_multi_span_recall(self):
        retrieved = [
            SpanMatch("doc.txt", 0, 50),
            SpanMatch("doc.txt", 50, 100),
        ]
        gt = [SpanMatch("doc.txt", 0, 100)]
        assert compute_recall(retrieved, gt) == 1.0


class TestBenchmarkResults:
    def test_precision_at_k(self):
        results = BenchmarkResults(results=[
            QueryResult(
                query="test",
                tags=["test_ds"],
                ground_truth=[SpanMatch("doc.txt", 0, 100)],
                retrieved=[
                    SpanMatch("doc.txt", 200, 300),  # miss
                    SpanMatch("doc.txt", 0, 50),      # hit
                    SpanMatch("doc.txt", 50, 100),     # hit
                ],
            ),
        ])

        # P@1: miss (0/100 = 0%)
        assert results.precision_at_k(1) == 0.0

        # P@3: (50+50)/(100+50+50) = 50%
        assert results.precision_at_k(3) == 0.5

    def test_recall_at_k(self):
        results = BenchmarkResults(results=[
            QueryResult(
                query="test",
                tags=["test_ds"],
                ground_truth=[SpanMatch("doc.txt", 0, 100)],
                retrieved=[
                    SpanMatch("doc.txt", 0, 50),      # 50 chars overlap
                    SpanMatch("doc.txt", 50, 100),     # 50 chars overlap
                ],
            ),
        ])

        # R@1: 50/100 = 50%
        assert results.recall_at_k(1) == 0.5

        # R@2: 100/100 = 100%
        assert results.recall_at_k(2) == 1.0

    def test_tag_filtering(self):
        results = BenchmarkResults(results=[
            QueryResult(
                query="q1", tags=["ds_a"],
                ground_truth=[SpanMatch("a.txt", 0, 100)],
                retrieved=[SpanMatch("a.txt", 0, 100)],
            ),
            QueryResult(
                query="q2", tags=["ds_b"],
                ground_truth=[SpanMatch("b.txt", 0, 100)],
                retrieved=[SpanMatch("b.txt", 200, 300)],  # miss
            ),
        ])

        assert results.avg_precision("ds_a") == 1.0
        assert results.avg_precision("ds_b") == 0.0
        assert results.avg_precision() == 0.5  # overall average
