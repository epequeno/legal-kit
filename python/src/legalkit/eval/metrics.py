"""Evaluation metrics for LegalBench-RAG.

Character-span-level precision and recall, matching the official
LegalBench-RAG evaluation methodology.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpanMatch:
    """A ground truth or retrieved span within a document."""

    file_path: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def span_overlap(a: SpanMatch, b: SpanMatch) -> int:
    """Compute character overlap between two spans in the same file.

    Returns 0 if different files or no overlap.
    """
    if a.file_path != b.file_path:
        return 0
    common_min = max(a.start, b.start)
    common_max = min(a.end, b.end)
    return max(0, common_max - common_min)


def compute_precision(
    retrieved: list[SpanMatch],
    ground_truth: list[SpanMatch],
) -> float:
    """Character-level precision: fraction of retrieved chars that are relevant.

    precision = |retrieved ∩ ground_truth| / |retrieved|
    """
    total_retrieved_len = sum(s.length for s in retrieved)
    if total_retrieved_len == 0:
        return 0.0

    relevant_retrieved_len = 0
    for ret in retrieved:
        for gt in ground_truth:
            relevant_retrieved_len += span_overlap(ret, gt)

    return relevant_retrieved_len / total_retrieved_len


def compute_recall(
    retrieved: list[SpanMatch],
    ground_truth: list[SpanMatch],
) -> float:
    """Character-level recall: fraction of relevant chars that were retrieved.

    recall = |retrieved ∩ ground_truth| / |ground_truth|
    """
    total_relevant_len = sum(s.length for s in ground_truth)
    if total_relevant_len == 0:
        return 0.0

    relevant_retrieved_len = 0
    for gt in ground_truth:
        for ret in retrieved:
            relevant_retrieved_len += span_overlap(ret, gt)

    return relevant_retrieved_len / total_relevant_len


@dataclass
class QueryResult:
    """Result of a single benchmark query."""

    query: str
    tags: list[str] = field(default_factory=list)
    ground_truth: list[SpanMatch] = field(default_factory=list)
    retrieved: list[SpanMatch] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return compute_precision(self.retrieved, self.ground_truth)

    @property
    def recall(self) -> float:
        return compute_recall(self.retrieved, self.ground_truth)


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""

    results: list[QueryResult] = field(default_factory=list)
    run_name: str = ""
    config: dict = field(default_factory=dict)

    def avg_precision(self, tag: str | None = None) -> float:
        filtered = self._filter(tag)
        if not filtered:
            return 0.0
        return sum(r.precision for r in filtered) / len(filtered)

    def avg_recall(self, tag: str | None = None) -> float:
        filtered = self._filter(tag)
        if not filtered:
            return 0.0
        return sum(r.recall for r in filtered) / len(filtered)

    def precision_at_k(self, k: int, tag: str | None = None) -> float:
        """Average precision when only top-k retrieved spans are considered."""
        filtered = self._filter(tag)
        if not filtered:
            return 0.0
        precisions = []
        for r in filtered:
            top_k = r.retrieved[:k]
            precisions.append(compute_precision(top_k, r.ground_truth))
        return sum(precisions) / len(precisions)

    def recall_at_k(self, k: int, tag: str | None = None) -> float:
        """Average recall when only top-k retrieved spans are considered."""
        filtered = self._filter(tag)
        if not filtered:
            return 0.0
        recalls = []
        for r in filtered:
            top_k = r.retrieved[:k]
            recalls.append(compute_recall(top_k, r.ground_truth))
        return sum(recalls) / len(recalls)

    def _filter(self, tag: str | None) -> list[QueryResult]:
        if tag is None:
            return self.results
        return [r for r in self.results if tag in r.tags]

    @property
    def tags(self) -> list[str]:
        all_tags: set[str] = set()
        for r in self.results:
            all_tags.update(r.tags)
        return sorted(all_tags)

    def summary_table(self) -> str:
        """Generate a summary table of results."""
        lines = []
        ks = [1, 2, 4, 8, 16, 32, 64]

        # Header
        header = f"{'Dataset':<15}" + "".join(f"{'P@'+str(k):>8}" for k in ks)
        header += "".join(f"{'R@'+str(k):>8}" for k in ks)
        lines.append(header)
        lines.append("-" * len(header))

        # Per-tag rows
        for tag in self.tags:
            row = f"{tag:<15}"
            row += "".join(f"{self.precision_at_k(k, tag)*100:>7.2f}%" for k in ks)
            row += "".join(f"{self.recall_at_k(k, tag)*100:>7.2f}%" for k in ks)
            lines.append(row)

        # Overall row
        row = f"{'ALL':<15}"
        row += "".join(f"{self.precision_at_k(k)*100:>7.2f}%" for k in ks)
        row += "".join(f"{self.recall_at_k(k)*100:>7.2f}%" for k in ks)
        lines.append("-" * len(header))
        lines.append(row)

        return "\n".join(lines)
