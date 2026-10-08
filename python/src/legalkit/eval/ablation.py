"""Ablation study — measure the contribution of each pipeline component.

Runs the LegalBench-RAG retrieval benchmark across a structured set of
configurations where one component at a time is added on top of a minimal
baseline, revealing how much each contributes to overall recall.

Default ablation suite (no LLM required):

    semantic        — embedding cosine similarity only
    bm25            — BM25 keyword search only
    hybrid          — semantic + BM25 via RRF          (+BM25 contribution)
    hybrid+xref     — hybrid + cross-reference expansion (+xref contribution)

The suite is designed to be additive: each config builds on the previous,
so deltas directly show the marginal value of each component.

Typical usage::

    legalkit ablation --max-per-dataset 50 --output results/ablation.json
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from legalkit.eval.benchmark import BenchmarkQuery, run_benchmark
from legalkit.eval.metrics import BenchmarkResults


REPORT_KS = [1, 2, 4, 8, 16, 32, 64]


@dataclass
class AblationConfig:
    """A single pipeline configuration for ablation."""

    label: str
    """Short human-readable name shown in result tables."""

    retrieval_method: str = "hybrid"
    """One of 'semantic', 'bm25', 'hybrid'."""

    use_cross_references: bool = False
    """Whether to expand cross-referenced sections after retrieval."""

    description: str = ""
    """One-line description of what this config adds over the previous."""

    def __repr__(self) -> str:
        return f"AblationConfig({self.label!r})"


# The default ablation suite — additive, each config adds one component
DEFAULT_CONFIGS: list[AblationConfig] = [
    AblationConfig(
        label="semantic",
        retrieval_method="semantic",
        use_cross_references=False,
        description="Baseline: embedding cosine similarity only",
    ),
    AblationConfig(
        label="bm25",
        retrieval_method="bm25",
        use_cross_references=False,
        description="BM25 keyword search only (no embeddings)",
    ),
    AblationConfig(
        label="hybrid",
        retrieval_method="hybrid",
        use_cross_references=False,
        description="Semantic + BM25 via Reciprocal Rank Fusion",
    ),
    AblationConfig(
        label="hybrid+xref",
        retrieval_method="hybrid",
        use_cross_references=True,
        description="Hybrid + cross-reference section expansion",
    ),
]


@dataclass
class AblationConfigResult:
    """Benchmark result for a single ablation configuration."""

    config: AblationConfig
    results: BenchmarkResults
    elapsed_seconds: float

    def recall_at_k(self, k: int, tag: str | None = None) -> float:
        return self.results.recall_at_k(k, tag)

    def precision_at_k(self, k: int, tag: str | None = None) -> float:
        return self.results.precision_at_k(k, tag)


@dataclass
class AblationResults:
    """Results of a full ablation study."""

    config_results: list[AblationConfigResult]
    queries_evaluated: int
    ks: list[int] = field(default_factory=lambda: list(REPORT_KS))

    @property
    def configs(self) -> list[AblationConfig]:
        return [cr.config for cr in self.config_results]

    def best_config(self, k: int = 64, metric: str = "recall") -> AblationConfigResult:
        fn = "recall_at_k" if metric == "recall" else "precision_at_k"
        return max(self.config_results, key=lambda cr: getattr(cr, fn)(k))

    def component_deltas(self, k: int = 64, tag: str | None = None) -> list[dict]:
        """Compute marginal recall gain of each config vs the previous one.

        Returns a list of dicts with keys: label, recall, delta_pp, description.
        """
        deltas = []
        prev_recall = 0.0
        for cr in self.config_results:
            recall = cr.recall_at_k(k, tag)
            delta = recall - prev_recall
            deltas.append({
                "label": cr.config.label,
                "recall": recall,
                "delta_pp": delta,
                "description": cr.config.description,
            })
            prev_recall = recall
        return deltas

    def comparison_table(
        self,
        ks: list[int] | None = None,
        tag: str | None = None,
        metric: str = "recall",
    ) -> str:
        """Render a plain-text R@k / P@k comparison table."""
        ks = ks or self.ks
        fn = "recall_at_k" if metric == "recall" else "precision_at_k"
        m = "R" if metric == "recall" else "P"

        headers = ["Config"] + [f"{m}@{k}" for k in ks] + ["Time"]
        widths = [18] + [8] * len(ks) + [8]

        def row(cells: list[str]) -> str:
            return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

        lines = [row(headers), row(["─" * w for w in widths])]

        best_k = ks[-1]
        best_score = max(getattr(cr, fn)(best_k, tag) for cr in self.config_results)

        for cr in self.config_results:
            vals = [f"{getattr(cr, fn)(k, tag)*100:.2f}%" for k in ks]
            marker = " ◀" if getattr(cr, fn)(best_k, tag) == best_score else ""
            lines.append(row(
                [cr.config.label] + vals + [f"{cr.elapsed_seconds:.0f}s{marker}"]
            ))

        return "\n".join(lines)

    def delta_table(self, k: int = 64, tag: str | None = None) -> str:
        """Render a marginal-contribution table at a given k."""
        deltas = self.component_deltas(k=k, tag=tag)
        lines = [
            f"Marginal recall contribution at R@{k}:",
            "",
            f"  {'Config':<18}  {'R@'+str(k):>7}  {'Δ (pp)':>9}  Description",
            "  " + "─" * 70,
        ]
        for d in deltas:
            sign = "+" if d["delta_pp"] >= 0 else ""
            lines.append(
                f"  {d['label']:<18}  {d['recall']*100:>6.2f}%  "
                f"  {sign}{d['delta_pp']*100:>5.2f}pp  {d['description']}"
            )
        return "\n".join(lines)


def run_ablation(
    queries: list[BenchmarkQuery],
    configs: list[AblationConfig] | None = None,
    max_chunks_per_query: int = 64,
    min_chars: int = 100,
    max_chars: int = 1500,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> AblationResults:
    """Run the retrieval benchmark for each ablation configuration.

    Args:
        queries: LegalBench-RAG queries to evaluate on.
        configs: Ablation configurations. Defaults to DEFAULT_CONFIGS.
        max_chunks_per_query: Retrieval depth for each run.
        min_chars: Minimum chunk size (applied to all configs).
        max_chars: Maximum chunk size (applied to all configs).
        progress_callback: Optional ``(stage, current, total) -> None``.

    Returns:
        :class:`AblationResults` with per-config benchmark results.
    """
    configs = configs or DEFAULT_CONFIGS
    config_results: list[AblationConfigResult] = []

    for ci, config in enumerate(configs):
        if progress_callback:
            progress_callback(f"config: {config.label}", ci, len(configs))

        t0 = time.time()

        results = run_benchmark(
            queries=queries,
            max_chunks_per_query=max_chunks_per_query,
            min_chars=min_chars,
            max_chars=max_chars,
            retrieval_method=config.retrieval_method,
            use_cross_references=config.use_cross_references,
            progress_callback=(
                (lambda s, c, t, lbl=config.label: progress_callback(f"[{lbl}] {s}", c, t))
                if progress_callback else None
            ),
        )

        config_results.append(AblationConfigResult(
            config=config,
            results=results,
            elapsed_seconds=time.time() - t0,
        ))

    if progress_callback:
        progress_callback("done", len(configs), len(configs))

    return AblationResults(
        config_results=config_results,
        queries_evaluated=len(queries),
    )
