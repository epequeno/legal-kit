"""Chunk size sweep experiment.

Runs the LegalBench-RAG retrieval benchmark across multiple chunk size
configurations and reports P@k / R@k for each, making it easy to identify
the optimal (min_chars, max_chars) pair for the legal corpus.

Typical usage::

    legalkit chunk-sweep --chunk-sizes 500 800 1500 3000 --top-k 64

The sweep re-uses query embeddings across configurations (queries don't
change, only the document chunking does), saving ~N_configs-1 embedding
calls per query.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from legalkit.eval.benchmark import BenchmarkQuery, run_benchmark
from legalkit.eval.metrics import BenchmarkResults


# Default configurations to sweep if none specified
DEFAULT_CONFIGS: list[tuple[int, int]] = [
    (50,  500),
    (100, 800),
    (100, 1500),
    (100, 3000),   # current default
]

# k values reported in result tables
REPORT_KS = [1, 2, 4, 8, 16, 32, 64]


@dataclass
class ChunkConfig:
    """A single chunk size configuration to benchmark."""

    min_chars: int
    max_chars: int

    @property
    def label(self) -> str:
        return f"{self.min_chars}–{self.max_chars}"

    def __repr__(self) -> str:
        return f"ChunkConfig(min={self.min_chars}, max={self.max_chars})"


@dataclass
class ConfigResult:
    """Benchmark result for a single chunk configuration."""

    config: ChunkConfig
    results: BenchmarkResults
    elapsed_seconds: float
    avg_chunks_per_doc: float = 0.0

    def precision_at_k(self, k: int, tag: str | None = None) -> float:
        return self.results.precision_at_k(k, tag)

    def recall_at_k(self, k: int, tag: str | None = None) -> float:
        return self.results.recall_at_k(k, tag)


@dataclass
class SweepResults:
    """Results of a full chunk size sweep."""

    config_results: list[ConfigResult]
    queries_evaluated: int
    ks: list[int] = field(default_factory=lambda: list(REPORT_KS))

    @property
    def configs(self) -> list[ChunkConfig]:
        return [cr.config for cr in self.config_results]

    def best_config(self, k: int = 64, metric: str = "recall") -> ConfigResult:
        """Return the config with the best recall (or precision) at k."""
        fn = "recall_at_k" if metric == "recall" else "precision_at_k"
        return max(self.config_results, key=lambda cr: getattr(cr, fn)(k))

    def comparison_table(
        self,
        ks: list[int] | None = None,
        tag: str | None = None,
        metric: str = "recall",
    ) -> str:
        """Render a plain-text comparison table of R@k (or P@k) per config.

        Args:
            ks: k values to include. Defaults to self.ks.
            tag: Dataset tag to filter on (None = overall).
            metric: "recall" or "precision".

        Returns:
            Formatted table string.
        """
        ks = ks or self.ks
        fn = "recall_at_k" if metric == "recall" else "precision_at_k"
        label = "R@k" if metric == "recall" else "P@k"

        header_cols = ["Config (min–max chars)"] + [f"{label[:-1]}@{k}" for k in ks] + ["Time (s)"]
        col_widths = [max(len(h), 22) for h in header_cols]
        col_widths[0] = max(col_widths[0], 22)

        def fmt_row(cells: list[str]) -> str:
            return "  ".join(c.ljust(w) for c, w in zip(cells, col_widths))

        lines = [
            fmt_row(header_cols),
            fmt_row(["─" * w for w in col_widths]),
        ]

        best_k = ks[-1]
        best_score = max(getattr(cr, fn)(best_k, tag) for cr in self.config_results)

        for cr in self.config_results:
            values = [f"{getattr(cr, fn)(k, tag) * 100:.2f}%" for k in ks]
            marker = " ◀ best" if getattr(cr, fn)(best_k, tag) == best_score else ""
            lines.append(fmt_row(
                [cr.config.label] + values + [f"{cr.elapsed_seconds:.1f}s{marker}"]
            ))

        return "\n".join(lines)

    def delta_table(
        self,
        baseline_max_chars: int = 3000,
        k: int = 64,
        tag: str | None = None,
    ) -> str:
        """Show recall delta vs the baseline config at a given k.

        Args:
            baseline_max_chars: max_chars of the config to use as baseline.
            k: Retrieval depth to compare at.
            tag: Dataset tag (None = overall).

        Returns:
            Formatted delta table string.
        """
        baseline = next(
            (cr for cr in self.config_results if cr.config.max_chars == baseline_max_chars),
            self.config_results[-1],
        )
        baseline_recall = baseline.recall_at_k(k, tag)

        lines = [
            f"Recall delta vs baseline ({baseline.config.label}) at R@{k}:",
            "",
        ]
        for cr in self.config_results:
            delta = cr.recall_at_k(k, tag) - baseline_recall
            sign = "+" if delta >= 0 else ""
            marker = " (baseline)" if cr is baseline else ""
            lines.append(
                f"  {cr.config.label:<20}  {cr.recall_at_k(k, tag)*100:6.2f}%  "
                f"({sign}{delta*100:.2f}pp){marker}"
            )
        return "\n".join(lines)


def run_chunk_sweep(
    queries: list[BenchmarkQuery],
    configs: list[ChunkConfig] | None = None,
    max_chunks_per_query: int = 64,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> SweepResults:
    """Run the retrieval benchmark for each chunk size configuration.

    Args:
        queries: LegalBench-RAG queries to evaluate on.
        configs: Configurations to sweep. Defaults to DEFAULT_CONFIGS.
        max_chunks_per_query: Retrieval depth for each run.
        progress_callback: Optional ``(stage, current, total) -> None``.

    Returns:
        :class:`SweepResults` with per-config benchmark results.
    """
    if configs is None:
        configs = [ChunkConfig(mn, mx) for mn, mx in DEFAULT_CONFIGS]

    config_results: list[ConfigResult] = []

    for ci, config in enumerate(configs):
        if progress_callback:
            progress_callback(f"config {config.label}", ci, len(configs))

        t0 = time.time()

        results = run_benchmark(
            queries=queries,
            max_chunks_per_query=max_chunks_per_query,
            min_chars=config.min_chars,
            max_chars=config.max_chars,
            progress_callback=(
                (lambda s, c, t, cfg=config.label: progress_callback(f"[{cfg}] {s}", c, t))
                if progress_callback else None
            ),
        )

        elapsed = time.time() - t0

        config_results.append(ConfigResult(
            config=config,
            results=results,
            elapsed_seconds=elapsed,
        ))

    if progress_callback:
        progress_callback("done", len(configs), len(configs))

    return SweepResults(
        config_results=config_results,
        queries_evaluated=len(queries),
    )
