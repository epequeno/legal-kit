"""Tests for chunk size sweep experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from legalkit.eval.chunk_sweep import (
    DEFAULT_CONFIGS,
    REPORT_KS,
    ChunkConfig,
    ConfigResult,
    SweepResults,
    run_chunk_sweep,
)
from legalkit.eval.metrics import BenchmarkResults, QueryResult, SpanMatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_results(recall_score: float = 0.8, n_queries: int = 5) -> BenchmarkResults:
    """Create a fake BenchmarkResults with a fixed recall score."""
    results = []
    for i in range(n_queries):
        gt = SpanMatch(file_path="doc.txt", start=0, end=100)
        # Retrieve a span that perfectly overlaps ground truth
        if recall_score >= 1.0:
            retrieved = [SpanMatch(file_path="doc.txt", start=0, end=100)]
        elif recall_score <= 0.0:
            retrieved = [SpanMatch(file_path="doc.txt", start=500, end=600)]
        else:
            # partial overlap
            overlap_end = int(100 * recall_score)
            retrieved = [SpanMatch(file_path="doc.txt", start=0, end=overlap_end)]
        results.append(QueryResult(
            query=f"query {i}",
            tags=["test"],
            ground_truth=[gt],
            retrieved=retrieved,
        ))
    return BenchmarkResults(results=results)


def _make_config_result(
    min_chars: int = 100,
    max_chars: int = 1000,
    recall: float = 0.8,
    elapsed: float = 5.0,
) -> ConfigResult:
    return ConfigResult(
        config=ChunkConfig(min_chars=min_chars, max_chars=max_chars),
        results=_make_results(recall_score=recall),
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# ChunkConfig
# ---------------------------------------------------------------------------


class TestChunkConfig:
    def test_label_format(self):
        c = ChunkConfig(min_chars=100, max_chars=3000)
        assert c.label == "100–3000"

    def test_repr(self):
        c = ChunkConfig(min_chars=50, max_chars=800)
        assert "50" in repr(c)
        assert "800" in repr(c)


# ---------------------------------------------------------------------------
# SweepResults
# ---------------------------------------------------------------------------


class TestSweepResults:
    def _make_sweep(self) -> SweepResults:
        return SweepResults(
            config_results=[
                _make_config_result(max_chars=500,  recall=0.70, elapsed=3.0),
                _make_config_result(max_chars=800,  recall=0.85, elapsed=4.0),
                _make_config_result(max_chars=1500, recall=0.80, elapsed=5.0),
                _make_config_result(max_chars=3000, recall=0.75, elapsed=6.0),
            ],
            queries_evaluated=20,
        )

    def test_configs_property(self):
        sweep = self._make_sweep()
        assert len(sweep.configs) == 4
        assert all(isinstance(c, ChunkConfig) for c in sweep.configs)

    def test_best_config_by_recall(self):
        sweep = self._make_sweep()
        best = sweep.best_config(k=64, metric="recall")
        assert best.config.max_chars == 800

    def test_comparison_table_returns_string(self):
        sweep = self._make_sweep()
        table = sweep.comparison_table(ks=[1, 4, 16, 64])
        assert isinstance(table, str)
        assert len(table) > 0

    def test_comparison_table_contains_all_configs(self):
        sweep = self._make_sweep()
        table = sweep.comparison_table(ks=[64])
        for cr in sweep.config_results:
            assert cr.config.label in table

    def test_comparison_table_marks_best(self):
        sweep = self._make_sweep()
        table = sweep.comparison_table(ks=[64])
        assert "best" in table

    def test_delta_table_returns_string(self):
        sweep = self._make_sweep()
        delta = sweep.delta_table(baseline_max_chars=3000, k=64)
        assert isinstance(delta, str)
        assert "baseline" in delta

    def test_delta_table_baseline_shows_zero_delta(self):
        sweep = self._make_sweep()
        delta = sweep.delta_table(baseline_max_chars=3000, k=64)
        assert "(+0.00pp)" in delta or "(baseline)" in delta

    def test_delta_table_falls_back_to_last_when_baseline_missing(self):
        sweep = self._make_sweep()
        # No config has max_chars=9999; should fall back gracefully
        delta = sweep.delta_table(baseline_max_chars=9999, k=64)
        assert isinstance(delta, str)


# ---------------------------------------------------------------------------
# run_chunk_sweep
# ---------------------------------------------------------------------------


class TestRunChunkSweep:
    def _make_mock_queries(self, n: int = 3):
        """Return minimal fake BenchmarkQuery objects."""
        from legalkit.eval.benchmark import BenchmarkQuery, GroundTruthSnippet
        return [
            BenchmarkQuery(
                query=f"query {i}",
                snippets=[GroundTruthSnippet(file_path="doc.txt", span=(0, 100))],
                tags=["test"],
            )
            for i in range(n)
        ]

    def test_runs_one_benchmark_per_config(self):
        configs = [
            ChunkConfig(100, 500),
            ChunkConfig(100, 800),
            ChunkConfig(100, 3000),
        ]
        queries = self._make_mock_queries()

        with patch("legalkit.eval.chunk_sweep.run_benchmark") as mock_bench:
            mock_bench.return_value = _make_results()
            sweep = run_chunk_sweep(queries, configs=configs)

        assert mock_bench.call_count == len(configs)

    def test_returns_correct_number_of_config_results(self):
        configs = [ChunkConfig(100, 500), ChunkConfig(100, 1500)]
        queries = self._make_mock_queries()

        with patch("legalkit.eval.chunk_sweep.run_benchmark") as mock_bench:
            mock_bench.return_value = _make_results()
            sweep = run_chunk_sweep(queries, configs=configs)

        assert len(sweep.config_results) == 2

    def test_passes_min_and_max_chars_to_benchmark(self):
        config = ChunkConfig(min_chars=75, max_chars=600)
        queries = self._make_mock_queries()

        with patch("legalkit.eval.chunk_sweep.run_benchmark") as mock_bench:
            mock_bench.return_value = _make_results()
            run_chunk_sweep(queries, configs=[config])

        call_kwargs = mock_bench.call_args.kwargs
        assert call_kwargs["min_chars"] == 75
        assert call_kwargs["max_chars"] == 600

    def test_passes_max_chunks_per_query(self):
        queries = self._make_mock_queries()

        with patch("legalkit.eval.chunk_sweep.run_benchmark") as mock_bench:
            mock_bench.return_value = _make_results()
            run_chunk_sweep(queries, configs=[ChunkConfig(100, 800)], max_chunks_per_query=32)

        call_kwargs = mock_bench.call_args.kwargs
        assert call_kwargs["max_chunks_per_query"] == 32

    def test_uses_default_configs_when_none_given(self):
        queries = self._make_mock_queries()

        with patch("legalkit.eval.chunk_sweep.run_benchmark") as mock_bench:
            mock_bench.return_value = _make_results()
            sweep = run_chunk_sweep(queries, configs=None)

        assert mock_bench.call_count == len(DEFAULT_CONFIGS)

    def test_progress_callback_called(self):
        configs = [ChunkConfig(100, 500), ChunkConfig(100, 800)]
        queries = self._make_mock_queries()
        calls: list[tuple] = []

        with patch("legalkit.eval.chunk_sweep.run_benchmark") as mock_bench:
            mock_bench.return_value = _make_results()
            run_chunk_sweep(
                queries,
                configs=configs,
                progress_callback=lambda s, c, t: calls.append((s, c, t)),
            )

        assert len(calls) > 0

    def test_elapsed_seconds_recorded(self):
        queries = self._make_mock_queries()

        with patch("legalkit.eval.chunk_sweep.run_benchmark") as mock_bench:
            mock_bench.return_value = _make_results()
            sweep = run_chunk_sweep(queries, configs=[ChunkConfig(100, 800)])

        assert sweep.config_results[0].elapsed_seconds >= 0.0

    def test_queries_evaluated_matches_input(self):
        queries = self._make_mock_queries(n=7)

        with patch("legalkit.eval.chunk_sweep.run_benchmark") as mock_bench:
            mock_bench.return_value = _make_results()
            sweep = run_chunk_sweep(queries, configs=[ChunkConfig(100, 800)])

        assert sweep.queries_evaluated == 7


# ---------------------------------------------------------------------------
# benchmark.py — min/max_chars params
# ---------------------------------------------------------------------------


class TestBenchmarkChunkParams:
    def test_run_benchmark_accepts_chunk_size_params(self):
        """Smoke test that run_benchmark passes chunk sizes through."""
        from legalkit.eval.benchmark import run_benchmark, BenchmarkQuery, GroundTruthSnippet

        queries = [
            BenchmarkQuery(
                query="test query",
                snippets=[GroundTruthSnippet(file_path="doc.txt", span=(0, 50))],
                tags=["test"],
            )
        ]

        captured = {}

        def mock_chunk(root, doc_id, min_chars=100, max_chars=3000):
            captured["min_chars"] = min_chars
            captured["max_chars"] = max_chars
            return []  # no chunks → index skipped

        with (
            patch("legalkit.eval.benchmark.parse_document") as mock_parse,
            patch("legalkit.eval.benchmark.chunk_document", side_effect=mock_chunk),
            patch("legalkit.eval.benchmark.CORPUS_DIR") as mock_dir,
        ):
            mock_parse.return_value = MagicMock()
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.read_text.return_value = "sample legal text"
            mock_path.stem = "doc"
            mock_dir.__truediv__ = lambda self, other: mock_path

            run_benchmark(queries, min_chars=42, max_chars=777)

        assert captured.get("min_chars") == 42
        assert captured.get("max_chars") == 777
