"""Tests for the ablation study."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from legalkit.eval.ablation import (
    DEFAULT_CONFIGS,
    AblationConfig,
    AblationConfigResult,
    AblationResults,
    run_ablation,
)
from legalkit.eval.metrics import BenchmarkResults, QueryResult, SpanMatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_results(recall: float = 0.8, n: int = 5, tag: str = "test") -> BenchmarkResults:
    results = []
    for i in range(n):
        gt = SpanMatch(file_path="doc.txt", start=0, end=100)
        end = int(100 * min(recall, 1.0))
        retrieved = [SpanMatch(file_path="doc.txt", start=0, end=end)]
        results.append(QueryResult(
            query=f"q{i}", tags=[tag], ground_truth=[gt], retrieved=retrieved,
        ))
    return BenchmarkResults(results=results)


def _make_cr(label: str, recall: float = 0.8, elapsed: float = 5.0) -> AblationConfigResult:
    cfg = AblationConfig(label=label, description=f"desc for {label}")
    return AblationConfigResult(
        config=cfg, results=_make_results(recall=recall), elapsed_seconds=elapsed,
    )


def _make_queries(n: int = 3):
    from legalkit.eval.benchmark import BenchmarkQuery, GroundTruthSnippet
    return [
        BenchmarkQuery(
            query=f"q{i}",
            snippets=[GroundTruthSnippet(file_path="doc.txt", span=(0, 100))],
            tags=["test"],
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# AblationConfig
# ---------------------------------------------------------------------------


class TestAblationConfig:
    def test_default_retrieval_method(self):
        c = AblationConfig(label="test")
        assert c.retrieval_method == "hybrid"

    def test_default_no_cross_references(self):
        c = AblationConfig(label="test")
        assert c.use_cross_references is False

    def test_repr(self):
        c = AblationConfig(label="hybrid+xref")
        assert "hybrid+xref" in repr(c)

    def test_default_configs_are_additive(self):
        """Each default config should add exactly one component."""
        labels = [c.label for c in DEFAULT_CONFIGS]
        assert "semantic" in labels
        assert "bm25" in labels
        assert "hybrid" in labels
        assert "hybrid+xref" in labels

    def test_hybrid_xref_config_has_cross_references(self):
        xref_config = next(c for c in DEFAULT_CONFIGS if c.label == "hybrid+xref")
        assert xref_config.use_cross_references is True
        assert xref_config.retrieval_method == "hybrid"

    def test_semantic_config_has_no_cross_references(self):
        sem = next(c for c in DEFAULT_CONFIGS if c.label == "semantic")
        assert sem.retrieval_method == "semantic"
        assert sem.use_cross_references is False


# ---------------------------------------------------------------------------
# AblationResults
# ---------------------------------------------------------------------------


class TestAblationResults:
    def _make_sweep(self) -> AblationResults:
        return AblationResults(
            config_results=[
                _make_cr("semantic",    recall=0.60),
                _make_cr("bm25",        recall=0.55),
                _make_cr("hybrid",      recall=0.75),
                _make_cr("hybrid+xref", recall=0.80),
            ],
            queries_evaluated=20,
        )

    def test_best_config(self):
        abl = self._make_sweep()
        best = abl.best_config(k=64)
        assert best.config.label == "hybrid+xref"

    def test_component_deltas_length(self):
        abl = self._make_sweep()
        deltas = abl.component_deltas(k=64)
        assert len(deltas) == 4

    def test_first_delta_has_no_previous(self):
        """First config delta is measured from 0."""
        abl = self._make_sweep()
        deltas = abl.component_deltas(k=64)
        # semantic recall ≈ 0.60, delta from 0 ≈ 0.60
        assert deltas[0]["delta_pp"] > 0

    def test_deltas_are_sequential(self):
        """Each delta measures improvement over previous config."""
        abl = self._make_sweep()
        deltas = abl.component_deltas(k=64)
        recalls = [d["recall"] for d in deltas]
        computed_deltas = [d["delta_pp"] for d in deltas]

        # First delta = first recall (from baseline 0)
        assert abs(computed_deltas[0] - recalls[0]) < 0.01
        # Subsequent deltas = diff from previous
        for i in range(1, len(recalls)):
            expected = recalls[i] - recalls[i - 1]
            assert abs(computed_deltas[i] - expected) < 0.01

    def test_comparison_table_returns_string(self):
        abl = self._make_sweep()
        table = abl.comparison_table(ks=[1, 8, 64])
        assert isinstance(table, str)
        assert "semantic" in table
        assert "hybrid+xref" in table

    def test_comparison_table_marks_best(self):
        abl = self._make_sweep()
        table = abl.comparison_table(ks=[64])
        assert "◀" in table

    def test_delta_table_returns_string(self):
        abl = self._make_sweep()
        dt = abl.delta_table(k=64)
        assert isinstance(dt, str)
        assert "Marginal" in dt
        assert "semantic" in dt

    def test_configs_property(self):
        abl = self._make_sweep()
        assert len(abl.configs) == 4
        assert all(isinstance(c, AblationConfig) for c in abl.configs)


# ---------------------------------------------------------------------------
# run_ablation
# ---------------------------------------------------------------------------


class TestRunAblation:
    def test_runs_once_per_config(self):
        configs = [
            AblationConfig("semantic", retrieval_method="semantic"),
            AblationConfig("hybrid", retrieval_method="hybrid"),
        ]
        with patch("legalkit.eval.ablation.run_benchmark") as mock_bench:
            mock_bench.return_value = _make_results()
            run_ablation(_make_queries(), configs=configs)
        assert mock_bench.call_count == 2

    def test_uses_default_configs_when_none(self):
        with patch("legalkit.eval.ablation.run_benchmark") as mock_bench:
            mock_bench.return_value = _make_results()
            run_ablation(_make_queries(), configs=None)
        assert mock_bench.call_count == len(DEFAULT_CONFIGS)

    def test_passes_retrieval_method(self):
        configs = [AblationConfig("bm25", retrieval_method="bm25")]
        with patch("legalkit.eval.ablation.run_benchmark") as mock_bench:
            mock_bench.return_value = _make_results()
            run_ablation(_make_queries(), configs=configs)
        kwargs = mock_bench.call_args.kwargs
        assert kwargs["retrieval_method"] == "bm25"

    def test_passes_cross_references_flag(self):
        configs = [AblationConfig("hybrid+xref", use_cross_references=True)]
        with patch("legalkit.eval.ablation.run_benchmark") as mock_bench:
            mock_bench.return_value = _make_results()
            run_ablation(_make_queries(), configs=configs)
        kwargs = mock_bench.call_args.kwargs
        assert kwargs["use_cross_references"] is True

    def test_passes_chunk_size_params(self):
        configs = [AblationConfig("hybrid")]
        with patch("legalkit.eval.ablation.run_benchmark") as mock_bench:
            mock_bench.return_value = _make_results()
            run_ablation(_make_queries(), configs=configs, min_chars=75, max_chars=1200)
        kwargs = mock_bench.call_args.kwargs
        assert kwargs["min_chars"] == 75
        assert kwargs["max_chars"] == 1200

    def test_returns_correct_number_of_results(self):
        configs = [AblationConfig("a"), AblationConfig("b"), AblationConfig("c")]
        with patch("legalkit.eval.ablation.run_benchmark") as mock_bench:
            mock_bench.return_value = _make_results()
            abl = run_ablation(_make_queries(), configs=configs)
        assert len(abl.config_results) == 3

    def test_queries_evaluated_matches_input(self):
        with patch("legalkit.eval.ablation.run_benchmark") as mock_bench:
            mock_bench.return_value = _make_results()
            abl = run_ablation(_make_queries(n=7))
        assert abl.queries_evaluated == 7

    def test_elapsed_seconds_recorded(self):
        with patch("legalkit.eval.ablation.run_benchmark") as mock_bench:
            mock_bench.return_value = _make_results()
            abl = run_ablation(_make_queries(), configs=[AblationConfig("test")])
        assert abl.config_results[0].elapsed_seconds >= 0.0

    def test_progress_callback_called(self):
        calls = []
        with patch("legalkit.eval.ablation.run_benchmark") as mock_bench:
            mock_bench.return_value = _make_results()
            run_ablation(
                _make_queries(),
                configs=[AblationConfig("a"), AblationConfig("b")],
                progress_callback=lambda s, c, t: calls.append((s, c, t)),
            )
        assert len(calls) > 0


# ---------------------------------------------------------------------------
# benchmark._retrieve_chunks dispatch
# ---------------------------------------------------------------------------


class TestRetrieveChunksDispatch:
    """Unit tests for the _retrieve_chunks helper in benchmark.py."""

    def _make_index(self, n_chunks: int = 5):
        from legalkit.pipeline.index import DocumentIndex
        from legalkit.models.retrieval import Chunk

        chunks = [
            Chunk(
                id=f"c{i}", text=f"text {i}", doc_id="doc",
                node_id=f"s{i}", char_start=i * 100, char_end=i * 100 + 100,
            )
            for i in range(n_chunks)
        ]
        rng = np.random.default_rng(42)
        emb = rng.random((n_chunks, 8), dtype=np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        index = DocumentIndex()
        index.build(chunks, emb)
        return index, chunks

    def test_semantic_calls_search_semantic(self):
        from legalkit.eval.benchmark import _retrieve_chunks

        index, chunks = self._make_index()
        q_emb = np.random.default_rng(0).random(8, dtype=np.float32)

        with patch.object(index, "search_semantic", wraps=index.search_semantic) as mock_sem:
            _retrieve_chunks(index, None, "query", q_emb, top_k=3,
                             retrieval_method="semantic", use_cross_references=False)
        mock_sem.assert_called_once()

    def test_bm25_calls_search_bm25(self):
        from legalkit.eval.benchmark import _retrieve_chunks

        index, chunks = self._make_index()
        q_emb = np.random.default_rng(0).random(8, dtype=np.float32)

        with patch.object(index, "search_bm25", wraps=index.search_bm25) as mock_bm25:
            _retrieve_chunks(index, None, "query", q_emb, top_k=3,
                             retrieval_method="bm25", use_cross_references=False)
        mock_bm25.assert_called_once()

    def test_hybrid_calls_search_hybrid(self):
        from legalkit.eval.benchmark import _retrieve_chunks

        index, chunks = self._make_index()
        q_emb = np.random.default_rng(0).random(8, dtype=np.float32)

        with patch.object(index, "search_hybrid", wraps=index.search_hybrid) as mock_hybrid:
            _retrieve_chunks(index, None, "query", q_emb, top_k=3,
                             retrieval_method="hybrid", use_cross_references=False)
        mock_hybrid.assert_called_once()

    def test_top_k_respected(self):
        from legalkit.eval.benchmark import _retrieve_chunks

        index, chunks = self._make_index(n_chunks=10)
        q_emb = np.random.default_rng(0).random(8, dtype=np.float32)

        result = _retrieve_chunks(index, None, "query", q_emb, top_k=3,
                                  retrieval_method="hybrid", use_cross_references=False)
        assert len(result) <= 3

    def test_returns_chunk_objects(self):
        from legalkit.eval.benchmark import _retrieve_chunks
        from legalkit.models.retrieval import Chunk

        index, chunks = self._make_index()
        q_emb = np.random.default_rng(0).random(8, dtype=np.float32)

        result = _retrieve_chunks(index, None, "query", q_emb, top_k=5,
                                  retrieval_method="semantic", use_cross_references=False)
        assert all(isinstance(c, Chunk) for c in result)
