"""Evaluate embedding models on LegalBench-RAG retrieval.

Compares the fine-tuned local model against the API baseline
(text-embedding-3-large) on the standard benchmark metrics.
"""

from __future__ import annotations

import numpy as np
from typing import Callable

from legalkit.eval.benchmark import load_benchmark, run_benchmark, DATASET_NAMES


def run_embedding_comparison(
    datasets: list[str] | None = None,
    max_per_dataset: int = 50,
    local_model_path: str = "models/legal-embeddings/model",
    top_k: int = 64,
    progress_callback=None,
) -> dict:
    """Run LegalBench-RAG benchmark with both local and API embeddings.
    
    Args:
        datasets: Which datasets to evaluate (None = all).
        max_per_dataset: Max queries per dataset.
        local_model_path: Path to fine-tuned sentence-transformers model.
        top_k: Retrieval depth.
        progress_callback: Optional callback(stage, current, total).
    
    Returns comparison dict:
    {
        'local_model': {'recall@k': ..., 'precision@k': ..., per_dataset: ...},
        'api_model': {'recall@k': ..., ...},
        'improvement': {'overall_recall@64': delta, ...}
    }
    """
    from legalkit.pipeline.embed import embed_texts_local, embed_texts
    
    # Load queries
    queries = load_benchmark(
        datasets=datasets,
        mini=True,
        max_per_dataset=max_per_dataset,
    )
    
    # Build embedding functions
    def local_embed_fn(texts: list[str]) -> np.ndarray:
        """Embed using local model."""
        return embed_texts_local(texts, model_path=local_model_path)
    
    def api_embed_fn(texts: list[str]) -> np.ndarray:
        """Embed using API model."""
        return embed_texts(texts)
    
    # Run benchmark with local model
    if progress_callback:
        progress_callback("local_model", 0, 1)
    
    local_results = run_benchmark(
        queries=queries,
        max_chunks_per_query=top_k,
        embedding_fn=local_embed_fn,
        progress_callback=lambda stage, curr, total: (
            progress_callback(f"local: {stage}", curr, total) if progress_callback else None
        ),
    )
    
    if progress_callback:
        progress_callback("local_model", 1, 1)
    
    # Run benchmark with API model
    if progress_callback:
        progress_callback("api_model", 0, 1)
    
    api_results = run_benchmark(
        queries=queries,
        max_chunks_per_query=top_k,
        embedding_fn=api_embed_fn,
        progress_callback=lambda stage, curr, total: (
            progress_callback(f"api: {stage}", curr, total) if progress_callback else None
        ),
    )
    
    if progress_callback:
        progress_callback("api_model", 1, 1)
    
    # Compute comparison metrics
    ks = [1, 2, 4, 8, 16, 32, 64]
    ks = [k for k in ks if k <= top_k]
    
    local_metrics = {
        "recall@k": {k: local_results.recall_at_k(k) for k in ks},
        "precision@k": {k: local_results.precision_at_k(k) for k in ks},
        "per_dataset": {
            tag: {
                "recall@k": {k: local_results.recall_at_k(k, tag) for k in ks},
                "precision@k": {k: local_results.precision_at_k(k, tag) for k in ks},
            }
            for tag in local_results.tags
        },
    }
    
    api_metrics = {
        "recall@k": {k: api_results.recall_at_k(k) for k in ks},
        "precision@k": {k: api_results.precision_at_k(k) for k in ks},
        "per_dataset": {
            tag: {
                "recall@k": {k: api_results.recall_at_k(k, tag) for k in ks},
                "precision@k": {k: api_results.precision_at_k(k, tag) for k in ks},
            }
            for tag in api_results.tags
        },
    }
    
    # Calculate improvement deltas
    improvement = {
        "overall_recall@64": local_results.recall_at_k(64) - api_results.recall_at_k(64),
        "overall_precision@64": local_results.precision_at_k(64) - api_results.precision_at_k(64),
        "per_k": {
            k: {
                "recall_delta": local_results.recall_at_k(k) - api_results.recall_at_k(k),
                "precision_delta": local_results.precision_at_k(k) - api_results.precision_at_k(k),
            }
            for k in ks
        },
    }
    
    return {
        "local_model": local_metrics,
        "api_model": api_metrics,
        "improvement": improvement,
        "queries_evaluated": len(queries),
        "datasets": datasets or DATASET_NAMES,
    }
