"""LegalBench-RAG benchmark runner.

Downloads, ingests, and evaluates against the LegalBench-RAG dataset.
Results are character-span-level precision and recall.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from legalkit.eval.metrics import BenchmarkResults, QueryResult, SpanMatch
from legalkit.pipeline.parse import parse_document
from legalkit.pipeline.chunk import chunk_document
from legalkit.pipeline.embed import embed_texts, embed_query
from legalkit.pipeline.index import DocumentIndex


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
CORPUS_DIR = DATA_DIR / "corpus"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"

DATASET_NAMES = ["privacy_qa", "contractnli", "maud", "cuad"]
MINI_SAMPLES_PER_DATASET = 194  # 776 total for mini


@dataclass
class GroundTruthSnippet:
    """A ground truth snippet from the benchmark."""
    file_path: str
    span: tuple[int, int]
    answer: str = ""
    """Ground-truth answer text (from benchmark annotation, if available)."""


@dataclass
class BenchmarkQuery:
    """A single benchmark query with ground truth."""
    query: str
    snippets: list[GroundTruthSnippet]
    tags: list[str] = field(default_factory=list)


def load_benchmark(
    datasets: list[str] | None = None,
    mini: bool = True,
    max_per_dataset: int | None = None,
) -> list[BenchmarkQuery]:
    """Load LegalBench-RAG benchmark queries.

    Args:
        datasets: Which datasets to load. None = all four.
        mini: If True, load 194 per dataset (776 total).
        max_per_dataset: Override max queries per dataset.

    Returns:
        List of BenchmarkQuery objects.
    """
    datasets = datasets or DATASET_NAMES
    max_n = max_per_dataset or (MINI_SAMPLES_PER_DATASET if mini else None)

    all_queries: list[BenchmarkQuery] = []

    for name in datasets:
        bench_file = BENCHMARKS_DIR / f"{name}.json"
        if not bench_file.exists():
            raise FileNotFoundError(
                f"Benchmark file not found: {bench_file}\n"
                f"Run 'legalkit bench --download' to download the dataset."
            )

        with open(bench_file) as f:
            data = json.load(f)

        tests = data.get("tests", data) if isinstance(data, dict) else data
        if isinstance(tests, dict):
            tests = tests.get("tests", [])

        for test in tests[:max_n]:
            snippets = []
            for s in test["snippets"]:
                span = s["span"]
                if isinstance(span, list):
                    span = tuple(span)
                snippets.append(GroundTruthSnippet(
                    file_path=s["file_path"],
                    span=span,
                    answer=s.get("answer", ""),
                ))

            all_queries.append(BenchmarkQuery(
                query=test["query"],
                snippets=snippets,
                tags=[name],
            ))

    return all_queries


def get_corpus_files(queries: list[BenchmarkQuery]) -> set[str]:
    """Get the set of corpus file paths needed for a set of queries."""
    files: set[str] = set()
    for q in queries:
        for s in q.snippets:
            files.add(s.file_path)
    return files


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


@dataclass
class CorpusIndex:
    """Index for a single corpus document."""
    file_path: str
    index: DocumentIndex
    chunks: list  # list[Chunk]
    root: object = None  # DocumentNode — populated when cross-ref expansion is needed


def run_benchmark(
    queries: list[BenchmarkQuery],
    max_chunks_per_query: int = 64,
    min_chars: int = 100,
    max_chars: int = 1500,
    retrieval_method: str = "hybrid",
    use_cross_references: bool = False,
    use_ml_parser: bool = False,
    ml_model_path: str | None = None,
    embedding_fn=None,
    progress_callback=None,
) -> BenchmarkResults:
    """Run the full benchmark.

    For each corpus document referenced by queries:
    1. Parse and chunk the document
    2. Embed chunks and build index
    3. Run all queries against relevant documents
    4. Compute character-level precision and recall

    Args:
        queries: Benchmark queries to evaluate.
        max_chunks_per_query: Maximum chunks to retrieve per query.
        min_chars: Minimum chunk size in characters (passed to chunk_document).
        max_chars: Maximum chunk size in characters (passed to chunk_document).
        retrieval_method: One of "semantic", "bm25", or "hybrid".
        use_cross_references: If True, expand cross-references from retrieved chunks.
        use_ml_parser: If True, use ML+regex hybrid parser instead of regex-only.
        ml_model_path: Path to ML model directory (None = use default location).
        embedding_fn: Optional callable(texts: list[str]) -> np.ndarray for custom embeddings (None = default API).
        progress_callback: Optional callback(stage, current, total).

    Returns:
        BenchmarkResults with per-query and aggregate metrics.
    """
    corpus_files = get_corpus_files(queries)

    # Group queries by their target files
    file_to_queries: dict[str, list[int]] = {}
    for i, q in enumerate(queries):
        for s in q.snippets:
            file_to_queries.setdefault(s.file_path, []).append(i)

    # Build indices for each corpus document
    corpus_indices: dict[str, CorpusIndex] = {}
    file_list = sorted(corpus_files)

    if progress_callback:
        progress_callback("ingesting", 0, len(file_list))

    for fi, file_path in enumerate(file_list):
        full_path = CORPUS_DIR / file_path
        if not full_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {full_path}")

        text = full_path.read_text()
        doc_id = full_path.stem

        # Parse & chunk
        if use_ml_parser:
            from legalkit.pipeline.parse import parse_document_with_ml
            root = parse_document_with_ml(text, doc_id=doc_id, model_path=ml_model_path)
        else:
            root = parse_document(text, doc_id=doc_id)
        chunks = chunk_document(root, doc_id=doc_id, min_chars=min_chars, max_chars=max_chars)

        if not chunks:
            continue

        # Embed
        embedding_texts = [c.embedding_text for c in chunks]
        if embedding_fn:
            embeddings = embedding_fn(embedding_texts)
        else:
            embeddings = embed_texts(embedding_texts)

        # Build index
        index = DocumentIndex()
        index.build(chunks, embeddings)

        corpus_indices[file_path] = CorpusIndex(
            file_path=file_path,
            index=index,
            chunks=chunks,
            root=root,
        )

        if progress_callback:
            progress_callback("ingesting", fi + 1, len(file_list))

    # Pre-embed all queries in batches (much faster than one call per query)
    if progress_callback:
        progress_callback("embedding queries", 0, len(queries))

    query_texts = [bq.query for bq in queries]
    if embedding_fn:
        all_query_embeddings = embedding_fn(query_texts)
    else:
        all_query_embeddings = embed_texts(query_texts)

    if progress_callback:
        progress_callback("embedding queries", len(queries), len(queries))

    # Run queries
    results: list[QueryResult] = []

    if progress_callback:
        progress_callback("querying", 0, len(queries))

    for qi, bq in enumerate(queries):
        # Find which corpus files this query relates to
        target_files = {s.file_path for s in bq.snippets}

        # Use pre-computed embedding
        q_emb = all_query_embeddings[qi]

        # Retrieve from all relevant corpus indices
        all_retrieved: list[SpanMatch] = []

        for file_path in target_files:
            if file_path not in corpus_indices:
                continue

            ci = corpus_indices[file_path]

            # Retrieve using the configured method
            retrieved_chunks = _retrieve_chunks(
                index=ci.index,
                root=ci.root,
                query_text=bq.query,
                query_embedding=q_emb,
                top_k=max_chunks_per_query,
                retrieval_method=retrieval_method,
                use_cross_references=use_cross_references,
            )

            # Convert chunk results to SpanMatch
            for chunk in retrieved_chunks:
                all_retrieved.append(SpanMatch(
                    file_path=file_path,
                    start=chunk.char_start,
                    end=chunk.char_end,
                ))

        # Sort by score (already sorted from hybrid), limit to max_chunks
        all_retrieved = all_retrieved[:max_chunks_per_query]

        # Build ground truth spans
        gt_spans = [
            SpanMatch(file_path=s.file_path, start=s.span[0], end=s.span[1])
            for s in bq.snippets
        ]

        results.append(QueryResult(
            query=bq.query,
            tags=bq.tags,
            ground_truth=gt_spans,
            retrieved=all_retrieved,
        ))

        if progress_callback and (qi + 1) % 10 == 0:
            progress_callback("querying", qi + 1, len(queries))

    if progress_callback:
        progress_callback("querying", len(queries), len(queries))

    return BenchmarkResults(results=results)


# ---------------------------------------------------------------------------
# Retrieval dispatch
# ---------------------------------------------------------------------------


def _retrieve_chunks(
    index: DocumentIndex,
    root: object,
    query_text: str,
    query_embedding,
    top_k: int,
    retrieval_method: str,
    use_cross_references: bool,
):
    """Run retrieval with the configured method and optional cross-ref expansion.

    Returns a flat list of Chunk objects (deduplicated, capped at top_k).
    """
    from legalkit.models.retrieval import Chunk

    if retrieval_method == "semantic":
        results = index.search_semantic(query_embedding, top_k=top_k)
        chunks = [index.chunks[idx] for idx, _ in results]

    elif retrieval_method == "bm25":
        results = index.search_bm25(query_text, top_k=top_k)
        chunks = [index.chunks[idx] for idx, _ in results]

    else:  # hybrid (default)
        results = index.search_hybrid(
            query_embedding=query_embedding,
            query_text=query_text,
            top_k=top_k,
        )
        chunks = [index.chunks[idx] for idx, _, _, _ in results]

    if use_cross_references and root is not None:
        from legalkit.models.retrieval import RankedChunk, RetrievalMethod
        from legalkit.pipeline.retrieve import _expand_cross_references
        from legalkit.utils.text import normalize_section_id

        ranked = [
            RankedChunk(chunk=c, score=0.0, method=RetrievalMethod.HYBRID)
            for c in chunks
        ]
        xref = _expand_cross_references(
            ranked_chunks=ranked,
            index=index,
            doc_tree=root,
            max_chunks=min(10, top_k - len(chunks)),
        )
        xref_chunks = [rc.chunk for rc in xref]
        seen = {c.id for c in chunks}
        for c in xref_chunks:
            if c.id not in seen:
                chunks.append(c)
                seen.add(c.id)

    return chunks[:top_k]


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

# The dataset is not redistributed with this repository. LegalBench-RAG is published
# upstream; point LEGALBENCH_RAG_URL at a local copy or mirror to use download_dataset().
# A previously hardcoded mirror URL has been removed: it pointed at personal storage.
LEGALBENCH_RAG_SOURCE = "https://github.com/zeroentropy-ai/legalbenchrag"
LEGALBENCH_RAG_URL_ENV = "LEGALBENCH_RAG_URL"


def download_dataset(target_dir: Path | None = None) -> None:
    """Download the LegalBench-RAG dataset.

    Downloads and extracts the corpus and benchmark files.

    Requires LEGALBENCH_RAG_URL to be set to a dataset zip (see LEGALBENCH_RAG_SOURCE);
    no dataset URL is bundled with this repository.
    """
    import os
    import urllib.request
    import zipfile
    import tempfile

    url = os.environ.get(LEGALBENCH_RAG_URL_ENV)
    if not url:
        raise SystemExit(
            f"Set {LEGALBENCH_RAG_URL_ENV} to a LegalBench-RAG dataset zip. "
            f"Upstream: {LEGALBENCH_RAG_SOURCE}"
        )

    target = target_dir or DATA_DIR
    target.mkdir(parents=True, exist_ok=True)

    print(f"Downloading LegalBench-RAG dataset...")
    print(f"  Source: {url}")
    print(f"  Target: {target}")

    # Download the zip
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        urllib.request.urlretrieve(url, tmp.name)
        tmp_path = Path(tmp.name)

    print(f"  Downloaded: {tmp_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Extract
    with zipfile.ZipFile(tmp_path) as zf:
        zf.extractall(target)

    tmp_path.unlink()
    print(f"  Extracted to: {target}")

    # Verify
    corpus_exists = (target / "corpus").exists() or (CORPUS_DIR).exists()
    bench_exists = (target / "benchmarks").exists() or (BENCHMARKS_DIR).exists()
    print(f"  Corpus: {'✓' if corpus_exists else '✗'}")
    print(f"  Benchmarks: {'✓' if bench_exists else '✗'}")
