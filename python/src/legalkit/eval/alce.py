"""ALCE-style citation quality evaluation.

Adapts the ALCE (Enabling LLMs to Generate Text with Citations) evaluation
framework — originally designed for open-domain QA — to measure citation
quality in legal document answers produced by this pipeline.

Reference:
    Gao et al. (2023) "Enabling Large Language Models to Generate Text with
    Citations" — EMNLP 2023, Princeton NLP.
    https://arxiv.org/abs/2305.14627

## Metrics

**Citation Recall (Cit-R)**
Fraction of non-gap claims that have at least one verified (or partially
verified) citation. Measures whether claims are backed by evidence.

**Citation Precision (Cit-P)**
Fraction of all citations that are verified or partially verified. Measures
whether citations actually support the claims they're attached to.

**Citation F1 (Cit-F1)**
Harmonic mean of citation recall and precision.

**Answer Correctness (String F1)**
Token-level F1 between the generated answer and the ground-truth answer
text from the benchmark. Measures factual coverage without requiring an
LLM judge.

**Hallucination Rate**
Fraction of non-gap claims that have *no* verified citations. The
complement of citation recall — claims the pipeline made without
substantiation.

## Comparison with published ALCE results

ALCE's published baselines (on ASQA/QAMPARI/ELI5) show citation recall
typically 50-70% and precision 60-80% for general-purpose LLMs. Our target
on legal documents is citation recall ≥ 0.85 and precision ≥ 0.80.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from legalkit.models.citation import VerificationStatus
from legalkit.models.report import EvidenceReport


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class ALCEMetrics:
    """ALCE-style citation quality metrics for a single query-answer pair."""

    citation_recall: float
    """Fraction of non-gap claims with ≥1 verified citation (0–1)."""

    citation_precision: float
    """Fraction of all citations that are verified (0–1)."""

    citation_f1: float
    """Harmonic mean of recall and precision (0–1)."""

    answer_string_f1: float | None
    """Token-level F1 vs ground truth answer (0–1). None if no ground truth."""

    hallucination_rate: float
    """Fraction of non-gap claims with 0 verified citations (0–1)."""

    n_claims: int
    """Total non-gap claims in the answer."""

    n_citations: int
    """Total citations across all claims."""

    n_verified_citations: int
    """Citations that are VERIFIED or PARTIALLY_VERIFIED."""

    @property
    def has_ground_truth(self) -> bool:
        return self.answer_string_f1 is not None

    def __repr__(self) -> str:
        return (
            f"ALCEMetrics("
            f"cit_r={self.citation_recall:.3f} "
            f"cit_p={self.citation_precision:.3f} "
            f"cit_f1={self.citation_f1:.3f} "
            f"str_f1={self.answer_string_f1:.3f if self.answer_string_f1 is not None else 'N/A'} "
            f"halluc={self.hallucination_rate:.3f})"
        )


@dataclass
class ALCEQueryResult:
    """ALCE evaluation result for a single query."""

    query: str
    dataset: str
    answer: str
    ground_truth: str | None
    metrics: ALCEMetrics
    elapsed_seconds: float = 0.0
    doc_id: str = ""


@dataclass
class ALCEReport:
    """Aggregate ALCE evaluation results across multiple queries."""

    results: list[ALCEQueryResult]
    source: str = ""

    # ---------------------------------------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------------------------------------

    @property
    def avg_citation_recall(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.metrics.citation_recall for r in self.results) / len(self.results)

    @property
    def avg_citation_precision(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.metrics.citation_precision for r in self.results) / len(self.results)

    @property
    def avg_citation_f1(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.metrics.citation_f1 for r in self.results) / len(self.results)

    @property
    def avg_answer_string_f1(self) -> float | None:
        scored = [r.metrics.answer_string_f1 for r in self.results
                  if r.metrics.answer_string_f1 is not None]
        if not scored:
            return None
        return sum(scored) / len(scored)

    @property
    def avg_hallucination_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.metrics.hallucination_rate for r in self.results) / len(self.results)

    @property
    def total_claims(self) -> int:
        return sum(r.metrics.n_claims for r in self.results)

    @property
    def total_citations(self) -> int:
        return sum(r.metrics.n_citations for r in self.results)

    def per_dataset(self) -> dict[str, "ALCEReport"]:
        """Split results by dataset tag."""
        by_ds: dict[str, list[ALCEQueryResult]] = {}
        for r in self.results:
            by_ds.setdefault(r.dataset, []).append(r)
        return {ds: ALCEReport(results=rs, source=ds) for ds, rs in by_ds.items()}

    def summary_table(self) -> str:
        """Plain-text summary table."""
        lines = [
            f"ALCE Citation Quality  ({self.source})",
            f"Queries evaluated: {len(self.results)}",
            "",
            f"  Citation Recall:     {self.avg_citation_recall:.3f}  "
            f"({self.avg_citation_recall:.1%}) — claims with ≥1 verified citation",
            f"  Citation Precision:  {self.avg_citation_precision:.3f}  "
            f"({self.avg_citation_precision:.1%}) — citations that actually support claims",
            f"  Citation F1:         {self.avg_citation_f1:.3f}",
            f"  Hallucination Rate:  {self.avg_hallucination_rate:.3f}  "
            f"({self.avg_hallucination_rate:.1%}) — claims with NO verified citation",
        ]
        if self.avg_answer_string_f1 is not None:
            lines.append(
                f"  Answer String F1:    {self.avg_answer_string_f1:.3f}  "
                f"({self.avg_answer_string_f1:.1%}) — token overlap vs ground truth"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        gt_f1 = self.avg_answer_string_f1
        return {
            "source": self.source,
            "n_queries": len(self.results),
            "total_claims": self.total_claims,
            "total_citations": self.total_citations,
            "avg_citation_recall": round(self.avg_citation_recall, 4),
            "avg_citation_precision": round(self.avg_citation_precision, 4),
            "avg_citation_f1": round(self.avg_citation_f1, 4),
            "avg_answer_string_f1": round(gt_f1, 4) if gt_f1 is not None else None,
            "avg_hallucination_rate": round(self.avg_hallucination_rate, 4),
            "per_query": [
                {
                    "query": r.query[:100],
                    "dataset": r.dataset,
                    "doc_id": r.doc_id,
                    "citation_recall": round(r.metrics.citation_recall, 4),
                    "citation_precision": round(r.metrics.citation_precision, 4),
                    "citation_f1": round(r.metrics.citation_f1, 4),
                    "answer_string_f1": round(r.metrics.answer_string_f1, 4)
                    if r.metrics.answer_string_f1 is not None else None,
                    "hallucination_rate": round(r.metrics.hallucination_rate, 4),
                    "n_claims": r.metrics.n_claims,
                    "n_citations": r.metrics.n_citations,
                    "elapsed_seconds": round(r.elapsed_seconds, 2),
                }
                for r in self.results
            ],
        }


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

_VERIFIED = {VerificationStatus.VERIFIED, VerificationStatus.PARTIALLY_VERIFIED}


def compute_alce_metrics(
    report: EvidenceReport,
    ground_truth: str | None = None,
) -> ALCEMetrics:
    """Compute ALCE-style metrics from an EvidenceReport.

    Args:
        report: The pipeline's evidence report for a single query.
        ground_truth: Ground-truth answer text (from benchmark) for string
            F1 computation. Pass None to skip answer correctness.

    Returns:
        :class:`ALCEMetrics`.
    """
    # Filter to substantive claims (not gap acknowledgments)
    substantive = [c for c in report.claims if not c.is_gap]

    n_claims = len(substantive)
    n_citations = sum(len(c.citations) for c in substantive)
    n_verified = sum(
        1 for c in substantive
        for cit in c.citations
        if cit.status in _VERIFIED
    )

    # Citation recall: fraction of claims with ≥1 verified citation
    if n_claims == 0:
        citation_recall = 0.0
    else:
        claims_with_verified = sum(
            1 for c in substantive
            if any(cit.status in _VERIFIED for cit in c.citations)
        )
        citation_recall = claims_with_verified / n_claims

    # Citation precision: fraction of citations that are verified
    citation_precision = (n_verified / n_citations) if n_citations > 0 else 0.0

    # F1
    if citation_recall + citation_precision > 0:
        citation_f1 = (
            2 * citation_recall * citation_precision
            / (citation_recall + citation_precision)
        )
    else:
        citation_f1 = 0.0

    # Hallucination rate: claims with NO verified citations
    if n_claims == 0:
        hallucination_rate = 0.0
    else:
        claims_no_verified = sum(
            1 for c in substantive
            if not any(cit.status in _VERIFIED for cit in c.citations)
        )
        hallucination_rate = claims_no_verified / n_claims

    # String F1 vs ground truth
    str_f1 = None
    if ground_truth:
        str_f1 = _string_f1(report.answer, ground_truth)

    return ALCEMetrics(
        citation_recall=round(citation_recall, 4),
        citation_precision=round(citation_precision, 4),
        citation_f1=round(citation_f1, 4),
        answer_string_f1=round(str_f1, 4) if str_f1 is not None else None,
        hallucination_rate=round(hallucination_rate, 4),
        n_claims=n_claims,
        n_citations=n_citations,
        n_verified_citations=n_verified,
    )


# ---------------------------------------------------------------------------
# Full pipeline runner
# ---------------------------------------------------------------------------


def run_alce_eval(
    datasets: list[str] | None = None,
    n_per_dataset: int = 10,
    top_k: int = 20,
    llm=None,  # LLMClient
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ALCEReport:
    """Run end-to-end ALCE evaluation on benchmark queries.

    For each selected query:
    1. Ingests the relevant corpus document (parse → chunk → embed → index)
    2. Runs the full pipeline (retrieve → generate → verify)
    3. Computes ALCE metrics from the EvidenceReport

    Uses the ground-truth answer text from the benchmark for string F1.

    Args:
        datasets: Which datasets to use. Defaults to all four.
        n_per_dataset: Queries per dataset (keep small for speed).
        top_k: Retrieval depth.
        llm: :class:`~legalkit.llm.client.LLMClient`. Created automatically
            if not provided.
        progress_callback: Optional ``(current, total, msg) -> None``.

    Returns:
        :class:`ALCEReport`.
    """
    import time as _time

    from legalkit.eval.benchmark import load_benchmark, CORPUS_DIR
    from legalkit.llm.client import LLMClient
    from legalkit.pipeline.chunk import chunk_document
    from legalkit.pipeline.definitions import DefinitionIndex
    from legalkit.pipeline.embed import embed_texts, embed_query
    from legalkit.pipeline.index import DocumentIndex
    from legalkit.pipeline.parse import parse_document
    from legalkit.pipeline.report import build_evidence_report
    from legalkit.pipeline.retrieve import retrieve

    if llm is None:
        llm = LLMClient()

    datasets = datasets or ["contractnli", "cuad", "maud", "privacy_qa"]

    queries = load_benchmark(
        datasets=datasets,
        mini=True,
        max_per_dataset=n_per_dataset,
    )

    results: list[ALCEQueryResult] = []
    doc_cache: dict[str, tuple] = {}  # file_path → (index, root, def_index)

    for qi, bq in enumerate(queries):
        if progress_callback:
            progress_callback(qi, len(queries), bq.query[:60])

        # Get the first snippet's file and ground truth answer
        if not bq.snippets:
            continue
        snippet = bq.snippets[0]
        file_path = snippet.file_path
        ground_truth = snippet.answer if hasattr(snippet, "answer") else None

        # Build or retrieve cached index
        if file_path not in doc_cache:
            full_path = CORPUS_DIR / file_path
            if not full_path.exists():
                continue
            text = full_path.read_text()
            doc_id = full_path.stem
            root = parse_document(text, doc_id=doc_id)
            chunks = chunk_document(root, doc_id=doc_id)
            embeddings = embed_texts([c.embedding_text for c in chunks])
            index = DocumentIndex()
            index.build(chunks, embeddings)
            def_index = DefinitionIndex()
            def_index.build(root)
            doc_cache[file_path] = (index, root, def_index, doc_id)

        index, root, def_index, doc_id = doc_cache[file_path]

        # Run pipeline
        t0 = _time.perf_counter()
        try:
            q_emb = embed_query(bq.query)
            retrieval = retrieve(
                query=bq.query,
                query_embedding=q_emb,
                index=index,
                doc_tree=root,
                top_k=top_k,
                definition_index=def_index,
            )
            report = build_evidence_report(
                question=bq.query,
                retrieval_result=retrieval,
                doc_tree=root,
                llm=llm,
                doc_id=doc_id,
            )
            elapsed = _time.perf_counter() - t0

            metrics = compute_alce_metrics(report, ground_truth=ground_truth)
            results.append(ALCEQueryResult(
                query=bq.query,
                dataset=bq.tags[0] if bq.tags else "",
                answer=report.answer,
                ground_truth=ground_truth,
                metrics=metrics,
                elapsed_seconds=elapsed,
                doc_id=doc_id,
            ))
        except Exception as exc:
            # Don't let one query failure abort the whole run
            elapsed = _time.perf_counter() - t0
            if progress_callback:
                progress_callback(qi, len(queries), f"ERROR: {exc}")

    if progress_callback:
        progress_callback(len(queries), len(queries), "done")

    return ALCEReport(
        results=results,
        source=f"{', '.join(datasets)} ({n_per_dataset}/dataset)",
    )


# ---------------------------------------------------------------------------
# String F1 helper
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> Counter:
    """Tokenize text to lowercase words for F1 computation."""
    tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return Counter(tokens)


def _string_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth.

    Standard QA string F1: measures token overlap, ignoring order.
    Commonly used in SQuAD, ALCE, and similar benchmarks.

    Args:
        prediction: The model's generated answer.
        ground_truth: The reference answer text.

    Returns:
        F1 score in [0, 1].
    """
    pred_tokens = _tokenize(prediction)
    gt_tokens = _tokenize(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0

    # Common tokens (intersection)
    common = sum((pred_tokens & gt_tokens).values())
    if common == 0:
        return 0.0

    precision = common / sum(pred_tokens.values())
    recall = common / sum(gt_tokens.values())
    return 2 * precision * recall / (precision + recall)
