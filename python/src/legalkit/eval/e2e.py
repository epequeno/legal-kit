"""End-to-end evaluation: the real product test.

Runs queries through the full pipeline (retrieval → generation → verification)
and compares against:
1. Ground truth snippets from LegalBench-RAG
2. A "full context" baseline (whole document in context, no RAG)

Measures what actually matters:
- Answer correctness (does the answer cover the ground truth?)
- Citation accuracy (do citations point to real, supporting text?)
- Hallucination detection (does the system flag its own errors?)
- Gap detection (does the system know what it doesn't know?)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from legalkit.eval.benchmark import (
    BenchmarkQuery,
    load_benchmark,
    CORPUS_DIR,
)
from legalkit.llm.client import LLMClient
from legalkit.models.citation import VerificationStatus
from legalkit.pipeline.parse import parse_document
from legalkit.pipeline.chunk import chunk_document
from legalkit.pipeline.embed import embed_texts, embed_query
from legalkit.pipeline.generate import generate_answer, parse_claims
from legalkit.pipeline.index import DocumentIndex
from legalkit.pipeline.retrieve import retrieve
from legalkit.pipeline.verify import verify_claims


# ---------------------------------------------------------------------------
# Data models for e2e results
# ---------------------------------------------------------------------------

@dataclass
class AnswerJudgment:
    """LLM judge assessment of answer quality."""
    correctness: str  # "correct", "partially_correct", "incorrect", "not_answerable"
    correctness_score: float  # 0.0-1.0
    explanation: str
    covers_ground_truth: bool
    adds_unsupported_claims: bool


@dataclass
class E2EQueryResult:
    """Result of a single e2e evaluation query."""
    query: str
    dataset: str
    ground_truth_text: str

    # RAG pipeline results
    rag_answer: str = ""
    rag_claims_total: int = 0
    rag_citations_total: int = 0
    rag_citations_verified: int = 0
    rag_citations_partial: int = 0
    rag_citations_unverified: int = 0
    rag_gaps_found: int = 0
    rag_judgment: AnswerJudgment | None = None
    rag_elapsed: float = 0.0

    # Full context baseline results
    baseline_answer: str = ""
    baseline_judgment: AnswerJudgment | None = None
    baseline_elapsed: float = 0.0

    @property
    def rag_verification_rate(self) -> float:
        total = self.rag_citations_total
        if total == 0:
            return 0.0
        return (self.rag_citations_verified + self.rag_citations_partial) / total

    @property
    def rag_strict_verification_rate(self) -> float:
        total = self.rag_citations_total
        if total == 0:
            return 0.0
        return self.rag_citations_verified / total


@dataclass
class E2EResults:
    """Aggregated e2e evaluation results."""
    results: list[E2EQueryResult] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    total_elapsed: float = 0.0

    # --- RAG metrics ---

    @property
    def rag_correctness_rate(self) -> float:
        judged = [r for r in self.results if r.rag_judgment]
        if not judged:
            return 0.0
        correct = sum(
            1 for r in judged
            if r.rag_judgment.correctness in ("correct", "partially_correct")
        )
        return correct / len(judged)

    @property
    def rag_strict_correctness_rate(self) -> float:
        judged = [r for r in self.results if r.rag_judgment]
        if not judged:
            return 0.0
        return sum(1 for r in judged if r.rag_judgment.correctness == "correct") / len(judged)

    @property
    def rag_avg_verification_rate(self) -> float:
        rated = [r for r in self.results if r.rag_citations_total > 0]
        if not rated:
            return 0.0
        return sum(r.rag_verification_rate for r in rated) / len(rated)

    @property
    def rag_avg_strict_verification_rate(self) -> float:
        rated = [r for r in self.results if r.rag_citations_total > 0]
        if not rated:
            return 0.0
        return sum(r.rag_strict_verification_rate for r in rated) / len(rated)

    @property
    def rag_hallucination_rate(self) -> float:
        rated = [r for r in self.results if r.rag_citations_total > 0]
        if not rated:
            return 0.0
        return sum(r.rag_citations_unverified for r in rated) / sum(r.rag_citations_total for r in rated)

    # --- Baseline metrics ---

    @property
    def baseline_correctness_rate(self) -> float:
        judged = [r for r in self.results if r.baseline_judgment]
        if not judged:
            return 0.0
        correct = sum(
            1 for r in judged
            if r.baseline_judgment.correctness in ("correct", "partially_correct")
        )
        return correct / len(judged)

    @property
    def baseline_strict_correctness_rate(self) -> float:
        judged = [r for r in self.results if r.baseline_judgment]
        if not judged:
            return 0.0
        return sum(1 for r in judged if r.baseline_judgment.correctness == "correct") / len(judged)

    def to_dict(self) -> dict:
        results_list = []
        for r in self.results:
            d = {
                "query": r.query,
                "dataset": r.dataset,
                "ground_truth_text": r.ground_truth_text[:500],
                "rag": {
                    "answer": r.rag_answer,
                    "claims": r.rag_claims_total,
                    "citations_total": r.rag_citations_total,
                    "citations_verified": r.rag_citations_verified,
                    "citations_partial": r.rag_citations_partial,
                    "citations_unverified": r.rag_citations_unverified,
                    "verification_rate": round(r.rag_verification_rate, 3),
                    "gaps_found": r.rag_gaps_found,
                    "elapsed": r.rag_elapsed,
                },
                "baseline": {
                    "answer": r.baseline_answer,
                    "elapsed": r.baseline_elapsed,
                },
            }
            if r.rag_judgment:
                d["rag"]["judgment"] = {
                    "correctness": r.rag_judgment.correctness,
                    "score": r.rag_judgment.correctness_score,
                    "explanation": r.rag_judgment.explanation,
                    "covers_ground_truth": r.rag_judgment.covers_ground_truth,
                    "adds_unsupported_claims": r.rag_judgment.adds_unsupported_claims,
                }
            if r.baseline_judgment:
                d["baseline"]["judgment"] = {
                    "correctness": r.baseline_judgment.correctness,
                    "score": r.baseline_judgment.correctness_score,
                    "explanation": r.baseline_judgment.explanation,
                    "covers_ground_truth": r.baseline_judgment.covers_ground_truth,
                    "adds_unsupported_claims": r.baseline_judgment.adds_unsupported_claims,
                }
            results_list.append(d)

        return {
            "config": self.config,
            "total_elapsed": round(self.total_elapsed, 2),
            "summary": {
                "num_queries": len(self.results),
                "rag": {
                    "correctness_rate": round(self.rag_correctness_rate, 3),
                    "strict_correctness_rate": round(self.rag_strict_correctness_rate, 3),
                    "avg_verification_rate": round(self.rag_avg_verification_rate, 3),
                    "avg_strict_verification_rate": round(self.rag_avg_strict_verification_rate, 3),
                    "hallucination_rate": round(self.rag_hallucination_rate, 3),
                },
                "baseline": {
                    "correctness_rate": round(self.baseline_correctness_rate, 3),
                    "strict_correctness_rate": round(self.baseline_strict_correctness_rate, 3),
                },
            },
            "results": results_list,
        }


# ---------------------------------------------------------------------------
# LLM judge for answer correctness
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are an expert legal evaluator. Given a question, a ground truth passage \
(the correct source text), and a generated answer, assess the answer's quality.

Evaluate along these dimensions:
1. Does the answer correctly represent the information in the ground truth passage?
2. Does the answer add claims not supported by the ground truth?

Respond in exactly this format:
CORRECTNESS: <correct|partially_correct|incorrect|not_answerable>
SCORE: <0.0 to 1.0>
COVERS_GROUND_TRUTH: <true|false>
ADDS_UNSUPPORTED: <true|false>
EXPLANATION: <one sentence>"""

JUDGE_USER = """\
Question: {question}

Ground truth passage (the correct source text from the document):
\"\"\"{ground_truth}\"\"\"

Generated answer to evaluate:
\"\"\"{answer}\"\"\"

Assessment:"""


def judge_answer(
    question: str,
    ground_truth: str,
    answer: str,
    llm: LLMClient,
) -> AnswerJudgment:
    """Use LLM to judge answer correctness against ground truth."""
    import re

    prompt = JUDGE_USER.format(
        question=question,
        ground_truth=ground_truth[:3000],
        answer=answer[:3000],
    )

    try:
        response = llm.generate(
            system=JUDGE_SYSTEM,
            user=prompt,
            max_tokens=300,
        )

        # Parse response
        correctness = "incorrect"
        score = 0.0
        covers = False
        unsupported = False
        explanation = ""

        m = re.search(r"CORRECTNESS:\s*(correct|partially_correct|incorrect|not_answerable)", response, re.I)
        if m:
            correctness = m.group(1).lower()

        m = re.search(r"SCORE:\s*([\d.]+)", response)
        if m:
            score = min(1.0, max(0.0, float(m.group(1))))

        m = re.search(r"COVERS_GROUND_TRUTH:\s*(true|false)", response, re.I)
        if m:
            covers = m.group(1).lower() == "true"

        m = re.search(r"ADDS_UNSUPPORTED:\s*(true|false)", response, re.I)
        if m:
            unsupported = m.group(1).lower() == "true"

        m = re.search(r"EXPLANATION:\s*(.+?)(?:\n|$)", response)
        if m:
            explanation = m.group(1).strip()

        return AnswerJudgment(
            correctness=correctness,
            correctness_score=score,
            explanation=explanation,
            covers_ground_truth=covers,
            adds_unsupported_claims=unsupported,
        )

    except Exception as e:
        return AnswerJudgment(
            correctness="error",
            correctness_score=0.0,
            explanation=f"Judge error: {e}",
            covers_ground_truth=False,
            adds_unsupported_claims=False,
        )


# ---------------------------------------------------------------------------
# Full context baseline (no RAG — just pass the whole document)
# ---------------------------------------------------------------------------

BASELINE_SYSTEM = """\
You are a legal analyst. Answer the question based ONLY on the provided \
document. Cite specific sections or quote relevant passages to support \
your answer. If the document does not contain enough information, say so."""

BASELINE_USER = """\
Question: {question}

Document:
{document}

Answer:"""


def run_baseline(
    question: str,
    document_text: str,
    llm: LLMClient,
) -> tuple[str, float]:
    """Run the full-context baseline: whole document in prompt, no RAG."""
    # Truncate document if needed (stay under ~150K chars for safety)
    doc_text = document_text[:150000]

    prompt = BASELINE_USER.format(
        question=question,
        document=doc_text,
    )

    t0 = time.time()
    answer = llm.generate(
        system=BASELINE_SYSTEM,
        user=prompt,
    )
    elapsed = time.time() - t0

    return answer, elapsed


# ---------------------------------------------------------------------------
# Main e2e runner
# ---------------------------------------------------------------------------


def run_e2e_evaluation(
    queries: list[BenchmarkQuery],
    llm: LLMClient,
    run_baseline_comparison: bool = True,
    top_k: int = 20,
    progress_callback=None,
) -> E2EResults:
    """Run end-to-end evaluation on benchmark queries.

    For each query:
    1. Load the corpus document
    2. Run full RAG pipeline (parse → chunk → embed → retrieve → generate → verify)
    3. Optionally run full-context baseline
    4. Judge both answers against ground truth

    Args:
        queries: Benchmark queries with ground truth.
        llm: LLM client for generation and judging.
        run_baseline_comparison: Whether to run the full-context baseline.
        top_k: Chunks to retrieve per query.
        progress_callback: Optional callback(current, total, status_msg).

    Returns:
        E2EResults with per-query and aggregate metrics.
    """
    t0 = time.time()
    results: list[E2EQueryResult] = []

    # Group queries by corpus file for efficient ingestion
    file_queries: dict[str, list[tuple[int, BenchmarkQuery]]] = {}
    for i, bq in enumerate(queries):
        for s in bq.snippets:
            file_queries.setdefault(s.file_path, []).append((i, bq))

    # Process per corpus file, deduplicating queries
    processed_query_indices: set[int] = set()
    query_count = 0
    total = len(queries)

    for file_path, fq_list in file_queries.items():
        corpus_path = CORPUS_DIR / file_path
        if not corpus_path.exists():
            continue

        doc_text = corpus_path.read_text()
        doc_id = corpus_path.stem

        # Parse, chunk, embed, index this document
        root = parse_document(doc_text, doc_id=doc_id)
        chunks = chunk_document(root, doc_id=doc_id)

        if not chunks:
            continue

        embedding_texts = [c.embedding_text for c in chunks]
        embeddings = embed_texts(embedding_texts)

        index = DocumentIndex()
        index.build(chunks, embeddings)

        # Run each query against this document (skip already-processed)
        for qi, bq in fq_list:
            if qi in processed_query_indices:
                continue
            processed_query_indices.add(qi)
            query_count += 1
            dataset = bq.tags[0] if bq.tags else "unknown"

            # Get ground truth text
            gt_texts = []
            for s in bq.snippets:
                if s.file_path == file_path:
                    gt_text = doc_text[s.span[0]:s.span[1]]
                    gt_texts.append(gt_text)
            ground_truth = "\n\n".join(gt_texts)

            if progress_callback:
                progress_callback(query_count, total, f"[{dataset}] {bq.query[:50]}...")

            result = E2EQueryResult(
                query=bq.query,
                dataset=dataset,
                ground_truth_text=ground_truth,
            )

            # --- RAG pipeline ---
            try:
                rag_t0 = time.time()

                q_emb = embed_query(bq.query)
                retrieval_result = retrieve(
                    query=bq.query,
                    query_embedding=q_emb,
                    index=index,
                    doc_tree=root,
                    top_k=top_k,
                )

                raw_answer, claims = generate_answer(
                    question=bq.query,
                    ranked_chunks=retrieval_result.all_chunks,
                    llm=llm,
                )

                verified_claims = verify_claims(
                    claims=claims,
                    doc_tree=root,
                    ranked_chunks=retrieval_result.all_chunks,
                    llm=llm,
                    local_nli_model=local_nli,
                )

                result.rag_answer = raw_answer
                result.rag_elapsed = time.time() - rag_t0
                result.rag_claims_total = len(verified_claims)

                for vc in verified_claims:
                    for cit in vc.citations:
                        result.rag_citations_total += 1
                        if cit.status == VerificationStatus.VERIFIED:
                            result.rag_citations_verified += 1
                        elif cit.status == VerificationStatus.PARTIALLY_VERIFIED:
                            result.rag_citations_partial += 1
                        else:
                            result.rag_citations_unverified += 1

                result.rag_gaps_found = sum(1 for vc in verified_claims if vc.is_gap)

                # Judge RAG answer
                result.rag_judgment = judge_answer(
                    question=bq.query,
                    ground_truth=ground_truth,
                    answer=raw_answer,
                    llm=llm,
                )

            except Exception as e:
                result.rag_answer = f"ERROR: {e}"
                result.rag_elapsed = time.time() - rag_t0

            # --- Full context baseline ---
            if run_baseline_comparison:
                try:
                    baseline_answer, baseline_elapsed = run_baseline(
                        question=bq.query,
                        document_text=doc_text,
                        llm=llm,
                    )
                    result.baseline_answer = baseline_answer
                    result.baseline_elapsed = baseline_elapsed

                    result.baseline_judgment = judge_answer(
                        question=bq.query,
                        ground_truth=ground_truth,
                        answer=baseline_answer,
                        llm=llm,
                    )
                except Exception as e:
                    result.baseline_answer = f"ERROR: {e}"

            results.append(result)

    return E2EResults(
        results=results,
        config={
            "top_k": top_k,
            "run_baseline": run_baseline_comparison,
            "num_queries": len(queries),
        },
        total_elapsed=time.time() - t0,
    )
    "run_baseline": run_baseline_comparison,
            "num_queries": len(queries),
        },
        total_elapsed=time.time() - t0,
    )
