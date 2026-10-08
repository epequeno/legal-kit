# ADR-007: LegalBench-RAG as Primary Benchmark

## Status

Accepted

## Date

2025-03-17

## Context

To validate our citation engine, we need an objective, reproducible evaluation framework. Several benchmarks exist:

| Benchmark | Focus | Size | Legal-Specific | Open Source |
|-----------|-------|------|----------------|-------------|
| **LegalBench-RAG** | Retrieval precision/recall in legal domain | 6,858 queries / 79M chars | ✅ | ✅ |
| **LegalBench** | LLM reasoning on legal tasks | 162 tasks | ✅ | ✅ |
| **ALCE** | Citation quality (fluency, correctness, citation) | 3 datasets | ❌ | ✅ |
| **CRAG** | Comprehensive RAG accuracy | Large | ❌ | ✅ |
| **RAGBench** | RAG with TRACe evaluation | 100K examples | ❌ | ✅ |

Key considerations:
- We need to evaluate **retrieval quality** separately from **generation quality**
- Legal-domain evaluation is critical because general-purpose rerankers actually hurt legal retrieval performance (LegalBench-RAG finding)
- We need character-level precision, not just document-level recall
- A mini variant for rapid iteration is essential during development

## Decision

### Primary Benchmark: LegalBench-RAG

Use **LegalBench-RAG** as our primary evaluation framework, specifically:

1. **LegalBench-RAG-mini** (776 queries) for development iteration
2. **Full LegalBench-RAG** (6,858 queries) for final validation

**Why**:
- Only benchmark that evaluates retrieval at **character-level span precision** — exactly what we need for citation accuracy
- Covers four legal datasets: CUAD (contracts), MAUD (M&A), ContractNLI (NDAs), PrivacyQA (privacy policies)
- Human-annotated ground truth by legal experts
- Published baselines to compare against (Precision@k, Recall@k at k=1,2,4,8,16,32,64)
- Open source with reproducible evaluation code
- Mini variant enables fast iteration during development

### Secondary Benchmark: ALCE (for generation quality)

Use **ALCE** to evaluate citation quality in generated responses:
- Citation recall: % of claims that have at least one supporting citation
- Citation precision: % of citations that actually support their claim
- Fluency and correctness metrics

### Custom Evaluation: End-to-End Verification

In addition to standard benchmarks, measure our verification pipeline's accuracy:
- **Verification true positive rate**: When verification says "verified", is the citation actually correct?
- **Verification true negative rate**: When verification flags a problem, is there actually a problem?
- Requires a hand-labeled subset (~100 queries) with claim-level annotations

### Metrics We Report

#### Retrieval
- Precision@k and Recall@k at k=1,2,4,8,16,32,64
- Mean Reciprocal Rank (MRR)
- Per-dataset breakdown (CUAD, MAUD, ContractNLI, PrivacyQA)

#### Generation
- Citation recall (% of claims with supporting citation)
- Citation precision (% of citations that support their claim)
- ALCE fluency and correctness scores

#### Verification
- Verification accuracy (TP, TN, FP, FN rates)
- Confidence calibration (do confidence scores predict actual accuracy?)

#### Targets

| Metric | Published Baseline | Our Target |
|--------|-------------------|------------|
| Retrieval Precision@1 | 6.4% | >15% |
| Retrieval Recall@64 | 62.2% | >80% |
| Citation verification pass rate | ~83% (commercial tools) | >85% |
| End-to-end hallucination-free | 63% (CRAG SOTA) | >85% |

## Alternatives Considered

### Build our own benchmark
Rejected for PoC. Too expensive, not reproducible by others, and defeats the purpose of objective evaluation. May contribute additional legal domain test cases to LegalBench-RAG later.

### Use only CRAG or RAGBench
Rejected. These are general-purpose and don't test legal-specific challenges (terminology, cross-references, structural complexity).

### Skip benchmarks, just demo with sample documents
Rejected. Qualitative demos are misleading — they only show the cases that work. Quantitative benchmarks expose weaknesses and give us honest metrics.

## Consequences

### Positive
- Objective, reproducible evaluation from day one
- Published baselines provide clear targets
- Mini variant enables fast development iteration
- Results are directly comparable to other legal AI systems
- Quantitative evidence for marketing and investor conversations

### Negative
- LegalBench-RAG only covers contracts and policies, not litigation or case law
- Benchmark performance may not perfectly predict real-world user satisfaction
- Character-level evaluation is strict — our system may be "useful" even when precision scores are low
- Running full benchmark takes time and API costs

### Process
- Run LegalBench-RAG-mini after every significant pipeline change
- Run full LegalBench-RAG before any milestone/release
- Store all benchmark results in `eval/` with timestamps for tracking progress
- Include ablation results (with/without each pipeline component) to understand contribution of each layer
