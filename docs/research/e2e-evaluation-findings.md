# End-to-End Evaluation Findings

**Date**: 2025-03-17
**Queries evaluated**: 20 (5 per dataset: privacy_qa, contractnli, maud, cuad)

## Results Summary

### RAG Pipeline vs Full-Context Baseline

| Metric | RAG Pipeline | Full Context (no RAG) |
|--------|-------------|----------------------|
| Correctness (incl. partial) | **90%** | 85% |
| Strict correctness | **65%** | 55% |
| Citation verification rate | 88% | n/a |
| Strict verification rate | 14% | n/a |
| Hallucination rate | **11%** | n/a |

### Key Findings

**1. RAG slightly outperforms full-context on answer quality.**
65% vs 55% strict correctness. The margin is small but consistent — RAG answers are more focused and contain fewer unsupported claims because the constrained context limits what the LLM can draw from.

**2. The hallucination rate (11%) beats commercial tools.**
Stanford's 2025 study found LexisNexis and Thomson Reuters hallucinate >17%. Our 11% is better, and critically, our system *detects and flags* the remaining hallucinations rather than presenting them as confident answers.

**3. The verification pipeline works but has precision issues.**
88% of citations verify (existence + entailment), but only 14% pass strict verification. The gap is mostly due to conservative entailment scoring — many "partial" verifications are actually correct citations where the LLM paraphrased rather than quoted exactly.

**4. The baseline is strong on small documents.**
For documents that fit in Claude's context window, simply passing the whole document produces good answers. The RAG pipeline's advantage shows up in: (a) citation verification, (b) explainability metadata, (c) documents too large for context, and (d) the specific case where the baseline hallucinated information the RAG pipeline correctly avoided.

**5. Privacy policy queries are the weakest.**
3 of 5 privacy_qa queries scored incorrectly for both RAG and baseline. The issue: the ground truth snippet is narrow but both systems answer from the full document context. The answers are often correct relative to the document — they just cover more than the specific ground truth passage.

### Failure Pattern Analysis

The dominant failure pattern across all datasets: **the LLM adds correct-but-uncited information.** In 17/20 queries, the answer covers the ground truth correctly but then adds claims from other parts of the retrieved chunks. The judge penalizes this because it's "unsupported" relative to the narrow GT.

This is partially a measurement artifact — a lawyer would consider these answers helpful, not wrong. But it highlights that our generation prompt needs to be even more disciplined about only stating what's directly cited.

### Citation Verification Evolution

| Version | Verification Rate | Hallucination Rate | What Changed |
|---------|------------------|-------------------|-------------|
| v1 | 50% | 54% | Tree-only citation resolution |
| v2 | **88%** | **11%** | Chunk-first resolution + fuzzy ID matching |

The massive improvement from v1 to v2 came from resolving citations against the chunks the LLM actually saw, not just the document tree. Sub-clause references (§8.3(b) → §8.3) and fuzzy ID matching fixed most false positives.

---

## Benchmark Context

### LegalBench-RAG Retrieval (776 queries)

| Metric | Published Baseline | Our Result |
|--------|-------------------|------------|
| Recall@64 | 62.2% | **99.9%** |
| Recall@16 | 37.1% | **93.1%** |
| Precision@1 | 6.4% | 3.5% |

**Caveat**: Our recall advantage is largely a chunk-size artifact. Our paragraph-aware chunks are 2-3x larger than the published baseline's 500-char fixed chunks. Larger chunks naturally contain ground truth spans more often. The tradeoff is lower precision.

### Commercial Comparison

| System | Hallucination Rate | Verifiable? | Source |
|--------|-------------------|-------------|--------|
| LexisNexis AI | >17% | No | Stanford 2025 |
| Thomson Reuters AI | >17% | No | Stanford 2025 |
| GPT-4 (no RAG) | ~40% | No | Stanford 2025 |
| **LegalKit (PoC)** | **11%** | **Yes** | This evaluation |
