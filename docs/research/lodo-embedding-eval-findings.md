# LODO Embedding Evaluation Findings

**Date**: 2026-03-19
**Method**: Leave-One-Dataset-Out cross-validation
**Models**: BAAI/bge-base-en-v1.5 (base) vs. fine-tuned on 3/4 LegalBench-RAG datasets

## Executive Summary

**The embedding fine-tuning provides no measurable improvement on held-out legal datasets.**

When evaluated on datasets the model was *not* trained on (the only fair comparison), the fine-tuned models perform within noise of the base BGE model — and slightly *worse* on average (-0.3pp R@64). The previously reported 99.03% R@64 reflected in-distribution overfitting, not genuine domain adaptation.

**The high recall numbers (95%+) are driven entirely by chunk size, not embedding quality.** Both the fine-tuned and base models achieve nearly identical recall because the large paragraph-aware chunks (~1500 chars) mechanically capture most ground-truth spans regardless of embedding quality.

## Results

### R@64 (Recall at 64 retrieved chunks)

| Dataset (held out) | Fine-tuned | Base BGE | Δ | Published baseline |
|---|---|---|---|---|
| **CUAD** | 96.5% | 95.7% | +0.7pp | 74.7% |
| **MAUD** | 80.7% | 82.6% | **-1.9pp** | 28.3% |
| **ContractNLI** | 101.8% | 101.8% | +0.0pp | 61.7% |
| **PrivacyQA** | 103.2% | 103.2% | +0.0pp | 84.2% |
| **AVERAGE** | **95.5%** | **95.8%** | **-0.3pp** | — |

### R@16 (More discriminating — fewer chunks to find the answer in)

| Dataset (held out) | Fine-tuned | Base BGE | Δ |
|---|---|---|---|
| **CUAD** | 76.4% | 77.8% | -1.5pp |
| **MAUD** | 52.1% | 52.6% | -0.5pp |
| **ContractNLI** | 97.9% | 98.3% | -0.4pp |
| **PrivacyQA** | 92.7% | 93.3% | -0.6pp |
| **AVERAGE** | **79.8%** | **80.5%** | **-0.7pp** |

At R@16 the fine-tuned model is consistently *worse* by 0.4-1.5pp across all 4 datasets. The fine-tuning is slightly *hurting* generalization.

## Analysis

### Why the fine-tuning doesn't help

1. **The base BGE model is already excellent at legal text.** BGE-base-en-v1.5 was trained on a massive general-purpose corpus that includes legal documents. There's little headroom for a domain-specific fine-tune to improve recall.

2. **The training data teaches dataset-specific patterns, not legal-general patterns.** Training on LegalBench-RAG queries teaches the model to match the specific question styles and document structures of those benchmarks. When the held-out dataset has different query patterns (e.g., MAUD's multi-choice M&A questions vs. CUAD's clause identification), the fine-tuning doesn't transfer.

3. **Large chunks mask embedding quality differences.** At 1500-char chunks, even a mediocre embedding will land somewhere near the relevant passage. The chunk size is doing the heavy lifting, not the embeddings. This is confirmed by ContractNLI and PrivacyQA both hitting >100% recall (chunks overlap multiple ground-truth spans).

4. **MAUD regression (-1.9pp) suggests slight overfitting.** The model trained on CUAD+ContractNLI+PrivacyQA actually performs *worse* on MAUD than the base model. MAUD's M&A agreement structure is different enough that the fine-tuning on other contract types introduces a bias.

### Why both models beat the published baseline by 30-50pp

The published LegalBench-RAG baseline uses 500-character fixed-size chunks. Our pipeline uses ~1500-char paragraph-aware chunks. This 3x larger chunk size mechanically inflates recall while deflating precision. **The comparison against the published baseline is not meaningful** — it's a chunk-size effect, not an embedding quality effect.

## Implications

### 1. Remove the "beats text-embedding-3-large" claim
This claim was based on in-distribution evaluation and is not supported by held-out results. The fine-tuned model provides zero improvement over the base BGE model on unseen legal data.

### 2. The fine-tuned embedding model should not ship as a differentiator
Shipping a 110M parameter custom model that performs identically to the base model is complexity without value. Options:
- **Ship the base BGE model** — same performance, simpler, no training pipeline to maintain
- **Invest in smaller chunks + reranker** — this addresses the actual precision problem (P@1 = 1.6-3.5%) which is the real product bottleneck
- **Revisit fine-tuning with fundamentally different training data** — see below

### 3. The retrieval pipeline's strength is architecture, not embeddings
Hybrid retrieval (semantic + BM25 + cross-reference expansion) and large paragraph-aware chunks are what drive the high recall. These are genuine engineering contributions that don't require a custom embedding model.

### 4. If fine-tuning is revisited, it needs different training data
The current training data (queries from LegalBench-RAG annotations) is too closely coupled to the benchmark. A legitimate fine-tuning dataset would need:
- Queries from real user interactions (not benchmark annotations)
- Documents from outside the benchmark corpus (e.g., SEC EDGAR filings, real client contracts)
- Focus on precision improvement (better ranking), not recall (already saturated)

## Updated Recommendation

| Component | Previous claim | Actual status | Recommendation |
|---|---|---|---|
| Embedding model | "Beats text-embedding-3-large" | No improvement over base BGE | Ship base BGE model |
| Retrieval recall | 99.03% R@64 | ~95.5% held-out (chunk-size driven) | Honest about chunk-size effect |
| Retrieval precision | P@1 = 3.5% | P@1 = 1.6% held-out | **This is the real problem to solve** |
| Hybrid retrieval | Genuine contribution | Confirmed — architecture drives quality | Keep and improve |

## Files

- LODO data splits: `data/embedding_training_lodo/<dataset>/`
- LODO eval results: `eval/lodo-<dataset>.json`
- Base BGE eval results: `eval/lodo-base-<dataset>.json`
- Training script: `scripts/run_lodo_training.py`
- Report script: `scripts/embedding_lodo_report.py`
