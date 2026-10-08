# LegalKit

A native Mac app for AI-powered contract analysis. Files never leave your machine.

## Status

**Archived — not maintained.** Development stopped 2026-04-17. Rust/Tauri native app, pivoted
from a Python/FastAPI web app on 2026-03-19. Kept as a record of the work; see the repository
README for what was and was not finished.

## Quick Navigation

### 🏗 Architecture
- [[architecture/adrs/index|ADR Index]] — all architecture decisions, current status
- [[architecture/adrs/adr-009-native-mac-app|ADR-009]] — why Rust/Tauri (most recent major decision)
- [[architecture/adrs/adr-010-local-models|ADR-010]] — bundled models, Ollama, API fallback
- [[architecture/adrs/adr-011-ui-design-system|ADR-011]] — three-panel layout, dark default, confidence UI

### 🔬 Research
- [[research/lodo-embedding-eval-findings|LODO Embedding Evaluation]] — the held-out result that retracted the 99.03% claim
- [[research/finetuned-model-evaluation|Fine-tuned Model Evaluation]] — legal-entailment on hand-built edge cases
- [[research/e2e-evaluation-findings|E2E Evaluation Findings]] — pipeline accuracy on LegalBench-RAG
- [[research/citation-accuracy-landscape|Citation Accuracy Landscape]] — competitive benchmarks

### 📚 Reference
- [[reference/glossary|Glossary]]
- [[reference/benchmarks|Benchmarks & Datasets]]
- [[reference/papers|Papers]] — the RL and retrieval reading list (links)

### 🗂 History
- [[history/EMBEDDING_TRAINING_SUMMARY|Embedding Training]] — how legal-embeddings-bge-base was built
- [[history/FRONTEND_IMPLEMENTATION_SUMMARY|Web Frontend]] — the superseded web app

## Key Numbers

In-distribution and held-out figures are kept apart on purpose — see
[[research/lodo-embedding-eval-findings|the LODO evaluation]] for why.

| Metric | Value | Evaluation |
|---|---|---|
| Retrieval R@64 | 95.5% held-out (99.03% in-distribution) | held-out is within noise of base BGE (−0.3pp); recall is a chunk-size effect |
| Retrieval R@16 | 79.8% held-out | the fine-tune is 0.7pp *worse* than base BGE |
| Retrieval P@1 | 1.6% held-out (10.93% in-distribution) | the real bottleneck; a cross-encoder reranker was the planned fix |
| legal-entailment 4-class accuracy | 75.8% fine-tuned vs 39.4% DeBERTa-base | 33 hand-built legal edge cases, in-distribution |
| End-to-end strict correctness | 65% RAG vs 55% full-context | 20 queries |
| Citation verification rate | 88% | 20 queries; 14% pass strict verification |
| Hallucination rate | 11% | 20 queries |
| Embedding model size | 110M params, ~440MB | not bundled in this repository |
| Chunk size (optimized) | 1,500 chars (~375 tokens) | |
| Default retrieval top-k | 20 chunks | |
| Flat-file store location | `~/.legalkit/docs/` | |
