# Document Structure Model: Research Findings

**Date**: 2026-03-18  
**Sources**: NotebookLM research — 20 papers on legal document structure, document layout analysis, and hierarchical document parsing  
**Notebook**: `legal-doc-structure`

---

## Executive Summary

The key finding: **don't build a separate "document structure model" from scratch.** The right approach is to use the regex parser we already have as a *weak supervisor* to generate training labels automatically from the 462 CUAD contracts, then fine-tune a lightweight transformer on those labels. The transformer learns to generalize from the regex's good outputs, so it works even when documents deviate from the patterns the regex was written for.

This converts a multi-month ML project into a ~2-week data pipeline + fine-tune job.

---

## Problem Statement

The current regex parser (`src/legalkit/pipeline/parse.py`) works well on well-formatted contracts with standard section numbering. It breaks on:
- Non-standard numbering (roman numerals inline, mixed numbering)
- Compressed/single-paragraph formats (common in SaaS agreements)
- Exhibit headers embedded mid-document
- NDA clause structures without numeric section numbering
- Multi-column layouts (post-OCR text)

The goal: a model that detects the structural hierarchy of any legal contract, regardless of formatting — and outperforms the regex parser on the LegalBench-RAG corpus.

---

## Recommended Architecture: Two-Stage BIO Tagging

Based on the research, the best approach for plain-text contract parsing is:

### Stage 1: Heading Detection (token classification)
Fine-tune a transformer to classify each line/token as a heading or non-heading, then assign the heading type:

```
B-ARTICLE_HEADER    — "ARTICLE IV - INDEMNIFICATION"
I-ARTICLE_HEADER
B-SECTION_HEADER    — "Section 4.2 Cap on Liability"  
I-SECTION_HEADER
B-SUBSECTION_HEADER — "4.2.1 General Rule"
I-SUBSECTION_HEADER
B-PREAMBLE          — "This Agreement is entered into..."
I-PREAMBLE
B-RECITAL           — "WHEREAS, Seller desires to sell..."
I-RECITAL
B-EXHIBIT_HEADER    — "EXHIBIT A — BILL OF SALE"
I-EXHIBIT_HEADER
O                   — everything else (body text, blank lines, etc.)
```

### Stage 2: Hierarchy Assembly (rule-based, post-processing)
Once heading boundaries are identified, the hierarchy is assembled by:
1. Inserting each heading at its correct depth based on numbering type (Article > Section > Subsection) — the HELD "root-to-leaf" traversal pattern
2. Attaching non-heading body text as children of their preceding heading
3. Detecting definition blocks via the existing `_DEFINITION_RE` regex (this still works well and doesn't need ML)

This matches the TocBERT approach (heading NER) combined with HELD-style tree assembly.

---

## Why Not End-to-End ML?

Two approaches rejected:
1. **Full LayoutLM/LayoutLMv3** — requires bounding boxes from PDF OCR. Our corpus is plain text, so 2D position embeddings add no value. Overkill.
2. **LLM-based segmentation (LLaMA-3.1-8B)** — achieves ROUGE F1=0.95 on NDAs but requires GPU at inference, ~500ms per doc, and $0.01+ per call. Too slow/expensive for pipeline use at scale.

The transformer NER approach (legal-roberta-base or similar, ~125M params) runs on CPU at ~200ms/doc with 75-85% F1 on hierarchical detection — sufficient to handle the edge cases the regex parser misses.

---

## Training Data Strategy: Weak Supervision

This is the key insight from the research. **TocBERT** was trained exactly this way:

1. **Generate silver labels**: Run the existing regex parser over all 462 CUAD contracts. Where the regex fires confidently (e.g., matches `_ARTICLE_RE`, `_SECTION_RE`, `_SUBSECTION_RE`), auto-label those tokens with BIO tags. This gives ~hundreds of thousands of labeled tokens essentially for free.

2. **Create gold validation/test set**: Manually review and correct 40-50 contracts with difficult formatting (the ones where the regex parser currently fails). These become the held-out test set that actually measures improvement over the baseline.

3. **Fine-tune transformer**: Train on silver labels, evaluate on gold. The model learns the contextual patterns behind the regex's correct outputs and generalises.

**Why this works**: The regex parser is ~95% accurate on well-formatted documents (CUAD's standard contracts). Fine-tuning a transformer on the regex outputs forces it to learn *semantic* features — "this text looks like a section header because of its structural position, capitalization, and the fact it's followed by body text" — rather than exact pattern matching. The trained transformer then correctly identifies headers in contracts where the regex would fail.

---

## State-of-the-Art Benchmarks

| Task | Model | Metric | Score |
|---|---|---|---|
| NDA clause segmentation | LLaMA-3.1-8B | ROUGE F1 | 0.95 ± 0.004 |
| Section heading detection | TocBERT (BERT-based) | F1 (hierarchical) | 72.8% |
| Section heading detection | TocBERT (BERT-based) | F1 (linear) | 84.6% |
| Hierarchical structure | MTD (multimodal) | F1 | 88.1% |
| Hierarchical structure | HELD | Path accuracy | 0.73 (English fin.) |
| Legal sentence boundary | NUPunkt (statistical) | Precision | 91.1% |
| Legal sentence boundary | CharBoundary (RF) | F1 | 0.782 |

**Our target**: 80%+ F1 on section heading detection across diverse contract types. TocBERT hits 84.6% on linear detection with a base BERT model — achievable with legal-domain pre-training.

---

## Model Choice: legal-roberta-base

Best fit for our constraints:
- **legal-roberta-base** (HuggingFace: `lexlms/legal-roberta-base`) — pre-trained on legal domain text (contracts, statutes, case law)
- 125M parameters — runs on CPU at ~200ms/doc
- Trained on CUAD, ECtHR, and US case law corpora — already knows legal terminology
- Tested and confirmed effective on CUAD clause classification (F1: 0.85 in NDA classification task)

Alternative: `nlpaueb/legal-bert-base-uncased` — similar performance, slightly less legal domain pre-training. Either works.

---

## Relevant Datasets

| Dataset | Size | What it provides |
|---|---|---|
| **CUAD** | 462+ contracts, 41 clause types | Clause span annotations — use as silver label source |
| **MAUD** | 152 merger agreements, 47K+ labels | M&A structural annotations |
| **Kleister-NDA** | 726 NDAs | NDA segmentation — directly usable |
| **LegalBench-RAG corpus** | 79M chars, 60+ contracts | Our primary eval target |
| **MultiLegalSBD** | 130K+ sentences | Sentence boundary training data |

We already have CUAD and LegalBench-RAG locally (`data/corpus/`). Kleister-NDA and MAUD are available on HuggingFace.

---

## Implementation Plan

### Phase A: Training Data Generation (Week 1)

1. **`src/legalkit/eval/structure_annotator.py`** — runs the regex parser over a corpus of contracts and produces BIO-labelled token sequences in JSONL format. Each line: `{text, tokens, labels, doc_id, contract_type}`.

2. Run over all 462 CUAD contracts + MAUD contracts + ContractNLI NDAs → target ~5,000+ labelled documents.

3. **Gold test set**: Select 50 contracts that the regex parser struggles with (non-standard formatting), manually verify their labels. These are excluded from training.

### Phase B: Model Training (Week 1-2)

4. **`src/legalkit/eval/structure_model.py`** — fine-tuning pipeline using `legal-roberta-base` with token classification head (BIO labels). Modal A10G, ~30 min training.

5. Evaluate on gold test set: target F1 ≥ 80% on heading detection.

### Phase C: Integration (Week 2)

6. **`src/legalkit/pipeline/parse_ml.py`** — drop-in replacement for the regex-heavy sections of `parse.py`, using the fine-tuned model for heading detection, with regex fallback if model confidence is low.

7. Run LegalBench-RAG mini benchmark with ML parser vs regex parser to measure improvement on retrieval recall.

---

## Key Tradeoffs vs Current Regex Approach

| Dimension | Regex Parser | ML Parser |
|---|---|---|
| Well-formatted contracts | ✓ ~95% accurate | ✓ ~90-95% (may be slightly worse — learns from regex) |
| Non-standard formatting | ✗ fails silently | ✓ ~80%+ generalisation |
| New contract types | ✗ needs new patterns | ✓ generalises |
| Speed | ✓ <5ms/doc | ~ 150-200ms/doc on CPU |
| Interpretability | ✓ fully transparent | ~ confidence scores available |
| Dependency | ✓ no GPU needed | ✓ no GPU needed (CPU inference) |

**Decision**: Run both in parallel. Use regex parser as primary, ML parser as fallback for documents where the regex returns fewer than N nodes (indicating parsing failure). Eventually ML parser becomes primary after evaluation confirms quality.

---

## References (from NotebookLM)

- Cao et al. (2022). *Extracting Variable-Depth Logical Document Hierarchy from Long Documents: HELD.* JCST.
- Hu et al. *Multimodal Tree Decoder for Table of Contents Extraction.* arXiv.
- Baghdadi et al. *TocBERT: Medical Document Structure Extraction Using Bidirectional Transformers.* arXiv.
- Bhattacharya et al. *Semantic Segmentation of Legal Documents via Rhetorical Roles.* ACL.
- Koreeda & Manning. *Structural Text Segmentation of Legal Documents.* arXiv.
- De Araujo et al. (2025). *Two-Stage Architecture for NDA Analysis: LLaMA Segmentation + Legal-RoBERTa Classification.* arXiv.
- Pfitzmann et al. (2022). *DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis.*
- Xu et al. (2020). *LayoutLM: Pre-training of Text and Layout for Document Image Understanding.*

---

*See also: [[benchmark-results]], [[finetuned-model-evaluation]], [[adr-007-benchmark-strategy]]*
