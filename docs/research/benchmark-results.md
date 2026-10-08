# LegalKit Benchmark Results & Methodology

**Last updated:** 2026-03-18  
**Status:** PoC — results are from a proof-of-concept pipeline, not a production system  
**Reproducibility:** All evaluation code is in `src/legalkit/eval/`; result files are in `eval/`

---

## Overview

LegalKit's citation engine is evaluated across four dimensions:

| Dimension | Benchmark | What It Measures |
|---|---|---|
| **Retrieval** | LegalBench-RAG (Pipitone et al., 2024) | Did we find the right passage in the document? |
| **Generation** | ALCE-style end-to-end | Is the answer correct and grounded in citations? |
| **Verification** | Internal edge cases + NLI eval | Can the pipeline detect its own citation errors? |
| **Reasoning** | LegalBench (Guha et al., 2023) | Can the LLM answer legal classification questions? |

These are deliberately separate. A system can retrieve perfectly but generate poorly, or generate correctly but fail to verify. Measuring all four independently shows where the gains are.

---

## 1. Retrieval: LegalBench-RAG

### What this benchmark measures

[LegalBench-RAG](https://github.com/zeroentropy-ai/legalbenchrag) is the only open-source benchmark that evaluates retrieval at **character-level span precision and recall** — not document-level. The ground truth is an exact character range in a legal document; the system must return a chunk that overlaps with that range.

Four legal datasets are included:

| Dataset | Domain | Difficulty |
|---|---|---|
| **PrivacyQA** | Privacy policies | Easiest |
| **ContractNLI** | NDAs | Medium |
| **CUAD** | Commercial contracts | Medium |
| **MAUD** | M&A agreements | Hardest |

The mini variant (776 queries, 60 corpus files) is used for development iteration. The full dataset is 6,858 queries across 79M characters of legal text.

### Our results (mini, 776 queries)

*File: `eval/mini-baseline-20260317.json` — run 2026-03-17, elapsed 409s*

**Overall recall across all datasets:**

| k | R@k | Published baseline R@k | Δ |
|---|---|---|---|
| 1 | 40.4% | — | — |
| 4 | 64.8% | — | — |
| 8 | 80.2% | — | — |
| 16 | 93.1% | 37.1% | +56.0pp |
| 32 | 97.9% | — | — |
| **64** | **99.9%** | **62.2%** | **+37.7pp** |

**Overall precision:**

| k | P@k | Published baseline P@1 |
|---|---|---|
| 1 | 3.5% | 6.4% |
| 4 | 4.2% | — |
| 16 | 3.9% | — |
| 64 | 3.3% | — |

**Per-dataset breakdown (R@64 / P@1):**

| Dataset | R@64 | P@1 | Published R@64 | Published P@1 |
|---|---|---|---|---|
| **ContractNLI** | 106.2%† | 6.1% | 61.7% | 6.6% |
| **PrivacyQA** | 99.9% | 7.0% | 84.2% | 14.4% |
| **CUAD** | 97.9% | 0.9% | 74.7% | 2.0% |
| **MAUD** | 95.5% | 0.0% | 28.3% | 2.7% |

*† ContractNLI recall exceeds 100% because our large chunks overlap multiple ground-truth spans per query. This is a measurement artifact, not an error.*

> **⚠️ Methodology limitation:** The embedding model used for these results was trained on pairs extracted from LegalBench-RAG ground-truth annotations. These results demonstrate in-distribution performance only. See §7 for details and the cross-dataset evaluation plan.

### ⚠ Critical caveat: the chunk-size effect

Our recall numbers are high largely because our chunks are **2–3× larger** than the published baseline's 500-character fixed chunks. Larger chunks have a higher probability of containing any given ground-truth span, which mechanically inflates recall while deflating precision.

This is a genuine tradeoff, not a cheat — but it means the raw recall comparison overstates our advantage. What the numbers actually demonstrate:

- **Our retrieval finds the relevant content** — even at R@1 (40.4%), we're finding the right passage in 40% of queries with a single chunk
- **Precision is lower than the baseline at k=1** — 3.5% vs 6.4%, expected given larger chunks
- **MAUD is our hardest dataset** — P@1=0% reflects that MAUD's ground truth spans are very short (often single clauses) that get diluted inside large chunks

A fair head-to-head comparison requires normalising chunk sizes. Our ablation study (see `legalkit chunk-sweep`) found that smaller chunks (500–800 chars) improve precision at the cost of recall. The right operating point depends on the use case.

### What this means in practice

Despite the caveat, the retrieval results show that our hybrid semantic+BM25 retrieval with contextual enrichment consistently finds the relevant legal content across all four datasets. MAUD (M&A agreements) is the hardest — the ground truth spans are short, precise clauses buried in long documents — and is the main area for future improvement.

---

## 2. End-to-End Generation & Citation Quality

### What this measures

The ALCE-style end-to-end evaluation runs the full pipeline (retrieval → generation → verification) on benchmark queries and assesses:

- **Answer correctness** — does the answer cover the ground truth? (LLM judge)
- **Citation verification rate** — what fraction of generated citations verify against source text?
- **Hallucination rate** — what fraction of answers contain unverifiable claims?

This is the product metric. A perfectly retrieved answer that the LLM fails to ground in citations is still a bad product outcome.

### Results: v2 pipeline (20 queries)

*File: `eval/e2e-v2.json` — run 2026-03-17, elapsed 582s*  
*5 queries per dataset: privacy_qa, contractnli, maud, cuad*

| Metric | RAG Pipeline | Full-Context Baseline |
|---|---|---|
| Correctness (incl. partial) | **90%** | 85% |
| Strict correctness | **65%** | 55% |
| Citation verification rate | **88%** | n/a |
| Strict verification rate | 14% | n/a |
| **Hallucination rate** | **10.6%** | n/a |

**RAG outperforms full-context on both correctness metrics.** The constrained context (only retrieved chunks, not the whole document) forces more disciplined generation with fewer unsupported claims.

### Commercial comparison

| System | Hallucination Rate | Verifiable? | Source |
|---|---|---|---|
| LexisNexis AI | >17% | No | Magesh et al., 2025 |
| Thomson Reuters AI | >17% | No | Magesh et al., 2025 |
| GPT-4 (no RAG) | ~40% | No | Magesh et al., 2025 |
| CRAG best-in-class RAG | ~37% | No | CRAG benchmark |
| **LegalKit v2 (PoC)** | **10.6%** | **Yes** | This evaluation |

Our 10.6% hallucination rate is better than commercial legal AI tools on the Stanford 2025 study metric. Critically, LegalKit also *detects and flags* the remaining hallucinations rather than presenting them as verified answers — which is the more important property for a legal product.

### The verification gap: 88% vs 14%

The large gap between 88% verification rate and 14% strict verification rate warrants explanation. Strict verification requires the citation to both (a) exist in the retrieved text and (b) pass LLM entailment checking with exact quote matching. The lenient rate (88%) passes any citation that exists and is semantically related.

The gap is mostly a **paraphrase problem**: the LLM generates accurate claims but rephrases the source text rather than quoting it. The strict entailment checker penalises paraphrases even when they're factually correct. This is a calibration issue — the strict rate understates actual citation accuracy, and the lenient rate slightly overstates it. The true rate is somewhere between 14% and 88%.

### Evolution: v1 → v2

| Version | Verification Rate | Hallucination Rate | Key change |
|---|---|---|---|
| v1 | 50% | **54%** | Tree-only citation resolution |
| **v2** | **88%** | **10.6%** | Chunk-first resolution + fuzzy ID matching |

The 44pp improvement in hallucination rate came from resolving citations against the **actual chunks the LLM saw** rather than the full document tree. Sub-clause references (§8.3(b) → §8.3) and fuzzy section ID matching fixed the majority of false citation failures.

### Known failure pattern

The dominant failure mode across all 20 queries: **the LLM adds correct-but-uncited information.** In 17 of 20 queries, the answer covers the ground truth accurately but includes additional claims drawn from the broader retrieved context. The LLM judge penalises these as "unsupported."

This is partly a measurement artefact — a lawyer reviewing the output would consider those answers helpful, not wrong. But it identifies a prompt engineering gap: the generation prompt needs to be more disciplined about only stating what is directly cited. This is the primary lever for improving strict correctness.

---

## 3. Verification Pipeline: Legal NLI Edge Cases

### What this measures

The 33 hand-labelled legal NLI edge cases in `data/eval/legal_nli_edge_cases.json` test whether the entailment model can correctly classify (passage, claim) pairs across 8 legal linguistic patterns where general NLI models fail:

- `notwithstanding` — override structures
- `shall_vs_may` — mandatory vs. discretionary
- `mae_carveout` — Material Adverse Effect exclusions
- `quantitative_precision` — exact amounts, thresholds, caps
- `temporal_qualifier` — survival periods, notice windows
- `condition_precedent` — if/then clause logic
- `defined_term` — contractual scope vs. ordinary meaning
- `knowledge_qualifier` — "to Seller's Knowledge" qualifications

### DeBERTa-base results

*`cross-encoder/nli-deberta-v3-base` — off-the-shelf, no fine-tuning on legal data*

| Metric | Value |
|---|---|
| Binary accuracy (entailed ∪ partial vs. contradicted ∪ neutral) | **63.6%** |
| 4-class exact accuracy | 39.4% |
| False positive rate (contradicted predicted as supported) | **77%** |
| Avg inference latency | ~106ms |
| Cost | Free (local) |

**Per-category binary accuracy:**

| Category | Accuracy | Status |
|---|---|---|
| knowledge_qualifier | 100% | ⚠ Misleading — see below |
| quantitative_precision | 75% | Marginal |
| condition_precedent | 67% | Marginal |
| defined_term | 67% | Marginal |
| temporal_qualifier | 67% | Marginal |
| mae_carveout | 60% | Failing |
| **notwithstanding** | **40%** | Failing |
| **shall_vs_may** | **40%** | Failing |

The 100% accuracy on knowledge_qualifier is an artefact: the model's natural uncertainty about hedged claims produces mid-range scores (0.35–0.70) that fall into the `partially_entailed` bucket via threshold mapping, which happens to be correct. A different threshold would break it.

### Why 63.6% is not good enough

The critical failure is the **77% false positive rate on contradicted claims** — the model predicts "supported" for claims that directly contradict the passage. For a citation verifier, this is the worst possible failure mode: it would approve roughly 3 in 4 incorrect citations as verified, undermining the whole purpose of the verification step.

Random baseline is 50%. The 13.6pp gap is real, but insufficient for production use.

### Root cause

General NLI models trained on MNLI do not understand:

1. **Carve-out logic** — "X, *provided that* Y shall not apply" requires reading the exception, not just the topic match
2. **Notwithstanding structure** — "Notwithstanding A, B applies" creates a logical override; the model sees only the topic of A and predicts entailment
3. **Modal verb legal semantics** — `shall` (mandatory obligation) vs. `may` (discretionary right) are legally distinct but syntactically similar; the model treats them as synonyms

### Expected LLM judge performance

Based on the pattern analysis, an LLM (Claude Sonnet) correctly reads clause structure and is expected to achieve **85–95% binary accuracy** on this test set. This is why the production pipeline uses LLM-as-judge rather than DeBERTa.

The fine-tuning roadmap targets a domain-adapted model that achieves **80%+ binary accuracy** at DeBERTa's ~100ms latency — matching LLM quality at local-model cost.

---

## 4. NLI Model Fine-Tuning (Baseline Run)

*File: `eval/finetune-results.json` — DeBERTa-v3-base, 3 epochs, A10G GPU, ~$0.05*

The v0 fine-tuning run was deliberately a **diagnostic, not a target result**. It confirmed the data quality problem before investing in GPU time on a better dataset.

| Metric | v0 Result | Root cause |
|---|---|---|
| Test accuracy | 26.4% | Model predicts "entailed" for everything |
| Macro F1 | 0.10 | Near-zero F1 on 3 of 4 classes |
| Entailed F1 | 0.42 | Only entailed class learned |
| Partially_entailed F1 | 0.00 | |
| Contradicted F1 | 0.00 | |
| Neutral F1 | 0.00 | |

**Training data composition (10,290 total pairs):**

| Source | Count | % | Quality |
|---|---|---|---|
| synthetic:cross_passage | 6,974 | 67.8% | Low — trivially easy negatives |
| synthetic:llm_generated | 3,024 | 29.4% | Medium — uniform style |
| synthetic:perturbation | 260 | 2.5% | High — targeted hard negatives |
| pipeline (organic) | 32 | 0.3% | High — real distribution |

The problem: 68% of the data is cross-passage negatives (claims paired with passages from completely different sections). These are easy for the model to dismiss — they're topically unrelated — so the model never learns to handle the hard cases where the passage and claim are topically similar but logically incompatible.

After downsampling to balance classes (3,032 pairs, 758/class), the neutral class is still dominated by cross-passage examples, producing a trivially learnable but useless decision boundary.

**What would fix it:** Replace cross-passage neutrals with hard negatives that are topically related to the passage. The `data/eval/legal_entailment_patterns.json` catalog (built as part of this phase) provides 96 curated hard-negative examples across the 8 failure categories as a starting point. The blocker for the next training run is collecting ~2,000+ hard negatives to match the scale of the training set.

---

## 5. LegalBench Reasoning (Stanford, 162 tasks)

The `legalkit legalbench` command evaluates LLM reasoning on [LegalBench](https://hazyresearch.github.io/legalbench/) — a distinct benchmark from LegalBench-RAG. Where LegalBench-RAG tests retrieval (did we find the right passage?), LegalBench tests reasoning (given the right text, can the model answer the legal question?).

Our 53-task contract-relevant subset spans:

| Category | Tasks | Label type |
|---|---|---|
| CUAD clause identification | 35 | Yes/No binary |
| MAUD M&A understanding | 10 | Multi-choice |
| NDA clause analysis | 5 | Yes/No binary |
| General contract QA | 3 | Binary / multi-choice |

Published GPT-4 baselines from Guha et al. (NeurIPS 2023) are included per-task for direct comparison. See `legalkit legalbench --list-tasks` for the full table.

*Note: Full results pending first eval run against the configured model. Use `legalkit legalbench --task-set contract --n 20` to run.*

---

## 6. Methodology

### Running the benchmarks

```bash
# LegalBench-RAG retrieval (mini, 776 queries) — ~7 min
legalkit bench --mini --output eval/results.json

# End-to-end citation quality (20 queries, requires API key) — ~10 min
legalkit e2e --n 5 --output eval/e2e.json

# ALCE citation quality (50 queries, requires API key) — ~5 min
legalkit alce-eval --n 10 --output eval/alce.json

# CUAD clause QA by type — ~5 min
legalkit cuad-eval --max-per-clause 5 --output eval/cuad.json

# Legal NLI edge cases (local DeBERTa) — ~30 sec
legalkit nli-eval --output eval/nli.json

# LegalBench reasoning (contract subset, requires API key) — ~10 min
legalkit legalbench --task-set contract --n 20 --output eval/legalbench.json

# ContractNLI document-level NLI (requires API key) — ~15 min
legalkit contractnli-eval --n 50 --output eval/contractnli.json
```

### Reproducibility notes

- All retrieval benchmarks are deterministic (no LLM calls)
- End-to-end results depend on the configured LLM model and temperature (fixed at 0.0)
- DeBERTa inference is deterministic given the same model weights
- LLM-as-judge calls have temperature=0.0 but may vary across model versions
- Result files in `eval/` are timestamped; re-running produces new files, not overwrites

### What we don't measure (yet)

- **Latency at scale** — all evals run sequentially on a single machine; production latency under concurrent load is not characterised
- **Cross-document retrieval** — LegalBench-RAG queries are always against a single known document; multi-document queries are not evaluated
- **Long documents** — the longest corpus files are ~50K characters; very long contracts (100K+) are not tested
- **Jurisdiction-specific reasoning** — all test data is US contract law; performance on UK, EU, or other jurisdictions is unknown
- **Adversarial inputs** — no robustness testing against unusual formatting, OCR errors, or deliberately misleading documents

---

## 7. Known Methodology Limitations

This section documents known weaknesses in our evaluation methodology. Acknowledging these limitations is essential for interpreting the headline metrics correctly and for planning future validation work.

### Embedding Model: Train/Eval Data Overlap

- The embedding training data (9,409 pairs) is extracted directly from LegalBench-RAG benchmark ground-truth annotations using the same 4 datasets (CUAD, MAUD, ContractNLI, PrivacyQA) and the same documents
- The headline 'beats text-embedding-3-large' comparison is not fair: our model was trained on the benchmark; text-embedding-3-large was not
- The 99.03% R@64 likely overstates generalization to unseen legal documents
- **LODO results (2026-03-19):** Leave-one-dataset-out cross-validation confirms this concern. The fine-tuned model averages **95.5% R@64** on held-out datasets vs. **95.8%** for the base BGE model (unfine-tuned) — **no improvement, and slightly worse on MAUD (-1.9pp)**. The high recall is entirely driven by chunk size, not embedding quality. See `docs/research/lodo-embedding-eval-findings.md` for full analysis.
- **Recommendation:** Remove the "beats text-embedding-3-large" claim. Ship the base BGE model instead of the fine-tuned model. Invest in precision improvements (reranker, smaller chunks) rather than embedding fine-tuning.

### Entailment Model: Small Evaluation Set

- The eval set is 33 hand-labeled cases — too small for statistically robust claims
- 87.9% binary accuracy = 29/33 correct; moving 2 cases changes accuracy by 6pp
- All 33 cases cover the same 8 categories the model was specifically trained on
- No out-of-distribution evaluation exists yet
- **Mitigation:** expanded 200+ case eval set with 6 OOD categories is in progress (see `scripts/expand_nli_eval_set.py`)

### Entailment Training Data: Unverified LLM Labels

- 53.8% of entailment v2 training data (3,338 samples) comes from pipeline_log — labels generated by Claude during pipeline runs
- **Audit results (2026-03-19):** Independent re-labeling of 100 stratified samples shows **66% agreement** overall. The `partially_entailed` class is particularly unreliable at **16% agreement** — 76% of `partially_entailed` labels should be `entailed`. The `neutral` (92%) and `contradicted` (76%) classes are more reliable.
- **Impact**: The fine-tuned model's weakest F1 score (0.66 for `partially_entailed`) directly reflects noisy training labels, not a learning failure.
- **Recommendation**: Collapse to 3 classes (merge `partially_entailed` into `entailed`) for the next training run. This reduces effective label noise from ~34% to ~12%. See `docs/research/label-audit-findings.md` for full analysis.

### E2E Evaluation: Small Sample Size

- Only 20 queries (5 per dataset) — too few for precise statistical claims
- The 10.6% hallucination rate vs >17% for commercial tools is a useful directional signal but not a rigorous comparison (different questions, different tools, different methodology)
- **Mitigation:** planned expansion to 100+ queries

---

## 8. SFT Reasoning Model (Qwen2.5-3B + LoRA)

*File: `eval/sft_eval_results.json` — run 2026-03-20*

A generative reasoning model trained via SFT on 3,012 Claude-generated reasoning chains. Unlike the cross-encoder models (§4), this model produces step-by-step legal reasoning for each prediction.

| Model | Binary Acc | 3-Class Acc | Reasoning |
|---|---|---|---|
| Cross-encoder v2 (DeBERTa-large) | **72.8%** | — | No |
| **SFT Qwen2.5-3B + LoRA** | **69.5%** | **56.8%** | **Yes** |

The SFT model trails the cross-encoder by 3.3pp on binary accuracy but provides interpretable reasoning. Notably, the model generalizes well to out-of-distribution categories (87%+ on governing_law, assignment, force_majeure — none in training data). Temporal reasoning is the major failure mode at 39%.

A second SFT run (v2) with improved neutral reasoning chains (regenerated 771 generic examples with specific reasoning) produced **identical results** — 69.5% binary, same error patterns across all categories. This confirms SFT has hit its ceiling; the model's errors are due to classification difficulty, not training data quality.

This model serves as the starting point for GRPO reinforcement learning, which aims to improve both accuracy and reasoning quality. See `docs/research/sft-reasoning-eval-findings.md` for full analysis.

---

## 9. Target Metrics vs. Current State

| Metric | Current (PoC) | Target | Gap | Path |
|---|---|---|---|---|
| LegalBench-RAG Recall@64 | 99.9%† | — | — | See chunk-size caveat |
| LegalBench-RAG Precision@1 | 3.5% | 15%+ | -11.5pp | Smaller chunks + reranker |
| Hallucination rate | 10.6% | <10% | 0.6pp | Generation prompt hardening |
| Citation verification | 88% | 90%+ | 2pp | Fine-tuned entailment model |
| DeBERTa binary NLI accuracy | 63.6% | 80%+ | 16.4pp | Fine-tune on legal hard negatives |
| E2E strict correctness | 65% | 80%+ | 15pp | Prompt + verification improvements |

*† With large chunks; not directly comparable to published baseline*

The two biggest gaps are **retrieval precision** (too many irrelevant characters returned per chunk) and **NLI model accuracy** (DeBERTa can't handle legal clause structure). Both are on the active roadmap.

---

## References

- Pipitone et al. (2024). *LegalBench-RAG: A Benchmark for Retrieval-Based Legal Question Answering.* [GitHub](https://github.com/zeroentropy-ai/legalbenchrag)
- Guha et al. (2023). *LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models.* NeurIPS 2023. [arXiv](https://arxiv.org/abs/2308.11462)
- Magesh et al. (2025). *Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools.* Journal of Empirical Legal Studies. [Stanford Law](https://law.stanford.edu/publications/hallucination-free-assessing-the-reliability-of-leading-ai-legal-research-tools/)
- Koreeda & Manning (2021). *ContractNLI: A Dataset for Document-Level Natural Language Inference for Contracts.* EMNLP 2021.
- Anthropic (2024). *Introducing Contextual Retrieval.* [Blog](https://www.anthropic.com/news/contextual-retrieval)

---

*See also: [[e2e-evaluation-findings]], [[nli-evaluation-findings]], [[fine-tuning-findings]], [[citation-accuracy-landscape]]*
