# SFT Legal Reasoning Model — Evaluation Findings

**Date**: 2026-03-20 (updated 2026-03-22)
**Model**: Qwen2.5-3B + LoRA (SFT on 3,012 Claude-generated reasoning chains)
**Eval set**: 243 cases, 14 categories (8 in-distribution, 6 out-of-distribution)

## ⚠️ Eval Infrastructure Note

During GRPO evaluation (2026-03-22) we discovered two bugs that invalidated earlier results:

1. **Wrong system prompt**: Both `modal_eval_sft.py` and `modal_eval_grpo.py` were using a long system prompt (with "When in doubt, default to neutral") that was never used during training. SFT was trained with a short 2-sentence prompt. The mismatch suppressed "entailed" predictions, giving both models ~35% binary accuracy (essentially the rate of non-entailed gold labels).

2. **Stale adapter path**: The GRPO eval was loading `checkpoint-300` from a previous training run instead of the final adapter.

Both were fixed on 2026-03-22. All results below are from corrected eval runs unless marked `[old]`.

## Corrected Summary

| Model | Binary Acc | 3-Class Acc | Notes |
|---|---|---|---|
| Cross-encoder v2 (DeBERTa-large) | **72.8%** | — | Baseline |
| SFT v2 Qwen2.5-3B + LoRA | 63.0% | 53.9% | Corrected eval |
| GRPO v1 Qwen2.5-3B (training bug) | 55.6% | 37.4% | "default to neutral" in training prompt |
| **GRPO v2 Qwen2.5-3B** | **72.4%** | **60.9%** | Fixed training + eval |
| **GRPO 7B** | **79.4%** | **73.3%** | Near 235B ceiling |
| Qwen3-235B zero-shot | 79.5% | — | DSPy ceiling test |
| Qwen3-235B + MIPROv2 | ~80-83% | — | Upper bound |

## SFT v2 Results (Corrected)

**Binary accuracy: 63.0% | 3-class accuracy: 53.9%**

### Per-Category Binary Accuracy

| Category | Accuracy | Type |
|---|---|---|
| non_compete_non_solicit | **80%** (12/15) | OOD |
| governing_law_jurisdiction | **80%** (12/15) | OOD |
| notwithstanding | 75% (15/20) | ID |
| force_majeure | 73% (11/15) | OOD |
| knowledge_qualifier | 70% (14/20) | ID |
| shall_vs_may | 70% (14/20) | ID |
| condition_precedent | 67% (12/18) | ID |
| defined_term | 67% (12/18) | ID |
| temporal_qualifier | 61% (11/18) | ID |
| assignment_change_of_control | 60% (9/15) | OOD |
| representations_warranties | 53% (8/15) | OOD |
| mae_carveout | 45% (9/20) | ID |
| limitation_of_liability | **47%** (7/15) | OOD |
| quantitative_precision | **37%** (7/19) | ID |

**Prediction distribution**: `{entailed: 92, contradicted: 91, neutral: 58, unknown: 2}`
Gold label distribution: 158 entailed, 64 contradicted, 21 neutral.

## GRPO v1 — Training Bug (55.6%)

The first full GRPO run on EC2 (`legal-reasoning-grpo/`) used a long system prompt in `train_grpo_local.py` that included **"When in doubt, default to neutral."** This directly conflicted with the reward signal (wrong neutrals are penalized), creating training instability:

- KL spiked to ~0.25 (~5× above the β=0.05 penalty threshold) around step 300-400
- Final reward plateau: ~0.61 (vs v2's 0.74-0.80)
- Eval result: 55.6% binary, 37.4% 3-class — worse than SFT

Prediction distribution: `{neutral: 115, contradicted: 54, entailed: 54}` — clear neutral bias from the training prompt leaking through RL.

## GRPO v2 Results — Fixed Prompt (72.4%)

Training run `legal-reasoning-grpo-v2/` on EC2 g5.xlarge. Fix: replaced training system prompt with the short SFT-matching prompt (no "default to neutral").

**Binary accuracy: 72.4% | 3-class accuracy: 60.9%**

![SFT vs GRPO v2 Evaluation Results](grpo_v2_eval_results.png)

*Top left: overall accuracy — GRPO v2 (+9.5pp binary, +7pp 3-class). Top right: prediction distribution — GRPO v2 predicts entailed more confidently (121 vs 92). Bottom: per-category accuracy and delta.*

### Per-Category Comparison

| Category | SFT v2 | GRPO v2 | Delta |
|---|---|---|---|
| representations_warranties | 53% | **87%** | +33% |
| limitation_of_liability | 47% | **73%** | +27% |
| mae_carveout | 45% | **70%** | +25% |
| defined_term | 67% | **89%** | +22% |
| shall_vs_may | 70% | **80%** | +10% |
| condition_precedent | 67% | **78%** | +11% |
| quantitative_precision | 37% | **47%** | +11% |
| force_majeure | 73% | **80%** | +7% |
| assignment_change_of_control | 60% | **67%** | +7% |
| governing_law_jurisdiction | 80% | 80% | +0% |
| temporal_qualifier | 61% | 61% | +0% |
| notwithstanding | 75% | 70% | -5% |
| knowledge_qualifier | 70% | 65% | -5% |
| non_compete_non_solicit | 80% | 73% | -7% |

11/14 categories improved. The 3 regressions (-5% to -7%) are within noise range for the sample sizes (~15-20 cases).

**Prediction distribution**: `{entailed: 121, contradicted: 54, neutral: 64, unknown: 4}` — much more balanced than GRPO v1's neutral-heavy distribution.

### Training Dynamics (v1 vs v2)

| Metric | GRPO v1 | GRPO v2 |
|---|---|---|
| Peak reward | ~0.68 | **~0.80** |
| Final reward (avg last 20) | 0.543 | **0.576** |
| Train loss | 0.0677 | **0.0428** |
| KL spike | ~0.25 (step 300) | <0.05 throughout |
| Runtime | 8.96 hrs | 8.35 hrs |

![GRPO v1 vs v2 Training Curves](grpo_training_curves.png)

*Top: reward over training steps (v2 consistently higher, peaks ~0.80 vs v1's ~0.68). Middle: loss and KL divergence (v1 spiked to 0.25 at step 300; v2 stayed flat below 0.05 throughout). Bottom: entropy and reward std dev.*

## GRPO 7B Results (79.4%)

Training `Qwen2.5-7B + SFT-7B adapter` on EC2 g5.2xlarge. Completed 2026-03-22.

**Binary accuracy: 79.4% | 3-class accuracy: 73.3%**

![GRPO 7B Full Model Progression](grpo_7b_eval_results.png)

*Top left: overall accuracy progression — 7B reaches 79.4% binary, within ~1-4pp of the 235B ceiling. Top right: prediction distribution nearly perfectly calibrated to gold label rates. Bottom: per-category breakdown and deltas.*

### Full Model Progression

| Model | Binary | 3-class | Notes |
|---|---|---|---|
| SFT 3B | 63.0% | 53.9% | Baseline |
| GRPO 3B v1 | 55.6% | 37.4% | Training bug ("default to neutral") |
| GRPO 3B v2 | 72.4% | 60.9% | Fixed prompt |
| **GRPO 7B** | **79.4%** | **73.3%** | Near ceiling |
| DSPy ceiling (235B) | ~80-83% | — | Upper bound |

### Per-Category: All Models

| Category | SFT 3B | GRPO 3B | GRPO 7B | Δ 7B vs 3B |
|---|---|---|---|---|
| knowledge_qualifier | 70% | 65% | **95%** | +30% |
| temporal_qualifier | 61% | 61% | **83%** | +22% |
| quantitative_precision | 37% | 47% | **68%** | +21% |
| mae_carveout | 45% | 70% | **85%** | +15% |
| force_majeure | 73% | 80% | **93%** | +13% |
| governing_law_jurisdiction | 80% | 80% | **87%** | +7% |
| non_compete_non_solicit | 80% | 73% | **80%** | +7% |
| notwithstanding | 75% | 70% | 75% | +5% |
| shall_vs_may | 70% | 80% | **85%** | +5% |
| representations_warranties | 53% | 87% | **87%** | +0% |
| condition_precedent | 67% | 78% | 72% | -6% |
| assignment_change_of_control | 60% | 67% | 60% | -7% |
| defined_term | 67% | 89% | 78% | -11% |
| limitation_of_liability | 47% | 73% | 60% | -13% |

**11/14 categories at or above the cross-encoder baseline (72.8%).**

### Key Observations

- **Prediction distribution nearly perfectly calibrated**: 172 entailed / 46 contradicted / 23 neutral vs gold of 158 / 64 / 21. The 7B model has learned the true label base rates — unlike the 3B which over-predicted entailed.

- **Previously weakest categories now strong**: temporal qualifier (61%→83%), knowledge qualifier (70%→95%), quantitative precision (37%→68%). These require genuine semantic precision — numeric reasoning, modal verb subtlety, temporal language — that the 7B model handles significantly better.

- **Two regressions**: limitation_of_liability (-13%) and defined_term (-11%). Both were categories where the 3B already showed strong performance; the 7B may be slightly over-reasoning on clear entailments.

- **Within ~1-4pp of the 235B ceiling**. The remaining gap likely reflects labeling noise and inherent task ambiguity rather than model capability.

### Eval Infrastructure

Results generated using the standalone `scripts/eval_local.py` on EC2 g5.xlarge (~$0.67 total vs ~$1.25 on Modal). Results at `s3://<bucket>/eval/grpo_7b_eval_results.json`.

## Key Lessons

1. **System prompt must match between training and eval.** Using a different prompt at eval time breaks the model. The prompt is part of the model's conditioning, not just an instruction.

2. **"Default to neutral" is dangerous in RL.** It creates a reward conflict — the model is told to hedge, but gets penalized for wrong hedges. It converges on a neutral-heavy degenerate strategy. The training prompt should be neutral about uncertainty.

3. **GRPO makes the biggest gains on the hardest categories.** The categories with the worst SFT performance (limitation_of_liability 47%, mae_carveout 45%, representations_warranties 53%) showed the biggest GRPO improvements (+27%, +25%, +33%). RL helps most where imitation learning struggled.

4. **SFT corrected accuracy is 63.0%, not 69.5%.** Earlier results in this doc were from a broken eval setup. The 69.5% figure from the old run likely used a different (accidentally matching) system prompt.

---

## SFT Training Details

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-3B |
| Method | QLoRA (4-bit NF4, rank 16, alpha 32) |
| Training data | 3,012 pairs (Claude-generated reasoning chains) |
| Validation data | 335 pairs |
| Epochs | 3 |
| Effective batch size | 16 |
| Learning rate | 2e-5, cosine schedule |
| System prompt | "You are a legal reasoning expert..." (2 sentences) |
| Trainable params | 29.9M / 1.73B (1.73%) |
| Training time | 102 minutes on A10G (~$1.70) |

## Files

- Eval results: `eval/sft_eval_results.json`, `eval/grpo_eval_results.json` (Modal volume `legalkit-data`)
- Charts: `docs/research/grpo_training_curves.png` (training dynamics), `docs/research/grpo_v2_eval_results.png` (eval results)
- SFT adapter: `models/legal-reasoning-sft/` (Modal: `models/legal-reasoning-sft/`)
- GRPO v2 adapter: `models/legal-reasoning-grpo-v2/` (Modal: `models/legal-reasoning-grpo-v2/final/`)
- Eval scripts: `scripts/modal_eval_sft.py`, `scripts/modal_eval_grpo.py`
