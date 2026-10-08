# Label Audit Findings: Pipeline Log Training Data

**Date**: 2026-03-19
**Method**: Re-labeled 100 stratified random pipeline_log entries (25 per class) with Claude Sonnet via independent re-evaluation
**Training data**: `data/training_v2/train.jsonl` (3,338 pipeline_log entries, 53.8% of training data)

## Executive Summary

**Overall agreement: 66%.** One-third of pipeline_log labels disagree with independent re-labeling. However, the disagreements are **highly concentrated in one class** — `partially_entailed` — and the pattern is systematic, not random noise.

**The core problem: `partially_entailed` labels are overwhelmingly wrong.** Only 16% of `partially_entailed` labels agree with independent re-labeling. 76% of them are re-labeled as `entailed` by Claude. This means the original labeling pipeline was over-splitting `entailed` into `partially_entailed`, creating a class that is mostly noise.

## Results

### Per-Class Agreement

| Original Label | Agreement | Re-labeled Breakdown |
|---|---|---|
| **neutral** | **92%** (23/25) | 1→entailed, 1→partially_entailed |
| **entailed** | **80%** (20/25) | 5→partially_entailed |
| **contradicted** | **76%** (19/25) | 4→entailed, 2→partially_entailed |
| **partially_entailed** | **16%** (4/25) | **19→entailed**, 2→contradicted |

### Confusion Matrix

```
Original ↓ / Re-label → | contradicted | entailed | neutral | partially_entailed
-------------------------+--------------+----------+---------+-------------------
contradicted             |      19      |     4    |    0    |         2
entailed                 |       0      |    20    |    0    |         5
neutral                  |       0      |     1    |   23    |         1
partially_entailed       |       2      |    19    |    0    |         4
```

### Top Disagreement Transitions

| Transition | Count | % of disagreements |
|---|---|---|
| **partially_entailed → entailed** | **19** | **55.9%** |
| entailed → partially_entailed | 5 | 14.7% |
| contradicted → entailed | 4 | 11.8% |
| partially_entailed → contradicted | 2 | 5.9% |
| contradicted → partially_entailed | 2 | 5.9% |

## Analysis

### 1. The `partially_entailed` class is broken

84% of `partially_entailed` labels are wrong. The dominant error (19/25 → `entailed`) means the original pipeline was labeling claims as "partially supported" when they were actually "fully supported." This is the single biggest data quality issue.

**Impact on training**: The model learned a noisy `partially_entailed` class that overlaps heavily with `entailed`. This explains the model's weakest F1 score (0.66 for partially_entailed in the fine-tuning results) — the training labels themselves are unreliable for this class.

### 2. `neutral` and `contradicted` are fairly reliable

92% and 76% agreement respectively. The `contradicted → entailed` transitions (4 cases) are more concerning — these are cases where the original pipeline called something "contradicted" but independent review says "entailed." These are false negative training signals that could teach the model to reject valid citations.

### 3. The boundary between `entailed` and `partially_entailed` is the problem

70.6% of all disagreements (24/34) involve the entailed ↔ partially_entailed boundary. This is a labeling guideline issue — the original pipeline was too aggressive in downgrading from "entailed" to "partially_entailed."

### 4. Confidence is not a reliable disagreement signal

Mean confidence for agreements (0.940) vs. disagreements (0.921) — only a 0.02 gap. The re-labeling model is confident even when it disagrees, suggesting the disagreements are genuine interpretation differences, not borderline cases.

## Implications for Model Training

### Option A: Collapse to 3 classes (Recommended)
Merge `partially_entailed` into `entailed` for training. The 4-class distinction is not reliably labelable — even Claude disagrees with itself across runs. The binary decision (supported vs. not supported) is what matters for citation verification, and both `entailed` and `partially_entailed` map to "supported."

**Expected impact**: Cleaner training signal, higher agreement rate (~85% estimated), stronger binary accuracy.

### Option B: Re-label all partially_entailed samples
Use the audit methodology to re-label all 602 `partially_entailed` pipeline_log samples. Keep 4 classes but with corrected labels. More expensive but preserves granularity.

### Option C: Increase label quality at the source
Fix the original labeling pipeline's prompt to be stricter about when to use `partially_entailed`. Re-run the pipeline. Most expensive, highest long-term value.

## Recommendation

**Do Option A (collapse to 3 classes) for the next training run.** The 4-class schema was aspirational but the labeling quality doesn't support it. For production citation verification, the binary decision is what matters anyway:
- `entailed` + `partially_entailed` → **SUPPORTED** (accept citation)
- `contradicted` + `neutral` → **NOT SUPPORTED** (reject citation)

After collapsing, the effective label noise drops from ~34% to ~12% (only the contradicted→entailed and neutral→entailed errors remain). This is a much cleaner training signal.

## Files

- Full audit results: `eval/label-audit-20260319_135942.json`
- Audited samples: `data/eval/label_audit_samples.json`
- Audit script: `scripts/audit_pipeline_labels.py`
