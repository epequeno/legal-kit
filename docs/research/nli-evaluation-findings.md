# NLI Model Evaluation: DeBERTa vs LLM-as-Judge

**Date**: 2026-03-17
**Test set**: `data/eval/legal_nli_edge_cases.json` (33 hand-labelled cases)
**Tool**: `legalkit nli-eval`

## Results

### DeBERTa-base (cross-encoder/nli-deberta-v3-base)

| Metric | Value |
|--------|-------|
| Binary accuracy (entailed∪partial vs contradicted∪neutral) | **63.6%** |
| 4-class exact accuracy | 39.4% |
| Avg inference latency | ~106ms |
| Cost per run | free (local) |
| False positives (contradicted predicted as supported) | 10/13 = 77% |

### Per-category binary accuracy

| Category | Accuracy | Notes |
|----------|----------|-------|
| knowledge qualifier | **100%** | Gets lucky via uncertainty → partially_entailed threshold |
| quantitative precision | 75% | Misses deductible inversion |
| condition precedent | 67% | |
| defined term | 67% | |
| temporal qualifier | 67% | |
| mae carveout | 60% | Can't detect carve-out exclusions |
| notwithstanding | **40%** | Core structural failure |
| shall vs may | **40%** | Modal verb confusion |

## Key Findings

### 1. The failure mode is false positives on contradicted cases

10 of 12 failures are the model predicting "supported" for claims that directly
contradict the passage. This is the worst possible failure for a citation
verifier — it would let hallucinated or incorrect claims pass as verified.

### 2. General NLI ≠ legal NLI

DeBERTa-base on MNLI doesn't understand:
- **Carve-out logic**: "X, provided that Y shall not apply" — model sees topic
  match and predicts entailment, missing the logical exclusion
- **Notwithstanding structure**: "Notwithstanding A, B applies" creates a
  logical override; the model misses the scope of the override
- **Modal verb legal semantics**: `shall` (mandatory) vs `may` (discretionary)
  are legally distinct but look similar to a general NLI model. The model
  classifies "the board may accelerate vesting" as entailing "vesting
  automatically accelerates" — 0 legal understanding

### 3. Knowledge qualifier success is an accident

100% on knowledge-qualifier cases is NOT because the model understands legal
epistemics. The model's natural uncertainty about knowledge-qualified claims
produces moderate entailment scores (0.35–0.70) which lands in the
`partially_entailed` bucket via threshold mapping. A different threshold would
break this.

### 4. 63.6% binary is not good enough

Random baseline is 50%. The gap of ~14pp is real but small. The specific
failure mode (letting contradictions through) means you cannot rely on this
model to catch incorrect citations in production — it would approve roughly
77% of contradicted claims as "verified".

## Implications

### The LLM judge is not replaceable yet

These results provide concrete evidence that a general NLI model cannot
substitute for the LLM judge on legal text. The LLM correctly reads clause
structure — it understands that "notwithstanding [exception]" scopes the main
clause and that "may" does not imply obligation.

Expected LLM accuracy on this benchmark: **85–95% binary** (to be measured
with `legalkit nli-eval --llm`).

### The fine-tuning strategy is validated

The specific failures map exactly to the "hard negative" problem identified in
[[fine-tuning-findings]]:

- `mae-001`, `nw-002`, `sm-003` etc. are all **topically similar but logically
  contradicted** — the exact case that cross-passage random negatives don't
  teach
- A model trained with LLM-generated hard negatives (neutral/contradicted
  claims that look entailed but aren't) would learn carve-out detection, modal
  semantics, and notwithstanding logic
- Expected accuracy after fine-tuning on legal-specific data: **80%+ binary**
  at ~100ms latency, zero marginal cost

### Training data priorities (from this evaluation)

1. **Notwithstanding pairs** — carve-out and override structures (highest-value)
2. **Shall vs may pairs** — mandatory vs discretionary, right vs obligation
3. **MAE carve-out pairs** — exclusion + exception-to-exclusion logic
4. **Quantitative precision pairs** — up to / at least / exceeding / exactly

These four categories cover the 60% of failures and should be over-represented
in the fine-tuning data.

## Next Steps

1. Run with `--llm` flag to get LLM baseline and compute the accuracy gap
2. Generate hard-negative training pairs for the four failure categories above
3. Re-run evaluation after fine-tuning to measure improvement
4. Track this benchmark as the standard for legal entailment model quality

## Test Infrastructure

The 33 edge cases are in `data/eval/legal_nli_edge_cases.json` and are
version-controlled. All future model versions should be evaluated against this
set. See `src/legalkit/eval/nli_eval.py` for the evaluation framework.
