# Fine-Tuning Findings: Legal Entailment Model v0

**Date**: 2025-03-17
**Infrastructure**: Modal A10G GPU, ~4 minutes training time, ~$0.05 cost

## Dataset

| Metric | Value |
|--------|-------|
| Raw samples collected | 10,290 |
| Balanced dataset | 3,032 (758/class) |
| Train / Val / Test | 2,426 / 303 / 303 |
| Classes | entailed, partially_entailed, contradicted, neutral |

### Data Sources

| Source | Samples | % |
|--------|---------|---|
| synthetic:cross_passage | 6,974 | 68% |
| synthetic:llm_generated | 3,024 | 29% |
| synthetic:perturbation | 260 | 3% |
| pipeline (organic) | 32 | <1% |

### Data Generation Strategies

1. **LLM-generated** (Claude Haiku): Given a real legal passage, generate one claim per entailment class. ~4 claims per passage, ~$0.001 per passage.

2. **Cross-passage negatives**: Pair a claim from one section with a passage from a different section. Free, but produces obvious negatives (topically unrelated text).

3. **Perturbation-based**: Take an entailed claim and systematically alter it — swap amounts ($50M → $100M), swap parties (Buyer ↔ Seller), negate conditions (shall → shall not). Free, high quality, but requires existing entailed pairs.

## Training Results

| Metric | Value |
|--------|-------|
| Test accuracy | 26.4% |
| Test macro F1 | 0.10 |
| entailed F1 | 0.42 |
| partially_entailed F1 | 0.00 |
| contradicted F1 | 0.00 |
| neutral F1 | 0.00 |
| Base model | microsoft/deberta-v3-base |
| Epochs | 3 |
| Training loss (final) | 2.86 |

## Analysis: Why It Failed

**The model essentially learned to predict "entailed" for everything.** Three root causes:

### 1. Cross-passage neutrals are too easy
68% of raw data is cross-passage negatives — passages paired with completely unrelated text from different sections. After downsampling, these still dominate the neutral class. They don't teach the model anything useful because they're trivially distinguishable. A real legal entailment model needs *hard negatives* — passages that are topically related but don't actually support the claim.

### 2. Synthetic distribution doesn't match real distribution
The LLM-generated claims are well-formed but uniform in style. Real entailment pairs from pipeline usage have much more variation — paraphrases, partial quotes, inferential claims, context-dependent interpretations. 32 organic samples isn't enough to anchor the distribution.

### 3. 3,032 samples is small for DeBERTa from scratch
DeBERTa-v3-base has 86M parameters. Fine-tuning from a non-NLI checkpoint with only ~600 samples per class is asking a lot. Starting from a model already trained on MNLI (like `microsoft/deberta-v3-base-mnli`) would give better initial representations.

## What Would Fix It

**Priority order:**

1. **Start from DeBERTa-MNLI checkpoint** instead of base. The model already understands general entailment — we just need to adapt it to legal patterns. This alone might get us to 60-70% accuracy.

2. **Replace cross-passage neutrals with hard negatives.** Use the LLM to generate neutral claims that are *topically related* but not supported. E.g., for an indemnification passage, a neutral claim about termination provisions (same deal, different topic).

3. **Collect more organic pipeline samples.** Run the full e2e evaluation on all 776 LegalBench-RAG-mini queries. That would generate ~2,000-3,000 organic entailment pairs with real distribution characteristics.

4. **Human review of a calibration set.** Even 100 manually-reviewed samples would let us measure how accurate the LLM-generated labels are and calibrate accordingly.

5. **More perturbation variety.** Add date changes, jurisdiction swaps, threshold modifications, defined-term substitutions.

## Infrastructure Validation

The Modal training pipeline works well:
- **Upload**: Training data mounts to volume in seconds
- **Training**: 3 epochs on A10G in ~4 minutes
- **Cost**: ~$0.05 per training run
- **Iteration**: Change data → re-run → results in 5 minutes

This means we can iterate quickly once the data quality improves. The bottleneck is data, not compute.

## Next Steps

The fine-tuning infrastructure was ready; the data quality was the blocker. The remaining gate was a manual review process for the training set, which was never completed.
