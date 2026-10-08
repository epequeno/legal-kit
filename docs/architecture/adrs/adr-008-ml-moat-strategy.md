# ADR-008: ML/AI Moat Strategy

## Status

Accepted

## Date

2025-03-17

## Context

LegalKit is open-source (AGPLv3). The code itself is not a moat — the core pipeline is ~2000 lines of Python using published techniques that anyone could reproduce. We need defensible value that compounds over time and benefits from being open-source rather than being weakened by it.

The PoC has demonstrated:
- 90% answer correctness, 88% citation verification, 11% hallucination rate
- RAG pipeline slightly outperforms full-context baseline on strict correctness
- The verification and explainability layers are the real product differentiator

The question is: what ML/AI investments create the most defensible long-term value?

## Decision

Invest in four compounding ML assets, in priority order:

### 1. Legal Entailment Model (Phase 2 — Highest Priority)

**What**: A fine-tuned model that determines whether a legal passage supports a specific claim.

**Why it's a moat**:
- Legal entailment differs from general NLI ("notwithstanding" inverts meaning, "shall" vs "may" is critical, "material adverse effect" has jurisdiction-dependent thresholds)
- The training data is the real asset — (passage, claim, entailment_label) triples validated by lawyers
- Faster (50ms vs 2-5s), cheaper (local), and more accurate than LLM-as-judge on legal patterns
- Runnable offline for air-gapped deployments (a key selling point)

**Open-source strategy**: Publish model weights on HuggingFace. Community files issues when it's wrong → corrections improve training data → model gets better. We become the standard for legal entailment.

### 2. Legal Document Structure Model (Phase 3)

**What**: A trained model that segments legal documents into structural elements regardless of formatting conventions.

**Why it's a moat**:
- Our regex parser fails on non-standard formats (privacy policies, M&A agreements with unusual numbering)
- A structure model trained on thousands of contracts across formats and jurisdictions would be genuinely valuable
- Grunt work with massive compounding value — nobody wants to build it but everyone needs it

**Open-source strategy**: The model becomes infrastructure that the legal AI ecosystem depends on.

### 3. Legal Retrieval Embeddings (Phase 3)

**What**: An embedding model fine-tuned on legal query-passage pairs.

**Why it's a moat**:
- LegalBench-RAG showed general-purpose rerankers hurt legal retrieval
- Training data from LegalBench-RAG + CUAD + MAUD + synthetic pairs from public corpora
- Becomes the standard legal embedding model if published on HuggingFace

### 4. Confidence Calibration (Phase 4)

**What**: Calibrated confidence scores where score=0.85 means the claim is actually correct 85% of the time.

**Why it's a moat**:
- Requires labeled data at scale: (claim, citation, source_text, actually_correct) tuples reviewed by lawyers
- Extremely expensive to produce — competitive moat through investment, not cleverness
- The cloud product has a structural advantage: user feedback (thumbs up/down) feeds calibration

**Open-source strategy**: The calibration methodology is open, but the managed training pipeline (cloud ingests feedback → retrains → deploys) is the commercial advantage.

## The Data Flywheel

All four investments share a common pattern:

```
More users → More queries/judgments → Better training data →
Better models → Better product → More users
```

Open-sourcing the models accelerates this flywheel because:
- Community contributions expand the training data beyond what we could collect alone
- Adoption makes our models the standard, so competitors are measured against our work
- Bug reports from diverse legal contexts surface edge cases we'd never find internally

## What Stays Commercial

- **Managed training pipeline** — automated retraining on user feedback (cloud product)
- **Calibrated confidence at scale** — requires production traffic to calibrate
- **Enterprise features** — SSO, audit logs, compliance, SLA
- **Hosted inference** — managed GPU infrastructure for the fine-tuned models

## Alternatives Considered

### Keep models proprietary
Rejected. In legal AI, trust is paramount. Open models that can be inspected, audited, and validated by the legal community build more trust than proprietary black boxes. This is the explicit lesson from Harvey/LexisNexis being caught overstating their "hallucination-free" claims.

### Focus on prompt engineering instead of model training
Rejected as sole strategy. Prompt engineering is table stakes and easily replicated. Fine-tuned models create durable advantages. However, prompt engineering remains important for the generation stage.

### Build a proprietary legal knowledge graph
Deferred. Knowledge graphs are valuable but require massive upfront investment and don't benefit from open-source community dynamics in the same way. Revisit after Phase 4.

## Consequences

### Positive
- Each phase produces a publishable, citable asset (model, benchmark, dataset)
- Open-source publication builds brand and community before the cloud product launches
- The data flywheel compounds — early investment pays increasing returns
- Air-gapped / self-hosted deployments become genuinely viable with local models
- Competitive moat deepens with each model release

### Negative
- Fine-tuning requires GPU infrastructure and ML engineering expertise
- Training data collection from lawyers is expensive and slow
- Models need ongoing maintenance as legal standards evolve
- Open-sourcing means competitors can use our models (but not our data pipeline)

### Metrics to Track
- Legal entailment model accuracy vs LLM-as-judge (target: parity or better)
- Legal entailment model latency (target: <100ms per pair)
- Legal structure model accuracy on non-standard document formats
- Legal embedding model retrieval scores on LegalBench-RAG vs general embeddings
- Confidence calibration error (target: <5% deviation from actual accuracy)
- HuggingFace download counts for published models (community adoption signal)
