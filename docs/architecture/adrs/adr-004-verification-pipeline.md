# ADR-004: Multi-Layer Citation Verification

## Status

Accepted

## Date

2025-03-17

## Context

Most RAG systems stop at retrieval — they feed retrieved chunks to an LLM and trust the output. This is insufficient for legal applications where every citation must be verifiable. The Stanford 2025 study showed that even commercial legal AI tools with RAG hallucinate >17% of the time.

The gap isn't just in retrieval — it's in **verification**. The LLM may:
1. Cite a section that doesn't exist (fabricated reference)
2. Quote text that doesn't appear in the cited section (fabricated quote)
3. Cite a section that exists but doesn't actually support the claim (misleading citation)
4. Miss relevant sections entirely (incomplete answer)

Each of these failure modes requires a different verification approach.

## Decision

Implement four independent verification checks on every claim-citation pair, plus a completeness check on the overall answer.

### Check 1: Citation Existence
**Question**: Does the cited section/clause ID exist in the document?

- Look up the section identifier (e.g., "§4.2(a)") in our structural document tree
- Binary: exists or doesn't
- If it doesn't exist, the citation is hallucinated — flag as ❌ **unverified**
- Cost: zero (local lookup)

### Check 2: Quote Verification
**Question**: If the LLM quoted text, does that text actually appear at the cited location?

- Extract quoted text from the LLM's response (text in quotation marks attributed to a citation)
- Fuzzy match against the source text at the cited location using normalized Levenshtein distance
- Threshold: ≥85% similarity = match, 70-85% = partial match, <70% = no match
- Cost: zero (local string matching)

### Check 3: Entailment Verification
**Question**: Does the cited passage actually support the claim being made?

- Use a Natural Language Inference (NLI) model (`microsoft/deberta-v3-large-mnli`)
- Premise: the cited source text
- Hypothesis: the claim text
- Classification: ENTAILED / PARTIALLY_ENTAILED / CONTRADICTED / NEUTRAL
- Score: softmax probability of ENTAILED class
- Cost: local GPU inference (~50ms per pair)

**Why NLI over LLM-as-a-Judge (for PoC)**:
- NLI is deterministic — same input always produces same output
- No API cost — runs locally
- Faster — batch process all claims
- Well-studied — DeBERTa on MNLI is a known-good baseline

We will add LLM-as-a-Judge as an optional comparison mode to evaluate whether it catches nuances that NLI misses (e.g., legal-specific entailment patterns). If it proves significantly better, we can make it the default for cloud users.

### Check 4: Completeness Check
**Question**: Are there relevant passages the answer didn't cite?

- For each claim in the answer, run a retrieval query using the claim text
- If high-relevance chunks (score > threshold) appear that weren't cited in the answer, flag them
- Report as "potentially relevant sections not cited"
- This catches the case where the LLM answered correctly but incompletely

### Confidence Scoring

Each claim gets a composite confidence score:

```
claim_confidence = weighted_average(
    citation_exists:      1.0 or 0.0     (weight: 0.3)
    quote_similarity:     0.0 - 1.0      (weight: 0.2)
    entailment_score:     0.0 - 1.0      (weight: 0.4)
    retrieval_score:      0.0 - 1.0      (weight: 0.1)
)
```

Mapped to levels:
- **High confidence** (≥ 0.85): Citation verified, quote matches, entailment strong
- **Medium confidence** (0.65 - 0.85): Citation exists but entailment partial or quote inexact
- **Low confidence** (< 0.65): Significant verification concerns

Additionally, **corroboration count** tracks how many independent chunks support the same claim. A claim supported by 3+ chunks is more trustworthy than one supported by a single chunk.

## Alternatives Considered

### LLM-as-a-Judge only
Rejected as primary approach. Non-deterministic, adds API cost per verification, and judge models can have their own biases. However, will be available as an optional mode for comparison.

### No verification (trust the RAG pipeline)
Rejected. This is what commercial tools do, and they hallucinate >17%. Verification is our core differentiator.

### HHEM (Hughes Hallucination Evaluation Model)
Considered. Lightweight classifier for hallucination detection. Less nuanced than NLI for entailment checking. May add as an additional signal in production.

### Human-in-the-loop verification
Out of scope for PoC. The system should flag uncertainty so humans know where to focus, but the verification pipeline itself is automated.

## Consequences

### Positive
- Every citation is independently verified — not just trusted
- Four orthogonal checks catch different failure modes
- Confidence scores give users actionable uncertainty information
- Completeness check catches omissions, not just errors
- NLI-based verification is deterministic and free (local inference)
- Architecture supports adding more verification methods over time

### Negative
- Verification adds latency (~2-5 seconds for NLI on all claims)
- DeBERTa NLI may not capture legal-specific entailment patterns (e.g., "notwithstanding" clauses)
- Completeness check requires additional retrieval passes (cost)
- Confidence weights are heuristic — will need calibration against human judgement

### Future Work
- Fine-tune an NLI model on legal entailment data
- Evaluate LLM-as-a-Judge vs NLI on legal-specific edge cases
- Calibrate confidence thresholds against lawyer assessments
- Add temporal verification (is this the current version of the clause, or has it been amended?)
