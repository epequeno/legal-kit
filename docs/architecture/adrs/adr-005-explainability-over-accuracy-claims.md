# ADR-005: Explainability Over Accuracy Claims

## Status

Accepted

## Date

2025-03-17

## Context

Commercial legal AI vendors (LexisNexis, Thomson Reuters, Casetext) have marketed their products with claims like "hallucination-free" and "eliminating hallucinations." Stanford's 2025 empirical study demonstrated these claims are overstated — both systems hallucinate >17% of the time.

This creates a trust problem. When a system claims to be perfect and isn't, users either:
1. Trust it blindly (dangerous in legal contexts), or
2. Lose trust entirely once they discover an error

There is a better model from healthcare: **explainability and uncertainty quantification.** Medical decision support systems don't claim to be infallible — they surface evidence, confidence intervals, and differential considerations so the clinician can make an informed decision. The AI augments expert judgment rather than replacing it.

## Decision

**LegalKit will never claim to be hallucination-free. Instead, we will make our system's reasoning transparent and its uncertainty visible.**

This manifests in three concrete layers:

### 1. Retrieval Transparency
- Show which chunks were retrieved and their relevance scores
- Show the retrieval method used (semantic, BM25, cross-reference)
- Flag when retrieval quality is low ("only 2 weakly relevant passages found")
- Show query reformulations if query expansion was applied

### 2. Citation Provenance
- For every claim, show the exact source text that supports it
- Show the structural location (section, clause, paragraph path)
- Provide char-span offsets for linking back to the original document
- Distinguish between directly-retrieved evidence and cross-referenced evidence
- Flag when a claim appears to draw on the LLM's parametric knowledge rather than retrieved evidence

### 3. Confidence & Uncertainty
- Per-claim confidence scores (high / medium / low) with underlying metrics
- Corroboration count (how many independent sources support this claim)
- Explicit flags for common risk patterns:
  - "Single source" — claim supported by only one passage
  - "Ambiguous entailment" — cited passage is related but interpretation is unclear
  - "Potential gap" — related sections exist that weren't cited
  - "Cross-reference unresolved" — chunk references a section that wasn't retrieved
- Explicit admission of gaps: "I could not find information addressing [aspect]"

### Product Positioning
- Marketing language: "Transparent legal AI" / "Verifiable legal research"
- Not: "Accurate" / "Hallucination-free" / "AI lawyer"
- Documentation will include our benchmark results with confidence intervals
- We will publish our evaluation methodology so anyone can reproduce our claims

## Alternatives Considered

### Claim high accuracy with caveats
Rejected. Even with caveats, accuracy claims invite scrutiny and eventual embarrassment. The Stanford study specifically targeted vendors who made such claims.

### No confidence scoring (just provide citations)
Rejected. Citations without confidence scores force the user to verify everything equally. Confidence scoring lets users focus their review time on the claims that need it most.

### Hide uncertainty to appear more polished
Rejected. In regulated industries (law, healthcare), uncertainty is a feature, not a bug. Professionals want to know what they should double-check.

## Consequences

### Positive
- Builds genuine trust through transparency
- Defensible positioning — we can't be "caught" overstating our capabilities
- Aligns with how lawyers actually work (verify, cross-reference, apply judgment)
- Uncertainty signals are actionable — users know where to focus review time
- Natural competitive moat — harder for competitors to copy a culture of transparency than a feature
- Regulatory alignment — bar associations are increasingly scrutinizing AI use in legal practice

### Negative
- May appear less polished than competitors who present confident-sounding answers
- Requires more sophisticated UI to present evidence reports (when we build the frontend)
- Some users may prefer a simple answer over a detailed evidence report
- Confidence calibration requires ongoing work to ensure scores are meaningful

### Implications for PoC
The PoC output format is an **evidence report**, not an answer. Every response is structured as:
- Answer text with inline citations
- Per-claim verification results
- Overall confidence assessment
- Identified gaps and related unresolved sections
- Retrieval metadata

Even as a CLI JSON output, this should be compelling enough to demonstrate the approach.
