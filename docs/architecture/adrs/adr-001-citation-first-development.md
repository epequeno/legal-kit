# ADR-001: Citation-First Development Strategy

## Status

Accepted

## Date

2025-03-17

## Context

We are building an open-source legal AI platform (LegalKit) to compete with Harvey.ai and other commercial legal AI tools. The original product plan identified four pillars: Assistant, Vault, Workflows, and Research. Building all four simultaneously would take months and produce nothing demo-able.

The core value proposition of any legal AI tool rests on a single claim: **AI can analyze legal documents and cite its sources accurately enough that a lawyer would trust it.** If we can't prove this, nothing else matters — not the UI, not the deployment story, not the pricing.

Stanford's 2025 study (Magesh et al.) demonstrated that commercial tools from LexisNexis and Thomson Reuters hallucinate >17% of the time despite claiming to be "hallucination-free." This means:
1. The bar is lower than expected
2. Citation accuracy is a genuinely unsolved problem
3. Proving we can do this well is the highest-leverage work

## Decision

**Build the citation accuracy engine first, as an isolated CLI proof of concept, before any other product work.**

Specifically:
- No UI, no database, no deployment infrastructure
- Python CLI that takes a document + question → evidence report with verified citations
- Validate against LegalBench-RAG-mini benchmark (776 queries)
- Only proceed to product development if citation verification pass rate exceeds 85%

## Consequences

### Positive
- Fastest path to validating the core thesis
- Forces us to solve the hardest problem first, not defer it
- Produces quantitative evidence we can use for marketing and fundraising
- Architecture decisions made under constraint tend to be better
- If it doesn't work, we fail fast and cheaply

### Negative
- No visible product for potential early users/community during this phase
- Risk of over-engineering the citation engine before understanding real user workflows
- PoC architecture decisions may need revision when building the full product

### Risks
- If citation accuracy can't reach 85%+, the product thesis is in question and we need to rethink our approach fundamentally
