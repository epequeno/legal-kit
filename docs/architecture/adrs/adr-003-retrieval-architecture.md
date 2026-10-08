# ADR-003: Hybrid Retrieval with Contextual Enrichment

## Status

Accepted

## Date

2025-03-17

## Context

Retrieval quality is the foundation of citation accuracy. The research landscape shows:

1. **Naive chunking destroys context** — a chunk reading "The company's revenue grew by 3%" is useless without knowing which company or time period. This is especially damaging in legal documents where clauses reference definitions and other sections constantly.

2. **Semantic search alone misses things** — a lawyer asks about "termination provisions" but the contract says "dissolution of agreement." BM25 keyword matching catches some of what semantic search misses, and vice versa.

3. **Anthropic's contextual retrieval** reduces retrieval failure by 67% (5.7% → 1.9%) when combining contextual embeddings + contextual BM25 + reranking.

4. **General-purpose rerankers hurt legal retrieval** — LegalBench-RAG found that Cohere's reranker actually decreased precision and recall on legal text. Domain specificity matters.

5. **Legal documents are structurally dense** — clauses cross-reference other sections, definitions are defined elsewhere, exhibits are attached. No existing benchmark evaluates cross-reference resolution, but it's how lawyers actually read contracts.

## Decision

Implement a four-layer retrieval architecture:

### Layer 1: Structural Parsing & Chunking
- Parse documents into a structural tree (sections, subsections, clauses, definitions)
- Chunk along structural boundaries, not arbitrary token counts
- Each chunk carries its structural lineage (e.g., `["Article IV", "§4.2", "§4.2(a)"]`)
- Detect cross-references within each chunk (regex for "Section X.Y", "as defined in", etc.)

### Layer 2: Contextual Enrichment
- For each chunk, use a cheap LLM (Claude Haiku) to generate 2-3 sentences of context:
  - What document this is from
  - What section this belongs to
  - Key entities/parties mentioned
- Prepend context to chunk text before embedding and BM25 indexing

### Layer 3: Hybrid Search with Reciprocal Rank Fusion
- **Semantic search**: embed query, find top-k by cosine similarity against contextual embeddings
- **BM25 search**: keyword match against contextual chunk text
- Combine results using Reciprocal Rank Fusion (RRF):
  ```
  score(chunk) = Σ 1/(k + rank_in_list) for each retrieval list
  ```
- Retrieve top-20 chunks (based on Anthropic's finding that 20 outperforms 5 or 10)

### Layer 4: Cross-Reference Expansion
- For each retrieved chunk, check its `cross_references` field
- If it references "Section 8.3" or "as defined in Article I", pull those chunks into the context
- Tag expanded chunks as "cross-reference" vs "directly retrieved"
- Cap total at 30 chunks to avoid context window overload

### Not using a general-purpose reranker
Based on LegalBench-RAG findings, general rerankers decrease performance on legal text. We skip this for now and re-evaluate with a domain-specific reranker later.

## Alternatives Considered

### Standard fixed-size chunking (500 tokens)
Rejected. LegalBench-RAG shows RCTS outperforms naive chunking. Structural chunking should outperform both because it respects document semantics.

### Semantic search only
Rejected. Legal terminology is precise and BM25 catches exact-match queries that semantic search misses. Hybrid consistently outperforms either alone in the literature.

### Late chunking
Considered. Processes entire document through embedding model before splitting. More efficient than contextual retrieval but research shows lower semantic preservation. Could revisit for self-hosted scenarios where LLM calls for contextualization are expensive.

### Dense passage retrieval with reranking
Partially rejected. No general-purpose reranker for now (hurts legal performance). Will evaluate domain-specific rerankers or fine-tuned models in production.

## Consequences

### Positive
- Contextual enrichment addresses the #1 failure mode (context loss)
- Hybrid search provides both semantic and lexical coverage
- Cross-reference expansion is a genuine differentiator — no published system does this
- Structural chunking preserves citation coordinates natively
- Each layer is independently testable and measurable

### Negative
- Contextual enrichment adds one LLM call per chunk during ingestion (~$0.05 per document)
- Cross-reference expansion increases context window size (cost + potential for distraction)
- Structural parsing requires document-type-specific heuristics
- More complex pipeline = more places for bugs

### Metrics to Track
- Precision@k and Recall@k on LegalBench-RAG, with and without each layer
- Ablation study: measure impact of contextual enrichment, hybrid search, and cross-reference expansion independently
