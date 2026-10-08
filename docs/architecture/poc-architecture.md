# LegalKit Citation Engine — Proof of Concept Architecture

## Goal

Build a CLI tool that takes a legal document + a question and returns a structured evidence report with verified, explainable citations. Validate against LegalBench-RAG-mini (776 queries).

**Success criteria:**
1. Beat LegalBench-RAG published retrieval baselines (Precision@1 > 6.4%, Recall@64 > 62%)
2. 85%+ of generated citations pass automated verification
3. Every claim has provenance, confidence score, and gap analysis
4. System correctly identifies and flags its own uncertainty

---

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    CLI Interface                         │
│   legalkit ingest <doc>                                  │
│   legalkit ask <question> --doc <doc>                    │
│   legalkit bench --dataset legalbench-rag-mini           │
└──────────────┬──────────────────────────┬───────────────┘
               │                          │
       ┌───────▼────────┐        ┌────────▼────────┐
       │  Document       │        │  Query           │
       │  Pipeline        │        │  Pipeline         │
       └───────┬────────┘        └────────┬────────┘
               │                          │
    ┌──────────▼──────────┐    ┌──────────▼──────────┐
    │ 1. Parse & Structure │    │ 4. Query Expansion   │
    │ 2. Chunk (structural)│    │ 5. Hybrid Retrieval  │
    │ 3. Contextualize     │    │ 6. Rerank            │
    │    & Embed + BM25    │    └──────────┬──────────┘
    └──────────┬──────────┘               │
               │                          │
               ▼                          ▼
    ┌─────────────────────────────────────────────────┐
    │              Document Store                      │
    │  PostgreSQL + pgvector + BM25 (pg_trgm / tsvec) │
    │  • chunk text                                    │
    │  • contextual embedding (vector)                 │
    │  • structural metadata (section, clause, path)   │
    │  • char offsets into original document            │
    └──────────────────────┬──────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │ 7. Generate  │
                    │    (LLM)     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ 8. Verify    │
                    │    & Score   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ 9. Evidence  │
                    │    Report    │
                    └─────────────┘
```

---

## Pipeline Stages (Detailed)

### Stage 1: Document Parsing & Structuring

**Input**: Raw document (PDF, DOCX, or plain text)
**Output**: Structural tree with stable identifiers

```python
@dataclass
class DocumentNode:
    id: str                    # e.g., "§4.2(a)", "Exhibit-B", "Recital-3"
    type: NodeType             # SECTION, SUBSECTION, CLAUSE, DEFINITION, EXHIBIT, RECITAL, PARAGRAPH
    title: str | None          # e.g., "Indemnification"
    text: str                  # full text content of this node
    char_start: int            # offset in original document
    char_end: int              # offset in original document
    children: list[DocumentNode]
    parent_id: str | None
    cross_references: list[str]  # other section IDs referenced in this node
```

**Approach**:
- For the PoC, focus on **plain text contracts** (LegalBench-RAG corpus is text)
- Use regex + heuristics for structural parsing of legal documents:
  - Section/article numbering patterns (1., 1.1, (a), (i), Article I, Section 1.01)
  - Definition sections (typically "defined terms", quoted terms with means/shall mean)
  - Exhibit/schedule markers
- Cross-reference detection: regex for "Section X.Y", "Article Z", "as defined in", "pursuant to"
- Preserve char offsets for every node → enables precise citation back to source

**NOT doing for PoC**: PDF parsing, OCR, complex table extraction. Text-in, text-out.

### Stage 2: Structural Chunking

**Input**: Document tree from Stage 1
**Output**: Chunks aligned to structural boundaries

```python
@dataclass
class Chunk:
    id: str                      # unique chunk ID
    text: str                    # chunk content
    context_text: str | None     # prepended context (filled in Stage 3)
    doc_id: str                  # which document
    node_id: str                 # which structural node this belongs to
    node_path: list[str]         # ancestry path, e.g., ["Article IV", "§4.2", "§4.2(a)"]
    char_start: int              # in original doc
    char_end: int                # in original doc
    cross_references: list[str]  # sections referenced within this chunk
```

**Strategy**:
- Chunk along structural boundaries (section/clause level), NOT arbitrary token counts
- If a section exceeds max chunk size (~800 tokens), split at paragraph boundaries within it
- Maintain overlap at split points (1 sentence)
- Each chunk carries its structural lineage (node_path) — this IS the citation coordinate

### Stage 3: Contextual Enrichment & Indexing

**Input**: Chunks from Stage 2
**Output**: Contextualized chunks with embeddings + BM25 index

For each chunk, generate a short context prefix using a cheap LLM call:

```
Prompt: "Given the following document and a specific chunk from it,
provide a short (2-3 sentence) context that situates this chunk
within the document. Include the document type, relevant parties,
and the section this chunk belongs to.

Document: {first 2000 chars of doc + table of contents if available}
Chunk location: {node_path}
Chunk text: {chunk.text}

Context:"
```

Then:
1. **Contextual embedding**: Embed `context_text + chunk.text` using embedding model
2. **BM25 index**: Index `context_text + chunk.text` for keyword search
3. Store everything in the document store with metadata

**Embedding model options** (PoC):
- OpenAI `text-embedding-3-large` (best quality, API cost)
- `BAAI/bge-large-en-v1.5` (good open-source option for self-hosted)
- Voyage `voyage-law-2` (legal-specific, if available)

**BM25 implementation** (PoC):
- Use `rank_bm25` Python library in-memory (simplest)
- Graduate to PostgreSQL full-text search later

### Stage 4: Query Processing

**Input**: User question
**Output**: Expanded query for retrieval

- **Query as-is** for semantic search
- **Query expansion**: Use LLM to generate 2-3 alternative phrasings with legal terminology
  - "What are the termination provisions?" → also search for "dissolution", "expiration", "cancellation of agreement"
- **Extract structural hints**: If the question mentions "Section 4" or "indemnification", use that to boost retrieval from matching structural nodes

### Stage 5: Hybrid Retrieval

**Input**: Processed query
**Output**: Ranked list of candidate chunks

Run two retrieval paths in parallel:

1. **Semantic search**: Query embedding vs chunk embeddings → top-k by cosine similarity
2. **BM25 search**: Query text vs chunk BM25 index → top-k by BM25 score

Combine using **Reciprocal Rank Fusion (RRF)**:

```python
def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)

# For each chunk, sum RRF scores across both result lists
# Sort by combined score
```

Retrieve **top-20 chunks** (based on Anthropic's finding that 20 is optimal).

### Stage 6: Cross-Reference Expansion

**Input**: Top-20 retrieved chunks
**Output**: Expanded set with cross-referenced sections

For each retrieved chunk:
- Check `cross_references` field
- If a chunk references "Section 8.3" or "as defined in Article I", pull those chunks into context too
- Cap at 30 total chunks to avoid context overload
- Tag expanded chunks as "cross-reference" vs "directly retrieved"

**This is our differentiator** — no benchmark currently does this, but it's how lawyers actually read contracts.

### Stage 7: Grounded Generation

**Input**: Question + ranked chunks with metadata
**Output**: Answer with inline citations

Prompt structure:

```
You are a legal analyst. Answer the question based ONLY on the
provided document excerpts. Follow these rules strictly:

1. Cite every factual claim using the section identifier in brackets,
   e.g., [§4.2(a)]
2. When quoting, use exact text from the source in quotation marks
3. If the excerpts don't contain enough information to answer
   fully, explicitly state what's missing
4. If excerpts are ambiguous or could support multiple interpretations,
   note the ambiguity
5. Never state facts not found in the provided excerpts

Question: {question}

Document excerpts (in order of relevance):

[§4.2(a)] (Relevance: 0.94, Source: directly retrieved)
{chunk text}

[§4.2(c)] (Relevance: 0.89, Source: directly retrieved)
{chunk text}

[§8.3] (Relevance: 0.71, Source: cross-reference from §4.2(a))
{chunk text}

...

Answer:
```

**Parse the response** to extract:
- Individual claims (sentence-level)
- Citation references per claim
- Quoted text per claim
- Hedging language / uncertainty expressions
- Explicit gaps acknowledged

### Stage 8: Verification & Scoring

**Input**: Parsed claims with citations
**Output**: Verification results per claim

For each claim-citation pair, run four checks:

#### Check 1: Citation Existence
Does the cited section ID exist in our document structure?
- ✅ Found
- ❌ Not found (hallucinated reference)

#### Check 2: Quote Verification
If the claim includes a quote, does it appear in the cited section?
- Fuzzy match using Levenshtein distance (threshold: 85% similarity)
- ✅ Exact/near match found
- ⚠️ Partial match (paraphrased)
- ❌ Quote not found in cited section

#### Check 3: Entailment Verification
Does the cited passage actually support the claim?
- Use NLI model (e.g., `microsoft/deberta-v3-large` fine-tuned on NLI)
- Or use a second LLM call:

```
Given the following passage and claim, determine if the passage
supports the claim.

Passage: {cited chunk text}
Claim: {claim text}

Classification:
- ENTAILED: The passage directly supports the claim
- PARTIALLY_ENTAILED: The passage is related but doesn't fully support
- CONTRADICTED: The passage contradicts the claim
- NEUTRAL: The passage is unrelated to the claim

Respond with the classification and a one-sentence explanation.
```

#### Check 4: Completeness Check
Are there relevant passages that the answer didn't cite?
- Re-run retrieval against each individual claim
- If high-relevance chunks appear that weren't in the original retrieval or weren't cited, flag as potential gaps

#### Confidence Scoring

```python
def compute_confidence(claim: Claim) -> ConfidenceScore:
    scores = []
    for citation in claim.citations:
        citation_score = (
            1.0 if citation.exists else 0.0,          # existence
            citation.quote_similarity,                  # 0.0-1.0
            citation.entailment_score,                  # 0.0-1.0
            citation.retrieval_score,                   # 0.0-1.0
        )
        scores.append(weighted_average(citation_score))

    claim_confidence = max(scores)  # best supporting citation
    corroboration = len([s for s in scores if s > 0.7])

    return ConfidenceScore(
        level="high" if claim_confidence > 0.85 else
              "medium" if claim_confidence > 0.65 else "low",
        score=claim_confidence,
        corroboration_count=corroboration,
        flags=collect_flags(claim)
    )
```

### Stage 9: Evidence Report

**Output**: Structured JSON (CLI) or formatted report

```python
@dataclass
class EvidenceReport:
    question: str
    answer: str                              # full text answer
    claims: list[VerifiedClaim]              # each claim with citations & scores
    overall_confidence: ConfidenceScore
    retrieval_summary: RetrievalSummary      # how many chunks, scores, methods
    gaps: list[str]                          # what the system couldn't find
    related_sections: list[str]              # potentially relevant but not cited
    metadata: dict                           # timing, model used, etc.

@dataclass
class VerifiedClaim:
    text: str                                # the claim
    citations: list[VerifiedCitation]
    confidence: ConfidenceScore
    flags: list[str]                         # e.g., "single source", "ambiguous"

@dataclass
class VerifiedCitation:
    section_id: str                          # e.g., "§4.2(a)"
    section_path: list[str]                  # e.g., ["Article IV", "Section 4.2", "(a)"]
    source_text: str                         # the actual text from the document
    char_span: tuple[int, int]               # offsets in original document
    retrieval_method: str                    # "semantic", "bm25", "cross-reference"
    retrieval_score: float
    quote_match: float                       # 0.0-1.0
    entailment: str                          # "entailed" | "partial" | "contradicted" | "neutral"
    entailment_score: float                  # 0.0-1.0
    verification_status: str                 # "verified" | "partially_verified" | "unverified"
```

---

## Technology Choices (PoC)

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Language** | Python 3.12 | AI/NLP ecosystem, fast iteration |
| **CLI** | `click` | Simple, well-known |
| **LLM (generation)** | Anthropic Claude 3.5 Sonnet | Best reasoning, good at following citation instructions |
| **LLM (contextualization)** | Anthropic Claude 3 Haiku | Cheap, fast, good enough for context generation |
| **Embeddings** | OpenAI `text-embedding-3-large` | Best general quality; swap later for legal-specific |
| **BM25** | `rank_bm25` in-memory | Simplest for PoC, no infra |
| **NLI verification** | `microsoft/deberta-v3-large-mnli` | Strong NLI model, runs locally |
| **Vector store** | `numpy` + cosine similarity | No database for PoC — everything in memory |
| **Document parsing** | Custom regex-based parser | Legal doc structure is pattern-based |
| **Benchmark** | LegalBench-RAG-mini | 776 queries, standard evaluation |
| **Package mgmt** | `uv` | Fast, modern |

### What We Explicitly Defer
- PostgreSQL / pgvector (use in-memory for PoC)
- PDF/DOCX parsing (text only)
- Web UI (CLI only)
- Docker deployment
- Multi-document workspaces
- Authentication
- Fine-tuned models

---

## Project Structure

```
legalkit/
├── pyproject.toml
├── README.md
├── src/
│   └── legalkit/
│       ├── __init__.py
│       ├── cli.py                    # CLI entry point
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── parse.py              # Stage 1: document parsing & structuring
│       │   ├── chunk.py              # Stage 2: structural chunking
│       │   ├── contextualize.py      # Stage 3: contextual enrichment
│       │   ├── index.py              # Stage 3: embedding & BM25 indexing
│       │   ├── query.py              # Stage 4: query processing
│       │   ├── retrieve.py           # Stage 5-6: hybrid retrieval + cross-ref
│       │   ├── generate.py           # Stage 7: grounded generation
│       │   ├── verify.py             # Stage 8: verification & scoring
│       │   └── report.py             # Stage 9: evidence report assembly
│       ├── models/
│       │   ├── __init__.py
│       │   ├── document.py           # DocumentNode, Chunk dataclasses
│       │   ├── retrieval.py          # RetrievalResult, RankedChunk
│       │   ├── citation.py           # Claim, Citation, VerifiedCitation
│       │   └── report.py             # EvidenceReport, ConfidenceScore
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── client.py             # LLM abstraction (Anthropic for now)
│       │   └── prompts.py            # All prompt templates
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── benchmark.py          # LegalBench-RAG runner
│       │   └── metrics.py            # Precision@k, Recall@k, citation accuracy
│       └── utils/
│           ├── __init__.py
│           └── text.py               # fuzzy matching, text utils
├── tests/
│   ├── test_parse.py
│   ├── test_chunk.py
│   ├── test_retrieve.py
│   ├── test_verify.py
│   └── fixtures/
│       └── sample_contract.txt       # test document
├── data/
│   └── .gitkeep                      # LegalBench-RAG data downloaded here
└── eval/
    └── .gitkeep                      # benchmark results stored here
```

---

## CLI Interface

```bash
# Ingest a document (parse, chunk, contextualize, index)
legalkit ingest contract.txt

# Ask a question against an ingested document
legalkit ask "What is the indemnification cap?" --doc contract.txt

# Ask with verbose output (show retrieval details)
legalkit ask "What are the termination provisions?" --doc contract.txt --verbose

# Run the benchmark
legalkit bench --dataset legalbench-rag-mini --output eval/results.json

# Show benchmark results
legalkit bench --report eval/results.json
```

---

## Implementation Order

### Phase 1: Data Models & Parsing (Day 1-2)
1. Define all dataclasses (document, chunk, citation, report)
2. Build the structural document parser (regex-based)
3. Build the structural chunker
4. Test against sample contracts from LegalBench-RAG corpus

### Phase 2: Indexing & Retrieval (Day 3-4)
5. Contextual enrichment (LLM context generation)
6. Embedding generation + in-memory vector store
7. BM25 index
8. Hybrid retrieval with RRF
9. Cross-reference expansion
10. Test retrieval against LegalBench-RAG-mini, measure Precision@k and Recall@k

### Phase 3: Generation & Verification (Day 5-7)
11. Grounded generation with citation prompting
12. Response parsing (extract claims, citations, quotes)
13. Citation existence check
14. Quote verification (fuzzy matching)
15. NLI entailment verification
16. Completeness check
17. Confidence scoring
18. Evidence report assembly

### Phase 4: Benchmark & Polish (Day 8-10)
19. Full LegalBench-RAG-mini benchmark runner
20. Metrics computation and reporting
21. CLI polish
22. Write-up of results

---

## Open Design Questions

1. **NLI model vs LLM-as-Judge for entailment**: NLI model is faster/cheaper and runs locally. LLM-as-Judge may be more nuanced for legal text. Could do both and compare. Start with NLI model, add LLM-as-Judge as a flag.

2. **Query expansion scope**: How aggressive? Legal terminology is precise — expanding too much could hurt precision. Start conservative (2 variants max).

3. **Cross-reference depth**: Follow references one level deep, or recursively? Start with one level, measure impact.

4. **Chunk size**: Structural boundaries may produce very large or very small chunks. Need min/max bounds. Suggest 200-1000 tokens, split large sections, merge tiny ones.

5. **Cost management**: Contextual enrichment requires one LLM call per chunk. For a 50-page contract with ~200 chunks, that's 200 Haiku calls (~$0.05). Acceptable for PoC. For production, consider batching or caching.

---

## Architectural Notes from Research

### On the RAG Ceiling (from MSA paper, 2026)

The retrieve-then-read pipeline architecture used here has a known performance ceiling: the retrieval objective (vector similarity) doesn't align with the generation/verification objective (does this passage support this claim?). EverMind's MSA paper provides strong empirical evidence — a 4B end-to-end trained model beats Qwen3-235B + SOTA reranker on most QA benchmarks.

**Implication for LegalKit**: The reasoning model (GRPO-trained Qwen) is the right long-term investment. As it improves, the cross-encoder becomes less critical — the reasoning model handles both "is this relevant?" and "is this entailed?" in one step. The current two-stage pipeline (cross-encoder → reasoning model) is pragmatic for now, but the reasoning model will eventually subsume the cross-encoder.

See: `docs/reference/msa-paper-notes.md`

### On Multi-Clause Verification (document-wise RoPE)

Stage 8 Check 3 (Entailment Verification) currently passes a single `<passage>` + `<claim>` pair. For claims that depend on multiple contract sections (e.g., a liability cap clause + its exceptions clause), a single-passage input is fundamentally insufficient.

When we extend to multi-passage inputs, use **document-wise RoPE**: assign independent position IDs (starting from 0) to each section rather than concatenating with global monotonically-increasing IDs. This prevents position ID explosion as more sections are added, and allows training on short multi-section examples while generalising to longer contracts.

### On Cross-Reference Resolution (Memory Interleave pattern)

Stage 6 (Cross-Reference Expansion) currently expands references one level deep via heuristic regex. A more principled approach — informed by MSA's Memory Interleave mechanism — would be to train the reasoning model to *identify* which cross-references it needs before reasoning, then fetch those sections, then reason. This mirrors how a lawyer reads a contract: follow references as needed rather than loading everything upfront.

Ablation evidence from MSA: removing iterative retrieval degrades multi-hop QA (HotpotQA) by 19.2%, with minimal impact on single-hop tasks. Our hardest eval categories (notwithstanding clauses, knowledge qualifiers) show the same profile — they're hard specifically because they involve implicit cross-references.
