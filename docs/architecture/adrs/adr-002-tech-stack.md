# ADR-002: PoC Technology Stack

## Status

Superseded by [[adr-009-native-mac-app|ADR-009]] (2026-03-19)

The Python/FastAPI backend has been replaced with a Rust/Tauri native Mac app.
The core pipeline logic is preserved but ported to Rust.

## Date

2025-03-17

## Context

We need to choose technologies for the citation engine PoC. The constraints are:
- CLI-only, no web UI in this phase
- Must integrate with LLM APIs and local NLI models
- Must run the LegalBench-RAG benchmark
- Speed of development is paramount — this is a validation exercise
- Frontend (when built) will use Elm + elm-css (decided separately, see [[adr-006-frontend-elm|ADR-006]])

## Decision

### Backend (PoC)

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.12 | AI/NLP ecosystem is Python-native. LegalBench-RAG tooling is Python. |
| CLI framework | `click` | Simple, well-documented, no magic |
| Package manager | `uv` | Fast, modern, handles venvs and lockfiles |
| LLM (generation) | Anthropic Claude 3.5 Sonnet | Best reasoning quality, strong instruction following for citation format |
| LLM (contextualization) | Anthropic Claude 3 Haiku | Cheap (~$0.25/M input tokens), fast, sufficient for context generation |
| Embeddings | OpenAI `text-embedding-3-large` | Best general quality; swap for legal-specific model later if needed |
| BM25 | `rank_bm25` (in-memory) | Zero infrastructure, good enough for PoC scale |
| NLI verification | `microsoft/deberta-v3-large-mnli` via HuggingFace | Strong NLI performance, runs locally, no API cost for verification |
| Vector store | `numpy` cosine similarity (in-memory) | No database dependency for PoC |
| Document parsing | Custom regex-based | Legal document structure follows predictable patterns |

### What We Explicitly Defer

- PostgreSQL / pgvector — unnecessary complexity for PoC
- PDF/DOCX parsing — PoC works with text files (LegalBench-RAG corpus is text)
- Docker / deployment infrastructure
- Multi-document workspaces
- Authentication / multi-tenancy
- Fine-tuned models

## Alternatives Considered

### LangChain / LlamaIndex
Rejected. These frameworks add abstraction layers that make debugging harder. For a PoC where we need to understand and control every step of the pipeline, direct API calls are better. We can evaluate frameworks for the production system.

### Local LLMs (Ollama)
Deferred. PoC needs best-possible quality to validate the thesis. If citation accuracy works with Claude Sonnet, we can evaluate open models later. If it doesn't work with the best model, it won't work with lesser ones.

### PostgreSQL + pgvector from the start
Rejected for PoC. Adds operational complexity. In-memory numpy is sufficient for the LegalBench-RAG corpus size (~79M chars, ~200-1000 chunks per document). Will be needed for production.

## Consequences

### Positive
- Minimal infrastructure — clone, install, run
- Fast iteration cycle
- Direct control over every pipeline step
- Easy to debug and profile

### Negative
- In-memory approach won't scale beyond PoC (hundreds of documents)
- API costs for LLM calls (~$0.05-0.10 per document ingestion, ~$0.02-0.05 per query)
- DeBERTa requires a GPU or will be slow on CPU (acceptable for PoC)
