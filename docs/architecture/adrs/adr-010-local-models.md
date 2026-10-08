# ADR-010: Bundled Local Models + Optional Cloud API

## Status

Accepted

## Date

2026-03-19

## Context

LegalKit's privacy positioning ("files never leave your machine") is undermined if queries must be
sent to OpenAI for embeddings. The custom `legal-embeddings-bge-base` model was fine-tuned on
9,409 legal query-passage pairs and benchmarked against `text-embedding-3-large`:

| Model | R@64 | P@1 |
|---|---|---|
| `text-embedding-3-large` (OpenAI API) | 97.93% | 5.49% |
| `legal-embeddings-bge-base` (local) | **99.03%** | **10.93%** |

Our model is better, smaller (110M params, ~440MB), and runs locally. There is no reason to call
the OpenAI embedding API by default.

## Decision

### Embeddings — local by default

Bundle `legal-embeddings-bge-base` with the app. Use HuggingFace `candle` crate to run inference
natively in Rust via Metal on Apple Silicon (or CPU fallback). No API key required for embeddings.

The embedding model is BERT-architecture with mean pooling, weights in `.safetensors` format.
`candle-transformers` supports this directly — no Python, no ONNX export, no subprocess.

```
models/legal-embeddings/   → bundled in app binary (or downloaded on first launch)
models/legal-entailment/   → bundled for citation verification
```

### Generation — user's choice, three tiers

| Tier | How | Privacy | Quality |
|---|---|---|---|
| **Ollama (local)** | llama3.3, qwen2.5, etc. via localhost:11434 | Complete — nothing leaves machine | Good; best on M3 Pro+ |
| **OpenRouter** | Single key, OpenAI-compatible API | Data sent to OpenRouter/Anthropic | Excellent |
| **Direct APIs** | Separate OPENAI_API_KEY + ANTHROPIC_API_KEY | Data sent to respective providers | Excellent |

Priority at runtime: Ollama (if detected) → OpenRouter → Direct APIs.

App detects Ollama by checking `GET http://localhost:11434/api/tags` at startup. If running,
offers a model picker populated from installed models. If not, falls back to API key prompt.

### Key detection at startup

```
OPENROUTER_API_KEY   → use for both embeddings fallback and generation
OPENAI_API_KEY       → embeddings fallback only
ANTHROPIC_API_KEY    → generation only (requires OPENAI_API_KEY for embeddings)
```

Embeddings only fall back to the API if the local model fails to load (e.g., first launch before
model is downloaded, or corrupted model file).

### Citation verification — local

`legal-entailment-deberta-v3-large` runs locally for citation verification. This is the NLI model
that checks whether a cited passage actually supports the generated claim. No API call needed.

## Consequences

### Positive
- Embedding quality is better than the paid API — no tradeoff to make
- Complete air-gap mode is possible: local embeddings + Ollama = zero network calls
- "No API key needed to try it" dramatically lowers the barrier to first use
- M-series Mac runs 110M BERT model fast enough to be imperceptible to the user
- Privacy story is unambiguous: "embeddings are computed on your machine"

### Negative
- App bundle is larger: ~440MB for embedding model, ~1.4GB for entailment model
- First-launch model download UX needs to be handled gracefully
- `candle` integration adds Rust complexity (tokenizer, model loading, inference loop)
- Generation quality with Ollama depends heavily on hardware; 7B models on older Macs are slow

### Model distribution options

1. **Bundle in binary** — simplest, large download (~2GB total)
2. **Download on first launch** — smaller initial download, requires internet on first use
3. **Lazy download** — download each model the first time it's needed

Recommendation: download on first launch with a progress indicator. Models are cached in
`~/.legalkit/models/`. User can re-download if corrupted.
