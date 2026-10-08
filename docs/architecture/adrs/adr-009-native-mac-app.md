# ADR-009: Native Mac App (Tauri + Rust)

## Status

Accepted — supersedes ADR-002 (Python/FastAPI stack) and ADR-006 (Elm web frontend)

## Date

2026-03-19

## Context

The Python/FastAPI web app architecture (ADR-002) hit a fundamental problem: browser-based file upload
requires a running local server, proxy configuration, and networking between the frontend dev server
and the API server. The first real test — uploading a contract — failed silently due to a URL mismatch
between the frontend and backend. This is a symptom of the wrong architecture, not a fixable bug.

More importantly, a competitive landscape review revealed:

- Every major legal AI tool (Harvey, Luminance, Kira, Spellbook, Ironclad, Evisort) is web SaaS
- The #1 objection to these tools from law firms is data privacy — attorney-client privilege, NDA exposure,
  deal confidentiality — files passing through third-party servers
- There is no serious native desktop competitor in the legal AI space
- The target early adopter segment (in-house counsel at tech companies, boutique M&A, solo practitioners)
  skews heavily Mac

A native Mac app solves the distribution problem (no server to run), the privacy problem (files never
leave the machine), and the cold-start problem (simpler to try than any web app that requires setup).

## Decision

Build LegalKit as a native Mac desktop application using **Tauri 2** (Rust backend, Elm frontend).

### Architecture

```
tauri-app/
  src-tauri/src/
    pipeline/       Rust: extract → parse → chunk → embed → index → retrieve → generate
    store.rs        Flat-file persistence: ~/.legalkit/docs/<id>/{meta,chunks,embeddings}
    commands.rs     Tauri IPC commands exposed to frontend
  src/              Elm frontend
  dist/             Compiled Elm + static assets (served by Tauri webview)
```

### Why Tauri over alternatives

| Option | Verdict |
|---|---|
| SwiftUI | Best native result, but requires learning Swift; team has no Swift experience |
| Electron | Cross-platform but ~150MB overhead, Chromium bundled, slow startup |
| Tauri | Rust backend + web frontend, ~5-15MB binary, uses system WebView, team knows web |
| Python GUI (customtkinter, textual) | Pure Python but limited distribution story, not native feel |

### Why Rust for the pipeline

The entire pipeline (parse, chunk, embed, index, retrieve, generate) is HTTP calls and data
transformation — no ML inference in Python. There is no PyTorch, no ONNX, no GPU runtime. This
makes the pipeline highly portable to Rust:

- `regex` crate — same syntax as Python's `re` module; all 6 parse patterns port directly
- `reqwest` — async HTTP for OpenAI/Anthropic/OpenRouter/Ollama API calls
- BM25 + cosine similarity — implemented from scratch in ~80 lines; no external ML libs needed
- `candle` (HuggingFace) — runs our `.safetensors` embedding model natively via Metal on Apple Silicon

### Why Elm stays

The Elm frontend from the web app is retained and adapted. The only change is replacing `Http` module
calls with Tauri ports (`window.__TAURI__.core.invoke`). A thin JS bridge (`main.js`) connects
the two. All reasoning for Elm from ADR-006 still applies.

### File access model

In a native app, the user selects a file via the OS native file picker (no upload). Tauri reads the
file directly from disk. No multipart encoding, no temp files, no port configuration.

### Persistence

Documents are stored in `~/.legalkit/docs/<doc-id>/`:
- `meta.json` — filename, size, chunk count, timestamp
- `chunks.json` — serialized chunks with structural metadata
- `embeddings.bin` — raw f32 array (little-endian, row-major)

This is intentionally simple — no SQLite, no external database. Fast to implement, easy to inspect,
straightforward to migrate if requirements grow.

## Consequences

### Positive
- Files never leave the machine — strongest possible privacy story
- No server to run, no ports to configure — single binary launch
- Native file picker replaces the broken upload widget
- Distribution via direct download or Mac App Store (notarization required)
- Rust pipeline is faster than Python for parse/chunk/index operations
- Apple Silicon (M-series) Metal acceleration for local embedding model inference

### Negative
- Rust learning curve — more verbose than Python for data manipulation
- Mac-only initially — Windows/Linux require additional testing and potential WinAPI work
- Mac App Store sandboxing may restrict arbitrary filesystem access; direct distribution avoids this
- No web version — users can't access from multiple machines (intentional for v1)

### Deferred

- Windows/Linux support (Tauri is cross-platform; the main work is CI/CD and testing)
- Mac App Store submission (requires notarization, sandbox review, icon assets)
- Multi-window support
