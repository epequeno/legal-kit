# LegalKit

**Contract analysis that runs entirely on your Mac.** Open a contract, ask a question, get an
answer with citations you can check. No document leaves the machine.

> **Status: archived, not maintained.** Development stopped 2026-04-17. This repository is
> published as a record of the work, not as a supported product. Read
> [What is not in this repository](#what-is-not-in-this-repository) before trying to build it —
> the model weights are not here and the app will not run without them.

## Can you run this?

**No, and the reason is stated rather than left to be discovered.** The model weights, the training and
evaluation corpora, and the GRPO adapters are not in this repository — some for licence reasons, some
because they were never committed. `just build` fails for exactly that reason; the full list is under
[What is not in this repository](#what-is-not-in-this-repository).

What you *can* read: the code, [`docs/history/`](docs/history/) — the decisions and what was learned —
and the evaluation write-ups. The `justfile` (`just --list`) still describes how the app was built,
trained and evaluated, which is a record of the work rather than a promise that it runs today.

## What it does

A Rust + Tauri desktop app with an Elm frontend, running local models on Metal with an
optional cloud LLM for generation.

```
File (TXT / PDF / DOCX)
  → extract text
  → parse structure     regex + structure-parser model (BERT-base)
  → chunk               boundary-aware, 1500 char max
  → embed               legal-embeddings (BGE-base, local/Metal) or an API
  → index               cosine + BM25, in-memory
  → persist             ~/.legalkit/docs/<id>/
Query
  → embed query
  → hybrid retrieve     RRF: semantic + BM25 + cross-reference expansion
  → generate            Anthropic / OpenRouter / Ollama
  → verify citations    legal-entailment model (DeBERTa-v3-large), one NLI check per citation
  → return report       every claim either cited-and-verified, or flagged
```

The generation step is the easy part. The work here is the **verification layer**: every
citation is resolved back to the chunk the model actually saw, then checked for entailment by
a local NLI model, so the app can say *"I could not verify this"* instead of presenting it
confidently. Citations are resolved chunk-first with fuzzy ID matching, which is what moved
verification from 50% to 88% ([`docs/research/e2e-evaluation-findings.md`](docs/research/e2e-evaluation-findings.md)).

## Results

Small-sample PoC numbers, with the caveats that belong beside them.

| What | Result | Scope and caveat |
|---|---|---|
| Citation verification rate | 88% | 20 queries; 14% pass *strict* verification |
| Hallucination rate | **11%** | 20 queries; vs >17% for LexisNexis and Thomson Reuters AI (Stanford 2025) |
| RAG vs full-context, strict correctness | 65% vs 55% | 20 queries across 4 datasets |
| legal-entailment, 4-class accuracy | 75.8% fine-tuned vs 39.4% DeBERTa-base | 33 hand-built legal edge cases, in-distribution |
| Retrieval R@64 | 95.5% held-out | **a chunk-size effect, not embedding quality** |
| Retrieval P@1 | 1.6% held-out | the real, unsolved bottleneck |

The full write-ups are in [`docs/research/`](docs/research/), and each number above is
reproducible from the raw runs in [`eval/`](eval/).

### A result I published, then retracted

The first evaluation of `legal-embeddings` reported **99.03% R@64, beating
`text-embedding-3-large` at 97.93%** — and it was published in those terms, in a
HuggingFace model card and in ADR-010.

A leave-one-dataset-out re-evaluation showed the claim was wrong. On datasets the model was
**not** trained on, the fine-tune is within noise of the base BGE model and slightly *worse*:
−0.3pp R@64, −0.7pp R@16. The original figure measured in-distribution overfitting.

Two things were hidden behind that number:

1. **The high recall is a chunk-size artifact.** Paragraph-aware chunks of ~1500 chars
   mechanically capture most ground-truth spans regardless of embedding quality. The
   published LegalBench-RAG baseline uses 500-char chunks, so the head-to-head comparison is
   not meaningful in either direction.
2. **Recall was never the problem.** P@1 is 1.6% held-out — the first result returned is
   almost always the wrong chunk. A cross-encoder reranker was the planned fix.

See [`docs/research/lodo-embedding-eval-findings.md`](docs/research/lodo-embedding-eval-findings.md).
ADR-010 is left accepted and unedited, because the record of a decision that was later
corrected is more useful than a tidied one — ADR-010 says what was believed on 2026-03-19,
and this README plus the LODO write-up carry the correction.

## Architecture decisions

11 ADRs in [`docs/architecture/adrs/`](docs/architecture/adrs/index.md), covering the
citation-first development strategy, hybrid retrieval, the verification pipeline, the pivot
from Python/FastAPI to a native Tauri app (ADR-009), bundled local models (ADR-010), and the
UI design direction (ADR-011).

## Repository layout

```
tauri-app/     Native Mac app — Rust core (17 files) + Elm frontend (17 files)
python/        Python reference implementation: the original PoC, 59 modules, 820 test functions
eval/          Raw evaluation runs (JSON) behind every number above
scripts/       Training and evaluation scripts (EC2 / Modal)
infra/         AWS CDK stack for training infrastructure
docs/          ADRs, research write-ups, benchmarks, glossary
```

## What is not in this repository

- **Model weights.** `models/` contains a README and nothing else. The three models the app
  expects — `legal-embeddings`, `legal-entailment`, `structure-parser` — are not included.
  A "download on first launch" flow was designed and never built.
- **Training and evaluation data.** The corpora, LegalBench-RAG splits and ContractNLI
  material are third-party datasets with their own terms, so they are not redistributed here.
  `scripts/` shows how they were consumed.
- **The reasoning-model adapters.** The GRPO fine-tunes described in `docs/history/` are not
  committed.

So: the code, the decisions and the evaluation write-ups are readable here; **the app is not
runnable from a fresh clone.** That is a deliberate choice at archive time, not an oversight.

## Reading path

1. [`docs/index.md`](docs/index.md) — everything, grouped, with the key numbers
2. [`docs/research/lodo-embedding-eval-findings.md`](docs/research/lodo-embedding-eval-findings.md) — the retraction above
3. [`docs/research/benchmark-results.md`](docs/research/benchmark-results.md) — retrieval, generation, verification, reasoning measured separately
4. [`docs/architecture/adrs/index.md`](docs/architecture/adrs/index.md) — the decisions
5. [`docs/reference/papers.md`](docs/reference/papers.md) — the reading list behind the RL work

## Tests

`python/tests/` holds 28 files and 820 test functions covering chunking, retrieval,
verification, calibration, the structure model and the benchmark harnesses.

**The Rust side has no tests.** The app was driven manually through the Tauri dev loop; the
CI workflow in `.github/workflows/ci.yml` runs an Elm type-check only.

## License

There is no `LICENSE` file in this repository.
[`docs/architecture/adrs/adr-008-ml-moat-strategy.md`](docs/architecture/adrs/adr-008-ml-moat-strategy.md)
records that the project was intended to be AGPL-3.0 — the pipeline itself is not the moat, so it
was deliberately open from the start — but the license file was never added before development
stopped. Nothing here is formally licensed.
