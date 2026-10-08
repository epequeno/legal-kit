# Paper Notes: MSA — Memory Sparse Attention

**Full title**: MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens  
**Authors**: Yu Chen, Runkai Chen et al. (Evermind / Shanda Group / Peking University)  
**Source**: not included — the paper PDF was not redistributed with this repository.  
**Date read**: 2026-03-22

---

## What It Does

MSA is an end-to-end trainable sparse attention architecture that scales LLM memory to 100M token contexts — approaching human lifelong memory capacity (~200-300M tokens estimated from cognitive science).

Built on **Qwen3-4B**, it outperforms SOTA RAG systems (including Qwen3-235B + KaLMv2 reranker) on most QA benchmarks at 60× fewer parameters, and maintains <9% accuracy degradation scaling from 16K to 100M tokens.

---

## Core Ideas

### 1. Sparse Attention via Learned Document Routing
Instead of attending to the full context, each attention layer routes to the top-k most relevant document chunks:
- A dedicated **Router K Projector** (W_KR) generates routing keys per document chunk
- A **Router Q Projector** (W_QR) generates a routing query from the user's question
- Cosine similarity selects Top-k documents; only their compressed KVs are loaded for attention
- Applied only to the **latter half** of model layers (early layers don't have semantic abstractions useful for retrieval)

### 2. Document-wise RoPE (Key Technique)
Standard global RoPE assigns monotonically increasing position IDs across all concatenated documents. When the number of documents grows at inference time (beyond what was seen in training), position IDs shift drastically → severe degradation.

MSA fix: **assign independent position IDs (starting from 0) to each document**. Each document is self-contained positionally. The active query/generation context uses Global RoPE, offset by the number of retrieved documents.

Result: train on 64K contexts, extrapolate to 100M tokens with no position ID explosion.

### 3. KV Cache Compression
Chunk each document into 64-token blocks and mean-pool K, V, and K_R within each chunk. This compresses the memory bank by 64× before storage, enabling 100M tokens on 2×A800 GPUs.

### 4. Memory Interleave (Multi-hop Reasoning)
For complex queries requiring evidence chains, inference alternates:
1. Generate document IDs (which docs are relevant?) 
2. Retrieve those docs, append to query context
3. Repeat until model decides it has enough evidence
4. Generate final answer

Ablation: removing Memory Interleave degrades HotpotQA (multi-hop) by **19.2%**. Single-hop tasks are barely affected.

### 5. Two-Stage Curriculum Training
- Stage 1: SFT at 8K context → establish instruction-following baseline
- Stage 2: Extend to 64K context with cleaned data → enables 100M inference extrapolation

MSA-S2 (both stages) outperforms MSA-S1 (stage 1 only) by 7.6% average, with **29.5% gain on MS MARCO** (7.34M token memory bank).

---

## Key Results

| Model | Params | Avg QA Score | Notes |
|---|---|---|---|
| MSA | 4B | **3.760** | End-to-end, @adaptive retrieval |
| KaLMv2 + Qwen3-235B (RR) | 235B | 3.580 | Best-of-breed RAG |
| KaLMv2 + Llama3.3-70B (RR) | 70B | 3.568 | |
| Same-backbone RAG (Qwen3-4B) | 4B | 3.372 | Pipeline baseline |

NIAH (Needle in Haystack): MSA maintains **94.84% accuracy at 1M tokens** vs baseline Qwen3-4B crashing to 24.69%.

---

## Why End-to-End Beats Pipeline RAG

The paper provides strong empirical validation of a key architectural principle:

> Discrete retrieval (embedding → top-k → separate LLM reader) has a fundamental performance ceiling because the retrieval objective doesn't align with the generation objective. End-to-end differentiation closes this gap.

MSA's routing is trained with a supervised contrastive auxiliary loss (L_aux) that explicitly aligns the routing space with the task. RAG's retrieval is based on model-agnostic embeddings that only weakly correlate with what the reader actually needs.

---

## Relevance to LegalKit

### High relevance — future architecture

**1. Validates the "reasoning model > better RAG" direction**  
LegalKit's current citation verification pipeline: embedding retrieval → cross-encoder → (optional) reasoning model. MSA's finding that a 4B end-to-end model beats 235B pipeline RAG supports investing in the reasoning model rather than endlessly improving the retrieval step.

See: `docs/architecture/poc-architecture.md`

**2. Document-wise RoPE — directly applicable to multi-clause verification**  
When verifying a claim that spans multiple contract sections (e.g., "the liability cap applies unless fraud"), the model currently sees them as one long concatenated sequence with global position IDs. If we extend to multi-clause inputs, document-wise RoPE prevents position ID explosion and enables training on short inputs while generalizing to longer ones.

Relevant for: next model version if we expand beyond single-passage inputs.

**3. Memory Interleave maps to contract cross-referencing**  
Legal documents are full of cross-references: §8.1 sets a cap, §8.3 lists exceptions, §1.1 defines terms. MSA's iterative retrieve-reason-retrieve loop directly models this pattern. The weakest categories in our eval (temporal qualifier, knowledge qualifier, notwithstanding) are often hard because of implicit cross-references the model has no way to resolve with a single-passage input.

Relevant for: long-term product vision where the model can follow references through a full contract.

**4. Two-stage curriculum (short → long context)**  
MSA's curriculum exactly mirrors the LegalKit approach: SFT on short examples → GRPO/extended training on longer contexts. Their 29.5% gain on large memory banks from the second stage supports the value of context-length curriculum.

### Not currently relevant

- The 100M token scale is far beyond clause-level verification (100-500 tokens per case)
- 158B token continuous pre-training is not feasible for LoRA fine-tuning
- Full MSA architecture requires new router projectors and modified attention — a significant model surgery

---

## Ideas to Steal

1. **Document-wise RoPE for multi-clause inputs** — low implementation cost, high payoff if we expand to multi-section verification. Add to the model training config when we move beyond single-passage.

2. **Router-style auxiliary loss for retrieval** — if we train a retrieval model, the L_aux supervised contrastive loss (positive documents vs negative documents) is a clean way to align retrieval with the reasoning objective. Better than cosine similarity on general-purpose embeddings.

3. **Iterative retrieval framing for hard cases** — for the ~15% of cases our model gets wrong, many involve cross-references. A two-pass approach (first pass: identify if cross-reference needed; second pass: resolve it) could target exactly these failure modes.

4. **Mean-pool KV compression** — if we ever need to handle full contracts (~50K tokens), chunked mean-pool compression is a principled way to fit them in context without truncation.
