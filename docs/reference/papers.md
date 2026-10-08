# Papers

The reading list behind the retrieval and RL work. **Links only** — full text is not
redistributed here.

| Paper | Link | Why it mattered |
|---|---|---|
| LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain | [arXiv:2408.10343](https://arxiv.org/abs/2408.10343) | The retrieval benchmark behind every R@k and P@k number in this repo |
| Drowning in Documents: Consequences of Scaling Reranker Inference | [arXiv:2411.11767](https://arxiv.org/abs/2411.11767) | Directly relevant to the P@1 bottleneck — reranker inference cost vs. ranking gain |
| DAPO: An Open-Source LLM Reinforcement Learning System at Scale | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) | Reference system for the GRPO-style training runs |
| Tricks or Traps? A Deep Dive into RL for LLM Reasoning | [arXiv:2508.08221](https://arxiv.org/abs/2508.08221) | Survey of what actually moves reasoning accuracy under RL |
| Beyond Correctness: Harmonizing Process and Outcome Rewards through RL Training | [arXiv:2509.03403](https://arxiv.org/abs/2509.03403) | Process vs. outcome reward shaping for the reasoning model |
| Entropy Collapse (Arbitrary Entropy Policy Optimization) | [arXiv:2510.08141](https://arxiv.org/abs/2510.08141) | Diagnosing the entropy collapse seen in the GRPO runs |
| Beyond Reasoning Gains: Mitigating General Capabilities Forgetting in RLVR | [arXiv:2510.21978](https://arxiv.org/abs/2510.21978) | Capability forgetting as a risk of RLVR fine-tuning |
| A Unified Framework for Rethinking Policy Divergence Measures in GRPO | [arXiv:2602.05494](https://arxiv.org/abs/2602.05494) | KL/divergence measures and what the choice of one implies |
| LawThinker: A Deep Research Legal Agent in Dynamic Environments | [arXiv:2602.12056](https://arxiv.org/abs/2602.12056) | The closest published system to this problem; used as a comparison point |

Related notes: [`msa-paper-notes.md`](msa-paper-notes.md) (MSA — Memory Sparse Attention,
read 2026-03-22), and the annotated bibliography in [`benchmarks.md`](benchmarks.md).
