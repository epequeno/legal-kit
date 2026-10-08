# Legal Entailment Benchmark Publication Infrastructure

**Date**: 2026-03-18  
**Status**: ✅ Complete — Ready for publication when benchmark dataset exists

---

## Summary

Built complete publication infrastructure for the Legal Entailment Benchmark, including:

1. ✅ **Pattern documentation** — Comprehensive research note on 14 legal NLI patterns
2. ✅ **HuggingFace publication script** — Automated dataset upload with dataset card generation
3. ✅ **Benchmark registry** — Updated reference documentation
4. ✅ **Leaderboard** — Community submission framework

All components are ready to use when `data/eval/legal_entailment_benchmark.json` becomes available.

---

## Deliverables

### 1. Pattern Documentation
**File**: `docs/research/legal-entailment-patterns.md`

Comprehensive 40KB research document covering:

- **Executive Summary**: What makes legal NLI different from general NLI
- **Pattern Taxonomy**: All 14 categories with training priorities
- **Detailed Patterns**: For each pattern:
  - Description and linguistic signature
  - Why general NLI fails (specific failure mode)
  - Example passage + entailed/contradicted claims
  - Training priority (1=critical, 2=important, 3=useful)
- **Failure Analysis**: DeBERTa-base (63.6%) vs fine-tuned (87.9%) vs LLM (~90%)
- **Training Recommendations**: Data priorities by pattern (500+ examples for P1, 300+ for P2, 100+ for P3)
- **Benchmark Design Notes**: Methodology, label definitions, difficulty tiers, QA process

**Key insight**: Fine-tuning with hard negatives achieves **97.7% of LLM accuracy** at zero cost by teaching legal structure reading.

### 2. HuggingFace Publication Script
**File**: `scripts/publish_benchmark_to_hf.py`

Automated publication tool that:

- ✅ Loads and validates `data/eval/legal_entailment_benchmark.json`
- ✅ Creates/updates HF repository at `epequeno/legal-entailment-benchmark`
- ✅ Uploads dataset JSON file
- ✅ Generates comprehensive README.md (dataset card) with:
  - YAML metadata header (task_categories, language, tags)
  - Dataset description and motivation
  - Usage examples (loading, evaluation, filtering)
  - Label definitions and category descriptions
  - Baseline results (DeBERTa-base, fine-tuned, LLM)
  - Citation information
- ✅ Graceful failure if benchmark file doesn't exist yet (with instructions)

**Usage**:
```bash
# Default repository
uv run python scripts/publish_benchmark_to_hf.py

# Custom repository
uv run python scripts/publish_benchmark_to_hf.py --repo username/repo-name

# Private repository
uv run python scripts/publish_benchmark_to_hf.py --private
```

**Authentication**: Requires `huggingface-cli login` with write access.

### 3. Benchmark Registry Update
**File**: `docs/reference/benchmarks.md`

Added new section at the top of the benchmarks document:

- **Overview**: Purpose, size, task type, categories
- **Pattern Categories**: Table of 14 patterns with priorities
- **Baseline Results**: Comparison table (base, fine-tuned, LLM)
- **Per-Category Performance**: Top improvements after fine-tuning
- **Usage**: Code examples and CLI commands
- **Related Resources**: Links to model, datasets, research notes

Positioned before the existing LegalBench-RAG section as the primary legal NLI benchmark.

### 4. Leaderboard Document
**File**: `docs/research/legal-entailment-leaderboard.md`

Community-facing leaderboard with:

- **Current Results**: Full benchmark (projected) and 33-case subset (measured)
- **Per-Category Breakdown**: Performance by legal pattern with improvement indicators
- **Failure Analysis**: Detailed breakdown of fine-tuned model errors
- **How to Submit Results**: Step-by-step submission guide with CLI commands
- **Model Categories**: Zero-shot, fine-tuned, LLM-as-judge
- **Benchmark Design Philosophy**: What it tests (and doesn't test)
- **Research Questions**: Key questions the leaderboard tracks
- **Contribution Guidelines**: How to extend the benchmark
- **Citation**: BibTeX entry

**Key features**:
- Clear submission process using `legalkit nli-eval`
- Reproducibility requirements
- Per-category tracking
- Community-friendly formatting

---

## File Structure

```
docs/
├── reference/
│   └── benchmarks.md                          # Updated with new benchmark
└── research/
    ├── legal-entailment-patterns.md           # NEW: Pattern taxonomy (40KB)
    └── legal-entailment-leaderboard.md        # NEW: Leaderboard (9KB)

scripts/
└── publish_benchmark_to_hf.py                 # NEW: Publication script (17KB)
```

---

## Pattern Taxonomy Overview

The documentation covers 14 legal NLI patterns across 3 priority tiers:

### Priority 1 — Critical (5 patterns)
- **notwithstanding**: Override clauses (40% → 100% after fine-tuning)
- **shall_vs_may**: Mandatory vs discretionary (40% → 100%)
- **quantitative_precision**: Exact numbers (75% → 100%)
- **condition_precedent**: Conditional obligations (67% → 67% — still hard)
- **knowledge_qualifier**: Epistemic limitations (100%* → 80%)

### Priority 2 — Important (5 patterns)
- **mae_carveout**: Material Adverse Effect exclusions (60% → 80%)
- **defined_term**: Contract-specific definitions (67% → 100%)
- **temporal_qualifier**: Time periods and deadlines (67% → 67%)
- **scope_limitation**: Restrictive clauses
- **negation_detection**: Logical negation in complex clauses

### Priority 3 — Useful (4 patterns)
- **anti_assignment**: Assignment restrictions
- **survival_clause**: Post-termination obligations
- **indemnification_scope**: Indemnification triggers
- **representation_warranty**: Reps vs covenants

---

## Key Findings from Documentation

### 1. Hard Negative Training Works
Fine-tuning with topically similar but logically contradicted claims successfully teaches legal structure reading:
- Notwithstanding: 40% → 100% (+60pp)
- Shall vs may: 40% → 100% (+60pp)
- Quantitative precision: 75% → 100% (+25pp)

### 2. Remaining Gaps Are Different
Temporal arithmetic and complex conditionals remain at 67% — fundamentally harder, may require:
- Specialized temporal normalizer
- More training data
- Hybrid approach (extract + normalize + compare)

### 3. Production-Ready
87.9% binary accuracy at 139ms latency and zero marginal cost makes the fine-tuned model suitable for deployment as primary citation verifier with LLM fallback for low-confidence cases.

### 4. ROI Is Clear
At 1,000 citations/month:
- Fine-tuned model: $0 marginal cost
- LLM-as-judge: $200/month
- **Savings: $2,400/year** while maintaining 87.9% accuracy (vs ~90% for LLM)

---

## Next Steps

### Immediate (when benchmark file exists)
1. Run publication script: `uv run python scripts/publish_benchmark_to_hf.py`
2. Verify dataset card renders correctly on HuggingFace
3. Test loading instructions from README
4. Update leaderboard with full benchmark results (currently shows 33-case projections)

### Future Enhancements
1. **Temporal reasoning module**: Address the 67% ceiling on temporal qualifier patterns
2. **Conditional logic dataset**: More training examples for condition precedent
3. **Confidence calibration**: Track prediction confidence for LLM fallback thresholds
4. **Multilingual expansion**: Extend benchmark to other legal languages (civil law systems)
5. **Additional patterns**: Identify new failure modes from production use

---

## Testing

✅ All tests pass (719 passed, 0 failed)
```bash
uv run pytest tests/ -q 2>&1 | tail -5
# 719 passed, 7 warnings in 6.59s
```

No regressions introduced by the new documentation and scripts.

---

## Related Work

**Existing evaluation infrastructure**:
- `data/eval/legal_nli_edge_cases.json` — 33 hand-labeled edge cases (frozen test set)
- `data/eval/legal_entailment_patterns.json` — 96 training pattern catalog
- `src/legalkit/eval/nli_eval.py` — Evaluation framework
- `docs/research/nli-evaluation-findings.md` — Base model results
- `docs/research/finetuned-model-evaluation.md` — Fine-tuned model results

**HuggingFace assets**:
- Model: `epequeno/legal-entailment-deberta-v3-large` (already published)
- Dataset: `epequeno/legal-entailment-benchmark` (ready for publication)

**Publication readiness**:
- ✅ Authentication: `huggingface-cli login` with write access to `epequeno` account
- ✅ Model published: [`epequeno/legal-entailment-deberta-v3-large`](https://huggingface.co/epequeno/legal-entailment-deberta-v3-large)
- ⏳ Dataset pending: Waiting for `data/eval/legal_entailment_benchmark.json` from parallel thread

---

## Success Criteria

All deliverables meet the specification:

1. ✅ **Pattern documentation** (`docs/research/legal-entailment-patterns.md`)
   - Executive summary ✓
   - 14-pattern taxonomy with descriptions, signatures, failure modes, examples ✓
   - Failure analysis by model type ✓
   - Training recommendations by priority ✓
   - Benchmark design notes ✓

2. ✅ **HuggingFace publication script** (`scripts/publish_benchmark_to_hf.py`)
   - Loads and validates benchmark JSON ✓
   - Creates HF dataset repo ✓
   - Uploads dataset file ✓
   - Generates comprehensive README with YAML header, usage examples, baselines ✓
   - Graceful failure with instructions if file missing ✓

3. ✅ **Benchmark registry update** (`docs/reference/benchmarks.md`)
   - New section for Legal Entailment Benchmark ✓
   - Overview, categories, baseline results, per-category performance ✓
   - Usage examples and related resources ✓
   - Positioned appropriately in the document ✓

4. ✅ **Leaderboard document** (`docs/research/legal-entailment-leaderboard.md`)
   - Current results table ✓
   - Per-category breakdown ✓
   - Failure analysis ✓
   - Submission instructions using `legalkit nli-eval` ✓
   - Contribution guidelines and citation ✓

---

**Infrastructure Status**: ✅ **READY FOR PUBLICATION**

When `data/eval/legal_entailment_benchmark.json` exists, run:
```bash
uv run python scripts/publish_benchmark_to_hf.py
```

The benchmark will be published to HuggingFace with complete documentation and community submission framework.
