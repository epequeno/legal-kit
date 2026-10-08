# NLI Evaluation Expansion - Implementation Summary

## Created Files

### 1. `expand_nli_eval_set.py` (16KB)
**Purpose**: Generate an expanded NLI evaluation set from 33 to 243 cases

**Key Features**:
- Loads existing 33 hand-labeled cases from `data/eval/legal_nli_edge_cases.json`
- Marks original cases as `reviewed: true` and `source: "hand_labeled"`
- Generates 15 new cases per existing category (8 categories × 15 = 120 cases)
- Generates 15 cases per NEW OOD category (6 categories × 15 = 90 cases)
- Uses Claude Sonnet 4 API with few-shot prompting for quality
- Marks generated cases as `source: "llm_generated"` and `reviewed: false`
- Annotates cases with `distribution: "in_distribution"` or `"out_of_distribution"`
- Outputs to `data/eval/legal_nli_eval_expanded.json`

**Original 8 Categories (In-Distribution)**:
1. notwithstanding
2. shall_vs_may
3. knowledge_qualifier
4. mae_carveout
5. quantitative_precision
6. temporal_qualifier
7. condition_precedent
8. defined_term

**New 6 Categories (Out-of-Distribution)**:
1. representations_warranties — Reps & warranties accuracy, survival, qualifiers
2. limitation_of_liability — Caps, exclusions, carve-outs
3. force_majeure — Trigger events, notice, mitigation
4. assignment_change_of_control — Consent requirements, permitted transfers
5. governing_law_jurisdiction — Choice of law, forum selection, arbitration
6. non_compete_non_solicit — Scope, duration, geographic limits

**Generation Strategy**:
- Selects 3 diverse few-shot examples per category (including label diversity)
- Prompts Claude to generate realistic legal contract text (50-200 words)
- Requires balanced label distribution across all 4 classes
- Validates JSON output and label correctness
- Provides detailed progress and summary statistics

**Total Output**: 243 cases (33 original + 210 generated)

---

### 2. `run_expanded_nli_eval.py` (14KB)
**Purpose**: Run fine-tuned DeBERTa model on expanded eval set with detailed breakdown

**Key Features**:
- Loads expanded eval set from `data/eval/legal_nli_eval_expanded.json`
- Runs fine-tuned model from `models/legal-entailment/model/` (or custom path)
- Computes detailed performance metrics:
  - Overall accuracy (4-class and binary)
  - In-distribution vs out-of-distribution breakdown
  - Per-category performance with ID/OOD labels
  - Per-label accuracy
  - Confusion matrix
- Highlights categories below accuracy threshold (default 70%)
- Saves results to `data/eval/nli-expanded-eval-<timestamp>.json`
- Supports GPU and CPU inference

**CLI Options**:
- `--model-path`: Custom model checkpoint path
- `--eval-path`: Custom eval set path
- `--device`: `cpu` or `cuda`
- `--no-save`: Skip saving results to JSON
- `--threshold`: Custom accuracy threshold for highlighting (default 0.70)

**Output Metrics**:
- Overall: total cases, 4-class accuracy, binary accuracy
- By distribution: ID cases, OOD cases, performance gap
- By category: cases per category, accuracy, distribution type
- By label: per-label accuracy
- Confusion patterns: most common misclassifications

**Progress Tracking**: Real-time progress bar during inference

---

### 3. `README_NLI_EVAL_EXPANSION.md` (11KB)
**Purpose**: Comprehensive documentation for the expansion workflow

**Contents**:
- Overview of expansion process
- Setup instructions (Python venv, dependencies)
- File descriptions with usage examples
- Complete workflow guide (Step 1: Generate, Step 2: Evaluate)
- Interpreting results and performance metrics
- Next steps (review, improve, continuous evaluation)
- Cost estimation for LLM generation
- Troubleshooting guide
- References to related files

**Key Sections**:
- Prerequisites and setup
- Category definitions (ID and OOD)
- Expected output examples
- Performance interpretation guidelines
- Review workflow for generated cases
- Continuous evaluation integration

---

## Implementation Details

### Data Format

**Expanded Eval Set Structure** (`legal_nli_eval_expanded.json`):
```json
{
  "version": "2.0",
  "description": "Expanded legal NLI evaluation set...",
  "generation_date": "2026-03-19T...",
  "generation_model": "claude-sonnet-4-20250514",
  "categories": ["notwithstanding", ..., "non_compete_non_solicit"],
  "labels": ["entailed", "partially_entailed", "contradicted", "neutral"],
  "cases": [
    {
      "id": "nw-001",
      "category": "notwithstanding",
      "passage": "...",
      "claim": "...",
      "label": "entailed",
      "explanation": "...",
      "source": "hand_labeled",
      "reviewed": true,
      "distribution": "in_distribution"
    },
    {
      "id": "gen-now-042",
      "category": "notwithstanding",
      "passage": "...",
      "claim": "...",
      "label": "partially_entailed",
      "explanation": "...",
      "source": "llm_generated",
      "reviewed": false,
      "distribution": "in_distribution"
    },
    ...
  ]
}
```

**Evaluation Results Structure** (`nli-expanded-eval-<timestamp>.json`):
```json
{
  "timestamp": "20260319-143022",
  "model_name": "legal-deberta-large:model",
  "model_path": "models/legal-entailment/model",
  "eval_set": "data/eval/legal_nli_eval_expanded.json",
  "eval_metadata": {...},
  "results": {
    "overall": {
      "accuracy_4class": 0.782,
      "accuracy_binary": 0.893,
      "total_cases": 243
    },
    "by_distribution": {
      "in_distribution": {...},
      "out_of_distribution": {...}
    },
    "by_category": {...},
    "by_label": {...}
  },
  "confusion_matrix": {...},
  "judgments": [...]
}
```

### Generation Approach

**Few-Shot Prompting**:
- Selects 3 diverse examples from original 33 cases
- Includes at least 1 example from target category (if ID)
- Ensures label diversity in examples
- Provides category description and requirements

**Quality Controls**:
- JSON schema validation
- Label validation (must be one of 4 valid labels)
- Field completeness checks (passage, claim, label, explanation)
- Error handling for API failures
- Progress tracking per category

**Label Distribution**:
Target: ~25% per label class across all generated cases
- entailed: ~60 cases
- partially_entailed: ~60 cases
- contradicted: ~60 cases
- neutral: ~60 cases

### Evaluation Approach

**Model Loading**:
- Uses `transformers.AutoModelForSequenceClassification`
- Label mapping: `[0=entailed, 1=partially_entailed, 2=contradicted, 3=neutral]`
- Supports both local paths and HuggingFace model IDs
- GPU/CPU inference with torch device selection

**Metrics Computation**:
- 4-class: Exact label match
- Binary: Positive (entailed/partially) vs Negative (contradicted/neutral)
- Per-category: Independent accuracy for each category
- Confusion matrix: Full 4×4 matrix of predictions

**Performance Analysis**:
- ID/OOD comparison to measure generalization
- Category-level drill-down for failure analysis
- Label-specific accuracy for class imbalance detection
- Confusion pattern analysis for systematic errors

---

## Usage Quick Reference

### Generate Expanded Set
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
cd python && source .venv/bin/activate && cd ..
python scripts/expand_nli_eval_set.py
```

### Run Evaluation
```bash
cd python && source .venv/bin/activate && cd ..
python scripts/run_expanded_nli_eval.py --device cuda
```

### Review Generated Cases
```python
import json
with open('data/eval/legal_nli_eval_expanded.json') as f:
    data = json.load(f)
unreviewed = [c for c in data['cases'] if not c.get('reviewed')]
print(f"{len(unreviewed)} cases need review")
```

---

## Expected Results

### Generation Output
- 243 total cases (33 original + 210 generated)
- 153 in-distribution cases (33 original + 120 generated)
- 90 out-of-distribution cases (all generated)
- Roughly balanced label distribution (~25% per label)
- Cost: ~$0.85 using Claude Sonnet 4

### Evaluation Results (Expected Ranges)

**Overall Performance**:
- 4-class accuracy: 75-85%
- Binary accuracy: 85-95%

**In-Distribution**:
- 4-class accuracy: 80-90% (model trained on these patterns)
- Binary accuracy: 90-95%

**Out-of-Distribution**:
- 4-class accuracy: 65-75% (novel patterns, generalization test)
- Binary accuracy: 80-90%

**Performance Gap**:
- Expected ID/OOD gap: 10-20% (indicates some overfitting to training categories)
- Gap > 25%: May indicate poor generalization
- Gap < 5%: Excellent generalization

**Label-Specific**:
- `entailed`: 80-90% (clearest case)
- `contradicted`: 75-85%
- `neutral`: 70-80%
- `partially_entailed`: 60-70% (most challenging due to subtle qualifiers)

---

## Integration Points

### Existing Code
- **Imports from**: `legalkit.eval.nli_eval` (load_edge_cases, run_finetuned_nli, NLITestCase, NLIEvalResult)
- **Model path**: `models/legal-entailment/model/` (from modal_finetune.py)
- **Original eval set**: `data/eval/legal_nli_edge_cases.json`
- **Test suite**: `python/tests/test_nli_eval.py` (for validation)

### New Outputs
- **Expanded eval set**: `data/eval/legal_nli_eval_expanded.json` (reusable across runs)
- **Evaluation results**: `data/eval/nli-expanded-eval-<timestamp>.json` (timestamped for comparison)

### Dependencies
- `anthropic` (for generation, already in pyproject.toml)
- `transformers` (for evaluation, already in pyproject.toml)
- `torch` (for model inference, already in pyproject.toml)

---

## Next Steps

1. **Generate the expanded set**:
   ```bash
   python scripts/expand_nli_eval_set.py
   ```

2. **Run initial evaluation**:
   ```bash
   python scripts/run_expanded_nli_eval.py
   ```

3. **Review generated cases**: Manually inspect LLM-generated cases for quality

4. **Iterate on low-performing categories**: Add training data or refine prompts for categories < 70%

5. **Integrate into CI/CD**: Add evaluation to model training pipeline

6. **Monitor performance over time**: Track ID/OOD gap as training data evolves

---

## File Locations

```
legal-kit/
├── data/
│   └── eval/
│       ├── legal_nli_edge_cases.json          # Original 33 hand-labeled cases
│       ├── legal_nli_eval_expanded.json       # Generated: 243 cases (NEW)
│       └── nli-expanded-eval-<timestamp>.json # Generated: eval results (NEW)
├── models/
│   └── legal-entailment/
│       └── model/                             # Fine-tuned DeBERTa checkpoint
├── scripts/
│   ├── expand_nli_eval_set.py                 # NEW: Generation script
│   ├── run_expanded_nli_eval.py               # NEW: Evaluation script
│   ├── README_NLI_EVAL_EXPANSION.md           # NEW: Documentation
│   └── NLI_EVAL_EXPANSION_SUMMARY.md          # NEW: This file
└── python/
    └── src/
        └── legalkit/
            └── eval/
                └── nli_eval.py                # Evaluation framework (existing)
```

---

## Validation Checklist

- [x] Scripts are executable (`chmod +x`)
- [x] Python syntax is valid (`python -m py_compile`)
- [x] Imports work (anthropic, legalkit.eval.nli_eval)
- [x] Original eval set exists and is properly formatted
- [x] Scripts use correct paths relative to project root
- [x] Documentation includes setup, usage, and troubleshooting
- [x] Error handling for missing files, API failures, invalid JSON
- [x] Progress tracking for user feedback
- [x] Detailed output summaries and statistics
- [ ] Test run of expand_nli_eval_set.py (requires ANTHROPIC_API_KEY)
- [ ] Test run of run_expanded_nli_eval.py (requires fine-tuned model)

---

## Cost & Performance Estimates

### Generation (expand_nli_eval_set.py)
- **Time**: ~5-10 minutes (14 categories × 15 cases, sequential API calls)
- **API Calls**: 14 (one per category)
- **Tokens**: ~115,500 input + ~42,000 output
- **Cost**: ~$0.85 (using Claude Sonnet 4)

### Evaluation (run_expanded_nli_eval.py)
- **Time (CPU)**: ~2-5 minutes (243 cases, depends on hardware)
- **Time (GPU)**: ~30-60 seconds (A10G or better)
- **Memory**: ~4GB for DeBERTa-large
- **Cost**: $0 (local inference)

### Total Workflow
- **One-time generation**: ~$0.85 + 10 minutes
- **Per evaluation run**: $0 + 1-5 minutes
- **Reusable**: Expanded eval set can be used indefinitely
