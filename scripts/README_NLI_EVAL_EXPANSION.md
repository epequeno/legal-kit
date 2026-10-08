# NLI Evaluation Set Expansion

This directory contains scripts to expand the legal NLI evaluation set from 33 hand-labeled cases to 200+ cases, including both in-distribution and out-of-distribution categories.

## Overview

The expansion process:

1. **Keeps** the original 33 hand-labeled cases (marked as `reviewed: true`)
2. **Generates** 15 new cases per existing category (8 categories × 15 = 120 cases)
3. **Generates** 15 cases per NEW category (6 OOD categories × 15 = 90 cases)
4. **Total**: 33 + 120 + 90 = **243 cases**

All generated cases are marked with:
- `source: "llm_generated"`
- `reviewed: false` (to flag for human review)
- `distribution: "in_distribution"` or `"out_of_distribution"`

## Files

### `expand_nli_eval_set.py`

Generates the expanded evaluation set using Claude API.

**Prerequisites:**
```bash
export ANTHROPIC_API_KEY="your-api-key"
```

**Usage:**
```bash
# Activate venv first
cd python && source .venv/bin/activate && cd ..

# Run the script
python scripts/expand_nli_eval_set.py
```

**Output:**
- `data/eval/legal_nli_eval_expanded.json` — Expanded evaluation set

**Categories:**

**In-Distribution (trained on):**
1. `notwithstanding` — Override clauses and exceptions
2. `shall_vs_may` — Mandatory vs discretionary obligations
3. `knowledge_qualifier` — Knowledge-qualified representations
4. `mae_carveout` — Material Adverse Effect carve-outs
5. `quantitative_precision` — Precise quantitative interpretation
6. `temporal_qualifier` — Time periods and deadlines
7. `condition_precedent` — Conditional obligations
8. `defined_term` — Defined term substitution

**Out-of-Distribution (novel):**
1. `representations_warranties` — Reps & warranties, survival, qualifiers
2. `limitation_of_liability` — Liability caps, exclusions, carve-outs
3. `force_majeure` — FM events, notice, mitigation
4. `assignment_change_of_control` — Assignment restrictions, consent requirements
5. `governing_law_jurisdiction` — Choice of law, forum selection, arbitration
6. `non_compete_non_solicit` — Non-compete scope, duration, geographic limits

### `run_expanded_nli_eval.py`

Runs the fine-tuned DeBERTa model against the expanded evaluation set and produces detailed performance breakdown.

**Prerequisites:**
- Expanded eval set must exist: `data/eval/legal_nli_eval_expanded.json`
- Fine-tuned model must exist: `models/legal-entailment/model/`

**Usage:**
```bash
# Activate venv first
cd python && source .venv/bin/activate && cd ..

# Run on CPU (default)
python scripts/run_expanded_nli_eval.py

# Run on GPU
python scripts/run_expanded_nli_eval.py --device cuda

# Custom model path
python scripts/run_expanded_nli_eval.py --model-path /path/to/model

# Don't save results
python scripts/run_expanded_nli_eval.py --no-save

# Custom threshold for highlighting
python scripts/run_expanded_nli_eval.py --threshold 0.80
```

**Output:**
- Console: Detailed performance report
- File: `data/eval/nli-expanded-eval-<timestamp>.json`

**Report includes:**
- Overall accuracy (4-class and binary)
- In-distribution vs out-of-distribution breakdown
- Per-category performance with ID/OOD labels
- Per-label accuracy
- Categories below threshold (default 70%)
- Most common confusion patterns

## Setup

### Prerequisites

The scripts require the legalkit Python environment:

```bash
# From project root
cd python

# Ensure dependencies are installed (using uv)
uv sync

# Or activate the existing virtual environment
source .venv/bin/activate
```

The `anthropic` package is already included in the project dependencies.

## Workflow

### Step 1: Generate Expanded Set

```bash
# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Activate Python environment (from python/ directory)
cd python && source .venv/bin/activate

# Generate expanded eval set (from project root)
cd ..
python scripts/expand_nli_eval_set.py
```

Expected output:
```
Loading original evaluation set...
  Loaded 33 original cases across 8 categories

Adding original cases (marking as reviewed)...

================================================================================
Generating cases for ORIGINAL categories (in-distribution)...
================================================================================
Generating 15 cases for category: notwithstanding
  ✓ Generated 15 valid cases
...

================================================================================
Generating cases for NEW categories (out-of-distribution)...
================================================================================
Generating 15 cases for category: representations_warranties
  ✓ Generated 15 valid cases
...

================================================================================
GENERATION COMPLETE
================================================================================

Output: /Users/steven/code/legal-kit/data/eval/legal_nli_eval_expanded.json

Total cases: 243

By source:
  hand_labeled: 33
  llm_generated: 210

By distribution:
  in_distribution: 153
  out_of_distribution: 90

By category:
  notwithstanding                    (ID ):  48
  shall_vs_may                       (ID ):  20
  ...
  representations_warranties         (OOD):  15
  limitation_of_liability            (OOD):  15
  ...

Label distribution:
  entailed            :  61 ( 25.1%)
  partially_entailed  :  59 ( 24.3%)
  contradicted        :  62 ( 25.5%)
  neutral             :  61 ( 25.1%)
```

### Step 2: Run Evaluation

```bash
# Ensure model exists
ls models/legal-entailment/model/

# Activate venv (if not already active)
cd python && source .venv/bin/activate && cd ..

# Run evaluation
python scripts/run_expanded_nli_eval.py
```

Expected output:
```
Loading expanded evaluation set from: data/eval/legal_nli_eval_expanded.json
  Version: 2.0
  Total cases: 243
  Categories: 14

Running evaluation with model: models/legal-entailment/model
Device: cpu

Progress:
  243/243 (100%)

================================================================================
EXPANDED NLI EVALUATION REPORT
================================================================================

### OVERALL PERFORMANCE
  Total cases: 243
  4-class accuracy: 78.2%
  Binary accuracy:  89.3%

### IN-DISTRIBUTION vs OUT-OF-DISTRIBUTION

  In-Distribution (trained categories):
    Cases: 153
    4-class accuracy: 84.3%
    Binary accuracy:  92.8%

  Out-of-Distribution (novel categories):
    Cases: 90
    4-class accuracy: 68.9%
    Binary accuracy:  83.3%

  Performance gap (ID - OOD): +15.4%

### PER-CATEGORY PERFORMANCE
Category                         Type  Cases  4-class   Binary
--------------------------------------------------------------------------------
notwithstanding                   ID      48    91.7%    95.8%
shall_vs_may                      ID      20    85.0%    95.0%
...
representations_warranties        OOD     15    72.0%    86.7%
limitation_of_liability           OOD     15    68.0%    80.0%
force_majeure                     OOD     15    62.0%    78.7%
...

  ⚠️  Categories below 70% threshold:
     - force_majeure: 62.0%
     - assignment_change_of_control: 66.7%

### PER-LABEL PERFORMANCE
Label                Correct    Total  Accuracy
--------------------------------------------------------------------------------
entailed                  53       61      86.9%
partially_entailed        40       59      67.8%
contradicted              51       62      82.3%
neutral                   46       61      75.4%

### MOST COMMON CONFUSIONS
  partially_entailed → entailed: 12
  neutral → contradicted: 8
  ...

================================================================================

✓ Results saved to: data/eval/nli-expanded-eval-20260319-143022.json
```

## Interpreting Results

### Performance Metrics

- **4-class accuracy**: Exact match on all 4 labels (entailed, partially_entailed, contradicted, neutral)
- **Binary accuracy**: Positive (entailed/partially) vs negative (contradicted/neutral)

### Expected Patterns

1. **ID > OOD**: In-distribution accuracy should be higher (model trained on these patterns)
2. **Partially_entailed errors**: Most challenging class (subtle qualifier differences)
3. **Category-specific failures**: Some OOD categories may be harder (e.g., force majeure event interpretation)

### Red Flags

- **ID accuracy < 80%**: Model may need retraining or more training data
- **OOD accuracy < 60%**: Poor generalization; may need broader training data
- **Large ID/OOD gap (>20%)**: Overfitting to training categories

## Next Steps

### Review Generated Cases

LLM-generated cases should be human-reviewed before using in production evals:

1. Filter unreviewed cases:
```python
import json
with open('data/eval/legal_nli_eval_expanded.json') as f:
    data = json.load(f)

unreviewed = [c for c in data['cases'] if not c.get('reviewed', False)]
print(f"{len(unreviewed)} cases need review")
```

2. Review cases manually (check for):
   - Realistic legal language
   - Correct label assignment
   - Clear explanations
   - No hallucinated legal terms

3. Mark as reviewed:
```python
# After manual review, set reviewed: true
for case in data['cases']:
    if case['id'] == 'gen-now-042':
        case['reviewed'] = True

with open('data/eval/legal_nli_eval_expanded.json', 'w') as f:
    json.dump(data, f, indent=2)
```

### Improve Low-Performing Categories

If a category scores < 70%:

1. **Add training data**: Generate more examples for that pattern
2. **Review errors**: Examine confusion matrix for that category
3. **Adjust model**: Consider category-specific fine-tuning

### Continuous Evaluation

Integrate into CI/CD:

```bash
# Run after each model training
cd python && source .venv/bin/activate && cd ..
python scripts/run_expanded_nli_eval.py --device cuda --threshold 0.75
```

## Cost Estimation

Using `claude-sonnet-4-20250514`:

- **Input**: ~350 tokens per case × 210 cases = 73,500 tokens
- **Output**: ~200 tokens per case × 210 cases = 42,000 tokens
- **Cost**: ~$0.22 + $0.63 = **~$0.85** per full generation

(14 categories × 15 cases × ~800 tokens avg per generation)

## Troubleshooting

### Error: ANTHROPIC_API_KEY not set
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Error: Model not found
```bash
# Check model path
ls -la models/legal-entailment/model/

# If missing, run training first
modal run scripts/modal_finetune.py
```

### Error: Eval set not found
```bash
# Generate it first
cd python && source .venv/bin/activate && cd ..
python scripts/expand_nli_eval_set.py
```

### Quick validation test
```bash
# Test that scripts can import properly
cd python && source .venv/bin/activate
python -c "import anthropic; print('✓ anthropic available')"
python -c "from legalkit.eval.nli_eval import run_finetuned_nli; print('✓ legalkit imports work')"
```

### Low-quality generated cases

- Increase few-shot examples (edit `n=3` in code)
- Use more specific category descriptions
- Add negative examples to prompts
- Review and filter manually

### Rate limiting

- Add retry logic with exponential backoff
- Reduce batch size (generate fewer cases per category)
- Use `claude-3-haiku` for faster/cheaper generation

## References

- Original eval set: `data/eval/legal_nli_edge_cases.json`
- Training script: `scripts/modal_finetune.py`
- Evaluation code: `python/src/legalkit/eval/nli_eval.py`
- Test suite: `python/tests/test_nli_eval.py`
