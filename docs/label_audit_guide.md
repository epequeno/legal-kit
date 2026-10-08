# Label Audit Guide

## Overview

The `audit_pipeline_labels.py` script audits LLM-generated labels in the entailment training data by having Claude Sonnet independently re-label a stratified sample and comparing the results.

## Purpose

53.8% of the training data (3,338 rows) comes from `pipeline_log` — these were labeled by Claude during pipeline runs. This script validates the quality of those labels by:

1. Sampling examples across all label classes
2. Re-labeling them with Claude Sonnet (blind to original labels)
3. Measuring agreement rates and identifying systematic biases
4. Flagging suspicious disagreements for human review

## Usage

### Basic usage (100 samples, 25 per class)
```bash
python scripts/audit_pipeline_labels.py
```

### Custom sample size
```bash
python scripts/audit_pipeline_labels.py --n 200
```

### Dry run (no API calls)
```bash
python scripts/audit_pipeline_labels.py --n 100 --dry-run
```

### Control batch size for API calls
```bash
python scripts/audit_pipeline_labels.py --batch-size 5
```

## Requirements

- `ANTHROPIC_API_KEY` environment variable must be set
- `anthropic` Python package (install: `pip install anthropic`)
- Training data at `data/training_v2/train.jsonl`

## Output

### 1. Full Results
**Location**: `eval/label-audit-<timestamp>.json`

Contains:
- Metadata (timestamp, model, parameters)
- Complete metrics (agreement rates, confusion matrix, transitions)
- All audited samples with both original and Claude labels

### 2. Review Samples
**Location**: `data/eval/label_audit_samples.json`

100 samples with:
- Original and Claude labels
- Confidence scores
- Explanations
- Agreement flags

Useful for manual inspection and understanding disagreements.

### 3. Console Summary

Printed to stdout:
- Overall agreement rate
- Per-class agreement rates
- Confusion matrix (original → Claude)
- Top label transitions (most common disagreements)
- Confidence distribution for agreements vs disagreements
- Most suspicious samples (high-confidence disagreements)

## Metrics Explained

### Overall Agreement
Percentage of samples where Claude's label exactly matches the original pipeline_log label.

### Per-Class Agreement
Agreement rate broken down by original label class. Shows if certain classes are more reliable than others.

### Confusion Matrix
Shows how original labels map to Claude's re-labels. Diagonal = agreements, off-diagonal = disagreements.

Example:
```
Original             | entailed | partial  | contra   | neutral
-------------------- | -------- | -------- | -------- | --------
entailed             |    23    |     2    |     0    |     0
partially_entailed   |     1    |    20    |     0    |     4
contradicted         |     0    |     1    |    22    |     2
neutral              |     0    |     3    |     1    |    21
```

### Label Transitions
Most common original→Claude transitions among disagreements. Reveals systematic biases.

Example:
- `neutral→contradicted: 15` (15% of disagreements)
- `partially_entailed→neutral: 12` (12% of disagreements)

### Confidence Distribution
- **Agreements**: High confidence = both models agree strongly
- **Disagreements**: High confidence = Claude strongly disagrees with original

### Most Suspicious Samples
Disagreements where Claude is highly confident (≥0.9). These are prime candidates for:
- Pipeline prompt tuning
- Label correction
- Human review

## Implementation Details

### Sampling Strategy
**Stratified sampling**: Equal samples from each label class (entailed, partially_entailed, contradicted, neutral) to avoid class imbalance bias.

### Re-labeling Process
- Model: `claude-sonnet-4-20250514`
- Temperature: 0 (deterministic)
- Claude sees ONLY premise and hypothesis (not original label)
- Prompt asks for: label, confidence (0-1), explanation
- Async batch processing with configurable concurrency

### Error Handling
- Up to 3 retries per API call
- Graceful handling of JSON parse errors
- Invalid responses marked as "error" and excluded from metrics

### Rate Limiting
- Configurable batch size (default: 10 concurrent calls)
- 0.5s delay between batches

## Interpreting Results

### High Agreement (>90%)
✓ Original labels are reliable
✓ Pipeline labeling is consistent with Claude's current understanding

### Moderate Agreement (70-90%)
⚠️ Some disagreements, investigate:
- Are disagreements concentrated in specific classes?
- Are transition patterns systematic?
- Are suspicious samples revealing edge cases?

### Low Agreement (<70%)
🚨 Significant labeling issues:
- Review pipeline prompts
- Check for prompt drift between pipeline and audit
- Consider re-labeling with current model
- Analyze transition patterns for systematic biases

### Common Patterns

**neutral→contradicted**: Pipeline may be too permissive with "neutral"
**partially_entailed→entailed**: Pipeline may be too strict about partial support
**entailed→partially_entailed**: Claude may require more explicit evidence

## Example Output

```
================================================================================
AUDIT SUMMARY
================================================================================

Total samples: 100
Valid re-labels: 98
Errors: 2

Overall Agreement: 84.7% (83/98)

Per-Class Agreement:
  entailed            :  92.0%  (n=25)
  partially_entailed  :  80.0%  (n=25)
  contradicted        :  88.0%  (n=25)
  neutral             :  79.2%  (n=24)

Confusion Matrix (original → claude):
Original             | entailed | partially_entailed | contradicted | neutral
entailed             |    23    |        2           |      0       |    0
partially_entailed   |     1    |       20           |      0       |    4
contradicted         |     0    |        1           |     22       |    2
neutral              |     0    |        3           |      1       |   20

Top Label Transitions (disagreements only):
  partially_entailed→neutral    :   4 (26.7% of disagreements)
  neutral→contradicted          :   2 (13.3% of disagreements)
  entailed→partially_entailed   :   2 (13.3% of disagreements)

Confidence Distribution:
Agreements:
  Mean: 0.912
  Range: [0.700, 1.000]
  Count: 83

Disagreements:
  Mean: 0.847
  Range: [0.600, 0.950]
  Count: 15

Most Suspicious (high-confidence disagreements):
1. partially_entailed → neutral (confidence: 0.950)
   Hypothesis: The agreement requires the recipient to notify the discloser...
   Explanation: The premise does not explicitly state notification requirements...
   Doc: NDA-Template-2023-v2
```

## Next Steps After Audit

1. **Review suspicious samples** in `data/eval/label_audit_samples.json`
2. **Identify systematic biases** from transition patterns
3. **Update pipeline prompts** if needed
4. **Re-label problematic samples** if necessary
5. **Re-train model** with corrected labels
6. **Re-run audit** to verify improvements

## Notes

- The audit uses the latest Claude Sonnet model, which may differ from the model used during pipeline runs
- Disagreements don't always mean the original label is wrong — sometimes both interpretations are defensible
- Use audit results to inform labeling guidelines and prompt engineering, not for automatic correction
