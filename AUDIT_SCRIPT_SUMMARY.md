# Label Audit Script - Implementation Summary

## Created Files

### 1. Main Script
**Path**: `scripts/audit_pipeline_labels.py` (522 lines)

Complete implementation with all requested features:
- ✅ Stratified sampling (25 per class by default)
- ✅ Async Claude API calls with batching
- ✅ Retry logic with exponential backoff
- ✅ Temperature=0 for deterministic re-labeling
- ✅ Blind labeling (Claude doesn't see original labels)
- ✅ Overall agreement rate computation
- ✅ Per-class agreement rates
- ✅ Confusion matrix (original → claude)
- ✅ Systematic bias detection (label transitions)
- ✅ Confidence distribution analysis
- ✅ Suspicious sample flagging
- ✅ Full results export to JSON
- ✅ Audit samples export for human review
- ✅ --dry-run flag
- ✅ --n flag for sample size
- ✅ --batch-size flag for concurrency control
- ✅ Comprehensive error handling

### 2. Documentation
**Path**: `docs/label_audit_guide.md` (221 lines)

Complete guide covering:
- Overview and purpose
- Usage examples
- Requirements
- Output formats
- Metrics explanation
- Interpretation guidelines
- Example output
- Next steps

## Quick Start

### 1. Set API key
```bash
export ANTHROPIC_API_KEY=your-key-here
```

### 2. Run audit
```bash
# Default: 100 samples (25 per class)
python scripts/audit_pipeline_labels.py

# Or dry-run first to see what will happen
python scripts/audit_pipeline_labels.py --dry-run

# Custom sample size
python scripts/audit_pipeline_labels.py --n 200

# Smaller batches for rate limiting
python scripts/audit_pipeline_labels.py --batch-size 5
```

### 3. Review results
- Console: Agreement rates, confusion matrix, suspicious samples
- `eval/label-audit-<timestamp>.json`: Full results with all metrics
- `data/eval/label_audit_samples.json`: 100 samples for human review

## Data Stats

Current training data (`data/training_v2/train.jsonl`):
- Total rows: 6,199
- Pipeline_log rows: 3,338 (53.8%)
- Sufficient samples in each class (min: 602)

Label distribution in pipeline_log:
- entailed: 652 (19.5%)
- partially_entailed: 602 (18.0%)
- contradicted: 854 (25.6%)
- neutral: 1,230 (36.8%)

## Technical Details

### API Configuration
- Model: `claude-sonnet-4-20250514`
- Temperature: 0 (deterministic)
- Max tokens: 500 (sufficient for label + confidence + explanation)
- Retries: 3 with delays

### Sampling Strategy
- **Stratified**: Equal samples from each class to avoid bias
- **Random seed**: Reproducible (default: 42)
- **Balanced**: Ensures fair representation across all labels

### Output Files

#### eval/label-audit-<timestamp>.json
```json
{
  "metadata": {
    "timestamp": "20260319_131500",
    "n_samples": 100,
    "model": "claude-sonnet-4-20250514",
    "temperature": 0
  },
  "metrics": {
    "overall_agreement": 0.847,
    "class_agreement": {...},
    "confusion_matrix": {...},
    "top_transitions": {...},
    "confidence_stats": {...},
    "most_suspicious": [...]
  },
  "samples": [...]
}
```

#### data/eval/label_audit_samples.json
```json
[
  {
    "premise": "...",
    "hypothesis": "...",
    "original_label": "entailed",
    "claude_label": "partially_entailed",
    "claude_confidence": 0.92,
    "claude_explanation": "...",
    "agreement": false,
    "doc_id": "...",
    "section_id": "..."
  }
]
```

## Example Workflow

```bash
# 1. Dry run to preview
python scripts/audit_pipeline_labels.py --n 100 --dry-run

# 2. Run actual audit
python scripts/audit_pipeline_labels.py --n 100

# 3. Review console output for high-level metrics

# 4. Inspect suspicious disagreements
cat data/eval/label_audit_samples.json | jq '.[] | select(.agreement == false and .claude_confidence > 0.9)'

# 5. Analyze full results
cat eval/label-audit-*.json | jq '.metrics.confusion_matrix'

# 6. If needed, run larger audit for more confidence
python scripts/audit_pipeline_labels.py --n 400
```

## Validation

Script has been validated for:
- ✅ Syntax correctness
- ✅ Required function definitions
- ✅ Async/batch processing
- ✅ Error handling
- ✅ Command-line argument parsing
- ✅ Training data compatibility
- ✅ Label mapping consistency
- ✅ Output directory creation

## Dependencies

- `anthropic>=0.85.0` (already installed in .venv)
- `asyncio` (stdlib)
- `json` (stdlib)
- `pathlib` (stdlib)

## Notes

- The script uses async API calls for speed (~10 samples/second with batch_size=10)
- Total cost for 100 samples: ~$0.05-0.10 (depending on input lengths)
- Run time: ~30-60 seconds for 100 samples
- No data is sent to Claude except premise/hypothesis pairs
- Results are deterministic (temperature=0, fixed seed)

## Next Steps

1. Run initial audit with default settings (100 samples)
2. Review overall agreement rate and confusion matrix
3. Investigate top label transitions
4. Examine suspicious high-confidence disagreements
5. Update pipeline prompts if systematic biases found
6. Re-run audit to validate improvements
7. Consider larger sample (200-400) if initial results unclear
