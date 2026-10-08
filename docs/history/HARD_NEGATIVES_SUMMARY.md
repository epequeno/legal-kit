# Hard-Negative Training Data Generation Pipeline - Implementation Summary

## Overview

Built a complete pipeline to replace low-quality cross-passage neutrals (68% of v0 training data) with topically-related hard negatives. This addresses the root cause of the v0 fine-tuning failure (26.4% accuracy).

## Components Implemented

### 1. `src/legalkit/eval/generate_hard_negatives.py`

Core generation module with 5 main functions:

- **`load_entailed_pairs()`**: Loads entailed + partially_entailed pairs from entailment log
  - Filters to high-quality sources (excludes cross_passage)
  - Supports source filtering and max_n limiting
  
- **`generate_perturbation_negatives()`**: Applies pattern-based perturbations
  - Detects patterns in passages (shall_vs_may, quantitative_precision, etc.)
  - Generates contradicted claims via strategic perturbations
  - Maps to training JSONL format (premise/hypothesis/label)
  
- **`load_catalog_as_training_pairs()`**: Converts 96-example curated catalog to training format
  - Maps PatternExample labels to integer labels
  - Sets source='curated:catalog'
  
- **`deduplicate()`**: Removes duplicate (premise, hypothesis) pairs
  - Keeps first occurrence
  
- **`generate_hard_negatives()`**: Main orchestration function
  - Loads entailed pairs → generates perturbations → loads catalog → deduplicates → saves
  - Returns comprehensive stats dict

### 2. Updated `src/legalkit/eval/prepare_training.py`

Enhanced `prepare_dataset()` with two new parameters:

- **`hard_negatives_path`**: Path to hard_negatives.jsonl for merging
- **`cross_passage_cap`**: Max fraction of neutral class from cross_passage (default 0.15 = 15%)

Features:
- Loads hard negatives and merges into training pool before balancing
- Caps cross_passage neutrals to prevent trivial negatives dominating
- Defaults to 1000 samples per class when hard negatives provided
- Adds source_distribution to stats output
- Fully backward compatible when hard_negatives_path=None

### 3. CLI Command: `legalkit generate-hard-negatives`

Added user-friendly CLI with:
- Progress reporting (load, perturb, catalog stages)
- Rich tables showing by_label and by_source distributions
- Detailed stats output
- Optional `--max-from-log` for testing

### 4. Comprehensive Test Suite: `tests/test_generate_hard_negatives.py`

22 tests covering:
- `load_entailed_pairs()`: label filtering, source filtering, max_n
- `generate_perturbation_negatives()`: schema, labels, source format, progress callback
- `load_catalog_as_training_pairs()`: label mapping, source setting
- `deduplicate()`: exact duplicate removal, first occurrence preservation
- `generate_hard_negatives()`: output format, deduplication, stats, JSONL validity
- `prepare_dataset()` integration: hard negative inclusion, cross_passage cap, backward compatibility

All 22 tests pass.

## Real-World Generation Results

Running against `data/entailment_logs/entailment_2026-03-17.jsonl`:

```
Input:
- 1,526 entailed pairs loaded (filtered to high-quality sources)
- 96 curated catalog examples

Output:
- 281 hard-negative training pairs
  - 225 contradicted (80%)
  - 40 entailed (14%)
  - 16 neutral (6%)

By Source:
- curated:catalog: 96 (34%)
- perturbation:shall_vs_may: 147 (52%)
- perturbation:quantitative_precision: 26 (9%)
- perturbation:temporal_qualifier: 9 (3%)
- perturbation:knowledge_qualifier: 3 (1%)

Deduplication:
- 4 duplicates removed
```

## Key Implementation Details

### Source Filtering
The entailment log uses "synthetic:" prefix for generated data. The pipeline filters to:
- `synthetic:llm_generated`
- `pipeline`
- `synthetic:perturbation:*` (negated, period_changed, amount_changed, parties_swapped)

Explicitly **excludes** `synthetic:cross_passage` (the root cause of v0 failure).

### Data Format
Training pairs use premise/hypothesis keys (not passage/claim):
```json
{
  "premise": "...",
  "hypothesis": "...",
  "label": 2,
  "label_name": "contradicted",
  "section_id": "§2.3",
  "doc_id": "sample_contract",
  "source": "perturbation:quantitative_precision"
}
```

### Pattern Detection & Perturbation
The pipeline leverages existing `entailment_patterns.py` infrastructure:
1. `detect_patterns(passage)` identifies which patterns are present
2. `generate_negatives_from_perturbation(passage, claim, category)` applies category-specific transformations
3. Perturbations are deterministic (use hash for reproducibility)

Current perturbation strategies:
- **shall_vs_may**: Swap shall ↔ may
- **quantitative_precision**: Change amounts by ±50%, invert thresholds (at least ↔ up to)
- **temporal_qualifier**: Change time periods (double/halve), swap prior to ↔ following
- **knowledge_qualifier**: Drop "to X's Knowledge" or "in all material respects"

## Usage

```bash
# Generate hard negatives
legalkit generate-hard-negatives

# Prepare training data with hard negatives
legalkit prepare-training --hard-negatives data/training/hard_negatives.jsonl

# Test with limited input
legalkit generate-hard-negatives --max-from-log 100
```

## Next Steps

To replace v0 training data:
1. Run `legalkit generate-hard-negatives` to create hard_negatives.jsonl
2. Run `legalkit prepare-training` with `--hard-negatives` flag to merge into balanced dataset
3. The cross_passage_cap will limit trivial negatives to 15% of neutral class
4. Fine-tune with the improved dataset (expect significant accuracy improvement over 26.4%)

## Files Modified

- `src/legalkit/eval/generate_hard_negatives.py` (new, 307 lines)
- `src/legalkit/eval/prepare_training.py` (updated, added hard_negatives_path + cross_passage_cap)
- `src/legalkit/cli.py` (added generate-hard-negatives command)
- `tests/test_generate_hard_negatives.py` (new, 652 lines, 22 tests)

## Impact

This pipeline addresses the v0 failure by:
1. **Reducing trivial negatives**: Cross-passage neutrals capped at 15% (down from 68%)
2. **Adding topically-related hard negatives**: 281 pairs that require careful legal reading
3. **Pattern-based perturbations**: Systematically target the 8 failure patterns from DeBERTa-MNLI eval
4. **Source transparency**: Every sample tracks its origin (perturbation:pattern_name or curated:catalog)

Expected outcome: Model learns to distinguish hard cases (shall vs may, quantitative precision, etc.) rather than relying on topical overlap heuristics.
