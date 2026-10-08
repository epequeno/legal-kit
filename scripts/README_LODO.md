# Leave-One-Dataset-Out (LODO) Cross-Validation

This directory contains scripts for evaluating embedding model generalization across different legal task types using leave-one-dataset-out (LODO) cross-validation.

## Overview

LODO cross-validation measures how well a fine-tuned embedding model generalizes to **unseen legal task types**. The LegalBench-RAG benchmark contains 4 datasets:

1. **cuad** (5,592 training pairs, 3,997 eval queries) - Contract Understanding Atticus Dataset
2. **maud** (2,206 pairs, 1,676 queries) - M&A Understanding Dataset  
3. **contractnli** (1,210 pairs, 977 queries) - Contract NLI
4. **privacy_qa** (401 pairs, 192 queries) - Privacy Policy QA

For each dataset, we:
- Train on the **other 3 datasets** (with 10% held out for validation)
- Evaluate on the **held-out dataset** (never seen during training)

This gives 4 models and 4 evaluation scores. The **aggregate** is the average across these 4 scores, providing a robust estimate of generalization.

## Workflow

### 1. Prepare LODO Splits

```bash
python scripts/embedding_cross_dataset_eval.py
```

This extracts training pairs from all 4 datasets and creates LODO splits in:
```
data/embedding_training_lodo/
├── cuad/
│   ├── train.jsonl  (3,839 pairs from maud+contractnli+privacy_qa)
│   └── val.jsonl    (426 pairs, 10% of train)
├── maud/
│   ├── train.jsonl  (7,199 pairs from cuad+contractnli+privacy_qa)
│   └── val.jsonl    (799 pairs)
├── contractnli/
│   ├── train.jsonl  (8,185 pairs from cuad+maud+privacy_qa)
│   └── val.jsonl    (909 pairs)
└── privacy_qa/
    ├── train.jsonl  (9,005 pairs from cuad+maud+contractnli)
    └── val.jsonl    (1,000 pairs)
```

Each JSONL file has the format:
```json
{
  "query": "Does the contract contain...",
  "positive": "The relevant passage...",
  "hard_negative": "An irrelevant passage from same doc...",
  "dataset": "cuad",
  "doc_id": "contract_123"
}
```

### 2. Train 4 Models

For each held-out dataset, copy its split to `data/embedding_training/` and train on Modal:

```bash
# Model 1: trained on maud+contractnli+privacy_qa, will eval on cuad
cp data/embedding_training_lodo/cuad/*.jsonl data/embedding_training/
modal run scripts/modal_embedding_train.py
modal volume get legalkit-data models/legal-embeddings/model models/legal-embeddings-lodo-cuad

# Model 2: trained on cuad+contractnli+privacy_qa, will eval on maud
cp data/embedding_training_lodo/maud/*.jsonl data/embedding_training/
modal run scripts/modal_embedding_train.py
modal volume get legalkit-data models/legal-embeddings/model models/legal-embeddings-lodo-maud

# Model 3: trained on cuad+maud+privacy_qa, will eval on contractnli
cp data/embedding_training_lodo/contractnli/*.jsonl data/embedding_training/
modal run scripts/modal_embedding_train.py
modal volume get legalkit-data models/legal-embeddings/model models/legal-embeddings-lodo-contractnli

# Model 4: trained on cuad+maud+contractnli, will eval on privacy_qa
cp data/embedding_training_lodo/privacy_qa/*.jsonl data/embedding_training/
modal run scripts/modal_embedding_train.py
modal volume get legalkit-data models/legal-embeddings/model models/legal-embeddings-lodo-privacy_qa
```

### 3. Evaluate Each Model

Run benchmark evaluation on the held-out dataset for each model:

```bash
legalkit bench --datasets cuad --embedding-model models/legal-embeddings-lodo-cuad \
  --output eval/lodo-cuad.json --mini

legalkit bench --datasets maud --embedding-model models/legal-embeddings-lodo-maud \
  --output eval/lodo-maud.json --mini

legalkit bench --datasets contractnli --embedding-model models/legal-embeddings-lodo-contractnli \
  --output eval/lodo-contractnli.json --mini

legalkit bench --datasets privacy_qa --embedding-model models/legal-embeddings-lodo-privacy_qa \
  --output eval/lodo-privacy_qa.json --mini
```

### 4. Generate Comparison Report

```bash
python scripts/embedding_lodo_report.py
```

This loads all 4 evaluation results and generates a comprehensive report showing:
- Per-dataset results (R@k and P@k for k=1,4,16,64)
- Aggregate metrics (average across 4 held-out datasets)
- Comparison to baseline (text-embedding-3-large) if available
- Improvement percentages (both absolute and relative)

Example output:
```
====================================================================================================
LODO (Leave-One-Dataset-Out) Cross-Validation Results
====================================================================================================

Per-Dataset Results (Fine-tuned on OTHER 3 datasets)
====================================================================================================

Dataset: cuad            (3997 queries)
----------------------------------------------------------------------------------------------------
Recall:    R@1  = 52.00%   R@4  = 73.00%   R@16 = 94.00%   R@64 = 99.50%   
Precision: P@1  =  4.50%   P@4  =  5.20%   P@16 =  4.70%   P@64 =  3.90%   

...

Aggregate Metrics (Average across 4 held-out datasets)
====================================================================================================

Fine-tuned (LODO average):
  Recall:    R@1  = 50.00%   R@4  = 71.00%   R@16 = 92.50%   R@64 = 99.25%   

Baseline (text-embedding-3-large average):
  Recall:    R@1  = 40.41%   R@4  = 64.81%   R@16 = 93.05%   R@64 = 99.85%   

Improvement (Fine-tuned vs Baseline):
  Metric      Fine-tuned   Baseline     Delta      Relative
  ------------------------------------------------------------
  R@1          50.00%      40.41%       +9.59%     +23.7%
  R@4          71.00%      64.81%       +6.19%      +9.5%
  R@16         92.50%      93.05%       -0.55%      -0.6%
  R@64         99.25%      99.85%       -0.60%      -0.6%
```

## Key Metrics

- **R@k (Recall at k)**: Fraction of ground-truth characters retrieved in top-k chunks
- **P@k (Precision at k)**: Fraction of retrieved characters that are ground-truth

The primary metric for comparison is **R@64**, which represents retrieval coverage when using the default 64 chunks per query.

## Expected Results

If fine-tuning improves generalization, you should see:
- ✅ Higher R@1 and R@4 (better focused retrieval)
- ✅ Similar or slightly lower R@64 (still high coverage)
- ⚠️ Lower precision (more chunks retrieved, more false positives)

A successful LODO evaluation demonstrates that the model learns **transferable** retrieval patterns rather than dataset-specific shortcuts.

## Files

- `embedding_cross_dataset_eval.py` - Prepares LODO data splits
- `embedding_lodo_report.py` - Generates comparison report from results
- `modal_embedding_train.py` - Trains embedding model on Modal GPU
- `README_LODO.md` - This file
