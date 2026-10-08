# Embedding Model Fine-Tuning Pipeline

## Summary

Built a complete end-to-end pipeline for fine-tuning legal domain embedding models and evaluating them against API baselines.

## Components Created

### 1. Modal Training Script (`scripts/modal_embedding_train.py`)
- **Purpose**: Fine-tune `BAAI/bge-base-en-v1.5` on legal query→passage pairs
- **GPU**: A10G with 75-minute timeout
- **Loss**: MultipleNegativesRankingLoss (in-batch negatives)
- **Evaluation**: InformationRetrievalEvaluator on validation set
- **Features**:
  - Auto batch size selection (64 for base, 32 for large models)
  - Training data upload from local `data/embedding_training/`
  - Model saved to Modal volume: `/vol/models/legal-embeddings/model`
  - Results saved locally: `eval/embedding-train-{timestamp}.json`
  - Graceful handling of missing training data

**Usage**:
```bash
modal run scripts/modal_embedding_train.py
modal run scripts/modal_embedding_train.py --model BAAI/bge-large-en-v1.5 --epochs 5
```

### 2. Embedding Evaluation Module (`src/legalkit/eval/embedding_eval.py`)
- **Purpose**: Compare local vs API embeddings on LegalBench-RAG
- **Function**: `run_embedding_comparison()`
- **Returns**: Comprehensive comparison dict with:
  - Local model metrics (recall@k, precision@k, per-dataset breakdown)
  - API model metrics
  - Improvement deltas
  
**Features**:
- Runs benchmark twice (local + API) with same queries
- Computes marginal improvement at multiple k values
- Per-dataset and overall metrics

### 3. Local Model Support (`src/legalkit/pipeline/embed.py`)
- **New Functions**:
  - `load_local_embedding_model()` - Load and cache sentence-transformers model
  - `embed_texts_local()` - Embed with cached local model, fallback to API
  - `embed_query_local()` - Single query embedding
  
**Features**:
- Module-level caching (no reload on subsequent calls)
- Automatic fallback to API if no model specified
  - Text truncation (10K chars max)
  - Empty string handling
  - Device selection (cpu/cuda/mps)

### 4. Benchmark Integration (`src/legalkit/eval/benchmark.py`)
- **Updated**: `run_benchmark()` now accepts `embedding_fn` parameter
- **Behavior**: Uses custom embedding function when provided, falls back to API
- **Impact**: Enables comparing any embedding model on LegalBench-RAG

### 5. CLI Support (`src/legalkit/cli.py`)
- **New Option**: `--embedding-model` on `legalkit bench` command
- **Usage**:
```bash
# API baseline
legalkit bench --mini

# Local model
legalkit bench --mini --embedding-model models/legal-embeddings/model

# Full dataset with local model
legalkit bench --full --embedding-model models/legal-embeddings/model
```

### 6. Comprehensive Tests (`tests/test_embedding_eval.py`)
- **10 tests covering**:
  - Model loading and caching
  - Fallback behavior
  - Empty string handling
  - Text truncation
  - Model reloading on path change
  - Integration with benchmark runner
  - Comparison dict structure
  - Query embedding
  - Error handling

**All 788 tests pass**, including new tests.

## Workflow

### Training Workflow
```bash
# 1. Generate training pairs (parallel thread builds data/embedding_training/)
legalkit build-embedding-pairs

# 2. Train on Modal GPU
modal run scripts/modal_embedding_train.py

# 3. Download trained model
modal volume get legalkit-data models/legal-embeddings/model

# 4. Evaluate on benchmark
legalkit bench --mini --embedding-model models/legal-embeddings/model
```

### Evaluation Workflow
```python
from legalkit.eval.embedding_eval import run_embedding_comparison

results = run_embedding_comparison(
    datasets=["privacy_qa", "contractnli"],
    max_per_dataset=50,
    local_model_path="models/legal-embeddings/model",
    top_k=64,
)

print(f"Recall@64 improvement: {results['improvement']['overall_recall@64']:.4f}")
```

## Architecture Decisions

### 1. **Sentence-Transformers Library**
- Industry standard for dense retrieval
- Built-in MultipleNegativesRankingLoss
- Efficient batch encoding
- Easy integration with HuggingFace models

### 2. **In-Batch Negatives**
- Each positive in the batch is a negative for other queries
- Effective batch size = `batch_size * (batch_size - 1)` negatives
- No need to mine hard negatives during training
- Symmetric loss (query→doc and doc→query)

### 3. **Module-Level Caching**
- Avoid reloading 400MB+ models on every call
- Thread-safe for single-process applications
- Clear cache by setting `_local_model = None`

### 4. **Graceful Degradation**
- Missing model? Fall back to API
- Missing training data? Clear error message
- Import errors? Tests mock dependencies

### 5. **Modal Volume Pattern**
- Training data: `/vol/embedding_training/`
- Trained models: `/vol/models/legal-embeddings/model`
- Results: `/vol/models/legal-embeddings/results.json`
- Same volume as structure training (`legalkit-data`)

## Training Data Schema

Expected format at `data/embedding_training/train.jsonl` and `val.jsonl`:

```json
{"query": "What is the refund policy?", "positive": "Section 5.2 states...", "hard_negative": "Section 3.1 discusses...", "dataset": "privacy_qa", "doc_id": "doc_123"}
```

**Fields**:
- `query`: User query string
- `positive`: Ground-truth relevant passage
- `hard_negative`: Optional hard negative (BM25/semantic near-miss)
- `dataset`: Source dataset name
- `doc_id`: Document identifier

## Next Steps

1. **Generate training data** (parallel thread): `legalkit build-embedding-pairs`
2. **Run training**: `modal run scripts/modal_embedding_train.py`
3. **Download model**: `modal volume get legalkit-data models/legal-embeddings/model`
4. **Evaluate**: `legalkit bench --embedding-model models/legal-embeddings/model`
5. **Compare results**: Check `eval/embedding-train-latest.json`

## Success Metrics

- **Recall@64** on LegalBench-RAG (target: >80%)
- **Training time**: <30 min for base model, <60 min for large
- **Model size**: ~400MB (base), ~1.3GB (large)
- **Inference speed**: 100+ queries/sec on CPU

## Dependencies

- `sentence-transformers>=3.0` (Modal image only)
- `torch>=2.2` (Modal image only)
- No new local dependencies required
