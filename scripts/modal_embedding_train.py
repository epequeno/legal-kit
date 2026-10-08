"""Fine-tune a legal retrieval embedding model on Modal A10G GPU.

Trains BAAI/bge-base-en-v1.5 on legal query→passage pairs using
MultipleNegativesRankingLoss from the sentence-transformers library.

Usage:
    modal run scripts/modal_embedding_train.py
    modal run scripts/modal_embedding_train.py --model BAAI/bge-large-en-v1.5
    modal run --detach scripts/modal_embedding_train.py
"""

import modal
from datetime import datetime
from pathlib import Path

app = modal.App("legalkit-embedding-train")

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTORCH_ALLOC_CONF": "expandable_segments:True"})
    .pip_install(
        "sentence-transformers>=3.0",
        "torch>=2.2",
        "transformers>=4.45",
        "numpy>=1.26",
        "accelerate>=1.1",
        "datasets>=2.0",
    )
)

vol = modal.Volume.from_name("legalkit-data", create_if_missing=True)


def upload_training_data_local() -> dict:
    """Upload embedding training data from local filesystem to Modal volume.
    
    Runs locally (not on Modal) so it can access local files.
    Returns stats dict.
    """
    local_train = Path("data/embedding_training/train.jsonl")
    local_val = Path("data/embedding_training/val.jsonl")

    if not local_train.exists():
        return {
            "status": "missing",
            "message": "Training data not found. Run first:\n  legalkit build-embedding-pairs",
        }

    stats: dict = {"uploaded": [], "status": "success"}

    with vol.batch_upload(force=True) as batch:
        batch.put_file(str(local_train), "/embedding_training/train.jsonl")
        stats["train_samples"] = sum(1 for _ in open(local_train))
        stats["uploaded"].append("train.jsonl")

        if local_val.exists():
            batch.put_file(str(local_val), "/embedding_training/val.jsonl")
            stats["val_samples"] = sum(1 for _ in open(local_val))
            stats["uploaded"].append("val.jsonl")

    return stats


@app.function(image=image, gpu="A10G", timeout=4500, volumes={"/vol": vol})
def train(
    base_model: str = DEFAULT_MODEL,
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 2e-5,
    max_seq_length: int = 256,
    warmup_ratio: float = 0.1,
):
    """Fine-tune embedding model with MultipleNegativesRankingLoss.
    
    Training approach:
    - Load train.jsonl from volume
    - Build InputExample list: texts=[query, positive] (hard_negative optional)
    - DataLoader with in-batch negatives
    - MultipleNegativesRankingLoss (symmetric: both query→positive and positive→query)
    - Evaluate on val set using InformationRetrievalEvaluator
    - Save best model to /vol/models/legal-embeddings/model
    
    Returns results dict with training metrics and evaluation scores.
    """
    import json
    import time
    from pathlib import Path
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from torch.utils.data import DataLoader
    
    print(f"Training {base_model} for {epochs} epochs")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max sequence length: {max_seq_length}")
    
    # Check for training data
    train_path = Path("/vol/embedding_training/train.jsonl")
    val_path = Path("/vol/embedding_training/val.jsonl")
    
    if not train_path.exists():
        return {
            "status": "error",
            "message": "Training data not found on volume. Run upload_training_data() first."
        }
    
    # Load training data
    print(f"\nLoading training data from {train_path}...")
    train_examples = []
    train_data = []
    
    with open(train_path) as f:
        for line in f:
            row = json.loads(line)
            train_data.append(row)
            # Build InputExample: texts=[query, positive, hard_negative?]
            texts = [row["query"], row["positive"]]
            if row.get("hard_negative"):
                texts.append(row["hard_negative"])
            train_examples.append(InputExample(texts=texts))
    
    print(f"  Loaded {len(train_examples)} training pairs")
    
    # Load validation data
    val_examples = []
    if val_path.exists():
        print(f"Loading validation data from {val_path}...")
        with open(val_path) as f:
            for line in f:
                row = json.loads(line)
                val_examples.append(row)
        print(f"  Loaded {len(val_examples)} validation pairs")
    
    # Load model
    print(f"\nLoading base model: {base_model}")
    model = SentenceTransformer(base_model)
    model.max_seq_length = max_seq_length
    
    # Build DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Loss: MultipleNegativesRankingLoss
    # Uses in-batch negatives - all other positives in the batch are treated as negatives
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Build evaluator — cap at 300 pairs to avoid OOM on large models
    import random as _random
    _random.seed(42)
    if len(val_examples) > 300:
        val_examples = _random.sample(val_examples, 300)

    evaluator = None
    if val_examples:
        print("\nBuilding validation evaluator...")
        # Build {qid: query}, {qid: {pid}}, {pid: passage} dicts for InformationRetrievalEvaluator
        queries = {}
        corpus = {}
        relevant_docs = {}
        
        for i, row in enumerate(val_examples):
            qid = f"q{i}"
            pid = f"p{i}"
            
            queries[qid] = row["query"]
            corpus[pid] = row["positive"]
            relevant_docs[qid] = {pid}
        
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name="val",
            show_progress_bar=True,
            batch_size=32,
        )
    
    # Calculate warmup steps
    warmup_steps = int(len(train_dataloader) * epochs * warmup_ratio)
    
    # Output path
    output_path = "/vol/models/legal-embeddings/model"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training...")
    print(f"  Total training steps: {len(train_dataloader) * epochs}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Output: {output_path}")
    
    t0 = time.time()
    
    # Train — use_amp=True for fp16 to cut memory ~50% on large models
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        save_best_model=True,
        evaluation_steps=len(train_dataloader),  # Once per epoch only
        use_amp=True,
    )
    
    elapsed = time.time() - t0
    
    print(f"\n✓ Training complete in {elapsed:.0f}s")
    
    # Save training results
    results = {
        "status": "success",
        "base_model": base_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "warmup_ratio": warmup_ratio,
        "train_samples": len(train_examples),
        "val_samples": len(val_examples),
        "elapsed_seconds": round(elapsed, 2),
        "output_path": output_path,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save results JSON
    results_path = Path("/vol/models/legal-embeddings/results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    vol.commit()
    
    return results


@app.local_entrypoint()
def main(
    epochs: int = 3,
    batch_size: int = 0,  # 0 = auto based on model size
    lr: float = 2e-5,
    model: str = DEFAULT_MODEL,
):
    """Upload training data, fine-tune, download results."""
    
    print("=" * 60)
    print("LegalKit Embedding Model Fine-Tuning")
    print("=" * 60)
    
    # Auto-select batch size if not specified
    if batch_size == 0:
        if "large" in model.lower():
            batch_size = 16  # Large models: fp16 + batch 16 fits A10G comfortably
        else:
            batch_size = 32
        print(f"\nAuto-selected batch size: {batch_size}")
    
    # Step 1: Upload training data (runs locally to access local files)
    print("\n[1/3] Uploading training data...")
    upload_stats = upload_training_data_local()

    if upload_stats.get("status") == "missing":
        print(f"\n❌ {upload_stats['message']}")
        return

    print(f"  ✓ Uploaded: {', '.join(upload_stats['uploaded'])}")
    print(f"  Train samples: {upload_stats.get('train_samples', 0)}")
    print(f"  Val samples: {upload_stats.get('val_samples', 0)}")
    
    # Step 2: Train on GPU
    print(f"\n[2/3] Training on A10G GPU...")
    print(f"  Model: {model}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    
    train_results = train.remote(
        base_model=model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
    )
    
    if train_results.get("status") == "error":
        print(f"\n❌ {train_results['message']}")
        return
    
    print(f"\n  ✓ Training complete!")
    print(f"  Elapsed: {train_results['elapsed_seconds']}s")
    print(f"  Model saved to: {train_results['output_path']}")
    
    # Step 3: Download results
    print("\n[3/3] Downloading results...")
    
    from pathlib import Path
    import json
    
    # Save results locally
    local_dir = Path("eval")
    local_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = local_dir / f"embedding-train-{timestamp}.json"
    latest_file = local_dir / "embedding-train-latest.json"
    
    with open(result_file, "w") as f:
        json.dump(train_results, f, indent=2)
    
    with open(latest_file, "w") as f:
        json.dump(train_results, f, indent=2)
    
    print(f"  ✓ Results saved to {result_file}")
    print(f"  ✓ Latest results: {latest_file}")
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Download model: modal volume get legalkit-data models/legal-embeddings/model")
    print("  2. Run evaluation: legalkit bench --embedding-model models/legal-embeddings/model")
    print("=" * 60)
