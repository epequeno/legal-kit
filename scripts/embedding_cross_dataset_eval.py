"""Leave-One-Dataset-Out (LODO) cross-validation for embedding model.

Prepares training data splits for cross-dataset evaluation. For each of the 4
LegalBench-RAG datasets (cuad, maud, contractnli, privacy_qa), this script:

1. Creates a train split from the OTHER 3 datasets
2. Withholds a 10% validation split from those 3 training datasets
3. The held-out dataset is used for evaluation (not included in train/val)

This allows us to measure how well the model generalizes to unseen legal
task types. The output is 4 sets of train/val JSONL files, one per held-out
dataset.

Usage:
    # Step 1: Prepare LODO splits
    python scripts/embedding_cross_dataset_eval.py

    # Step 2: Train 4 models (one per held-out dataset)
    # For each dataset, copy its split to data/embedding_training/ and train:
    
    cp data/embedding_training_lodo/cuad/*.jsonl data/embedding_training/
    modal run scripts/modal_embedding_train.py
    modal volume get legalkit-data models/legal-embeddings/model models/legal-embeddings-lodo-cuad
    
    cp data/embedding_training_lodo/maud/*.jsonl data/embedding_training/
    modal run scripts/modal_embedding_train.py
    modal volume get legalkit-data models/legal-embeddings/model models/legal-embeddings-lodo-maud
    
    # ... repeat for contractnli and privacy_qa

    # Step 3: Evaluate each model on its held-out dataset
    legalkit bench --datasets cuad --embedding-model models/legal-embeddings-lodo-cuad --output eval/lodo-cuad.json --mini
    legalkit bench --datasets maud --embedding-model models/legal-embeddings-lodo-maud --output eval/lodo-maud.json --mini
    legalkit bench --datasets contractnli --embedding-model models/legal-embeddings-lodo-contractnli --output eval/lodo-contractnli.json --mini
    legalkit bench --datasets privacy_qa --embedding-model models/legal-embeddings-lodo-privacy_qa --output eval/lodo-privacy_qa.json --mini

    # Step 4: Generate comparison report
    python scripts/embedding_lodo_report.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# Import the pair extraction function from the existing module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "python" / "src"))

from legalkit.eval.embedding_pairs import extract_pairs_from_benchmark, DATASET_NAMES


def prepare_lodo_splits(
    output_base_dir: Path | None = None,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Prepare leave-one-dataset-out training splits.
    
    For each dataset in DATASET_NAMES, creates:
    - Train split: pairs from the OTHER 3 datasets
    - Val split: 10% of the train split
    - The held-out dataset is NOT included (used for eval later)
    
    Args:
        output_base_dir: Base directory for output (default: data/embedding_training_lodo)
        val_ratio: Fraction of train data to hold out for validation
        seed: Random seed for reproducibility
    """
    output_base_dir = output_base_dir or Path("data/embedding_training_lodo")
    random.seed(seed)
    
    # Extract pairs from all datasets
    print("=" * 80)
    print("LODO Cross-Validation: Extracting training pairs from all datasets")
    print("=" * 80)
    print()
    
    all_pairs_by_dataset = {}
    total_pairs = 0
    
    for dataset_name in DATASET_NAMES:
        print(f"Extracting pairs from {dataset_name}...", end=" ", flush=True)
        pairs = extract_pairs_from_benchmark(dataset_name)
        all_pairs_by_dataset[dataset_name] = pairs
        total_pairs += len(pairs)
        print(f"{len(pairs)} pairs")
    
    print()
    print(f"Total pairs extracted: {total_pairs}")
    print()
    
    # Create LODO splits
    print("=" * 80)
    print("Creating LODO splits")
    print("=" * 80)
    print()
    
    summary_rows = []
    
    for held_out_dataset in DATASET_NAMES:
        print(f"\nHeld-out dataset: {held_out_dataset}")
        print("-" * 40)
        
        # Combine all OTHER datasets for training
        train_datasets = [ds for ds in DATASET_NAMES if ds != held_out_dataset]
        all_train_pairs = []
        
        for ds in train_datasets:
            pairs = all_pairs_by_dataset[ds]
            all_train_pairs.extend(pairs)
            print(f"  Training on {ds}: {len(pairs)} pairs")
        
        # Shuffle
        random.shuffle(all_train_pairs)
        
        # Split train/val
        n_val = int(len(all_train_pairs) * val_ratio)
        val_pairs = all_train_pairs[:n_val]
        train_pairs = all_train_pairs[n_val:]
        
        print(f"  Total train pairs: {len(train_pairs)}")
        print(f"  Total val pairs: {len(val_pairs)}")
        
        # Get eval dataset info
        eval_pairs = all_pairs_by_dataset[held_out_dataset]
        # Count unique queries in eval set
        eval_queries = len(set(p.query for p in eval_pairs))
        
        print(f"  Eval dataset ({held_out_dataset}): {eval_queries} unique queries")
        
        # Write files
        output_dir = output_base_dir / held_out_dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = output_dir / "train.jsonl"
        val_path = output_dir / "val.jsonl"
        
        with open(train_path, 'w') as f:
            for pair in train_pairs:
                f.write(json.dumps(pair.to_dict()) + '\n')
        
        with open(val_path, 'w') as f:
            for pair in val_pairs:
                f.write(json.dumps(pair.to_dict()) + '\n')
        
        print(f"  Written to: {output_dir}/")
        
        # Save summary row
        summary_rows.append({
            'held_out': held_out_dataset,
            'train_pairs': len(train_pairs),
            'val_pairs': len(val_pairs),
            'eval_queries': eval_queries,
            'train_datasets': ', '.join(train_datasets),
        })
    
    # Print summary table
    print()
    print("=" * 80)
    print("LODO Split Summary")
    print("=" * 80)
    print()
    print(f"{'Held-Out':<15} {'Train Pairs':<12} {'Val Pairs':<12} {'Eval Queries':<12} {'Train Datasets'}")
    print("-" * 80)
    
    for row in summary_rows:
        print(
            f"{row['held_out']:<15} "
            f"{row['train_pairs']:<12} "
            f"{row['val_pairs']:<12} "
            f"{row['eval_queries']:<12} "
            f"{row['train_datasets']}"
        )
    
    print()
    print("=" * 80)
    print("Next Steps: Training and Evaluation")
    print("=" * 80)
    print()
    print("NOTE: The modal_embedding_train.py script expects data in data/embedding_training/")
    print("      You need to copy each LODO split there before training.")
    print()
    print("1. Train 4 embedding models (one per held-out dataset):")
    print()
    
    for held_out in DATASET_NAMES:
        data_dir = f"data/embedding_training_lodo/{held_out}"
        print(f"   # Train model with {held_out} held out:")
        print(f"   cp {data_dir}/*.jsonl data/embedding_training/")
        print(f"   modal run scripts/modal_embedding_train.py")
        print(f"   # Download and save model as: models/legal-embeddings-lodo-{held_out}/")
        print(f"   modal volume get legalkit-data models/legal-embeddings/model models/legal-embeddings-lodo-{held_out}")
        print()
    
    print("2. Evaluate each model on its held-out dataset:")
    print()
    
    for held_out in DATASET_NAMES:
        print(f"   legalkit bench --datasets {held_out} \\")
        print(f"     --embedding-model models/legal-embeddings-lodo-{held_out} \\")
        print(f"     --output eval/lodo-{held_out}.json --mini")
        print()
    
    print("3. Generate comparison report:")
    print()
    print("   python scripts/embedding_lodo_report.py")
    print()
    print("=" * 80)


def main():
    """Run LODO split preparation."""
    prepare_lodo_splits()


if __name__ == "__main__":
    main()
