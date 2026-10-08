"""Run all 4 LODO embedding training jobs on Modal sequentially.

Copies each LODO split into data/embedding_training/, runs training via Modal,
and downloads the resulting model to models/legal-embeddings-lodo-<dataset>/.

Usage:
    python scripts/run_lodo_training.py                 # Run all 4
    python scripts/run_lodo_training.py --held-out maud # Run one
    python scripts/run_lodo_training.py --skip-existing  # Skip if model dir exists

Prerequisites:
    - Modal CLI authenticated (modal token new)
    - LODO splits prepared (python scripts/embedding_cross_dataset_eval.py)
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
LODO_DIR = ROOT / "data" / "embedding_training_lodo"
TRAINING_DIR = ROOT / "data" / "embedding_training"
MODELS_DIR = ROOT / "models"
EVAL_DIR = ROOT / "eval"

DATASETS = ["cuad", "maud", "contractnli", "privacy_qa"]


def run_one(held_out: str, skip_existing: bool = False) -> dict:
    """Train one LODO model with `held_out` dataset excluded from training."""
    
    model_dir = MODELS_DIR / f"legal-embeddings-lodo-{held_out}"
    
    if skip_existing and model_dir.exists() and (model_dir / "config.json").exists():
        print(f"\n  ⏭  Skipping {held_out} — model already exists at {model_dir}")
        return {"held_out": held_out, "status": "skipped"}
    
    lodo_split_dir = LODO_DIR / held_out
    if not lodo_split_dir.exists():
        print(f"\n  ❌ LODO split not found: {lodo_split_dir}")
        print(f"     Run: python scripts/embedding_cross_dataset_eval.py")
        return {"held_out": held_out, "status": "error", "message": "LODO split not found"}
    
    # Step 1: Copy LODO split to the standard training directory
    print(f"\n  [1/3] Copying LODO split for held_out={held_out}...")
    
    # Backup existing training data
    backup_train = TRAINING_DIR / "train.jsonl.bak"
    backup_val = TRAINING_DIR / "val.jsonl.bak"
    original_train = TRAINING_DIR / "train.jsonl"
    original_val = TRAINING_DIR / "val.jsonl"
    
    if original_train.exists():
        shutil.copy2(original_train, backup_train)
    if original_val.exists():
        shutil.copy2(original_val, backup_val)
    
    # Copy LODO split
    shutil.copy2(lodo_split_dir / "train.jsonl", original_train)
    shutil.copy2(lodo_split_dir / "val.jsonl", original_val)
    
    train_count = sum(1 for _ in open(original_train))
    val_count = sum(1 for _ in open(original_val))
    print(f"         Train: {train_count:,} pairs, Val: {val_count:,} pairs")
    
    # Step 2: Run Modal training
    print(f"  [2/3] Training on Modal (held_out={held_out})...")
    
    result = subprocess.run(
        ["modal", "run", "scripts/modal_embedding_train.py"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour max
    )
    
    if result.returncode != 0:
        print(f"  ❌ Modal training failed:")
        print(result.stderr[-500:] if result.stderr else "(no stderr)")
        # Restore backup
        _restore_backup(backup_train, backup_val, original_train, original_val)
        return {"held_out": held_out, "status": "error", "message": result.stderr[-200:]}
    
    print(result.stdout[-300:])
    
    # Step 3: Download model from Modal volume
    print(f"  [3/3] Downloading model to {model_dir}...")
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    dl_result = subprocess.run(
        ["modal", "volume", "get", "legalkit-data", "models/legal-embeddings/model", str(model_dir)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    
    if dl_result.returncode != 0:
        print(f"  ⚠️  Download may have failed: {dl_result.stderr[:200]}")
    else:
        print(f"  ✓ Model saved to {model_dir}")
    
    # Restore original training data
    _restore_backup(backup_train, backup_val, original_train, original_val)
    
    # Save training record
    record = {
        "held_out": held_out,
        "status": "success",
        "train_samples": train_count,
        "val_samples": val_count,
        "model_dir": str(model_dir),
        "timestamp": datetime.now().isoformat(),
    }
    
    record_path = EVAL_DIR / f"lodo-train-{held_out}.json"
    EVAL_DIR.mkdir(exist_ok=True)
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)
    
    return record


def _restore_backup(backup_train, backup_val, original_train, original_val):
    """Restore original training data from backup."""
    if backup_train.exists():
        shutil.move(str(backup_train), str(original_train))
    if backup_val.exists():
        shutil.move(str(backup_val), str(original_val))


def main():
    parser = argparse.ArgumentParser(description="Run LODO embedding training on Modal")
    parser.add_argument("--held-out", type=str, choices=DATASETS,
                        help="Run only one held-out dataset (default: all 4)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if model directory already exists")
    args = parser.parse_args()
    
    datasets = [args.held_out] if args.held_out else DATASETS
    
    print("=" * 70)
    print("LODO Embedding Training — Leave-One-Dataset-Out Cross-Validation")
    print("=" * 70)
    print(f"Datasets to train: {datasets}")
    print(f"Each model is trained on 3 datasets, evaluated on the held-out 4th.")
    
    results = []
    for i, held_out in enumerate(datasets):
        print(f"\n{'='*70}")
        print(f"  [{i+1}/{len(datasets)}] Training with {held_out} held out")
        print(f"{'='*70}")
        
        record = run_one(held_out, skip_existing=args.skip_existing)
        results.append(record)
        
        if record["status"] == "error":
            print(f"\n⚠️  Training failed for {held_out}. Continuing with next...")
    
    # Summary
    print(f"\n{'='*70}")
    print("LODO Training Summary")
    print(f"{'='*70}")
    for r in results:
        status = "✓" if r["status"] == "success" else "⏭" if r["status"] == "skipped" else "✗"
        print(f"  {status} {r['held_out']}: {r['status']}")
    
    succeeded = sum(1 for r in results if r["status"] == "success")
    print(f"\n{succeeded}/{len(datasets)} training jobs completed successfully.")
    
    if succeeded > 0:
        print(f"\nNext steps:")
        print(f"  1. Run eval for each held-out dataset:")
        for r in results:
            if r["status"] == "success":
                held = r["held_out"]
                print(f"     legalkit bench --datasets {held} \\")
                print(f"       --embedding-model models/legal-embeddings-lodo-{held} \\")
                print(f"       --output eval/lodo-{held}.json --mini")
        print(f"  2. Generate report: python scripts/embedding_lodo_report.py")


if __name__ == "__main__":
    main()
