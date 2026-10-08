"""Prepare entailment training data v3: 3-class with corrected labels.

Changes from v2:
  1. Collapse partially_entailed → entailed (audit showed 84% were mislabeled)
  2. 3 classes: entailed (supported), contradicted, neutral
  3. Compute class weights for balanced training
  4. Clean splits with no data leakage

Usage:
    python scripts/prepare_entailment_v3.py
    python scripts/prepare_entailment_v3.py --audit-corrections  # Apply audit re-labels too
"""

import json
import random
import argparse
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent

# 3-class label mapping
LABEL_MAP_V3 = {
    "entailed": 0,
    "contradicted": 1,
    "neutral": 2,
}
LABEL_NAMES_V3 = {v: k for k, v in LABEL_MAP_V3.items()}

# Old 4-class → new 3-class
COLLAPSE_MAP = {
    "entailed": "entailed",
    "partially_entailed": "entailed",  # THE KEY CHANGE
    "contradicted": "contradicted",
    "neutral": "neutral",
}


def load_v2_data() -> list[dict]:
    """Load all v2 training data (train + val + test merged for re-splitting)."""
    rows = []
    for split in ("train", "val", "test"):
        path = ROOT / "data" / "training_v2" / f"{split}.jsonl"
        if path.exists():
            with open(path) as f:
                for line in f:
                    rows.append(json.loads(line))
    return rows


def load_audit_corrections() -> dict[str, str]:
    """Load label corrections from the audit.
    
    Returns dict of (premise_hash, hypothesis_hash) → corrected_label
    for samples where Claude disagreed with the original label.
    """
    audit_path = ROOT / "data" / "eval" / "label_audit_samples.json"
    if not audit_path.exists():
        print("  Warning: No audit samples found, skipping corrections")
        return {}
    
    samples = json.load(open(audit_path))
    corrections = {}
    
    for s in samples:
        claude_result = s.get("claude_result", {})
        if claude_result.get("label_numeric", -1) == -1:
            continue  # Error in audit
        
        orig_label = s["label_name"]
        claude_label = claude_result["label_name"]
        
        if orig_label != claude_label:
            # Use a hash of premise+hypothesis as key
            key = (s["premise"][:100], s["hypothesis"][:100])
            # Map Claude's 4-class label to 3-class
            corrections[key] = COLLAPSE_MAP.get(claude_label, claude_label)
    
    return corrections


def convert_to_v3(rows: list[dict], corrections: dict = None) -> list[dict]:
    """Convert 4-class rows to 3-class, applying corrections."""
    corrections = corrections or {}
    converted = []
    stats = Counter()
    
    for row in rows:
        old_label = row["label_name"]
        
        # Check for audit correction first
        key = (row["premise"][:100], row["hypothesis"][:100])
        if key in corrections:
            new_label = corrections[key]
            stats["audit_corrected"] += 1
        else:
            new_label = COLLAPSE_MAP[old_label]
        
        if old_label == "partially_entailed":
            stats["collapsed_partial→entailed"] += 1
        
        converted.append({
            "premise": row["premise"],
            "hypothesis": row["hypothesis"],
            "label": LABEL_MAP_V3[new_label],
            "label_name": new_label,
            "source": row.get("source", "unknown"),
            "doc_id": row.get("doc_id", ""),
            "section_id": row.get("section_id", ""),
            "original_label": old_label,
        })
    
    return converted, stats


def compute_class_weights(rows: list[dict]) -> dict[str, float]:
    """Compute inverse-frequency class weights."""
    counts = Counter(r["label_name"] for r in rows)
    max_count = max(counts.values())
    weights = {label: max_count / count for label, count in counts.items()}
    return weights


def split_data(rows: list[dict], val_ratio: float = 0.12, test_ratio: float = 0.06, 
               seed: int = 42) -> tuple[list, list, list]:
    """Split data into train/val/test."""
    random.seed(seed)
    random.shuffle(rows)
    n = len(rows)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    return rows[n_val + n_test:], rows[:n_val], rows[n_val:n_val + n_test]


def write_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-corrections", action="store_true",
                        help="Apply label corrections from audit")
    parser.add_argument("--extra-data", type=str, nargs="+",
                        help="Additional JSONL files to merge (e.g. targeted negatives)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 70)
    print("Preparing Entailment Training Data v3 (3-class)")
    print("=" * 70)
    
    # Load v2 data
    print("\nLoading v2 data...")
    rows = load_v2_data()
    print(f"  Total rows: {len(rows):,}")
    
    v2_dist = Counter(r["label_name"] for r in rows)
    print("  v2 distribution:")
    for l, c in sorted(v2_dist.items()):
        print(f"    {l:<25} {c:>5} ({c/len(rows):.1%})")
    
    # Load corrections
    corrections = {}
    if args.audit_corrections:
        print("\nLoading audit corrections...")
        corrections = load_audit_corrections()
        print(f"  Corrections available: {len(corrections)}")
    
    # Convert to 3-class
    print("\nConverting to 3-class...")
    converted, stats = convert_to_v3(rows, corrections)
    
    v3_dist = Counter(r["label_name"] for r in converted)
    print("  v3 distribution:")
    for l, c in sorted(v3_dist.items()):
        print(f"    {l:<25} {c:>5} ({c/len(converted):.1%})")
    print(f"  Collapsed partial→entailed: {stats.get('collapsed_partial→entailed', 0)}")
    if args.audit_corrections:
        print(f"  Audit-corrected labels: {stats.get('audit_corrected', 0)}")
    
    # Merge extra data (e.g. targeted hard negatives)
    if args.extra_data:
        print(f"\nMerging extra training data...")
        for extra_path in args.extra_data:
            extra_rows = []
            with open(extra_path) as f:
                for line in f:
                    row = json.loads(line)
                    # Ensure 3-class label mapping
                    label_name = row.get("label_name", "")
                    if label_name == "partially_entailed":
                        label_name = "entailed"
                    if label_name not in LABEL_MAP_V3:
                        print(f"  Warning: skipping row with unknown label '{label_name}'")
                        continue
                    extra_rows.append({
                        "premise": row.get("premise", row.get("passage", "")),
                        "hypothesis": row.get("hypothesis", row.get("claim", "")),
                        "label": LABEL_MAP_V3[label_name],
                        "label_name": label_name,
                        "source": row.get("source", "extra"),
                        "doc_id": row.get("doc_id", ""),
                        "section_id": row.get("section_id", ""),
                        "original_label": row.get("original_label", label_name),
                    })
            converted.extend(extra_rows)
            extra_dist = Counter(r["label_name"] for r in extra_rows)
            print(f"  {extra_path}: {len(extra_rows)} rows")
            for l, c in sorted(extra_dist.items()):
                print(f"    {l}: {c}")
        
        # Recompute distribution
        v3_dist = Counter(r["label_name"] for r in converted)
        print(f"\n  Combined distribution:")
        for l, c in sorted(v3_dist.items()):
            print(f"    {l:<25} {c:>5} ({c/len(converted):.1%})")
    
    # Compute class weights
    weights = compute_class_weights(converted)
    print("\n  Class weights (for balanced training):")
    for l, w in sorted(weights.items()):
        print(f"    {l:<25} {w:.3f}")
    
    # Split
    train, val, test = split_data(converted, seed=args.seed)
    print(f"\n  Splits: train={len(train):,}  val={len(val):,}  test={len(test):,}")
    
    # Write
    out_dir = ROOT / "data" / "training_v3"
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(val, out_dir / "val.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")
    
    # Write metadata
    metadata = {
        "version": "v3",
        "description": "3-class entailment (collapsed partially_entailed → entailed)",
        "classes": list(LABEL_MAP_V3.keys()),
        "label_map": LABEL_MAP_V3,
        "class_weights": weights,
        "total_samples": len(converted),
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "distribution": dict(sorted(v3_dist.items())),
        "audit_corrections_applied": args.audit_corrections,
        "seed": args.seed,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  Written to {out_dir}/")
    print(f"\nNext: modal run scripts/modal_finetune.py --data-dir data/training_v3 --num-labels 3")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
