"""Prepare expanded entailment training data (v2).

Merges existing training set with high-confidence pipeline logs,
capping the dominant 'neutral' class to preserve balance.

Usage:
    python scripts/prepare_entailment_v2.py
    python scripts/prepare_entailment_v2.py --neutral-cap 2  # multiplier over rarest class
"""

import json
import random
import argparse
from collections import Counter
from pathlib import Path

LABEL_MAP = {
    "entailed": 0,
    "partially_entailed": 1,
    "contradicted": 2,
    "neutral": 3,
}

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


def load_existing(path: Path) -> list[dict]:
    """Load existing training data (premise/hypothesis/label format)."""
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            # Normalise: ensure label_name is present
            if "label_name" not in row:
                row["label_name"] = LABEL_NAMES[row["label"]]
            rows.append(row)
    return rows


def load_logs(path: Path, min_confidence: float = 0.9) -> list[dict]:
    """Load pipeline entailment logs (passage/claim format) and convert."""
    rows = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("confidence", 0) < min_confidence:
                continue
            label_name = entry["entailment_class"]
            if label_name not in LABEL_MAP:
                continue
            rows.append({
                "premise": entry["passage"],
                "hypothesis": entry["claim"],
                "label": LABEL_MAP[label_name],
                "label_name": label_name,
                "section_id": entry.get("section_id", ""),
                "doc_id": entry.get("doc_id", ""),
                "source": "pipeline_log",
            })
    return rows


def balance(rows: list[dict], neutral_cap_multiplier: int = 2) -> list[dict]:
    """Cap neutral class at multiplier × rarest class count."""
    by_class: dict[str, list] = {}
    for row in rows:
        by_class.setdefault(row["label_name"], []).append(row)

    non_neutral = {k: v for k, v in by_class.items() if k != "neutral"}
    rarest = min(len(v) for v in non_neutral.values())
    cap = rarest * neutral_cap_multiplier

    result = []
    for label_name, items in by_class.items():
        if label_name == "neutral" and len(items) > cap:
            random.shuffle(items)
            items = items[:cap]
        result.extend(items)
    return result


def split(rows: list[dict], val_ratio: float = 0.12, test_ratio: float = 0.06, seed: int = 42):
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
    print(f"  wrote {len(rows):,} rows → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neutral-cap", type=int, default=2,
                        help="Neutral cap as multiplier over rarest class (default: 2)")
    parser.add_argument("--min-confidence", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).parent.parent

    # Load existing training data (all splits merged for redistribution)
    print("Loading existing training data...")
    existing = []
    for split_name in ("train", "val", "test"):
        p = root / "data/training" / f"{split_name}.jsonl"
        if p.exists():
            rows = load_existing(p)
            existing.extend(rows)
            print(f"  {split_name}.jsonl: {len(rows):,} rows")

    # Load logs
    print("\nLoading pipeline logs...")
    log_path = root / "data/entailment_logs/entailment_2026-03-17.jsonl"
    log_rows = load_logs(log_path, min_confidence=args.min_confidence)
    print(f"  loaded {len(log_rows):,} rows (confidence >= {args.min_confidence})")

    classes = Counter(r["label_name"] for r in log_rows)
    print("  class distribution in logs:")
    for cls, count in sorted(classes.items()):
        print(f"    {cls}: {count} ({count/len(log_rows)*100:.1f}%)")

    # Balance logs
    print(f"\nBalancing logs (neutral cap = {args.neutral_cap}x rarest)...")
    log_balanced = balance(log_rows, neutral_cap_multiplier=args.neutral_cap)
    bc = Counter(r["label_name"] for r in log_balanced)
    print("  after balancing:")
    for cls, count in sorted(bc.items()):
        print(f"    {cls}: {count} ({count/len(log_balanced)*100:.1f}%)")

    # Merge
    all_rows = existing + log_balanced
    print(f"\nMerged: {len(existing):,} existing + {len(log_balanced):,} logs = {len(all_rows):,} total")
    mc = Counter(r["label_name"] for r in all_rows)
    print("  final class distribution:")
    for cls, count in sorted(mc.items()):
        print(f"    {cls}: {count} ({count/len(all_rows)*100:.1f}%)")

    # Split
    train, val, test = split(all_rows, seed=args.seed)
    print(f"\nSplits: train={len(train):,}  val={len(val):,}  test={len(test):,}")

    # Write
    out_dir = root / "data/training_v2"
    print(f"\nWriting to {out_dir}/")
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(val, out_dir / "val.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")

    print("\nDone. Next:")
    print("  modal run scripts/modal_finetune.py --data-dir data/training_v2")


if __name__ == "__main__":
    main()
