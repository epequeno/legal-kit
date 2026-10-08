#!/usr/bin/env python3
"""Run expanded NLI evaluation with detailed breakdown.

Evaluates the fine-tuned DeBERTa model on the expanded evaluation set
(200+ cases including out-of-distribution categories) and reports:
- Overall accuracy (4-class and binary)
- In-distribution vs out-of-distribution performance
- Per-category accuracy
- Performance by label class
- Highlights categories where accuracy < 70%

Results saved to data/eval/nli-expanded-eval-<timestamp>.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent to path to import legalkit
sys.path.insert(0, str(Path(__file__).parent.parent / "python/src"))

from legalkit.eval.nli_eval import (
    NLIEvalResult,
    NLITestCase,
    load_edge_cases,
    run_finetuned_nli,
)


# Paths
EXPANDED_EVAL_PATH = Path(__file__).parent.parent / "data/eval/legal_nli_eval_expanded.json"
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models/legal-entailment/model"
OUTPUT_DIR = Path(__file__).parent.parent / "data/eval"


def load_expanded_cases(path: Path = EXPANDED_EVAL_PATH) -> tuple[dict, list[dict]]:
    """Load the expanded evaluation set and return metadata + raw case dicts."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expanded eval set not found: {path}\n"
            f"Run scripts/expand_nli_eval_set.py first to generate it."
        )
    
    with open(path) as f:
        data = json.load(f)
    
    metadata = {
        "version": data["version"],
        "description": data["description"],
        "generation_date": data.get("generation_date"),
        "generation_model": data.get("generation_model"),
        "categories": data["categories"],
        "labels": data["labels"],
    }
    
    return metadata, data["cases"]


def cases_to_nli_test_cases(raw_cases: list[dict]) -> list[NLITestCase]:
    """Convert raw case dicts to NLITestCase objects."""
    return [
        NLITestCase(
            id=c["id"],
            category=c["category"],
            passage=c["passage"],
            claim=c["claim"],
            label=c["label"],
            explanation=c.get("explanation", ""),
        )
        for c in raw_cases
    ]


def compute_detailed_breakdown(
    result: NLIEvalResult,
    raw_cases: list[dict],
) -> dict[str, Any]:
    """Compute detailed performance breakdown."""
    
    # Build case metadata lookup
    case_meta = {c["id"]: c for c in raw_cases}
    
    # Group judgments by distribution type
    id_judgments = []
    ood_judgments = []
    id_cases = []
    ood_cases = []
    
    for judgment, test_case in zip(result.judgments, result.test_cases):
        meta = case_meta[judgment.case_id]
        if meta["distribution"] == "in_distribution":
            id_judgments.append(judgment)
            id_cases.append(test_case)
        else:
            ood_judgments.append(judgment)
            ood_cases.append(test_case)
    
    # Compute ID vs OOD metrics
    def compute_accuracy(judgments, cases):
        if not judgments:
            return {"accuracy_4class": 0.0, "accuracy_binary": 0.0, "n": 0}
        
        from legalkit.eval.nli_eval import _to_binary
        
        correct_4class = sum(
            1 for j, c in zip(judgments, cases)
            if j.predicted == c.label
        )
        
        correct_binary = sum(
            1 for j, c in zip(judgments, cases)
            if _to_binary(j.predicted) == c.binary_label
        )
        
        return {
            "accuracy_4class": correct_4class / len(judgments),
            "accuracy_binary": correct_binary / len(judgments),
            "n": len(judgments),
        }
    
    id_metrics = compute_accuracy(id_judgments, id_cases)
    ood_metrics = compute_accuracy(ood_judgments, ood_cases)
    
    # Per-category breakdown with distribution annotation
    category_breakdown = {}
    by_category = defaultdict(lambda: {"judgments": [], "cases": []})
    
    for judgment, test_case in zip(result.judgments, result.test_cases):
        by_category[test_case.category]["judgments"].append(judgment)
        by_category[test_case.category]["cases"].append(test_case)
    
    for category, data in sorted(by_category.items()):
        metrics = compute_accuracy(data["judgments"], data["cases"])
        meta = case_meta[data["judgments"][0].case_id]
        
        category_breakdown[category] = {
            **metrics,
            "distribution": meta["distribution"],
            "accuracy_4class_pct": f"{metrics['accuracy_4class']:.1%}",
            "accuracy_binary_pct": f"{metrics['accuracy_binary']:.1%}",
        }
    
    # Label-wise performance
    from legalkit.eval.nli_eval import _to_binary
    
    label_breakdown = {}
    by_label = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for judgment, test_case in zip(result.judgments, result.test_cases):
        by_label[test_case.label]["total"] += 1
        if judgment.predicted == test_case.label:
            by_label[test_case.label]["correct"] += 1
    
    for label, counts in sorted(by_label.items()):
        accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
        label_breakdown[label] = {
            "correct": counts["correct"],
            "total": counts["total"],
            "accuracy": accuracy,
            "accuracy_pct": f"{accuracy:.1%}",
        }
    
    # Confusion matrix (already in result, but extract for clarity)
    confusion = result.confusion_matrix
    
    return {
        "overall": {
            "accuracy_4class": result.accuracy_4class,
            "accuracy_binary": result.accuracy_binary,
            "accuracy_4class_pct": f"{result.accuracy_4class:.1%}",
            "accuracy_binary_pct": f"{result.accuracy_binary:.1%}",
            "total_cases": len(result.judgments),
        },
        "by_distribution": {
            "in_distribution": {
                **id_metrics,
                "accuracy_4class_pct": f"{id_metrics['accuracy_4class']:.1%}",
                "accuracy_binary_pct": f"{id_metrics['accuracy_binary']:.1%}",
            },
            "out_of_distribution": {
                **ood_metrics,
                "accuracy_4class_pct": f"{ood_metrics['accuracy_4class']:.1%}",
                "accuracy_binary_pct": f"{ood_metrics['accuracy_binary']:.1%}",
            },
        },
        "by_category": category_breakdown,
        "by_label": label_breakdown,
        "confusion_matrix": confusion,
    }


def print_report(breakdown: dict, highlight_threshold: float = 0.70) -> None:
    """Print human-readable evaluation report."""
    
    print("\n" + "="*80)
    print("EXPANDED NLI EVALUATION REPORT")
    print("="*80)
    
    # Overall
    print("\n### OVERALL PERFORMANCE")
    print(f"  Total cases: {breakdown['overall']['total_cases']}")
    print(f"  4-class accuracy: {breakdown['overall']['accuracy_4class_pct']}")
    print(f"  Binary accuracy:  {breakdown['overall']['accuracy_binary_pct']}")
    
    # ID vs OOD
    print("\n### IN-DISTRIBUTION vs OUT-OF-DISTRIBUTION")
    
    id_data = breakdown['by_distribution']['in_distribution']
    print(f"\n  In-Distribution (trained categories):")
    print(f"    Cases: {id_data['n']}")
    print(f"    4-class accuracy: {id_data['accuracy_4class_pct']}")
    print(f"    Binary accuracy:  {id_data['accuracy_binary_pct']}")
    
    ood_data = breakdown['by_distribution']['out_of_distribution']
    print(f"\n  Out-of-Distribution (novel categories):")
    print(f"    Cases: {ood_data['n']}")
    print(f"    4-class accuracy: {ood_data['accuracy_4class_pct']}")
    print(f"    Binary accuracy:  {ood_data['accuracy_binary_pct']}")
    
    # Performance gap
    gap = id_data['accuracy_4class'] - ood_data['accuracy_4class']
    print(f"\n  Performance gap (ID - OOD): {gap:+.1%}")
    
    # Per-category
    print("\n### PER-CATEGORY PERFORMANCE")
    print(f"{'Category':<32} {'Type':^5} {'Cases':>5} {'4-class':>8} {'Binary':>8}")
    print("-" * 80)
    
    below_threshold = []
    
    for category, data in sorted(breakdown['by_category'].items()):
        dist_abbr = "ID" if data['distribution'] == "in_distribution" else "OOD"
        print(f"{category:<32} {dist_abbr:^5} {data['n']:>5} {data['accuracy_4class_pct']:>8} {data['accuracy_binary_pct']:>8}")
        
        if data['accuracy_4class'] < highlight_threshold:
            below_threshold.append((category, data['accuracy_4class']))
    
    # Highlight low-performing categories
    if below_threshold:
        print(f"\n  ⚠️  Categories below {highlight_threshold:.0%} threshold:")
        for cat, acc in sorted(below_threshold, key=lambda x: x[1]):
            print(f"     - {cat}: {acc:.1%}")
    else:
        print(f"\n  ✓ All categories above {highlight_threshold:.0%} threshold")
    
    # Per-label
    print("\n### PER-LABEL PERFORMANCE")
    print(f"{'Label':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 80)
    
    for label, data in sorted(breakdown['by_label'].items()):
        print(f"{label:<20} {data['correct']:>8} {data['total']:>8} {data['accuracy_pct']:>10}")
    
    # Confusion matrix summary (just show where errors concentrate)
    print("\n### MOST COMMON CONFUSIONS")
    confusion = breakdown['confusion_matrix']
    
    confusions = []
    for true_label, preds in confusion.items():
        for pred_label, count in preds.items():
            if true_label != pred_label and count > 0:
                confusions.append((count, true_label, pred_label))
    
    confusions.sort(reverse=True)
    for count, true_label, pred_label in confusions[:10]:
        print(f"  {true_label} → {pred_label}: {count}")
    
    print("\n" + "="*80)


def run_expanded_evaluation(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    eval_path: Path = EXPANDED_EVAL_PATH,
    device: str = "cpu",
    save_results: bool = True,
) -> tuple[NLIEvalResult, dict]:
    """Run evaluation on the expanded test set.
    
    Args:
        model_path: Path to fine-tuned model checkpoint
        eval_path: Path to expanded evaluation JSON
        device: 'cpu' or 'cuda'
        save_results: Whether to save results to JSON
    
    Returns:
        (NLIEvalResult, detailed_breakdown dict)
    """
    
    # Load test cases
    print(f"Loading expanded evaluation set from: {eval_path}")
    metadata, raw_cases = load_expanded_cases(eval_path)
    
    print(f"  Version: {metadata['version']}")
    print(f"  Total cases: {len(raw_cases)}")
    print(f"  Categories: {len(metadata['categories'])}")
    
    # Convert to NLITestCase objects
    test_cases = cases_to_nli_test_cases(raw_cases)
    
    # Run evaluation
    print(f"\nRunning evaluation with model: {model_path}")
    print(f"Device: {device}")
    print("\nProgress:")
    
    def progress(completed, total):
        pct = 100 * completed / total
        print(f"  {completed}/{total} ({pct:.0f}%)", end="\r")
    
    result = run_finetuned_nli(
        test_cases=test_cases,
        model_path=str(model_path),
        device=device,
        progress_callback=progress,
    )
    
    print("\n")
    
    # Compute detailed breakdown
    breakdown = compute_detailed_breakdown(result, raw_cases)
    
    # Print report
    print_report(breakdown)
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = OUTPUT_DIR / f"nli-expanded-eval-{timestamp}.json"
        
        output_data = {
            "timestamp": timestamp,
            "model_name": result.model_name,
            "model_path": str(model_path),
            "eval_set": str(eval_path),
            "eval_metadata": metadata,
            "results": {
                "overall": breakdown["overall"],
                "by_distribution": breakdown["by_distribution"],
                "by_category": breakdown["by_category"],
                "by_label": breakdown["by_label"],
            },
            "confusion_matrix": breakdown["confusion_matrix"],
            "judgments": [
                {
                    "case_id": j.case_id,
                    "predicted": j.predicted,
                    "confidence": j.confidence,
                    "latency_ms": j.latency_ms,
                }
                for j in result.judgments
            ],
        }
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
    
    return result, breakdown


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run expanded NLI evaluation")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=EXPANDED_EVAL_PATH,
        help="Path to expanded evaluation JSON",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to JSON",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Accuracy threshold for highlighting low-performing categories",
    )
    
    args = parser.parse_args()
    
    # Check model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model not found at {args.model_path}", file=sys.stderr)
        print("\nExpected model at models/legal-entailment/model/", file=sys.stderr)
        print("Run training first or specify --model-path", file=sys.stderr)
        sys.exit(1)
    
    # Check eval set exists
    if not args.eval_path.exists():
        print(f"Error: Eval set not found at {args.eval_path}", file=sys.stderr)
        print("\nRun scripts/expand_nli_eval_set.py first to generate it", file=sys.stderr)
        sys.exit(1)
    
    # Run evaluation
    result, breakdown = run_expanded_evaluation(
        model_path=args.model_path,
        eval_path=args.eval_path,
        device=args.device,
        save_results=not args.no_save,
    )
    
    # Note about threshold
    if args.threshold != 0.70:
        print(f"\n(Using custom threshold: {args.threshold:.0%})")


if __name__ == "__main__":
    main()
