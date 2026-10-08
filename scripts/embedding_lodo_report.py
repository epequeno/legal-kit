"""LODO (Leave-One-Dataset-Out) evaluation results reporter.

Loads evaluation results from the 4 LODO experiments and generates a
comparison table showing how well each fine-tuned model performs on its
held-out dataset vs. the baseline embedding model.

Usage:
    python scripts/embedding_lodo_report.py

    # With custom file paths:
    python scripts/embedding_lodo_report.py \
        --lodo-results eval/lodo-*.json \
        --baseline eval/baseline.json

Input format: Each JSON file should contain benchmark results with structure:
    {
        "metrics": {
            "overall": {
                "recall": {"@1": ..., "@64": ...},
                "precision": {"@1": ..., "@64": ...}
            },
            "per_dataset": {
                "<dataset_name>": {
                    "recall": {...},
                    "precision": {...}
                }
            }
        },
        "num_queries": <int>,
        "datasets": [...]
    }
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any


DATASET_NAMES = ["cuad", "maud", "contractnli", "privacy_qa"]
REPORT_KS = [1, 4, 16, 64]  # Key recall@k values to report


def load_results(file_path: Path) -> dict[str, Any]:
    """Load benchmark results JSON file.
    
    Args:
        file_path: Path to results JSON
        
    Returns:
        Parsed JSON dict
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    with open(file_path) as f:
        return json.load(f)


def extract_metrics(
    results: dict[str, Any],
    dataset_name: str | None = None,
) -> dict[str, float]:
    """Extract R@k and P@k metrics from results.
    
    Args:
        results: Benchmark results dict
        dataset_name: If provided, extract per-dataset metrics; otherwise overall
        
    Returns:
        Dict with keys like 'R@1', 'R@4', 'P@64', etc.
    """
    metrics = results.get("metrics", {})
    
    if dataset_name:
        source = metrics.get("per_dataset", {}).get(dataset_name, {})
    else:
        source = metrics.get("overall", {})
    
    recall = source.get("recall", {})
    precision = source.get("precision", {})
    
    extracted = {}
    for k in REPORT_KS:
        r_key = f"@{k}"
        extracted[f"R@{k}"] = recall.get(r_key, 0.0)
        extracted[f"P@{k}"] = precision.get(r_key, 0.0)
    
    return extracted


def generate_report(
    lodo_files: list[Path],
    baseline_file: Path | None = None,
) -> None:
    """Generate LODO evaluation comparison report.
    
    Args:
        lodo_files: List of paths to LODO result JSON files
        baseline_file: Optional path to baseline results JSON
    """
    print("=" * 100)
    print("LODO (Leave-One-Dataset-Out) Cross-Validation Results")
    print("=" * 100)
    print()
    
    # Load baseline if available
    baseline_metrics = None
    if baseline_file and baseline_file.exists():
        print(f"Loading baseline results from: {baseline_file}")
        baseline_results = load_results(baseline_file)
        baseline_metrics = {}
        for dataset in DATASET_NAMES:
            baseline_metrics[dataset] = extract_metrics(baseline_results, dataset)
        print()
    elif baseline_file:
        print(f"Warning: Baseline file not found: {baseline_file}")
        print()
    
    # Load LODO results
    lodo_results = {}
    
    for file_path in lodo_files:
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
        
        # Infer dataset name from filename (e.g., lodo-cuad.json -> cuad)
        dataset_name = None
        for ds in DATASET_NAMES:
            if ds in file_path.stem:
                dataset_name = ds
                break
        
        if not dataset_name:
            print(f"Warning: Could not infer dataset name from {file_path}")
            continue
        
        print(f"Loading LODO results for {dataset_name}: {file_path}")
        results = load_results(file_path)
        
        # Extract metrics for the held-out dataset
        lodo_results[dataset_name] = {
            'file': file_path,
            'metrics': extract_metrics(results, dataset_name) if 'per_dataset' in results.get('metrics', {}) else extract_metrics(results),
            'num_queries': results.get('num_queries', 0),
        }
    
    print()
    
    if not lodo_results:
        print("Error: No LODO results found!")
        print()
        print("Expected files:")
        for ds in DATASET_NAMES:
            print(f"  eval/lodo-{ds}.json")
        print()
        print("Run the evaluation pipeline first:")
        print("  python scripts/embedding_cross_dataset_eval.py")
        print("  # ... train models ...")
        print("  # ... run evaluations ...")
        return
    
    # Print detailed per-dataset results
    print("=" * 100)
    print("Per-Dataset Results (Fine-tuned on OTHER 3 datasets)")
    print("=" * 100)
    print()
    
    for dataset in DATASET_NAMES:
        if dataset not in lodo_results:
            print(f"Dataset: {dataset:<15} [NO RESULTS]")
            print()
            continue
        
        lodo = lodo_results[dataset]
        metrics = lodo['metrics']
        
        print(f"Dataset: {dataset:<15} ({lodo['num_queries']} queries)")
        print("-" * 100)
        
        # Print recall metrics
        print("Recall:    ", end="")
        for k in REPORT_KS:
            val = metrics.get(f"R@{k}", 0.0) * 100
            print(f"R@{k:<2} = {val:5.2f}%   ", end="")
        print()
        
        # Print precision metrics
        print("Precision: ", end="")
        for k in REPORT_KS:
            val = metrics.get(f"P@{k}", 0.0) * 100
            print(f"P@{k:<2} = {val:5.2f}%   ", end="")
        print()
        
        # Compare to baseline if available
        if baseline_metrics and dataset in baseline_metrics:
            baseline = baseline_metrics[dataset]
            print()
            print("Baseline comparison (text-embedding-3-large):")
            print("  Metric      Fine-tuned   Baseline     Delta")
            
            for k in REPORT_KS:
                r_key = f"R@{k}"
                ft_val = metrics.get(r_key, 0.0) * 100
                bl_val = baseline.get(r_key, 0.0) * 100
                delta = ft_val - bl_val
                sign = "+" if delta > 0 else ""
                print(f"  {r_key:<11} {ft_val:6.2f}%      {bl_val:6.2f}%      {sign}{delta:5.2f}%")
        
        print()
    
    # Compute aggregate metrics (average across 4 held-out datasets)
    print("=" * 100)
    print("Aggregate Metrics (Average across 4 held-out datasets)")
    print("=" * 100)
    print()
    
    if len(lodo_results) < len(DATASET_NAMES):
        missing = set(DATASET_NAMES) - set(lodo_results.keys())
        print(f"Warning: Missing results for {len(missing)} dataset(s): {', '.join(missing)}")
        print()
    
    # Calculate averages
    aggregate_lodo = {}
    for k in REPORT_KS:
        r_key = f"R@{k}"
        p_key = f"P@{k}"
        
        r_values = [lodo_results[ds]['metrics'].get(r_key, 0.0) for ds in lodo_results]
        p_values = [lodo_results[ds]['metrics'].get(p_key, 0.0) for ds in lodo_results]
        
        aggregate_lodo[r_key] = sum(r_values) / len(r_values) if r_values else 0.0
        aggregate_lodo[p_key] = sum(p_values) / len(p_values) if p_values else 0.0
    
    print("Fine-tuned (LODO average):")
    print("  Recall:    ", end="")
    for k in REPORT_KS:
        val = aggregate_lodo.get(f"R@{k}", 0.0) * 100
        print(f"R@{k:<2} = {val:5.2f}%   ", end="")
    print()
    
    print("  Precision: ", end="")
    for k in REPORT_KS:
        val = aggregate_lodo.get(f"P@{k}", 0.0) * 100
        print(f"P@{k:<2} = {val:5.2f}%   ", end="")
    print()
    print()
    
    # Compare aggregates to baseline
    if baseline_metrics:
        print("Baseline (text-embedding-3-large average):")
        
        aggregate_baseline = {}
        for k in REPORT_KS:
            r_key = f"R@{k}"
            p_key = f"P@{k}"
            
            r_values = [baseline_metrics[ds].get(r_key, 0.0) for ds in baseline_metrics]
            p_values = [baseline_metrics[ds].get(p_key, 0.0) for ds in baseline_metrics]
            
            aggregate_baseline[r_key] = sum(r_values) / len(r_values) if r_values else 0.0
            aggregate_baseline[p_key] = sum(p_values) / len(p_values) if p_values else 0.0
        
        print("  Recall:    ", end="")
        for k in REPORT_KS:
            val = aggregate_baseline.get(f"R@{k}", 0.0) * 100
            print(f"R@{k:<2} = {val:5.2f}%   ", end="")
        print()
        
        print("  Precision: ", end="")
        for k in REPORT_KS:
            val = aggregate_baseline.get(f"P@{k}", 0.0) * 100
            print(f"P@{k:<2} = {val:5.2f}%   ", end="")
        print()
        print()
        
        print("Improvement (Fine-tuned vs Baseline):")
        print("  Metric      Fine-tuned   Baseline     Delta      Relative")
        print("  " + "-" * 60)
        
        for k in REPORT_KS:
            r_key = f"R@{k}"
            ft_val = aggregate_lodo.get(r_key, 0.0) * 100
            bl_val = aggregate_baseline.get(r_key, 0.0) * 100
            delta = ft_val - bl_val
            rel = ((ft_val / bl_val - 1) * 100) if bl_val > 0 else 0
            sign = "+" if delta > 0 else ""
            rel_sign = "+" if rel > 0 else ""
            print(f"  {r_key:<11} {ft_val:6.2f}%      {bl_val:6.2f}%      {sign}{delta:5.2f}%     {rel_sign}{rel:5.1f}%")
        
        print()
    
    print("=" * 100)
    print()
    
    # Summary interpretation
    if lodo_results:
        print("Interpretation:")
        print("-" * 100)
        print()
        print("LODO (Leave-One-Dataset-Out) cross-validation measures how well the fine-tuned")
        print("model generalizes to UNSEEN legal task types. Each model is trained on 3 datasets")
        print("and evaluated on the 4th held-out dataset.")
        print()
        print("Key metrics:")
        print("  - R@64: Recall at top-64 chunks (primary metric for retrieval coverage)")
        print("  - R@16: Recall at top-16 chunks (practical limit for LLM context)")
        print("  - R@4:  Recall at top-4 chunks (focused retrieval)")
        print()
        
        # Find best and worst performers
        if len(lodo_results) > 1:
            r64_scores = {ds: lodo_results[ds]['metrics'].get('R@64', 0.0) * 100 
                         for ds in lodo_results}
            best_ds = max(r64_scores, key=r64_scores.get)
            worst_ds = min(r64_scores, key=r64_scores.get)
            
            print(f"Best performance:  {best_ds:<15} (R@64 = {r64_scores[best_ds]:.2f}%)")
            print(f"Worst performance: {worst_ds:<15} (R@64 = {r64_scores[worst_ds]:.2f}%)")
            print()
        
        print("The aggregate metrics show the AVERAGE performance across all 4 held-out datasets,")
        print("giving a robust estimate of generalization to new legal tasks.")
        print()
        print("=" * 100)


def main():
    """CLI entrypoint for LODO report generation."""
    parser = argparse.ArgumentParser(
        description="Generate LODO cross-validation report for embedding models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--lodo-dir",
        type=Path,
        default=Path("eval"),
        help="Directory containing LODO result JSON files (default: eval/)",
    )
    
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("eval/mini-baseline.json"),
        help="Baseline results JSON file (default: eval/mini-baseline.json)",
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="lodo-*.json",
        help="Filename pattern for LODO results (default: lodo-*.json)",
    )
    
    args = parser.parse_args()
    
    # Find LODO result files
    lodo_files = sorted(args.lodo_dir.glob(args.pattern))
    
    if not lodo_files:
        print(f"Error: No LODO result files found matching {args.lodo_dir}/{args.pattern}")
        print()
        print("Expected files:")
        for ds in DATASET_NAMES:
            print(f"  {args.lodo_dir}/lodo-{ds}.json")
        print()
        print("Run the evaluation pipeline first:")
        print("  python scripts/embedding_cross_dataset_eval.py")
        return
    
    # Generate report
    generate_report(
        lodo_files=lodo_files,
        baseline_file=args.baseline if args.baseline.exists() else None,
    )


if __name__ == "__main__":
    main()
