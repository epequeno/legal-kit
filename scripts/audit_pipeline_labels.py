"""Audit LLM-generated labels in entailment training data.

Samples pipeline_log entries and re-labels them with Claude Sonnet to measure
agreement rates, identify systematic biases, and flag suspicious disagreements.

Usage:
  python scripts/audit_pipeline_labels.py
  python scripts/audit_pipeline_labels.py --n 200 --dry-run
  python scripts/audit_pipeline_labels.py --n 50 --batch-size 5
"""

import json
import random
import asyncio
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import os

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

if not HAS_ANTHROPIC and not HAS_HTTPX:
    print("ERROR: Need either 'anthropic' or 'httpx' package. Install with: pip install anthropic")
    exit(1)


LABEL_MAP = {
    "entailed": 0,
    "partially_entailed": 1,
    "contradicted": 2,
    "neutral": 3,
}

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

# Inverse mapping for response parsing
RESPONSE_TO_LABEL = {
    "entailed": 0,
    "partially_entailed": 1,
    "partially entailed": 1,
    "contradicted": 2,
    "neutral": 3,
}


AUDIT_PROMPT = """You are a legal document analyst. I will give you a PREMISE (text from a legal contract) and a HYPOTHESIS (a claim about that text).

Your task is to determine the entailment relationship between them. Use these definitions:

- **entailed**: The hypothesis is fully and directly supported by the premise. All key facts in the hypothesis are explicitly stated or necessarily implied.
- **partially_entailed**: The hypothesis is partially supported but contains elements that are not directly stated, are overgeneralized, or require additional inference.
- **contradicted**: The hypothesis directly contradicts information in the premise.
- **neutral**: The hypothesis is not addressed by the premise (neither supported nor contradicted).

PREMISE:
{premise}

HYPOTHESIS:
{hypothesis}

Respond with valid JSON only (no markdown, no code fences):
{{
  "label": "entailed" | "partially_entailed" | "contradicted" | "neutral",
  "confidence": 0.95,
  "explanation": "Brief explanation of your reasoning"
}}"""


async def call_claude(client: anthropic.AsyncAnthropic, premise: str, hypothesis: str, 
                      max_retries: int = 3) -> Dict[str, Any]:
    """Call Claude to re-label a single example."""
    prompt = AUDIT_PROMPT.format(premise=premise, hypothesis=hypothesis)
    
    for attempt in range(max_retries):
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.content[0].text.strip()
            
            # Try to parse JSON
            # Handle case where Claude wraps in markdown code blocks despite instructions
            if text.startswith("```"):
                # Extract JSON from code block
                lines = text.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block or (not line.strip().startswith("```")):
                        json_lines.append(line)
                text = "\n".join(json_lines).strip()
            
            result = json.loads(text)
            
            # Validate structure
            if "label" not in result or "confidence" not in result:
                raise ValueError(f"Missing required fields: {result}")
            
            # Normalize label
            label_str = result["label"].lower().strip()
            if label_str not in RESPONSE_TO_LABEL:
                raise ValueError(f"Invalid label: {label_str}")
            
            result["label_numeric"] = RESPONSE_TO_LABEL[label_str]
            result["label_name"] = LABEL_NAMES[result["label_numeric"]]
            
            return result
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            if attempt < max_retries - 1:
                print(f"  Parse error (attempt {attempt+1}/{max_retries}): {e}")
                await asyncio.sleep(1)
                continue
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return {
                    "label": "error",
                    "label_numeric": -1,
                    "label_name": "error",
                    "confidence": 0.0,
                    "explanation": f"Parse error: {e}",
                    "raw_response": text
                }
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  API error (attempt {attempt+1}/{max_retries}): {e}")
                await asyncio.sleep(2)
                continue
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return {
                    "label": "error",
                    "label_numeric": -1,
                    "label_name": "error",
                    "confidence": 0.0,
                    "explanation": f"API error: {e}"
                }
    
    # Should never reach here
    return {
        "label": "error",
        "label_numeric": -1,
        "label_name": "error",
        "confidence": 0.0,
        "explanation": "Unknown error"
    }


async def call_openrouter(premise: str, hypothesis: str, api_key: str,
                          max_retries: int = 3) -> Dict[str, Any]:
    """Call Claude via OpenRouter's OpenAI-compatible API."""
    prompt = AUDIT_PROMPT.format(premise=premise, hypothesis=hypothesis)
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "anthropic/claude-sonnet-4",
                        "max_tokens": 500,
                        "temperature": 0,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"].strip()
            
            # Handle markdown code blocks
            if text.startswith("```"):
                lines = text.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block or (not line.strip().startswith("```")):
                        json_lines.append(line)
                text = "\n".join(json_lines).strip()
            
            result = json.loads(text)
            
            if "label" not in result or "confidence" not in result:
                raise ValueError(f"Missing required fields: {result}")
            
            label_str = result["label"].lower().strip()
            if label_str not in RESPONSE_TO_LABEL:
                raise ValueError(f"Invalid label: {label_str}")
            
            result["label_numeric"] = RESPONSE_TO_LABEL[label_str]
            result["label_name"] = LABEL_NAMES[result["label_numeric"]]
            
            return result
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Error (attempt {attempt+1}/{max_retries}): {e}")
                await asyncio.sleep(2)
                continue
            else:
                return {
                    "label": "error",
                    "label_numeric": -1,
                    "label_name": "error",
                    "confidence": 0.0,
                    "explanation": f"Error: {e}"
                }


async def batch_audit(samples: List[Dict], batch_size: int = 10) -> List[Dict]:
    """Re-label samples in batches using Claude (via Anthropic API or OpenRouter)."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    if anthropic_key and HAS_ANTHROPIC:
        use_openrouter = False
        print("  Using Anthropic API directly")
        client = anthropic.AsyncAnthropic(api_key=anthropic_key)
    elif openrouter_key:
        use_openrouter = True
        print("  Using OpenRouter API (Claude Sonnet)")
    else:
        raise ValueError("Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY environment variable")
    
    results = []
    total = len(samples)
    
    for i in range(0, total, batch_size):
        batch = samples[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} "
              f"(samples {i+1}-{min(i+batch_size, total)}/{total})...")
        
        if use_openrouter:
            tasks = [
                call_openrouter(sample["premise"], sample["hypothesis"], openrouter_key)
                for sample in batch
            ]
        else:
            tasks = [
                call_claude(client, sample["premise"], sample["hypothesis"])
                for sample in batch
            ]
        
        batch_results = await asyncio.gather(*tasks)
        
        # Merge results with original samples
        for sample, claude_result in zip(batch, batch_results):
            sample["claude_result"] = claude_result
            results.append(sample)
        
        # Rate limiting: small delay between batches
        if i + batch_size < total:
            await asyncio.sleep(0.5)
    
    return results


def stratified_sample(rows: List[Dict], n_per_class: int, seed: int = 42) -> List[Dict]:
    """Sample n_per_class examples from each label class."""
    random.seed(seed)
    
    by_class = defaultdict(list)
    for row in rows:
        by_class[row["label"]].append(row)
    
    samples = []
    for label in sorted(by_class.keys()):
        class_rows = by_class[label]
        if len(class_rows) < n_per_class:
            print(f"  WARNING: Only {len(class_rows)} examples for label {LABEL_NAMES[label]}, "
                  f"requested {n_per_class}")
            samples.extend(class_rows)
        else:
            samples.extend(random.sample(class_rows, n_per_class))
    
    return samples


def compute_metrics(audited: List[Dict]) -> Dict[str, Any]:
    """Compute agreement metrics and confusion matrix."""
    
    # Filter out errors
    valid = [s for s in audited if s["claude_result"]["label_numeric"] != -1]
    n_errors = len(audited) - len(valid)
    
    if not valid:
        return {
            "error": "No valid re-labels",
            "n_errors": n_errors,
            "total": len(audited)
        }
    
    # Overall agreement
    agreements = sum(1 for s in valid 
                    if s["label"] == s["claude_result"]["label_numeric"])
    overall_agreement = agreements / len(valid)
    
    # Per-class agreement
    by_class = defaultdict(lambda: {"total": 0, "agreements": 0})
    for s in valid:
        orig_label = s["label"]
        claude_label = s["claude_result"]["label_numeric"]
        by_class[orig_label]["total"] += 1
        if orig_label == claude_label:
            by_class[orig_label]["agreements"] += 1
    
    class_agreement = {
        LABEL_NAMES[label]: {
            "agreement_rate": stats["agreements"] / stats["total"] if stats["total"] > 0 else 0,
            "n_samples": stats["total"]
        }
        for label, stats in by_class.items()
    }
    
    # Confusion matrix: original → claude
    confusion = defaultdict(lambda: defaultdict(int))
    for s in valid:
        orig = LABEL_NAMES[s["label"]]
        claude = s["claude_result"]["label_name"]
        confusion[orig][claude] += 1
    
    # Label transitions (disagreements only)
    transitions = defaultdict(int)
    for s in valid:
        if s["label"] != s["claude_result"]["label_numeric"]:
            orig = LABEL_NAMES[s["label"]]
            claude = s["claude_result"]["label_name"]
            transitions[f"{orig}→{claude}"] += 1
    
    # Confidence distribution
    agreement_confidences = [
        s["claude_result"]["confidence"]
        for s in valid
        if s["label"] == s["claude_result"]["label_numeric"]
    ]
    disagreement_confidences = [
        s["claude_result"]["confidence"]
        for s in valid
        if s["label"] != s["claude_result"]["label_numeric"]
    ]
    
    # Find most suspicious (high confidence disagreements)
    disagreements = [
        {
            "premise": s["premise"][:200] + "..." if len(s["premise"]) > 200 else s["premise"],
            "hypothesis": s["hypothesis"],
            "original_label": LABEL_NAMES[s["label"]],
            "claude_label": s["claude_result"]["label_name"],
            "confidence": s["claude_result"]["confidence"],
            "explanation": s["claude_result"]["explanation"],
            "doc_id": s.get("doc_id", ""),
            "section_id": s.get("section_id", "")
        }
        for s in valid
        if s["label"] != s["claude_result"]["label_numeric"]
    ]
    disagreements.sort(key=lambda x: x["confidence"], reverse=True)
    
    return {
        "n_total": len(audited),
        "n_valid": len(valid),
        "n_errors": n_errors,
        "overall_agreement": overall_agreement,
        "n_agreements": agreements,
        "n_disagreements": len(valid) - agreements,
        "class_agreement": class_agreement,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "top_transitions": dict(sorted(transitions.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True)[:10]),
        "confidence_stats": {
            "agreements": {
                "mean": sum(agreement_confidences) / len(agreement_confidences) if agreement_confidences else 0,
                "min": min(agreement_confidences) if agreement_confidences else 0,
                "max": max(agreement_confidences) if agreement_confidences else 0,
                "n": len(agreement_confidences)
            },
            "disagreements": {
                "mean": sum(disagreement_confidences) / len(disagreement_confidences) if disagreement_confidences else 0,
                "min": min(disagreement_confidences) if disagreement_confidences else 0,
                "max": max(disagreement_confidences) if disagreement_confidences else 0,
                "n": len(disagreement_confidences)
            }
        },
        "most_suspicious": disagreements[:20]
    }


def print_summary(metrics: Dict[str, Any]):
    """Print human-readable summary."""
    print("\n" + "="*80)
    print("AUDIT SUMMARY")
    print("="*80)
    
    if "error" in metrics:
        print(f"\nERROR: {metrics['error']}")
        return
    
    print(f"\nTotal samples: {metrics['n_total']}")
    print(f"Valid re-labels: {metrics['n_valid']}")
    print(f"Errors: {metrics['n_errors']}")
    
    print(f"\n{'OVERALL AGREEMENT':-^80}")
    print(f"Agreement rate: {metrics['overall_agreement']:.1%} "
          f"({metrics['n_agreements']}/{metrics['n_valid']})")
    
    print(f"\n{'PER-CLASS AGREEMENT':-^80}")
    for label, stats in sorted(metrics['class_agreement'].items()):
        print(f"  {label:20s}: {stats['agreement_rate']:6.1%}  (n={stats['n_samples']})")
    
    print(f"\n{'CONFUSION MATRIX (original → claude)':-^80}")
    all_labels = sorted(set(
        list(metrics['confusion_matrix'].keys()) + 
        [k for v in metrics['confusion_matrix'].values() for k in v.keys()]
    ))
    
    # Print header
    print(f"{'Original':<20s} | " + " ".join(f"{lbl[:6]:>6s}" for lbl in all_labels))
    print("-" * 80)
    
    for orig in all_labels:
        row_data = metrics['confusion_matrix'].get(orig, {})
        counts = [str(row_data.get(claude, 0)) for claude in all_labels]
        print(f"{orig:<20s} | " + " ".join(f"{c:>6s}" for c in counts))
    
    print(f"\n{'TOP LABEL TRANSITIONS (disagreements only)':-^80}")
    for transition, count in list(metrics['top_transitions'].items())[:10]:
        pct = count / metrics['n_disagreements'] * 100 if metrics['n_disagreements'] > 0 else 0
        print(f"  {transition:30s}: {count:3d} ({pct:5.1f}% of disagreements)")
    
    print(f"\n{'CONFIDENCE DISTRIBUTION':-^80}")
    print(f"Agreements:")
    print(f"  Mean: {metrics['confidence_stats']['agreements']['mean']:.3f}")
    print(f"  Range: [{metrics['confidence_stats']['agreements']['min']:.3f}, "
          f"{metrics['confidence_stats']['agreements']['max']:.3f}]")
    print(f"  Count: {metrics['confidence_stats']['agreements']['n']}")
    
    print(f"\nDisagreements:")
    print(f"  Mean: {metrics['confidence_stats']['disagreements']['mean']:.3f}")
    print(f"  Range: [{metrics['confidence_stats']['disagreements']['min']:.3f}, "
          f"{metrics['confidence_stats']['disagreements']['max']:.3f}]")
    print(f"  Count: {metrics['confidence_stats']['disagreements']['n']}")
    
    print(f"\n{'MOST SUSPICIOUS (high-confidence disagreements)':-^80}")
    for i, item in enumerate(metrics['most_suspicious'][:5], 1):
        print(f"\n{i}. {item['original_label']} → {item['claude_label']} "
              f"(confidence: {item['confidence']:.3f})")
        print(f"   Hypothesis: {item['hypothesis']}")
        print(f"   Explanation: {item['explanation']}")
        print(f"   Doc: {item['doc_id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, 
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=100,
                       help="Total number of samples (default: 100, split equally across 4 classes)")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Number of concurrent API calls (default: 10)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be audited without making API calls")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling (default: 42)")
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent
    train_path = root / "data/training_v2/train.jsonl"
    
    if not train_path.exists():
        print(f"ERROR: Training data not found at {train_path}")
        return 1
    
    # Load pipeline_log rows
    print(f"Loading training data from {train_path}...")
    pipeline_rows = []
    total_rows = 0
    
    with open(train_path) as f:
        for line in f:
            total_rows += 1
            row = json.loads(line)
            if row.get("source") == "pipeline_log":
                pipeline_rows.append(row)
    
    print(f"  Total rows: {total_rows:,}")
    print(f"  Pipeline_log rows: {len(pipeline_rows):,} ({len(pipeline_rows)/total_rows*100:.1f}%)")
    
    # Class distribution
    class_dist = Counter(LABEL_NAMES[row["label"]] for row in pipeline_rows)
    print(f"\n  Class distribution in pipeline_log:")
    for label in sorted(class_dist.keys()):
        count = class_dist[label]
        print(f"    {label:20s}: {count:5,} ({count/len(pipeline_rows)*100:.1f}%)")
    
    # Stratified sample
    n_per_class = args.n // 4
    print(f"\nSampling {n_per_class} examples per class (total: {n_per_class * 4})...")
    samples = stratified_sample(pipeline_rows, n_per_class, seed=args.seed)
    
    sample_dist = Counter(LABEL_NAMES[s["label"]] for s in samples)
    print(f"  Sampled distribution:")
    for label in sorted(sample_dist.keys()):
        print(f"    {label:20s}: {sample_dist[label]:3d}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would audit these samples:")
        for i, s in enumerate(samples[:3], 1):
            print(f"\n  Sample {i}:")
            print(f"    Label: {LABEL_NAMES[s['label']]}")
            print(f"    Hypothesis: {s['hypothesis'][:100]}...")
            print(f"    Premise: {s['premise'][:150]}...")
        print(f"\n  ... and {len(samples) - 3} more")
        print(f"\nWould make ~{len(samples)} API calls to Claude Sonnet")
        print("Re-run without --dry-run to execute")
        return 0
    
    # Audit with Claude
    print(f"\nRe-labeling {len(samples)} samples with Claude Sonnet...")
    print(f"Batch size: {args.batch_size}")
    print("(This may take a few minutes...)")
    
    audited = asyncio.run(batch_audit(samples, batch_size=args.batch_size))
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(audited)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Full results
    results_path = root / f"eval/label-audit-{timestamp}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "n_samples": len(samples),
                "n_per_class": n_per_class,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "model": "claude-sonnet-4-20250514",
                "temperature": 0
            },
            "metrics": metrics,
            "samples": audited
        }, f, indent=2)
    print(f"\nSaved full results → {results_path}")
    
    # Audit samples for human review
    samples_path = root / "data/eval/label_audit_samples.json"
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    
    review_samples = [
        {
            "premise": s["premise"],
            "hypothesis": s["hypothesis"],
            "original_label": LABEL_NAMES[s["label"]],
            "original_label_numeric": s["label"],
            "claude_label": s["claude_result"]["label_name"],
            "claude_label_numeric": s["claude_result"]["label_numeric"],
            "claude_confidence": s["claude_result"]["confidence"],
            "claude_explanation": s["claude_result"]["explanation"],
            "agreement": s["label"] == s["claude_result"]["label_numeric"],
            "doc_id": s.get("doc_id", ""),
            "section_id": s.get("section_id", "")
        }
        for s in audited
    ]
    
    with open(samples_path, "w") as f:
        json.dump(review_samples, f, indent=2)
    print(f"Saved audit samples → {samples_path}")
    
    # Print summary
    print_summary(metrics)
    
    print("\n" + "="*80)
    print("Done!")
    print(f"Full results: {results_path}")
    print(f"Review samples: {samples_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
