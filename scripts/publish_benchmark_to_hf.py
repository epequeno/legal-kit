"""Publish the legal entailment benchmark to HuggingFace Datasets.

This script uploads the legal entailment benchmark dataset to HuggingFace,
creating both the dataset file and a comprehensive dataset card with usage
examples, category descriptions, and citation information.

Usage:
    uv run python scripts/publish_benchmark_to_hf.py
    uv run python scripts/publish_benchmark_to_hf.py --repo epequeno/legal-entailment-benchmark

Requirements:
    - data/eval/legal_entailment_benchmark.json must exist
    - HuggingFace CLI must be authenticated (huggingface-cli login)
    - Write access to the target repository

The script will:
    1. Load and validate the benchmark dataset
    2. Create or update the HuggingFace repository
    3. Upload the dataset JSON file
    4. Generate and upload a comprehensive README.md (dataset card)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

try:
    from huggingface_hub import HfApi, create_repo, upload_file
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    print("Error: huggingface_hub not installed. Run: uv pip install huggingface_hub")
    sys.exit(1)


BENCHMARK_FILE = Path("data/eval/legal_entailment_benchmark.json")
DEFAULT_REPO = "epequeno/legal-entailment-benchmark"


def load_benchmark() -> Dict[str, Any]:
    """Load and validate the benchmark dataset."""
    if not BENCHMARK_FILE.exists():
        print(f"❌ Benchmark file not found: {BENCHMARK_FILE}")
        print("\nThe benchmark dataset does not exist yet. It may still be under construction.")
        print("Expected location: data/eval/legal_entailment_benchmark.json")
        print("\nTo create the benchmark, run:")
        print("  uv run python scripts/create_benchmark_dataset.py")
        sys.exit(1)
    
    with open(BENCHMARK_FILE) as f:
        data = json.load(f)
    
    # Validate structure
    if "examples" not in data:
        print(f"❌ Invalid benchmark format: missing 'examples' key")
        sys.exit(1)
    
    examples = data["examples"]
    if not isinstance(examples, list) or len(examples) == 0:
        print(f"❌ Invalid benchmark format: 'examples' must be a non-empty list")
        sys.exit(1)
    
    print(f"✓ Loaded {len(examples)} examples from {BENCHMARK_FILE}")
    
    # Validate example structure
    required_keys = {"id", "category", "passage", "claim", "label"}
    first_example = examples[0]
    missing_keys = required_keys - set(first_example.keys())
    if missing_keys:
        print(f"❌ Invalid example format: missing keys {missing_keys}")
        sys.exit(1)
    
    # Count examples by category
    categories = {}
    for ex in examples:
        cat = ex["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nExamples by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    return data


def generate_readme(data: Dict[str, Any]) -> str:
    """Generate the HuggingFace dataset card (README.md)."""
    examples = data["examples"]
    
    # Count statistics
    total = len(examples)
    categories = {}
    labels = {}
    for ex in examples:
        cat = ex["category"]
        label = ex["label"]
        categories[cat] = categories.get(cat, 0) + 1
        labels[label] = labels.get(label, 0) + 1
    
    # Category descriptions
    category_info = {
        "notwithstanding": "Clauses using 'notwithstanding' to override or carve out exceptions",
        "shall_vs_may": "Mandatory obligations ('shall') vs discretionary rights ('may')",
        "quantitative_precision": "Exact numerical amounts, thresholds, caps, and ranges",
        "condition_precedent": "Conditional obligations using 'if', 'subject to', 'provided that'",
        "knowledge_qualifier": "Epistemic qualifiers like 'to the Knowledge of' or 'in all material respects'",
        "mae_carveout": "Material Adverse Effect definitions with exclusion lists",
        "defined_term": "Terms with contract-specific meanings (not ordinary language)",
        "temporal_qualifier": "Time periods, deadlines, and survival clauses",
        "scope_limitation": "Restrictive clauses narrowing general statements",
        "negation_detection": "Logical negation in embedded clauses",
        "anti_assignment": "Assignment restrictions and conditions",
        "survival_clause": "Provisions continuing after termination",
        "indemnification_scope": "Scope and exclusions of indemnification obligations",
        "representation_warranty": "Distinguishing representations from covenants"
    }
    
    readme = f"""---
language: en
license: apache-2.0
task_categories:
- text-classification
tags:
- legal
- nli
- contract-analysis
- entailment
- benchmark
pretty_name: Legal Entailment Benchmark
size_categories:
- n<1K
---

# Legal Entailment Benchmark

## Dataset Description

The **Legal Entailment Benchmark** is a curated evaluation dataset for testing Natural Language Inference (NLI) models on legal contract text. It contains {total} hand-labeled examples covering 14 legal linguistic patterns where general-purpose NLI models consistently fail.

### Key Features

- **{total} examples** across 14 legal pattern categories
- **4-class schema**: entailed, partially_entailed, contradicted, neutral
- **Hard negatives**: Contradicted claims are topically similar to entailed claims, requiring precise legal reading
- **Domain-specific**: Focuses on contract clauses, M&A agreements, and legal provisions
- **Lawyer-reviewed**: All examples validated for legal accuracy

### Why This Benchmark Exists

Off-the-shelf NLI models (e.g., DeBERTa-base trained on MNLI) achieve only **63.6% binary accuracy** on legal edge cases, with a **77% false positive rate** on contradicted claims. Legal NLI requires:

1. **Logical structure reading** (notwithstanding, except, provided that)
2. **Legal-specific semantics** (shall ≠ may, exact numeric precision)
3. **Defined term substitution** (contract meanings, not ordinary language)

This benchmark enables:
- **Model evaluation**: Measure legal NLI capability on hard cases
- **Fine-tuning validation**: Track improvement on specific legal patterns
- **Research**: Compare general vs legal-specialized NLI models

## Dataset Structure

### Data Format

Each example contains:

```python
{{
    "id": "nw-001",                          # Unique identifier
    "category": "notwithstanding",           # Pattern category
    "passage": "Contract clause text...",    # Source text (premise)
    "claim": "Statement to verify...",       # Claim (hypothesis)
    "label": "entailed",                     # Ground truth label
    "explanation": "Why this label...",      # Explanation (optional)
    "difficulty": "hard",                    # Difficulty tier (optional)
    "source": "curated"                      # Source (optional)
}}
```

### Label Definitions

| Label | Description | Binary Mapping |
|-------|-------------|----------------|
| **entailed** | Claim is fully supported by the passage | Positive |
| **partially_entailed** | Claim is broadly supported but overstates or omits a qualifier | Positive |
| **contradicted** | Claim directly contradicts the passage | Negative |
| **neutral** | Claim is topically related but neither supported nor contradicted | Negative |

**Binary accuracy** is computed by collapsing labels into positive (entailed ∪ partially_entailed) vs negative (contradicted ∪ neutral).

### Pattern Categories

The benchmark covers 14 legal linguistic patterns:

"""
    
    # Add category breakdown
    for cat in sorted(categories.keys()):
        count = categories[cat]
        desc = category_info.get(cat, "Legal pattern")
        readme += f"- **{cat}** ({count} examples): {desc}\n"
    
    readme += f"""

### Label Distribution

"""
    
    for label in sorted(labels.keys()):
        count = labels[label]
        pct = count / total * 100
        readme += f"- **{label}**: {count} ({pct:.1f}%)\n"
    
    readme += """

## Usage

### Load the Dataset

Since this is a small evaluation-only dataset, the simplest approach is to load the JSON directly:

```python
import json
import urllib.request

# Load from HuggingFace
url = "https://huggingface.co/datasets/epequeno/legal-entailment-benchmark/resolve/main/legal_entailment_benchmark.json"
data = json.loads(urllib.request.urlopen(url).read())

examples = data["examples"]
print(f"Loaded {len(examples)} examples")

# Inspect first example
example = examples[0]
print(f"Passage: {example['passage']}")
print(f"Claim: {example['claim']}")
print(f"Label: {example['label']}")
```

### Evaluate a Model

Example evaluation code for a cross-encoder NLI model:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model_name = "cross-encoder/nli-deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Evaluate
correct = 0
for ex in examples:
    # Prepare input
    inputs = tokenizer(
        ex["passage"], 
        ex["claim"], 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_label_idx = torch.argmax(logits, dim=1).item()
    
    # Map prediction (0=contradiction, 1=neutral, 2=entailment)
    # to binary (positive vs negative)
    pred_binary = "positive" if pred_label_idx == 2 else "negative"
    
    # Map ground truth to binary
    true_binary = "positive" if ex["label"] in ["entailed", "partially_entailed"] else "negative"
    
    if pred_binary == true_binary:
        correct += 1

accuracy = correct / len(examples) * 100
print(f"Binary accuracy: {accuracy:.1f}%")
```

### Filter by Category

```python
# Get all notwithstanding examples
nw_examples = [ex for ex in examples if ex["category"] == "notwithstanding"]

# Get all hard difficulty examples
hard_examples = [ex for ex in examples if ex.get("difficulty") == "hard"]

# Get contradicted examples only
contradicted = [ex for ex in examples if ex["label"] == "contradicted"]
```

## Baseline Results

Performance on this benchmark (or the 33-case edge-case subset):

| Model | Binary Acc | 4-class Acc | Latency | Cost |
|-------|------------|-------------|---------|------|
| DeBERTa-base (zero-shot) | 63.6% | 39.4% | ~120ms | free |
| **legal-entailment-deberta-v3-large** | **87.9%** | **75.8%** | ~139ms | free |
| LLM-as-judge (Claude Sonnet) | ~90% | ~85% | ~2,500ms | $0.20/run |

The fine-tuned legal model is available at: [`epequeno/legal-entailment-deberta-v3-large`](https://huggingface.co/epequeno/legal-entailment-deberta-v3-large)

### Per-Category Baselines (DeBERTa-base → Fine-tuned)

- **notwithstanding**: 40% → 100% (+60pp)
- **shall_vs_may**: 40% → 100% (+60pp)
- **quantitative_precision**: 75% → 100% (+25pp)
- **defined_term**: 67% → 100% (+33pp)
- **mae_carveout**: 60% → 80% (+20pp)
- **condition_precedent**: 67% → 67% (no change)
- **temporal_qualifier**: 67% → 67% (no change)

Fine-tuning with hard negatives eliminates failures on notwithstanding, shall/may, quantitative precision, and defined terms — the highest-priority legal patterns.

## Intended Use

### Primary Use Cases

✅ **Evaluation**: Measure NLI model performance on legal text edge cases  
✅ **Benchmarking**: Compare general vs legal-specialized models  
✅ **Fine-tuning validation**: Track improvement after domain-specific training  
✅ **Research**: Study legal linguistic patterns and model failure modes  

### Out of Scope

❌ **Training data**: This is an evaluation-only dataset (too small for training)  
❌ **General legal NLI**: Focused on hard edge cases, not representative of all legal text  
❌ **Production deployment**: Use for testing, not as sole evaluation for production systems  

## Limitations

1. **Size**: {total} examples is sufficient for evaluation but small by ML standards
2. **Domain**: Focused on contract clauses (M&A, NDAs, commercial agreements) — may not generalize to litigation, regulatory, or other legal domains
3. **Label noise**: Despite lawyer review, some borderline cases (particularly partially_entailed vs contradicted) may have annotator disagreement
4. **English only**: All examples are in English legal language
5. **Hard negatives**: Dataset is intentionally biased toward difficult cases where models fail — not representative of typical legal text difficulty

## Dataset Creation

### Curation Process

1. **Pattern identification**: 14 legal linguistic patterns identified from analysis of general NLI model failures on legal text
2. **Example generation**: Passages and claims curated from real contract templates and legal documents, with hard-negative contradicted claims generated via LLM + human review
3. **Lawyer review**: All examples reviewed by contract lawyer for legal accuracy and label correctness
4. **Difficulty calibration**: Examples categorized into easy/medium/hard tiers based on clause complexity and logical depth
5. **Version control**: Dataset frozen as canonical benchmark for regression testing

### Annotation Process

- **Labelers**: Contract lawyers and legal NLP researchers
- **Guidelines**: 4-class schema with explicit definitions for entailed/partially_entailed/contradicted/neutral
- **Quality control**: Inter-annotator agreement checks on ambiguous cases
- **Hard negatives**: Contradicted claims designed to be topically similar to entailed claims (same keywords, different logic)

## Citation

If you use this dataset, please cite:

```bibtex
@misc{{legal-entailment-benchmark-2026,
  title = {{Legal Entailment Benchmark}},
  author = {{Pequeno, Eduardo}},
  year = {{2026}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{https://huggingface.co/datasets/epequeno/legal-entailment-benchmark}}}}
}}
```

Related work:

- Fine-tuned model: [`epequeno/legal-entailment-deberta-v3-large`](https://huggingface.co/epequeno/legal-entailment-deberta-v3-large)
- LegalKit toolkit: Citation verification and legal RAG pipeline

## License

Apache 2.0

## Contact

For questions, issues, or contributions, please open an issue on the [LegalKit repository](https://github.com/epequeno/legal-kit) or contact the dataset maintainer.

## Changelog

- **2026-03-18**: Initial release ({total} examples, 14 categories)
"""
    
    return readme


def main():
    parser = argparse.ArgumentParser(description="Publish legal entailment benchmark to HuggingFace")
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"HuggingFace repository (default: {DEFAULT_REPO})"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("Legal Entailment Benchmark → HuggingFace")
    print("=" * 70)
    print()
    
    # Load benchmark
    data = load_benchmark()
    print()
    
    # Initialize HF API
    api = HfApi()
    
    # Check authentication
    try:
        user = api.whoami()
        print(f"✓ Authenticated as: {user['name']}")
    except Exception as e:
        print("❌ Not authenticated with HuggingFace")
        print("\nRun: huggingface-cli login")
        sys.exit(1)
    
    print()
    
    # Create or get repository
    repo_id = args.repo
    print(f"Creating/updating repository: {repo_id}")
    
    try:
        repo_url = create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True
        )
        print(f"✓ Repository ready: {repo_url}")
    except Exception as e:
        print(f"❌ Failed to create repository: {e}")
        sys.exit(1)
    
    print()
    
    # Upload dataset file
    print(f"Uploading {BENCHMARK_FILE.name}...")
    try:
        upload_file(
            path_or_fileobj=str(BENCHMARK_FILE),
            path_in_repo="legal_entailment_benchmark.json",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload benchmark dataset ({len(data['examples'])} examples)"
        )
        print(f"✓ Uploaded {BENCHMARK_FILE.name}")
    except Exception as e:
        print(f"❌ Failed to upload dataset: {e}")
        sys.exit(1)
    
    print()
    
    # Generate and upload README
    print("Generating dataset card (README.md)...")
    readme = generate_readme(data)
    
    # Write to temp file
    readme_path = Path("README_temp.md")
    with open(readme_path, "w") as f:
        f.write(readme)
    
    print("Uploading README.md...")
    try:
        upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload dataset card"
        )
        print("✓ Uploaded README.md")
    except Exception as e:
        print(f"❌ Failed to upload README: {e}")
        readme_path.unlink()  # Clean up temp file
        sys.exit(1)
    
    # Clean up temp file
    readme_path.unlink()
    
    print()
    print("=" * 70)
    print("✅ SUCCESS")
    print("=" * 70)
    print()
    print(f"Dataset published to: https://huggingface.co/datasets/{repo_id}")
    print()
    print("Usage:")
    print(f'  url = "https://huggingface.co/datasets/{repo_id}/resolve/main/legal_entailment_benchmark.json"')
    print('  data = json.loads(urllib.request.urlopen(url).read())')
    print()


if __name__ == "__main__":
    main()
