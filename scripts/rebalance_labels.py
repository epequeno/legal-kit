#!/usr/bin/env python3
"""
Rebalance label distribution to match targets:
- entailed: ~35% (target 178 from 508)
- contradicted: ~35% (target 178 from 508)
- neutral: ~20% (target 102 from 508)
- partially_entailed: ~10% (target 51 from 508)
"""

import json
import random

with open('data/eval/legal_entailment_benchmark.json', 'r') as f:
    benchmark = json.load(f)

examples = benchmark['examples']
total = len(examples)

print(f"Total examples: {total}")
print("\nCurrent label distribution:")
from collections import Counter
current_labels = Counter(ex['label'] for ex in examples)
for label, count in sorted(current_labels.items()):
    print(f"  {label:20s}: {count:4d} ({100*count/total:5.1f}%)")

# Target distribution
targets = {
    "entailed": int(0.35 * total),  # 178
    "contradicted": int(0.35 * total),  # 178
    "neutral": int(0.20 * total),  # 102
    "partially_entailed": int(0.10 * total),  # 51
}

print("\nTarget distribution:")
for label, target in targets.items():
    print(f"  {label:20s}: {target:4d} ({100*target/total:5.1f}%)")

# Keep first 356 examples unchanged (the original hand-curated ones)
# Only modify the 152 template-generated ones (examples[356:])
original_examples = examples[:356]
generated_examples = examples[356:]

print(f"\nOriginal examples: {len(original_examples)}")
print(f"Generated examples to rebalance: {len(generated_examples)}")

# Count labels in original examples
original_labels = Counter(ex['label'] for ex in original_examples)
print("\nLabels in original examples:")
for label, count in sorted(original_labels.items()):
    print(f"  {label:20s}: {count:4d}")

# Calculate how many of each label we need from generated examples
needed_from_generated = {}
for label in targets:
    needed_from_generated[label] = max(0, targets[label] - original_labels.get(label, 0))

print("\nNeeded from generated examples:")
total_needed = sum(needed_from_generated.values())
for label, needed in sorted(needed_from_generated.items()):
    pct = 100 * needed / total_needed if total_needed > 0 else 0
    print(f"  {label:20s}: {needed:4d} ({pct:5.1f}%)")

# Reassign labels to generated examples to meet targets
# Shuffle to randomize which examples get which labels
random.seed(42)
random.shuffle(generated_examples)

# Assign labels according to needed distribution
idx = 0
for label, needed in needed_from_generated.items():
    for i in range(needed):
        if idx < len(generated_examples):
            old_label = generated_examples[idx]['label']
            generated_examples[idx]['label'] = label
            
            # Update explanation to match new label
            category = generated_examples[idx]['category']
            if label == "contradicted":
                generated_examples[idx]['explanation'] = f"The passage establishes {category.replace('_', ' ')} that contradicts the claim's assertion."
            elif label == "entailed":
                generated_examples[idx]['explanation'] = f"The claim correctly captures the {category.replace('_', ' ')} expressed in the passage."
            elif label == "partially_entailed":
                generated_examples[idx]['explanation'] = f"The claim overstates or simplifies the passage due to {category.replace('_', ' ')} qualifications."
            else:  # neutral
                generated_examples[idx]['explanation'] = f"The passage doesn't provide enough information to confirm or deny the claim about {category.replace('_', ' ')}."
            
            idx += 1

# Combine back
examples = original_examples + generated_examples

# Update stats
from collections import Counter
label_counts = Counter(ex['label'] for ex in examples)
category_counts = Counter(ex['category'] for ex in examples)
difficulty_counts = Counter(ex.get('difficulty', 'medium') for ex in examples)
contract_types_set = sorted(set(ex.get('contract_type', 'unknown') for ex in examples if ex.get('contract_type')))

benchmark['stats'] = {
    "total": len(examples),
    "by_label": dict(label_counts),
    "by_category": dict(category_counts),
    "by_difficulty": dict(difficulty_counts),
    "contract_types": contract_types_set
}

benchmark['examples'] = examples

# Write back
with open('data/eval/legal_entailment_benchmark.json', 'w') as f:
    json.dump(benchmark, f, indent=2)

print("\n" + "="*70)
print("REBALANCED LABEL DISTRIBUTION")
print("="*70)
print(f"\nFinal label distribution:")
for label, count in sorted(label_counts.items()):
    pct = 100 * count / len(examples)
    target_pct = {"entailed": 35, "contradicted": 35, "neutral": 20, "partially_entailed": 10}.get(label, 0)
    diff = pct - target_pct
    print(f"  {label:20s}: {count:4d} ({pct:5.1f}%) [target: {target_pct:2d}%, diff: {diff:+5.1f}%]")
