#!/usr/bin/env python3
"""Generate reasoning chains for batch 12 legal entailment triples."""

import json
import sys

def generate_reasoning(premise, hypothesis, label):
    """Generate concise legal reasoning for a triple."""
    
    # This will be populated with actual reasoning for each case
    # For now, creating a template structure
    reasoning_map = {}
    
    return reasoning_map.get((premise[:50], hypothesis[:50], label), 
                            f"Default reasoning for {label}")

def main():
    # Read input
    with open('/Users/steven/code/legal-kit/data/sft_reasoning/batches/batch_12.json', 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} triples")
    
    # Process each triple
    results = []
    for idx, item in enumerate(data, 1):
        premise = item['premise']
        hypothesis = item['hypothesis']
        label = item['label']
        
        # Generate reasoning
        reasoning = generate_reasoning(premise, hypothesis, label)
        
        # Count words
        word_count = len(reasoning.split())
        
        result = {
            'input': f"Passage: {premise}\n\nClaim: {hypothesis}",
            'output': label,
            'label': label,
            'reasoning': reasoning,
            'reasoning_word_count': word_count
        }
        
        results.append(result)
        print(f"Processed {idx}/50: {label} ({word_count} words)")
    
    # Write JSONL
    output_path = '/Users/steven/code/legal-kit/data/sft_reasoning/results/batch_12.jsonl'
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nWrote {len(results)} results to {output_path}")

if __name__ == '__main__':
    main()
