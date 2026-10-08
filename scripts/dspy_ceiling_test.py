#!/usr/bin/env python3
"""DSPy ceiling test — how well can an optimized prompt do on our eval set?

This is a diagnostic, not a product. We want to know:
1. What's the accuracy ceiling with a good model + optimized prompt?
2. What few-shot examples does the optimizer select?
3. What instructions does it discover?

Usage:
    cd python && uv run python ../scripts/dspy_ceiling_test.py
    cd python && uv run python ../scripts/dspy_ceiling_test.py --model qwen/qwen3-235b-a22b-2507
    cd python && uv run python ../scripts/dspy_ceiling_test.py --optimize  # Run MIPROv2 optimization
"""

import json
import argparse
import os
import random
from pathlib import Path
from collections import Counter

import dspy

ROOT = Path(__file__).parent.parent

# --- DSPy Signature ---

class LegalEntailment(dspy.Signature):
    """Given a passage from a legal contract and a claim about it,
    reason step-by-step about the legal relationship, then classify it."""
    
    passage: str = dspy.InputField(desc="A passage from a legal contract")
    claim: str = dspy.InputField(desc="A claim about the passage")
    reasoning: str = dspy.OutputField(desc="Step-by-step legal reasoning (50-150 words)")
    label: str = dspy.OutputField(desc="One of: entailed, contradicted, neutral")


# --- DSPy Module ---

class LegalReasoner(dspy.Module):
    def __init__(self):
        self.reason = dspy.ChainOfThought(LegalEntailment)
    
    def forward(self, passage, claim):
        result = self.reason(passage=passage, claim=claim)
        # Normalize label
        label = result.label.strip().lower()
        if label not in ('entailed', 'contradicted', 'neutral'):
            # Try to extract from the text
            for l in ('entailed', 'contradicted', 'neutral'):
                if l in label:
                    label = l
                    break
            else:
                label = 'neutral'  # fallback
        result.label = label
        return result


# --- Metric ---

LABEL_MAP = {
    "entailed": "entailed",
    "partially_entailed": "entailed",
    "contradicted": "contradicted",
    "neutral": "neutral",
}

def binary_accuracy(example, pred, trace=None):
    """Binary metric: entailed vs not-entailed."""
    gold = LABEL_MAP.get(example.label, example.label)
    predicted = pred.label.strip().lower()
    
    gold_binary = "entailed" if gold == "entailed" else "not_entailed"
    pred_binary = "entailed" if predicted == "entailed" else "not_entailed"
    
    return gold_binary == pred_binary

def three_class_accuracy(example, pred, trace=None):
    """3-class exact match."""
    gold = LABEL_MAP.get(example.label, example.label)
    predicted = pred.label.strip().lower()
    return gold == predicted


def load_eval_data():
    """Load our 243-case eval set as DSPy examples."""
    path = ROOT / "data" / "eval" / "legal_nli_eval_expanded.json"
    raw = json.load(open(path))
    cases = raw["cases"] if isinstance(raw, dict) and "cases" in raw else raw
    
    examples = []
    for c in cases:
        ex = dspy.Example(
            passage=c.get("passage", c.get("premise", ""))[:2500],
            claim=c.get("claim", c.get("hypothesis", "")),
            label=LABEL_MAP.get(c["label"], c["label"]),
            category=c.get("category", "unknown"),
        ).with_inputs("passage", "claim")
        examples.append(ex)
    
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen/qwen3-235b-a22b-2507",
                        help="OpenRouter model ID")
    parser.add_argument("--optimize", action="store_true",
                        help="Run MIPROv2 optimization")
    parser.add_argument("--n-eval", type=int, default=None,
                        help="Limit eval to N examples (default: all 243)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY")
        return
    
    print("=" * 60)
    print("DSPy Ceiling Test — Legal Entailment")
    print("=" * 60)
    
    # Configure DSPy with OpenRouter
    lm = dspy.LM(
        model=f"openrouter/{args.model}",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        max_tokens=400,
        temperature=0.7,
    )
    dspy.configure(lm=lm)
    print(f"\nModel: {args.model}")
    
    # Load data
    examples = load_eval_data()
    print(f"Eval set: {len(examples)} cases")
    
    if args.n_eval:
        examples = examples[:args.n_eval]
        print(f"Limited to: {args.n_eval} cases")
    
    # Split for optimization: 70% train, 30% test
    random.shuffle(examples)
    split = int(len(examples) * 0.7)
    trainset = examples[:split]
    testset = examples[split:]
    print(f"Train: {len(trainset)}, Test: {len(testset)}")
    
    program = LegalReasoner()
    
    if args.optimize:
        print(f"\n--- Running MIPROv2 Optimization ---")
        print(f"This will make many LLM calls (~$1-5 depending on model)")
        
        optimizer = dspy.MIPROv2(
            metric=binary_accuracy,
            auto="medium",
            num_threads=1,  # avoid SQLite lock issues
        )
        
        optimized = optimizer.compile(
            program,
            trainset=trainset,
        )
        
        print(f"\n--- Optimization Complete ---")
        
        # Evaluate optimized program
        print(f"\nEvaluating optimized program on test set ({len(testset)} cases)...")
        evaluator = dspy.Evaluate(
            devset=testset,
            metric=binary_accuracy,
            num_threads=1,
            display_progress=True,
        )
        
        optimized_score = evaluator(optimized)
        print(f"\nOptimized binary accuracy: {optimized_score}")
        
        # Also evaluate unoptimized as baseline
        print(f"\nEvaluating baseline (unoptimized) on test set...")
        baseline_score = evaluator(program)
        print(f"Baseline binary accuracy: {baseline_score}")
        
        print(f"\n--- Comparison ---")
        print(f"Baseline:  {baseline_score}")
        print(f"Optimized: {optimized_score}")
        print(f"\nOur fine-tuned Qwen2.5-3B: 69.5%")
        print(f"Cross-encoder DeBERTa:     72.8%")
        
        # Save optimized program
        out_dir = ROOT / "models" / "dspy_optimized"
        out_dir.mkdir(parents=True, exist_ok=True)
        optimized.save(str(out_dir / "legal_reasoner"))
        print(f"\nSaved optimized program to {out_dir / 'legal_reasoner'}")
        
        # Inspect what it learned
        print(f"\n--- Inspected Optimized Prompt ---")
        # Show the optimized instruction
        for name, param in optimized.named_parameters():
            print(f"\n{name}:")
            if hasattr(param, 'demos') and param.demos:
                print(f"  Few-shot examples: {len(param.demos)}")
                for i, demo in enumerate(param.demos[:2]):
                    print(f"    Demo {i}: label={demo.get('label', '?')}, "
                          f"reasoning={str(demo.get('reasoning', ''))[:100]}...")
        
    else:
        # Just run baseline evaluation
        print(f"\n--- Baseline Evaluation (no optimization) ---")
        print(f"Evaluating on {len(testset)} test cases...")
        
        evaluator = dspy.Evaluate(
            devset=testset,
            metric=binary_accuracy,
            num_threads=1,
            display_progress=True,
        )
        
        score = evaluator(program)
        print(f"\nBaseline binary accuracy: {score}")
        print(f"\nComparison:")
        print(f"  {args.model} (zero-shot): {score}")
        print(f"  Our fine-tuned Qwen2.5-3B:  69.5%")
        print(f"  Cross-encoder DeBERTa:      72.8%")
        
        # Per-category breakdown
        print(f"\nPer-category results:")
        by_cat = {}
        for ex in testset:
            cat = ex.category
            pred = program(passage=ex.passage, claim=ex.claim)
            correct = binary_accuracy(ex, pred)
            by_cat.setdefault(cat, {"correct": 0, "total": 0})
            by_cat[cat]["total"] += 1
            if correct:
                by_cat[cat]["correct"] += 1
        
        for cat in sorted(by_cat.keys()):
            s = by_cat[cat]
            pct = s["correct"] / s["total"] if s["total"] else 0
            print(f"  {cat:<35s} {s['correct']}/{s['total']} ({pct:.0%})")


if __name__ == "__main__":
    main()
