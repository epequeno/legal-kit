#!/usr/bin/env python3
"""Generate an expanded NLI evaluation set targeting 200+ cases.

Takes the existing 33 hand-labeled cases across 8 categories and:
1. Generates 15 additional cases per existing category (different scenarios)
2. Generates 15 cases per 6 NEW out-of-distribution categories
3. Uses Claude API with few-shot prompting for quality
4. Marks all generated cases with source='llm_generated' and reviewed=false
5. Marks original 33 cases with reviewed=true
6. Outputs to data/eval/legal_nli_eval_expanded.json

Total: 33 original + (8 + 6) × 15 generated = 243 cases
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

import httpx


# Paths
ORIGINAL_EVAL_PATH = Path(__file__).parent.parent / "data/eval/legal_nli_edge_cases.json"
OUTPUT_PATH = Path(__file__).parent.parent / "data/eval/legal_nli_eval_expanded.json"

# New out-of-distribution categories not in training data
OOD_CATEGORIES = {
    "representations_warranties": {
        "description": "Representations & warranties accuracy, survival periods, qualifiers like 'to the best of knowledge', and materiality thresholds",
        "examples": [
            "Seller reps that financial statements are accurate",
            "Reps survive closing for 18 months",
            "To the best of Seller's knowledge qualifier",
        ],
    },
    "limitation_of_liability": {
        "description": "Liability caps, exclusions of consequential damages, carve-outs for fraud or willful misconduct",
        "examples": [
            "Cap of $5M except for fraud",
            "No consequential damages except for breaches of confidentiality",
            "Joint and several liability vs. several only",
        ],
    },
    "force_majeure": {
        "description": "Force majeure trigger events, notice requirements, mitigation obligations, impossibility vs. impracticability",
        "examples": [
            "Acts of God, war, pandemic as FM events",
            "Party must provide written notice within 10 days",
            "Obligation to use reasonable efforts to mitigate",
        ],
    },
    "assignment_change_of_control": {
        "description": "Assignment restrictions, consent requirements (not to be unreasonably withheld), permitted transfers, change of control triggers",
        "examples": [
            "No assignment without prior written consent",
            "Consent not to be unreasonably withheld or delayed",
            "Assignment to affiliates permitted without consent",
        ],
    },
    "governing_law_jurisdiction": {
        "description": "Choice of law clauses, forum selection, exclusive vs. non-exclusive jurisdiction, arbitration provisions, waiver of jury trial",
        "examples": [
            "Governed by New York law without regard to conflicts principles",
            "Exclusive jurisdiction in Delaware courts",
            "Disputes resolved by binding arbitration in San Francisco",
        ],
    },
    "non_compete_non_solicit": {
        "description": "Non-compete scope, duration, geographic limits, non-solicitation of employees or customers, reasonableness standards",
        "examples": [
            "2-year non-compete within 50 miles",
            "No solicitation of employees for 18 months",
            "Limited to direct competitors in same industry",
        ],
    },
}

# Labels for generation
LABELS = ["entailed", "partially_entailed", "contradicted", "neutral"]


@dataclass
class GeneratedCase:
    """A generated NLI test case."""
    passage: str
    claim: str
    label: str
    explanation: str


def load_original_cases() -> tuple[dict, list[dict]]:
    """Load the original evaluation set and return metadata + cases."""
    if not ORIGINAL_EVAL_PATH.exists():
        raise FileNotFoundError(f"Original eval set not found: {ORIGINAL_EVAL_PATH}")
    
    with open(ORIGINAL_EVAL_PATH) as f:
        data = json.load(f)
    
    return {
        "version": data["version"],
        "description": data["description"],
        "categories": data["categories"],
        "labels": data["labels"],
    }, data["cases"]


def select_few_shot_examples(
    all_cases: list[dict],
    target_category: str | None = None,
    n: int = 3,
) -> list[dict]:
    """Select n diverse few-shot examples.
    
    If target_category is in the original 8, include at least 1 from that category.
    Always include examples covering different labels.
    """
    # Group by category
    by_category = defaultdict(list)
    for case in all_cases:
        by_category[case["category"]].append(case)
    
    examples = []
    
    # If target category exists in original set, take 1 from it
    if target_category and target_category in by_category:
        examples.append(random.choice(by_category[target_category]))
    
    # Fill remaining slots with diverse examples
    remaining = [c for c in all_cases if c not in examples]
    
    # Ensure label diversity
    by_label = defaultdict(list)
    for case in remaining:
        by_label[case["label"]].append(case)
    
    # Try to get one of each label type
    for label in LABELS:
        if len(examples) >= n:
            break
        if by_label[label]:
            examples.append(random.choice(by_label[label]))
    
    # Fill any remaining slots randomly
    while len(examples) < n:
        candidate = random.choice(remaining)
        if candidate not in examples:
            examples.append(candidate)
    
    return examples[:n]


def build_generation_prompt(
    category: str,
    category_description: str,
    few_shot_examples: list[dict],
    target_count: int = 15,
) -> str:
    """Build the prompt for Claude to generate test cases."""
    
    # Build few-shot examples section
    examples_text = ""
    for i, ex in enumerate(few_shot_examples, 1):
        examples_text += f"""
Example {i}:
Category: {ex['category']}
Passage: {ex['passage']}
Claim: {ex['claim']}
Label: {ex['label']}
Explanation: {ex['explanation']}
"""
    
    prompt = f"""You are an expert in legal contract language and natural language inference. Your task is to generate {target_count} high-quality test cases for evaluating legal NLI models on the category: **{category}**.

**Category Description:**
{category_description}

**Labels:**
- entailed: The claim is fully supported by the passage
- partially_entailed: The claim is partially supported but overstates, understates, or omits key qualifiers
- contradicted: The claim directly contradicts the passage
- neutral: The claim is not addressed by the passage

**Requirements:**
1. Generate realistic legal contract text (50-200 words) for each passage
2. Create precise claims that test the model's understanding of {category}
3. Distribute labels roughly evenly across the 4 classes
4. Include challenging edge cases where general NLI models might fail
5. Provide clear explanations for why each label is correct
6. Vary the contract types (M&A, employment, licensing, SaaS, etc.)

**Few-Shot Examples (from other categories):**
{examples_text}

**Output Format:**
Return a JSON array of {target_count} test cases, each with this structure:
{{
  "passage": "Full contract passage text...",
  "claim": "The claim to verify...",
  "label": "entailed|partially_entailed|contradicted|neutral",
  "explanation": "Why this label is correct..."
}}

Generate {target_count} diverse, challenging test cases for the **{category}** category:"""
    
    return prompt


def _call_llm(client, prompt: str, api_key: str = None) -> str:
    """Call Claude via Anthropic or OpenRouter."""
    if client is not None:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            temperature=1.0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    else:
        # OpenRouter
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "anthropic/claude-sonnet-4",
                "max_tokens": 8000,
                "temperature": 1.0,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def generate_cases_for_category(
    client,
    category: str,
    category_description: str,
    few_shot_examples: list[dict],
    target_count: int = 15,
    api_key: str = None,
) -> list[GeneratedCase]:
    """Generate test cases for a single category using Claude."""
    
    print(f"Generating {target_count} cases for category: {category}")
    
    prompt = build_generation_prompt(
        category=category,
        category_description=category_description,
        few_shot_examples=few_shot_examples,
        target_count=target_count,
    )
    
    try:
        content = _call_llm(client, prompt, api_key)
        
        # Try to find JSON array in the response
        import re
        json_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
        if not json_match:
            print(f"Warning: No JSON array found in response for {category}")
            print(f"Response: {content[:500]}...")
            return []
        
        cases_data = json.loads(json_match.group())
        
        # Validate and convert to GeneratedCase objects
        cases = []
        for case_dict in cases_data:
            if all(k in case_dict for k in ["passage", "claim", "label", "explanation"]):
                if case_dict["label"] in LABELS:
                    cases.append(GeneratedCase(**case_dict))
                else:
                    print(f"Warning: Invalid label '{case_dict['label']}' in generated case")
            else:
                print(f"Warning: Missing required fields in generated case: {case_dict.keys()}")
        
        print(f"  ✓ Generated {len(cases)} valid cases")
        return cases
        
    except Exception as e:
        print(f"Error generating cases for {category}: {e}")
        return []


def expand_evaluation_set() -> None:
    """Main function to expand the evaluation set."""
    
    # Load API key — prefer Anthropic, fall back to OpenRouter
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    
    if anthropic_key and HAS_ANTHROPIC:
        client = anthropic.Anthropic(api_key=anthropic_key)
        api_key = None
        print("Using Anthropic API directly")
    elif openrouter_key:
        client = None
        api_key = openrouter_key
        print("Using OpenRouter API (Claude Sonnet)")
    else:
        raise ValueError("Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY environment variable")
    
    # Load original cases
    print("Loading original evaluation set...")
    metadata, original_cases = load_original_cases()
    print(f"  Loaded {len(original_cases)} original cases across {len(metadata['categories'])} categories")
    
    # Prepare expanded dataset structure
    expanded_cases = []
    
    # Add original cases with reviewed=True
    print("\nAdding original cases (marking as reviewed)...")
    for case in original_cases:
        expanded_case = {
            **case,
            "source": "hand_labeled",
            "reviewed": True,
            "distribution": "in_distribution",
        }
        expanded_cases.append(expanded_case)
    
    # Generate cases for original 8 categories (in-distribution)
    print("\n" + "="*80)
    print("Generating cases for ORIGINAL categories (in-distribution)...")
    print("="*80)
    
    case_counter = len(original_cases)
    
    for category in metadata['categories']:
        few_shot = select_few_shot_examples(original_cases, target_category=category, n=3)
        
        # Category description (inferred from examples)
        category_descriptions = {
            "notwithstanding": "Notwithstanding clauses that override or carve out exceptions to general provisions",
            "shall_vs_may": "Distinctions between mandatory obligations (shall) and discretionary rights (may)",
            "knowledge_qualifier": "Knowledge-qualified representations (to the Knowledge of, to best of knowledge)",
            "mae_carveout": "Material Adverse Effect definitions with carve-outs and exceptions",
            "quantitative_precision": "Precise interpretation of quantitative terms (up to, at least, exceeding, etc.)",
            "temporal_qualifier": "Time periods, deadlines, survival periods, and temporal conditions",
            "condition_precedent": "Conditions that must be satisfied before obligations arise",
            "defined_term": "Proper interpretation and application of contractually defined terms",
        }
        
        desc = category_descriptions.get(category, f"Cases testing {category}")
        
        generated = generate_cases_for_category(
            client=client,
            category=category,
            category_description=desc,
            few_shot_examples=few_shot,
            target_count=15,
            api_key=api_key,
        )
        
        # Add generated cases to dataset
        for gen_case in generated:
            case_counter += 1
            expanded_cases.append({
                "id": f"gen-{category[:3]}-{case_counter:03d}",
                "category": category,
                "passage": gen_case.passage,
                "claim": gen_case.claim,
                "label": gen_case.label,
                "explanation": gen_case.explanation,
                "source": "llm_generated",
                "reviewed": False,
                "distribution": "in_distribution",
            })
    
    # Generate cases for NEW OOD categories
    print("\n" + "="*80)
    print("Generating cases for NEW categories (out-of-distribution)...")
    print("="*80)
    
    for category, info in OOD_CATEGORIES.items():
        few_shot = select_few_shot_examples(original_cases, target_category=None, n=3)
        
        generated = generate_cases_for_category(
            client=client,
            category=category,
            category_description=info["description"],
            few_shot_examples=few_shot,
            target_count=15,
            api_key=api_key,
        )
        
        # Add generated cases to dataset
        for gen_case in generated:
            case_counter += 1
            expanded_cases.append({
                "id": f"ood-{category[:3]}-{case_counter:03d}",
                "category": category,
                "passage": gen_case.passage,
                "claim": gen_case.claim,
                "label": gen_case.label,
                "explanation": gen_case.explanation,
                "source": "llm_generated",
                "reviewed": False,
                "distribution": "out_of_distribution",
            })
    
    # Build final output structure
    all_categories = metadata['categories'] + list(OOD_CATEGORIES.keys())
    
    output_data = {
        "version": "2.0",
        "description": "Expanded legal NLI evaluation set with LLM-generated cases covering both in-distribution (training categories) and out-of-distribution (novel categories) scenarios.",
        "generation_date": datetime.now().isoformat(),
        "generation_model": "claude-sonnet-4-20250514",
        "categories": all_categories,
        "labels": LABELS,
        "cases": expanded_cases,
    }
    
    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"\nTotal cases: {len(expanded_cases)}")
    
    # Breakdown by source
    by_source = defaultdict(int)
    for case in expanded_cases:
        by_source[case["source"]] += 1
    
    print("\nBy source:")
    for source, count in sorted(by_source.items()):
        print(f"  {source}: {count}")
    
    # Breakdown by distribution
    by_dist = defaultdict(int)
    for case in expanded_cases:
        by_dist[case["distribution"]] += 1
    
    print("\nBy distribution:")
    for dist, count in sorted(by_dist.items()):
        print(f"  {dist}: {count}")
    
    # Breakdown by category
    by_category = defaultdict(int)
    for case in expanded_cases:
        by_category[case["category"]] += 1
    
    print("\nBy category:")
    for category in all_categories:
        count = by_category[category]
        dist_type = "ID" if category in metadata['categories'] else "OOD"
        print(f"  {category:30s} ({dist_type:3s}): {count:3d}")
    
    # Label distribution
    by_label = defaultdict(int)
    for case in expanded_cases:
        by_label[case["label"]] += 1
    
    print("\nLabel distribution:")
    for label in LABELS:
        count = by_label[label]
        pct = 100 * count / len(expanded_cases)
        print(f"  {label:20s}: {count:3d} ({pct:5.1f}%)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    expand_evaluation_set()
