"""Generate targeted hard negatives for weak entailment categories.

Analyzes v3 eval failures and generates training pairs that specifically
address the model's weaknesses:
  - temporal_qualifier: model rejects valid time-based claims
  - knowledge_qualifier: model rejects "to Seller's knowledge" claims
  - quantitative_precision: model rejects valid numerical boundary claims
  - mae_carveout: model rejects valid MAE exception claims
  - defined_term: model confuses scope of defined terms

Usage:
    python scripts/generate_targeted_negatives.py
    python scripts/generate_targeted_negatives.py --n-per-category 50
    python scripts/generate_targeted_negatives.py --dry-run
"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime

try:
    import httpx
except ImportError:
    print("ERROR: httpx required. Install with: pip install httpx")
    exit(1)

ROOT = Path(__file__).parent.parent

# 3-class label mapping (v3)
LABEL_MAP = {"entailed": 0, "contradicted": 1, "neutral": 2}

CATEGORIES = {
    "temporal_qualifier": {
        "description": "Claims involving time periods, dates, deadlines, survival periods, expiration.",
        "failure_pattern": "Model predicts 'contradicted' for valid claims that correctly paraphrase time-based provisions. It struggles with time arithmetic (e.g., '60 days' = 'two months'), relative time references, and claims that correctly state a time provision without quoting it exactly.",
        "examples": [
            {
                "passage": "This Agreement shall terminate on the third anniversary of the Effective Date, unless earlier terminated pursuant to Section 12.",
                "claim": "The agreement has a three-year term from the Effective Date.",
                "label": "entailed",
                "explanation": "Third anniversary = three years. The claim correctly restates the duration."
            },
            {
                "passage": "The Seller's indemnification obligations under Section 8.1 shall survive for a period of eighteen (18) months following the Closing Date.",
                "claim": "Indemnification claims must be brought within two years of closing.",
                "label": "contradicted",
                "explanation": "18 months ≠ 2 years (24 months). The claim overstates the survival period."
            },
            {
                "passage": "Buyer shall have forty-five (45) calendar days from receipt of the Disclosure Schedules to complete its due diligence review.",
                "claim": "The due diligence period is approximately six weeks from receipt of schedules.",
                "label": "entailed",
                "explanation": "45 days ≈ 6.4 weeks. 'Approximately six weeks' is a reasonable paraphrase."
            },
        ],
    },
    "knowledge_qualifier": {
        "description": "Claims about representations qualified by 'to Seller's knowledge', 'to the best of Company's knowledge', awareness qualifiers.",
        "failure_pattern": "Model predicts 'contradicted' for claims that are partially entailed — the passage says 'to Seller's knowledge, there are no X' and the claim says 'there are no X' (dropping the knowledge qualifier). This should be 'entailed' (the claim is a weaker/broader version of what's stated) but the model treats the missing qualifier as a contradiction.",
        "examples": [
            {
                "passage": "To the Knowledge of the Company, there are no pending or threatened claims, actions, or proceedings against the Company that would reasonably be expected to result in liability exceeding $100,000.",
                "claim": "The Company is not aware of any significant pending litigation.",
                "label": "entailed",
                "explanation": "The claim correctly captures the substance — the Company doesn't know of material litigation. The knowledge qualifier is preserved in 'not aware of'."
            },
            {
                "passage": "To the best of Seller's knowledge, Seller has complied with all applicable Environmental Laws in all material respects.",
                "claim": "Seller has definitively complied with all environmental laws without exception.",
                "label": "contradicted",
                "explanation": "The claim removes both the knowledge qualifier ('to the best of knowledge') and the materiality qualifier ('in all material respects'), making it absolute."
            },
            {
                "passage": "To the Knowledge of the Company, all Intellectual Property owned by the Company is valid, subsisting, and enforceable.",
                "claim": "Based on the Company's awareness, its intellectual property rights are valid and enforceable.",
                "label": "entailed",
                "explanation": "The claim preserves the knowledge qualification ('based on awareness') and correctly restates the substance."
            },
        ],
    },
    "quantitative_precision": {
        "description": "Claims involving specific dollar amounts, percentages, thresholds, caps, numerical boundaries.",
        "failure_pattern": "Model predicts 'contradicted' for claims that correctly apply numerical provisions to specific scenarios. It struggles with boundary conditions (exactly at threshold), arithmetic derived from stated formulas, and claims that correctly restate a cap/floor without quoting the exact number.",
        "examples": [
            {
                "passage": "The aggregate liability of Seller under this Article VIII shall not exceed an amount equal to fifteen percent (15%) of the Purchase Price. The Purchase Price is $10,000,000.",
                "claim": "The maximum indemnification Seller must pay is $1,500,000.",
                "label": "entailed",
                "explanation": "15% × $10,000,000 = $1,500,000. The claim correctly computes the cap."
            },
            {
                "passage": "Licensee shall pay a royalty of 3% of Net Sales up to $1,000,000 and 5% of Net Sales in excess of $1,000,000.",
                "claim": "On $2,000,000 in Net Sales, the total royalty would be $80,000.",
                "label": "entailed",
                "explanation": "3% × $1M = $30K + 5% × $1M = $50K = $80K total. The claim correctly applies the tiered formula."
            },
            {
                "passage": "The deductible for indemnification claims shall be $250,000 (the 'Basket'). Buyer may only recover amounts in excess of the Basket.",
                "claim": "If Buyer has $300,000 in indemnifiable losses, Buyer can recover $300,000.",
                "label": "contradicted",
                "explanation": "With a $250K deductible/basket, Buyer can only recover $50K ($300K - $250K), not the full $300K."
            },
        ],
    },
    "mae_carveout": {
        "description": "Material Adverse Effect definitions, carve-outs, exceptions to exceptions, disproportionate impact qualifiers.",
        "failure_pattern": "Model predicts 'contradicted' for claims that correctly state what IS or IS NOT excluded from the MAE definition. It struggles with the nested logic: 'X is excluded from MAE, except if X disproportionately affects the Company' — the model doesn't track the exception-to-the-exception.",
        "examples": [
            {
                "passage": "Material Adverse Effect shall not include any effect arising from (i) changes in general economic conditions, (ii) changes in the industries in which the Company operates, or (iii) changes in applicable Law; provided, however, that clauses (i) through (iii) shall not apply to the extent such effects have a disproportionate impact on the Company relative to other participants in the industries in which the Company operates.",
                "claim": "A change in general economic conditions can constitute a Material Adverse Effect if it disproportionately impacts the Company compared to industry peers.",
                "label": "entailed",
                "explanation": "The 'provided however' clause creates an exception to the carve-out: general economic changes ARE excluded UNLESS they disproportionately affect the Company."
            },
            {
                "passage": "Material Adverse Effect excludes effects from (a) pandemics, (b) natural disasters, and (c) acts of war or terrorism.",
                "claim": "A pandemic cannot be considered a Material Adverse Effect under any circumstances.",
                "label": "entailed",
                "explanation": "Unlike the previous example, there is no 'disproportionate impact' exception here. Pandemics are unconditionally excluded."
            },
            {
                "passage": "Material Adverse Effect excludes effects from (a) pandemics, (b) natural disasters, and (c) acts of war; provided that the foregoing shall not apply to the extent such effects disproportionately affect the Company.",
                "claim": "A pandemic cannot be considered a Material Adverse Effect under any circumstances.",
                "label": "contradicted",
                "explanation": "The 'provided that' clause means pandemics CAN be MAE if they disproportionately affect the Company. The claim ignores this exception-to-the-exception."
            },
        ],
    },
    "defined_term": {
        "description": "Contractual defined terms, their scope, what is included/excluded from a definition.",
        "failure_pattern": "Model confuses the boundaries of defined terms — accepting claims that expand a definition beyond its stated scope, or rejecting claims that correctly narrow to the definition's actual scope.",
        "examples": [
            {
                "passage": "'Permitted Transfer' means any transfer of Shares (i) to a Family Member of the Holder, (ii) to a trust for the benefit of the Holder or any Family Member, or (iii) upon the death of the Holder to the Holder's estate.",
                "claim": "Transferring shares to a business partner is a Permitted Transfer.",
                "label": "contradicted",
                "explanation": "Business partners are not listed in the definition. Only family members, trusts for family, and estate transfers qualify."
            },
            {
                "passage": "'Confidential Information' means all non-public information disclosed by one party to the other, including but not limited to trade secrets, customer lists, financial data, and business plans. Confidential Information does not include information that (a) is or becomes publicly available, or (b) was known to the Receiving Party prior to disclosure.",
                "claim": "Customer lists shared during negotiations are Confidential Information.",
                "label": "entailed",
                "explanation": "Customer lists are explicitly included in the definition."
            },
        ],
    },
}

GENERATION_PROMPT = """You are generating training data for a legal entailment model. The model determines whether a legal passage SUPPORTS or CONTRADICTS a claim about it.

CATEGORY: {category}
DESCRIPTION: {description}
KNOWN FAILURE PATTERN: {failure_pattern}

EXAMPLES OF CORRECT LABELING:
{examples}

Generate {n} NEW training pairs. Requirements:
1. Each pair has: passage (realistic contract text, 50-300 words), claim (1-2 sentences), label (entailed/contradicted/neutral), explanation
2. Approximately balanced: ~40% entailed, ~35% contradicted, ~25% neutral
3. Focus on the FAILURE PATTERN above — generate cases that would teach a model to handle these correctly
4. For "entailed" cases: include claims that paraphrase, use approximate language, derive conclusions from the passage — things the model currently wrongly rejects
5. For "contradicted" cases: include subtle contradictions that look similar to the passage but change a key detail
6. Use realistic legal contract language — real-world clause structures, not simplified examples
7. Vary the contract types: M&A, NDAs, employment, licensing, real estate, services agreements

Return a JSON array only (no markdown, no code fences):
[
  {{"passage": "...", "claim": "...", "label": "entailed", "explanation": "..."}},
  ...
]"""


def generate_pairs(category: str, info: dict, n: int, api_key: str) -> list[dict]:
    """Generate training pairs for one category."""
    examples_text = "\n\n".join(
        f"Passage: {ex['passage']}\nClaim: {ex['claim']}\nLabel: {ex['label']}\nExplanation: {ex['explanation']}"
        for ex in info["examples"]
    )

    prompt = GENERATION_PROMPT.format(
        category=category,
        description=info["description"],
        failure_pattern=info["failure_pattern"],
        examples=examples_text,
        n=n,
    )

    # Call in chunks if n > 30 (LLM quality drops for very large batches)
    all_pairs = []
    remaining = n
    while remaining > 0:
        batch_n = min(remaining, 25)
        batch_prompt = prompt.replace(f"Generate {n} NEW", f"Generate {batch_n} NEW")

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
                "messages": [{"role": "user", "content": batch_prompt}],
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

        # Extract JSON array
        import re
        json_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
        if not json_match:
            print(f"    Warning: no JSON found in response, retrying...")
            continue

        try:
            pairs = json.loads(json_match.group())
            valid = [p for p in pairs if all(k in p for k in ["passage", "claim", "label", "explanation"])
                     and p["label"] in LABEL_MAP]
            all_pairs.extend(valid)
            remaining -= len(valid)
            print(f"    Generated {len(valid)} valid pairs ({len(all_pairs)}/{n} total)")
        except json.JSONDecodeError as e:
            print(f"    Warning: JSON parse error: {e}, retrying...")
            continue

    return all_pairs[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-category", type=int, default=40,
                        help="Training pairs per category (default: 40)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES.keys()),
                        help="Which categories to generate for")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY")
        return

    print("=" * 70)
    print("Generating Targeted Hard Negatives for Weak Categories")
    print("=" * 70)
    print(f"Categories: {args.categories}")
    print(f"Pairs per category: {args.n_per_category}")
    print(f"Expected total: ~{len(args.categories) * args.n_per_category} pairs")

    if args.dry_run:
        print("\n[DRY RUN] Would generate pairs for:")
        for cat in args.categories:
            info = CATEGORIES[cat]
            print(f"  {cat}: {info['description'][:80]}")
            print(f"    Failure: {info['failure_pattern'][:80]}...")
        return

    all_generated = []

    for cat in args.categories:
        info = CATEGORIES[cat]
        print(f"\n{'—'*60}")
        print(f"Category: {cat} ({args.n_per_category} pairs)")
        print(f"  Failure pattern: {info['failure_pattern'][:100]}...")

        pairs = generate_pairs(cat, info, args.n_per_category, api_key)

        for p in pairs:
            all_generated.append({
                "premise": p["passage"],
                "hypothesis": p["claim"],
                "label": LABEL_MAP[p["label"]],
                "label_name": p["label"],
                "source": f"targeted_hard_negative:{cat}",
                "category": cat,
                "explanation": p.get("explanation", ""),
            })

    # Save
    out_path = ROOT / "data" / "training_targeted" / "targeted_negatives.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in all_generated:
            f.write(json.dumps(row) + "\n")

    # Stats
    from collections import Counter
    label_dist = Counter(r["label_name"] for r in all_generated)
    cat_dist = Counter(r["category"] for r in all_generated)

    print(f"\n{'='*70}")
    print(f"Generated {len(all_generated)} targeted training pairs")
    print(f"Saved to: {out_path}")
    print(f"\nLabel distribution:")
    for l, c in sorted(label_dist.items()):
        print(f"  {l}: {c} ({c/len(all_generated):.0%})")
    print(f"\nPer category:")
    for c, n in sorted(cat_dist.items()):
        print(f"  {c}: {n}")
    print(f"\nNext steps:")
    print(f"  1. Review samples: head -5 {out_path}")
    print(f"  2. Merge with v3 data: python scripts/prepare_entailment_v3.py --extra-data {out_path}")
    print(f"  3. Retrain: modal run scripts/modal_finetune_v3.py && modal deploy ... && python ... --trigger")


if __name__ == "__main__":
    main()
