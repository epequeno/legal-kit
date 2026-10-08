#!/usr/bin/env python3
"""Generate synthetic temporal reasoning prompts deterministically.

No LLM needed — all passage/claim/label triples are computed from templates
and time unit conversions. Labels are mathematically correct.

Usage:
    python scripts/generate_temporal_prompts.py                    # preview stats
    python scripts/generate_temporal_prompts.py --output           # write to grpo_prompts
    python scripts/generate_temporal_prompts.py --output --merge   # merge into existing pool
"""

import json
import random
import argparse
from pathlib import Path
from itertools import product

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "data" / "grpo_prompts"

# ---------------------------------------------------------------------------
# Clause templates
# ---------------------------------------------------------------------------
# Each template is a dict with:
#   - name: category identifier
#   - passage_templates: list of format strings with {duration} placeholder
#   - claim_templates: dict of label -> list of format strings with {duration}
#     "same" templates get the same/equivalent duration (entailed)
#     "different" templates get a wrong duration (contradicted)
#     "unrelated" templates ignore the duration (neutral)

CLAUSE_TEMPLATES = [
    {
        "name": "survival_period",
        "passage": [
            "The representations and warranties contained herein shall survive the Closing for a period of {duration}.",
            "All representations and warranties made by the parties shall survive the Closing Date and continue in full force and effect for {duration} thereafter.",
            "The representations, warranties, and covenants of Seller shall survive for a period of {duration} following the Closing.",
        ],
        "claim_entailed": [
            "Representations and warranties survive closing for {duration}.",
            "The survival period for representations and warranties is {duration}.",
            "Warranty claims can be brought within {duration} of closing.",
        ],
        "claim_contradicted": [
            "Representations and warranties survive closing for {duration}.",
            "The survival period for representations and warranties is {duration}.",
            "Warranty claims expire after {duration} from closing.",
        ],
        "claim_neutral": [
            "The indemnification cap is limited to the purchase price.",
            "Seller must provide audited financial statements at closing.",
            "The representations cover environmental compliance matters.",
        ],
    },
    {
        "name": "cure_period",
        "passage": [
            "In the event of a material breach, the non-breaching party shall provide written notice and the breaching party shall have {duration} to cure such breach.",
            "Upon any material breach of this Agreement, the breaching party shall be afforded a cure period of {duration} following written notice thereof.",
            "A party in material breach shall have {duration} from receipt of written notice to cure the breach before the non-breaching party may terminate.",
        ],
        "claim_entailed": [
            "The cure period for material breach is {duration}.",
            "A breaching party has {duration} to remedy a material breach after notice.",
            "Material breaches must be cured within {duration} of notice.",
        ],
        "claim_contradicted": [
            "The cure period for material breach is {duration}.",
            "A breaching party has {duration} to remedy a material breach.",
            "Breaches must be cured within {duration} after notification.",
        ],
        "claim_neutral": [
            "Material breach includes failure to maintain insurance coverage.",
            "Written notice must be sent via certified mail.",
            "The non-breaching party may seek specific performance.",
        ],
    },
    {
        "name": "notice_period",
        "passage": [
            "Either party may terminate this Agreement for convenience by providing {duration} prior written notice to the other party.",
            "This Agreement may be terminated by either party upon {duration} advance written notice.",
            "Termination for convenience requires not less than {duration} written notice to the other party.",
        ],
        "claim_entailed": [
            "Termination for convenience requires {duration} advance notice.",
            "A party must give {duration} notice before terminating without cause.",
            "The notice period for convenience termination is {duration}.",
        ],
        "claim_contradicted": [
            "Termination for convenience requires {duration} advance notice.",
            "Only {duration} notice is needed to terminate without cause.",
            "The convenience termination notice period is {duration}.",
        ],
        "claim_neutral": [
            "Termination for cause requires a material breach.",
            "All notices must be delivered to the addresses specified in the Agreement.",
            "The agreement automatically renews unless terminated.",
        ],
    },
    {
        "name": "confidentiality_duration",
        "passage": [
            "The obligations of confidentiality shall remain in effect for a period of {duration} from the date of disclosure.",
            "Each party's confidentiality obligations under this Agreement shall survive for {duration} following disclosure of the Confidential Information.",
            "Confidentiality obligations shall continue for {duration} from the Effective Date, regardless of any termination of this Agreement.",
        ],
        "claim_entailed": [
            "Confidentiality obligations last for {duration}.",
            "The confidentiality period is {duration} from disclosure.",
            "Parties must maintain confidentiality for {duration}.",
        ],
        "claim_contradicted": [
            "Confidentiality obligations last only {duration}.",
            "The confidentiality period expires after {duration}.",
            "Parties are bound by confidentiality for {duration}.",
        ],
        "claim_neutral": [
            "Confidential information does not include publicly available data.",
            "The receiving party may disclose information pursuant to court order.",
            "Trade secrets are excluded from the time limitation.",
        ],
    },
    {
        "name": "non_compete",
        "passage": [
            "Employee agrees not to engage in any Competing Business for a period of {duration} following termination of employment.",
            "The non-competition restriction shall apply for {duration} after the termination or expiration of this Agreement.",
            "For a period of {duration} after separation, Employee shall not directly or indirectly compete with the Company.",
        ],
        "claim_entailed": [
            "The non-compete period is {duration} after termination.",
            "The employee cannot compete for {duration} after leaving.",
            "The non-competition restriction lasts {duration}.",
        ],
        "claim_contradicted": [
            "The non-compete restriction is only {duration}.",
            "The employee is restricted from competing for {duration}.",
            "Non-competition obligations last {duration} after termination.",
        ],
        "claim_neutral": [
            "The non-compete applies within a 50-mile radius of the office.",
            "The restriction covers solicitation of existing clients.",
            "Employee may work in an unrelated industry during the restriction period.",
        ],
    },
    {
        "name": "payment_terms",
        "passage": [
            "Payment of all invoiced amounts shall be due within {duration} of receipt of invoice.",
            "Client shall pay all undisputed invoices within {duration} after the date of invoice.",
            "All fees are payable net {duration} from the invoice date.",
        ],
        "claim_entailed": [
            "Invoices must be paid within {duration}.",
            "The payment term is {duration} from invoice receipt.",
            "Payment is due within {duration} of invoicing.",
        ],
        "claim_contradicted": [
            "Invoices must be paid within {duration}.",
            "The payment deadline is {duration} from receipt of invoice.",
            "Payments are due net {duration}.",
        ],
        "claim_neutral": [
            "Late payments accrue interest at 1.5% per month.",
            "Disputed amounts may be withheld pending resolution.",
            "All payments shall be made in US dollars by wire transfer.",
        ],
    },
    {
        "name": "warranty_period",
        "passage": [
            "Seller warrants that the Products shall be free from defects for a period of {duration} from the date of delivery.",
            "The warranty period for all goods delivered hereunder shall be {duration} from acceptance.",
            "Products are warranted against defects in materials and workmanship for {duration} following delivery.",
        ],
        "claim_entailed": [
            "The product warranty lasts {duration}.",
            "Warranty coverage extends for {duration} from delivery.",
            "Defects must be reported within the {duration} warranty period.",
        ],
        "claim_contradicted": [
            "The warranty period is {duration} from delivery.",
            "Products are warranted for {duration}.",
            "Warranty claims must be made within {duration} of delivery.",
        ],
        "claim_neutral": [
            "The warranty does not cover damage caused by misuse.",
            "Seller's sole remedy is repair or replacement of defective products.",
            "The warranty is void if the product is modified by the buyer.",
        ],
    },
    {
        "name": "option_exercise",
        "passage": [
            "The option to purchase must be exercised within {duration} of the grant date.",
            "Holder may exercise this option at any time during the {duration} period following the grant date.",
            "The option shall be exercisable for a period of {duration} commencing on the date of grant.",
        ],
        "claim_entailed": [
            "The option exercise window is {duration}.",
            "Options must be exercised within {duration} of grant.",
            "The option period lasts {duration} from the grant date.",
        ],
        "claim_contradicted": [
            "The option exercise period is {duration}.",
            "Options expire after {duration} from the grant date.",
            "Holders have {duration} to exercise the option.",
        ],
        "claim_neutral": [
            "The exercise price is determined by the board of directors.",
            "Options are not transferable without prior written consent.",
            "Unvested options are forfeited upon termination of employment.",
        ],
    },
    {
        "name": "lease_term",
        "passage": [
            "The initial term of this Lease shall be {duration}, commencing on the Commencement Date.",
            "This Lease is granted for a term of {duration} beginning on the Effective Date.",
            "Tenant shall have the right to occupy the Premises for an initial period of {duration}.",
        ],
        "claim_entailed": [
            "The lease term is {duration}.",
            "The initial lease period is {duration} from commencement.",
            "The tenant occupies the premises for {duration}.",
        ],
        "claim_contradicted": [
            "The lease term is {duration}.",
            "The initial lease runs for {duration}.",
            "The tenant's occupancy period is {duration}.",
        ],
        "claim_neutral": [
            "Rent increases by 3% annually during the lease term.",
            "The tenant is responsible for all utilities and maintenance.",
            "The lease includes an option to renew for additional terms.",
        ],
    },
    {
        "name": "indemnification_expiry",
        "passage": [
            "All indemnification claims must be asserted within {duration} of the Closing Date.",
            "The right to seek indemnification under this Agreement shall expire {duration} after Closing.",
            "No claim for indemnification may be brought after the expiration of {duration} following the Closing Date.",
        ],
        "claim_entailed": [
            "Indemnification claims expire {duration} after closing.",
            "The indemnification period is {duration} from the Closing Date.",
            "Claims must be brought within {duration} of closing.",
        ],
        "claim_contradicted": [
            "Indemnification claims must be filed within {duration} of closing.",
            "The indemnification deadline is {duration} after the Closing Date.",
            "The indemnity period expires after {duration}.",
        ],
        "claim_neutral": [
            "Indemnification is subject to a deductible of $500,000.",
            "The indemnifying party has the right to control the defense of any claim.",
            "Indemnification obligations are several, not joint.",
        ],
    },
    {
        "name": "escrow_release",
        "passage": [
            "The escrow funds shall be released to Seller {duration} after the Closing Date, subject to any pending claims.",
            "Escrow amounts will be disbursed upon the expiration of {duration} following Closing.",
            "The Escrow Agent shall release the escrow funds {duration} after Closing, less any amounts reserved for unresolved claims.",
        ],
        "claim_entailed": [
            "Escrow funds are released {duration} after closing.",
            "The escrow release occurs {duration} from the Closing Date.",
            "Seller receives the escrow {duration} after closing.",
        ],
        "claim_contradicted": [
            "The escrow is released {duration} after closing.",
            "Escrow funds are disbursed {duration} from Closing.",
            "The escrow release period is {duration}.",
        ],
        "claim_neutral": [
            "The escrow agent is a nationally recognized financial institution.",
            "Escrow funds earn interest for the benefit of the Seller.",
            "Disputes over escrow release are resolved by arbitration.",
        ],
    },
    {
        "name": "renewal_notice",
        "passage": [
            "This Agreement shall automatically renew unless either party provides written notice of non-renewal at least {duration} prior to the end of the then-current term.",
            "To prevent automatic renewal, a party must deliver notice of termination no later than {duration} before expiration of the current term.",
            "Non-renewal notice must be given at least {duration} in advance of the expiration of any renewal term.",
        ],
        "claim_entailed": [
            "Non-renewal requires {duration} advance notice.",
            "At least {duration} notice is required to prevent automatic renewal.",
            "The non-renewal notice period is {duration}.",
        ],
        "claim_contradicted": [
            "Non-renewal notice must be given {duration} in advance.",
            "Only {duration} notice is needed to prevent automatic renewal.",
            "The required non-renewal notice period is {duration}.",
        ],
        "claim_neutral": [
            "Renewal terms are on the same conditions as the original agreement.",
            "The agreement may be renewed for up to three additional terms.",
            "Renewal is contingent upon satisfactory performance.",
        ],
    },
]

# ---------------------------------------------------------------------------
# Duration system — all durations stored as days for comparison
# ---------------------------------------------------------------------------

class Duration:
    """A duration with a canonical value in days and a human-readable string."""

    def __init__(self, days: int, text: str):
        self.days = days
        self.text = text

    def __eq__(self, other):
        return self.days == other.days

    def __repr__(self):
        return f"Duration({self.days}d, '{self.text}')"


def make_duration_pairs():
    """Generate (equivalent, non-equivalent) duration pairs using different units.

    Returns list of dicts with:
      - passage_dur: Duration used in passage
      - claim_equiv: Duration that's equivalent in different units (entailed)
      - claim_diff: Duration that's different (contradicted)
    """
    pairs = []

    # --- Months <-> Years ---
    month_year_map = [
        (12, 1), (24, 2), (36, 3), (48, 4), (60, 5),
        (18, "one and a half years"), (6, "half a year"),
    ]
    for months, years in month_year_map:
        if isinstance(years, str):
            year_str = years
        else:
            year_str = f"{int(years)} year{'s' if years != 1 else ''}"
        month_dur = Duration(months * 30, f"{months} months")
        year_dur = Duration(months * 30, year_str)

        # Wrong values for contradicted
        wrong_months = months + random.choice([6, 12, -6]) if months > 6 else months + 12
        wrong_dur = Duration(wrong_months * 30, f"{wrong_months} months")

        pairs.append({
            "passage_dur": month_dur,
            "claim_equiv": year_dur,
            "claim_diff": wrong_dur,
            "conversion": "months_to_years",
        })
        # Also test years in passage, months in claim
        pairs.append({
            "passage_dur": year_dur,
            "claim_equiv": month_dur,
            "claim_diff": wrong_dur,
            "conversion": "years_to_months",
        })

    # --- Days <-> Weeks ---
    day_week_map = [
        (7, 1), (14, 2), (21, 3), (28, 4), (42, 6), (56, 8),
    ]
    for days, weeks in day_week_map:
        week_str = f"{weeks} week{'s' if weeks != 1 else ''}"
        day_dur = Duration(days, f"{days} days")
        week_dur = Duration(days, week_str)

        wrong_days = days + random.choice([7, 14, -7]) if days > 7 else days + 7
        wrong_dur = Duration(wrong_days, f"{wrong_days} days")

        pairs.append({
            "passage_dur": day_dur,
            "claim_equiv": week_dur,
            "claim_diff": wrong_dur,
            "conversion": "days_to_weeks",
        })
        pairs.append({
            "passage_dur": week_dur,
            "claim_equiv": day_dur,
            "claim_diff": wrong_dur,
            "conversion": "weeks_to_days",
        })

    # --- Days <-> Months (approximate: 30/60/90/180/365) ---
    day_month_map = [
        (30, "thirty (30) days", 1, "one month"),
        (60, "sixty (60) days", 2, "two months"),
        (90, "ninety (90) days", 3, "three months"),
        (180, "one hundred eighty (180) days", 6, "six months"),
        (365, "three hundred sixty-five (365) days", 12, "twelve months"),
    ]
    for days, day_text, months, month_text in day_month_map:
        day_dur = Duration(days, day_text)
        month_dur = Duration(days, month_text)  # treat as equivalent

        wrong_months = months + random.choice([1, 2, 3])
        wrong_dur = Duration(wrong_months * 30, f"{wrong_months} months")

        pairs.append({
            "passage_dur": day_dur,
            "claim_equiv": month_dur,
            "claim_diff": wrong_dur,
            "conversion": "days_to_months",
        })
        pairs.append({
            "passage_dur": month_dur,
            "claim_equiv": day_dur,
            "claim_diff": wrong_dur,
            "conversion": "months_to_days",
        })

    # --- Written-out numbers <-> digits ---
    written_map = [
        ("twelve (12) months", "1 year"),
        ("twenty-four (24) months", "2 years"),
        ("thirty-six (36) months", "3 years"),
        ("six (6) months", "180 days"),
        ("ninety (90) days", "3 months"),
        ("sixty (60) days", "2 months"),
        ("one (1) year", "12 months"),
        ("two (2) years", "24 months"),
        ("three (3) years", "36 months"),
        ("five (5) years", "60 months"),
    ]
    days_lookup = {
        "twelve (12) months": 360, "1 year": 360,
        "twenty-four (24) months": 720, "2 years": 720,
        "thirty-six (36) months": 1080, "3 years": 1080,
        "six (6) months": 180, "180 days": 180,
        "ninety (90) days": 90, "3 months": 90,
        "sixty (60) days": 60, "2 months": 60,
        "one (1) year": 360, "12 months": 360,
        "two (2) years": 720, "24 months": 720,
        "three (3) years": 1080, "36 months": 1080,
        "five (5) years": 1800, "60 months": 1800,
    }
    for written, equiv in written_map:
        d = days_lookup[written]
        passage_dur = Duration(d, written)
        claim_dur = Duration(d, equiv)

        # Pick a wrong duration
        wrong_options = [v for k, v in days_lookup.items() if v != d]
        wrong_d = random.choice(wrong_options) if wrong_options else d + 180
        wrong_text = next((k for k, v in days_lookup.items() if v == wrong_d and k != written), f"{wrong_d // 30} months")
        wrong_dur = Duration(wrong_d, wrong_text)

        pairs.append({
            "passage_dur": passage_dur,
            "claim_equiv": claim_dur,
            "claim_diff": wrong_dur,
            "conversion": "written_to_digit",
        })

    # --- "Within X" vs "at least X" comparisons ---
    # 18 months is within 2 years (entailed for "within" claims)
    within_pairs = [
        (Duration(540, "eighteen (18) months"), Duration(720, "two years"), "entailed"),
        (Duration(180, "six (6) months"), Duration(360, "one year"), "entailed"),
        (Duration(90, "ninety (90) days"), Duration(180, "six months"), "entailed"),
        (Duration(720, "twenty-four (24) months"), Duration(1080, "three years"), "entailed"),
    ]
    for shorter, longer, _ in within_pairs:
        pairs.append({
            "passage_dur": shorter,
            "claim_equiv": Duration(longer.days, f"within {longer.text}"),
            "claim_diff": Duration(shorter.days // 2, f"within {shorter.days // 60} months"),
            "conversion": "within_comparison",
        })

    return pairs


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

def generate_prompts(seed: int = 42) -> list[dict]:
    """Generate all temporal prompts deterministically."""
    rng = random.Random(seed)
    duration_pairs = make_duration_pairs()
    rng.shuffle(duration_pairs)

    prompts = []

    for pair in duration_pairs:
        passage_dur = pair["passage_dur"]
        claim_equiv = pair["claim_equiv"]
        claim_diff = pair["claim_diff"]

        # Pick a random clause template
        template = rng.choice(CLAUSE_TEMPLATES)

        passage_text = rng.choice(template["passage"]).format(duration=passage_dur.text)

        # --- Entailed: same duration, different units ---
        claim_text = rng.choice(template["claim_entailed"]).format(duration=claim_equiv.text)
        prompts.append({
            "passage": passage_text,
            "claim": claim_text,
            "label": "entailed",
            "source": "synthetic_temporal",
            "category": f"temporal_{template['name']}",
            "conversion_type": pair["conversion"],
        })

        # --- Contradicted: different duration ---
        claim_text = rng.choice(template["claim_contradicted"]).format(duration=claim_diff.text)
        prompts.append({
            "passage": passage_text,
            "claim": claim_text,
            "label": "contradicted",
            "source": "synthetic_temporal",
            "category": f"temporal_{template['name']}",
            "conversion_type": pair["conversion"],
        })

        # --- Neutral: unrelated claim ---
        claim_text = rng.choice(template["claim_neutral"])
        prompts.append({
            "passage": passage_text,
            "claim": claim_text,
            "label": "neutral",
            "source": "synthetic_temporal",
            "category": f"temporal_{template['name']}",
            "conversion_type": pair["conversion"],
        })

    return prompts


def print_stats(prompts: list[dict]):
    """Print summary statistics."""
    from collections import Counter

    print(f"Total prompts: {len(prompts)}")
    print()

    label_counts = Counter(p["label"] for p in prompts)
    print("Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({count/len(prompts)*100:.1f}%)")
    print()

    conv_counts = Counter(p["conversion_type"] for p in prompts)
    print("Conversion types:")
    for conv, count in sorted(conv_counts.items(), key=lambda x: -x[1]):
        print(f"  {conv}: {count}")
    print()

    cat_counts = Counter(p["category"] for p in prompts)
    print("Clause templates used:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    print()

    # Show a few examples
    rng = random.Random(99)
    samples = rng.sample(prompts, min(6, len(prompts)))
    print("=== Sample prompts ===")
    for s in samples:
        print(f"\n[{s['label'].upper()}] ({s['conversion_type']})")
        print(f"  Passage: {s['passage']}")
        print(f"  Claim:   {s['claim']}")


def merge_into_pool(new_prompts: list[dict]):
    """Merge new temporal prompts into the existing GRPO prompt pool."""
    pool_path = OUTPUT_DIR / "grpo_prompts.jsonl"

    # Load existing
    existing = []
    if pool_path.exists():
        with open(pool_path) as f:
            existing = [json.loads(line) for line in f if line.strip()]

    # Remove old synthetic temporal prompts
    filtered = [p for p in existing if p.get("source") != "synthetic_temporal"]
    removed = len(existing) - len(filtered)

    # Add new
    merged = filtered + [{k: v for k, v in p.items() if k != "conversion_type"} for p in new_prompts]

    # Shuffle deterministically
    rng = random.Random(42)
    rng.shuffle(merged)

    # Write
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(pool_path, "w") as f:
        for p in merged:
            f.write(json.dumps(p) + "\n")

    print(f"Existing prompts: {len(existing)} (removed {removed} old synthetic_temporal)")
    print(f"New temporal prompts: {len(new_prompts)}")
    print(f"Total pool: {len(merged)}")


def write_standalone(prompts: list[dict]):
    """Write temporal prompts to a standalone file."""
    out_path = OUTPUT_DIR / "temporal_prompts.jsonl"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(prompts)} prompts to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic temporal GRPO prompts")
    parser.add_argument("--output", action="store_true", help="Write prompts to file")
    parser.add_argument("--merge", action="store_true", help="Merge into existing GRPO pool")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    prompts = generate_prompts(seed=args.seed)
    print_stats(prompts)

    if args.output:
        write_standalone(prompts)
        if args.merge:
            merge_into_pool(prompts)


if __name__ == "__main__":
    main()
