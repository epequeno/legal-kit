# GRPO Temporal Qualifier Failure Analysis

**Date**: 2026-03-21
**Model**: Qwen2.5-3B + LoRA (SFT → GRPO)
**Category accuracy**: 28% 3-class (5/18), 50% binary (9/18)

## Summary

Temporal qualifier is the worst-performing category by a wide margin. Analysis of all 13 failures reveals **three distinct failure modes**, with the dominant one being trivially describable: the model cannot convert between time units.

## Failure Mode 1: Unit Conversion Blindness (7/13 failures)

The model treats numerically equivalent durations expressed in different units as incomparable or contradictory.

| ID | Passage says | Claim says | Gold | Pred | Issue |
|---|---|---|---|---|---|
| tq-002 | 36 months | 3 years | entailed | **contradicted** | 36 months = 3 years |
| gen-tem-109 | 60 days | 2 months | entailed | **neutral** | 60 days ≈ 2 months |
| gen-tem-115 | Jan 1 2024 – Dec 31 2026 (3 years) | 36 months | entailed | **neutral** | 3 years = 36 months |
| gen-tem-119 | 12 months | 1 year | entailed | **neutral** | 12 months = 1 year |
| gen-tem-121 | 25% at 1yr + monthly over 36mo | 4 years total | entailed | **neutral** | 1 + 3 = 4 years |
| gen-tem-122 | 3 installments over 18 months | within 2 years | entailed | **neutral** | 18 months < 2 years |
| gen-tem-111 | 45 days | 6 weeks (42 days) | contradicted | **neutral** | 45 ≠ 42, should catch this |

**Pattern**: When the passage uses one unit (months, days) and the claim uses another (years, weeks), the model defaults to "neutral" instead of converting. In tq-002, it explicitly states "three years is a different period than thirty-six months" — a factual error.

**Root cause**: Small LMs have weak numerical reasoning. The SFT training data likely contains few examples requiring unit conversion, so the model never learned these equivalences. GRPO couldn't fix this because the GRPO prompt pool probably has the same gap.

## Failure Mode 2: Reasoning-Label Disconnect (3/13 failures)

The model's reasoning correctly identifies the relationship but assigns the wrong label.

| ID | Gold | Pred | What happened |
|---|---|---|---|
| gen-tem-114 | contradicted | **neutral** | Reasoning says "30-day cure period is not supported by the passage" and "passage indicates immediate termination" — then labels neutral instead of contradicted |
| gen-tem-118 | entailed | **contradicted** | Reasoning correctly explains that the termination date is June 30, 2025 regardless of force majeure — then calls it contradicted because "summer 2025" ≠ "June 30, 2025" |
| gen-tem-113 | entailed | **neutral** | Reasoning notes the 2-year expiry with tax exception — then hedges to neutral |

**Pattern**: The reasoning chain arrives at the right conclusion but the final label doesn't match. This is a reward alignment issue — the GRPO consistency reward (0.1 weight) may not be strong enough to enforce reasoning-label agreement.

## Failure Mode 3: Scope/Semantic Confusion (3/13 failures)

The model misinterprets the semantic scope of a claim relative to the passage.

| ID | Gold | Pred | What happened |
|---|---|---|---|
| gen-tem-112 | neutral | **contradicted** | Passage: 12-month renewal terms. Claim: monthly billing. Model conflates renewal terms with billing cycles — these are independent concepts (neutral) |
| gen-tem-120 | contradicted | **entailed** | Passage: renewal window opens 12 months before March 31, 2025. Claim: opens in March 2024. Model says March 31, 2024 is "in March" but actually the window opens March 31, so the claim saying "March 2024" is technically ambiguous/wrong |
| tq-003 | contradicted | **neutral** | Passage: 60-day cure period required. Claim: immediate termination. Model recognizes the difference but won't commit to "contradicted" |

## Recommendations

### 1. Targeted GRPO Prompts for Unit Conversion
Generate 50-100 additional GRPO prompts specifically exercising time unit equivalences:
- months ↔ years (12, 24, 36, 48, 60 months)
- days ↔ weeks (7, 14, 21, 28, 30, 45, 60, 90 days)
- date arithmetic (specific dates → durations)
- "within X" vs "exactly X" (18 months < 2 years → entailed)

These should have unambiguous gold labels and high reward signal.

### 2. Increase Consistency Reward Weight
The reasoning-label disconnect suggests bumping the consistency reward from 0.1 → 0.2 and adding a specific penalty when the reasoning contains contradiction language ("not supported", "different from", "contradicts") but the label is "neutral" or "entailed".

### 3. Consider a Lightweight Tool
For the shipping product, a simple post-processing step that normalizes time units before the model sees them could eliminate failure mode 1 entirely. E.g., convert "36 months" → "36 months (3 years)" in both passage and claim.

## Raw Data

All 18 temporal qualifier cases with full passages, claims, reasoning, and labels are available in `eval/grpo_eval_results.json`.

### Correct predictions (5/18)
- tq-001: contradicted ✓
- gen-tem-110: entailed ✓
- gen-tem-116: contradicted ✓
- gen-tem-117: entailed ✓
- gen-tem-123: entailed ✓
