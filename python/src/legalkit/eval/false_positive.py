"""False positive analysis for citation verification.

A false positive occurs when the pipeline marks a claim as VERIFIED (or
PARTIALLY_VERIFIED) when the claim is actually wrong or not supported by
the cited passage.

This module provides:

1. A **failure taxonomy** — eight categories covering the patterns seen in
   legal NLI evaluation, grounded in the edge cases from
   ``data/eval/legal_nli_edge_cases.json``.

2. **Heuristic auto-categorization** — regex-based rules that assign a
   category to any (passage, claim) failure pair without requiring an LLM
   call. Accuracy is ~70-80%; precise enough to identify dominant failure
   modes at scale.

3. **FalsePositiveReport** — category distribution, representative
   examples, and actionable recommendations for each failure mode.

4. **Integration with the NLI edge cases** — the 12 labelled DeBERTa
   failures from ``legalkit nli-eval`` can be directly analyzed.

Typical usage::

    from legalkit.eval.false_positive import analyze_nli_failures
    from legalkit.eval.nli_eval import load_edge_cases, run_deberta_nli

    cases = load_edge_cases()
    result = run_deberta_nli(cases)
    report = analyze_nli_failures(result)
    print(report.summary_table())
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------


class FailureCategory(Enum):
    """Categories of verification false positives.

    Each category represents a distinct pattern in which a model (or human)
    incorrectly marks a claim as supported by a legal passage.
    """

    SURFACE_FORM_AGREEMENT = "surface_form_agreement"
    """Passage and claim share vocabulary/topic but the claim contradicts
    the passage. The model pattern-matches on shared words rather than
    reading the logical structure.
    Example: passage carves out 'economic conditions' from MAE; claim says
    economic downturn IS an MAE — both mention 'economic conditions'."""

    MODAL_CONFUSION = "modal_confusion"
    """Confusion between mandatory (shall) and discretionary (may) language,
    or between a right/permission and an obligation.
    Example: passage says board 'may' accelerate vesting; claim says vesting
    'automatically accelerates'."""

    QUALIFIER_OMISSION = "qualifier_omission"
    """Claim drops a critical qualifying phrase that limits the scope of the
    statement — knowledge qualifiers, 'material', 'in all material respects',
    'to the extent', etc.
    Example: passage says 'to Seller's Knowledge, no proceedings are pending';
    claim says 'no proceedings are pending'."""

    CARVEOUT_MISSED = "carveout_missed"
    """Claim ignores an exception, exclusion, or notwithstanding clause that
    limits or overrides the general rule.
    Example: passage says 'X, provided that Y shall not apply'; claim states
    the general rule X without the Y exclusion."""

    QUANTITATIVE_IMPRECISION = "quantitative_imprecision"
    """Claim misstates a numeric amount, threshold, or boundary — e.g.
    treating 'up to $X' as exactly '$X', confusing a deductible with a
    payment obligation, or converting 18 months to 2 years.
    Example: passage: 'up to $5M termination fee'; claim: 'fee is $5M'."""

    CONDITION_STRIPPED = "condition_stripped"
    """Claim omits a condition precedent, triggering event, or 'subject to'
    qualifier — making a conditional obligation appear unconditional.
    Example: passage: closing is 'subject to regulatory approval'; claim:
    'Buyer must close the transaction'."""

    SCOPE_OVERBROAD = "scope_overbroad"
    """Claim generalizes beyond the actual scope of the passage — saying
    'all X' when the passage only covers 'X in certain circumstances', or
    omitting a party-specific limitation.
    Example: passage: 'Buyer has no indemnification obligation for losses
    from Seller's gross negligence'; claim: 'Buyer has no indemnification
    obligation for any losses'."""

    NEGATION_MISSED = "negation_missed"
    """Claim misses a negation in the passage — 'shall not', 'no obligation',
    'excluded', 'does not constitute', etc.
    Example: passage: 'acts of war shall NOT constitute a Material Adverse
    Effect'; claim: 'acts of war constitute a Material Adverse Effect'."""

    UNKNOWN = "unknown"
    """Could not be categorized by heuristics. Requires manual review."""


# Human-readable descriptions for reporting
_CATEGORY_DESCRIPTIONS: dict[FailureCategory, str] = {
    FailureCategory.SURFACE_FORM_AGREEMENT:
        "Topic match without logical support — shared vocabulary hides contradiction",
    FailureCategory.MODAL_CONFUSION:
        "Shall vs may — mandatory obligation confused with discretionary right",
    FailureCategory.QUALIFIER_OMISSION:
        "Critical qualifier dropped — 'to knowledge of', 'material', 'in all material respects'",
    FailureCategory.CARVEOUT_MISSED:
        "Exception/exclusion ignored — 'notwithstanding', 'provided that', 'except'",
    FailureCategory.QUANTITATIVE_IMPRECISION:
        "Numeric misstatement — 'up to' vs exact, deductible inversion, time conversion",
    FailureCategory.CONDITION_STRIPPED:
        "Condition precedent omitted — 'subject to', 'conditioned upon', closing conditions",
    FailureCategory.SCOPE_OVERBROAD:
        "Scope generalized — 'any' vs 'certain', party-specific limits removed",
    FailureCategory.NEGATION_MISSED:
        "Negation missed — 'shall not', 'no obligation', 'excluded', 'does not constitute'",
    FailureCategory.UNKNOWN:
        "Uncategorized — requires manual review",
}

# Training data recommendations per category
_CATEGORY_RECOMMENDATIONS: dict[FailureCategory, str] = {
    FailureCategory.SURFACE_FORM_AGREEMENT:
        "Generate hard negatives: topically similar passages where claim is contradicted",
    FailureCategory.MODAL_CONFUSION:
        "Add shall/may pairs with matching content — 'may X' vs claim 'must X'",
    FailureCategory.QUALIFIER_OMISSION:
        "Add knowledge-qualified and materiality-qualified entailment pairs",
    FailureCategory.CARVEOUT_MISSED:
        "Add carve-out pairs: entailed (exception applies) vs contradicted (main rule claimed)",
    FailureCategory.QUANTITATIVE_IMPRECISION:
        "Add numeric precision pairs: caps, floors, deductibles, time conversions",
    FailureCategory.CONDITION_STRIPPED:
        "Add conditional pairs: closing conditions, subject-to, conditioned-upon",
    FailureCategory.SCOPE_OVERBROAD:
        "Add scope pairs: party-specific limits, 'certain circumstances' vs 'always'",
    FailureCategory.NEGATION_MISSED:
        "Add negation pairs: 'shall not X' ↔ claim 'X occurs', exclusion lists",
    FailureCategory.UNKNOWN:
        "Manual review required to assign category and add to training data",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class FailureCase:
    """A single false positive case with category label."""

    case_id: str
    """Identifier of the test case or entailment log entry."""

    passage: str
    """The legal passage (premise)."""

    claim: str
    """The claim being verified (hypothesis)."""

    predicted_label: str
    """What the model predicted (e.g. 'entailed')."""

    true_label: str
    """Ground truth label (e.g. 'contradicted')."""

    category: FailureCategory
    """Failure category assigned by heuristics or manual review."""

    confidence: float = 0.0
    """Model's confidence score for the prediction."""

    explanation: str = ""
    """Human-readable explanation of why this is a failure."""

    manually_reviewed: bool = False
    """Whether a human has verified the category assignment."""

    @property
    def is_false_positive(self) -> bool:
        """True if the model said supported but ground truth is not."""
        from legalkit.eval.nli_eval import _BINARY_POSITIVE
        return (
            self.predicted_label in _BINARY_POSITIVE
            and self.true_label not in _BINARY_POSITIVE
        )

    @property
    def is_false_negative(self) -> bool:
        """True if the model said not-supported but ground truth is supported."""
        from legalkit.eval.nli_eval import _BINARY_POSITIVE
        return (
            self.predicted_label not in _BINARY_POSITIVE
            and self.true_label in _BINARY_POSITIVE
        )


@dataclass
class FalsePositiveReport:
    """Analysis of false positives across a set of failure cases."""

    cases: list[FailureCase]
    """All failure cases analyzed."""

    source: str = ""
    """Description of where these cases came from."""

    @property
    def false_positives(self) -> list[FailureCase]:
        return [c for c in self.cases if c.is_false_positive]

    @property
    def false_negatives(self) -> list[FailureCase]:
        return [c for c in self.cases if c.is_false_negative]

    @property
    def category_counts(self) -> dict[FailureCategory, int]:
        return dict(Counter(c.category for c in self.false_positives))

    @property
    def top_categories(self) -> list[tuple[FailureCategory, int]]:
        return sorted(self.category_counts.items(), key=lambda x: -x[1])

    def examples_for(self, category: FailureCategory, n: int = 2) -> list[FailureCase]:
        return [c for c in self.false_positives if c.category == category][:n]

    def summary_table(self) -> str:
        """Render a plain-text summary of failure categories."""
        fps = self.false_positives
        fns = self.false_negatives
        total = len(self.cases)

        lines = [
            f"False Positive Analysis  ({self.source})",
            f"Total failures: {total}  |  False positives: {len(fps)}  |  False negatives: {len(fns)}",
            "",
            f"{'Category':<28}  {'N':>4}  {'%':>6}  Description",
            "─" * 90,
        ]

        for cat, count in self.top_categories:
            pct = count / len(fps) * 100 if fps else 0
            desc = _CATEGORY_DESCRIPTIONS[cat][:55]
            lines.append(f"  {cat.value:<26}  {count:>4}  {pct:>5.0f}%  {desc}")

        lines.append("─" * 90)
        lines.append("")
        lines.append("Recommendations by category:")
        for cat, _ in self.top_categories:
            if cat != FailureCategory.UNKNOWN:
                lines.append(f"  [{cat.value}]")
                lines.append(f"    → {_CATEGORY_RECOMMENDATIONS[cat]}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "total_failures": len(self.cases),
            "false_positives": len(self.false_positives),
            "false_negatives": len(self.false_negatives),
            "category_distribution": {
                cat.value: count for cat, count in self.top_categories
            },
            "cases": [
                {
                    "id": c.case_id,
                    "category": c.category.value,
                    "predicted": c.predicted_label,
                    "true": c.true_label,
                    "confidence": c.confidence,
                    "explanation": c.explanation,
                    "manually_reviewed": c.manually_reviewed,
                    "passage_snippet": c.passage[:120],
                    "claim_snippet": c.claim[:120],
                }
                for c in self.cases
            ],
        }


# ---------------------------------------------------------------------------
# Heuristic auto-categorization
# ---------------------------------------------------------------------------

# Ordered rules: (pattern_fn, category) — first match wins
# Each pattern_fn takes (passage: str, claim: str) and returns bool

def _has_notwithstanding(passage: str, claim: str) -> bool:
    return bool(re.search(r"\bnotwithstanding\b", passage, re.I))

def _has_carveout(passage: str, claim: str) -> bool:
    return bool(re.search(
        r"\b(provided[,\s]+(?:however|that)|except\s+(?:that|as|for|to)|"
        r"shall\s+(?:not|exclude|be\s+excluded)|"
        r"does\s+not\s+(?:constitute|include)|"
        r"shall\s+not\s+(?:constitute|apply|include))\b",
        passage, re.I,
    ))

def _has_negation_in_passage_not_claim(passage: str, claim: str) -> bool:
    passage_negations = len(re.findall(
        r"\b(not|no|neither|nor|never|cannot|excluded?|does\s+not)\b",
        passage, re.I,
    ))
    claim_negations = len(re.findall(
        r"\b(not|no|neither|nor|never|cannot|excluded?|does\s+not)\b",
        claim, re.I,
    ))
    return passage_negations > 0 and claim_negations < passage_negations - 1

def _has_modal_mismatch(passage: str, claim: str) -> bool:
    passage_shall = bool(re.search(r"\bshall\b", passage, re.I))
    passage_may = bool(re.search(r"\b(?:may|in\s+its\s+(?:sole\s+)?discretion)\b", passage, re.I))
    claim_mandatory = bool(re.search(
        r"\b(must|required|mandatory|automatically|shall|is\s+(?:obligated|required))\b",
        claim, re.I,
    ))
    claim_permissive = bool(re.search(r"\b(may|can|is\s+permitted|is\s+allowed)\b", claim, re.I))
    # shall in passage, mandatory claimed → likely correct
    # may in passage, mandatory claimed → modal confusion
    if passage_may and claim_mandatory:
        return True
    # shall in passage, permissive claim → also suspicious
    if passage_shall and claim_permissive:
        return True
    return False

def _has_knowledge_qualifier(passage: str, claim: str) -> bool:
    return bool(re.search(
        r"\bto\s+(?:the\s+)?(?:seller'?s?|buyer'?s?|company'?s?|party'?s?)?\s*knowledge\b",
        passage, re.I,
    ))

def _has_materiality_qualifier(passage: str, claim: str) -> bool:
    return bool(re.search(
        r"\b(in\s+all\s+material\s+respects|material\s+(?:adverse|breach|respect)|materially)\b",
        passage, re.I,
    ))

def _has_quantitative_mismatch(passage: str, claim: str) -> bool:
    return bool(re.search(
        r"\b(up\s+to|not\s+(?:to\s+)?exceed|at\s+least|no\s+more\s+than|"
        r"deductible|basket|cap|floor|exceed[s]?|exceeding)\b",
        passage, re.I,
    ))

def _has_condition(passage: str, claim: str) -> bool:
    return bool(re.search(
        r"\b(subject\s+to|conditioned?\s+(?:upon|on)|"
        r"is\s+subject\s+to|satisfaction\s+of|"
        r"shall\s+be\s+(?:subject\s+to|conditioned)|"
        r"obligation[s]?\s+(?:of|to).{0,30}(?:is|are)\s+subject)\b",
        passage, re.I,
    ))

def _has_scope_generalization(passage: str, claim: str) -> bool:
    """Claim uses 'all', 'any', 'every' but passage uses qualified scope."""
    claim_broad = bool(re.search(r"\b(any|all|every|always|never|under\s+any\s+circumstances)\b", claim, re.I))
    passage_qualified = bool(re.search(
        r"\b(certain|specific|particular|solely|only|limited\s+to|"
        r"except|unless|solely\s+for|in\s+connection\s+with)\b",
        passage, re.I,
    ))
    return claim_broad and passage_qualified


_CATEGORIZATION_RULES: list[tuple] = [
    # (condition_fn, category) — first match wins
    (_has_notwithstanding,              FailureCategory.CARVEOUT_MISSED),
    (_has_carveout,                     FailureCategory.CARVEOUT_MISSED),
    (_has_modal_mismatch,               FailureCategory.MODAL_CONFUSION),
    (_has_knowledge_qualifier,          FailureCategory.QUALIFIER_OMISSION),
    (_has_materiality_qualifier,        FailureCategory.QUALIFIER_OMISSION),
    (_has_quantitative_mismatch,        FailureCategory.QUANTITATIVE_IMPRECISION),
    (_has_condition,                    FailureCategory.CONDITION_STRIPPED),
    (_has_scope_generalization,         FailureCategory.SCOPE_OVERBROAD),
    (_has_negation_in_passage_not_claim, FailureCategory.NEGATION_MISSED),
]


def categorize_failure(
    passage: str,
    claim: str,
    predicted_label: str,
    true_label: str,
) -> FailureCategory:
    """Assign a failure category using heuristic regex rules.

    Applies rules in priority order; returns the first matching category.
    Falls back to ``UNKNOWN`` if no rule matches.

    This is deliberately conservative — it prefers specificity over coverage.
    Uncategorized cases should be manually reviewed.

    Args:
        passage: The legal passage text.
        claim: The claim text.
        predicted_label: What the model predicted.
        true_label: The ground truth label.

    Returns:
        :class:`FailureCategory`.
    """
    for condition_fn, category in _CATEGORIZATION_RULES:
        if condition_fn(passage, claim):
            return category
    return FailureCategory.UNKNOWN


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def analyze_nli_failures(
    nli_result,  # NLIEvalResult
    include_false_negatives: bool = True,
) -> FalsePositiveReport:
    """Analyze failures in an NLI evaluation result.

    Finds all cases where the model's binary prediction was wrong,
    categorizes each failure using heuristics, and returns a report.

    Args:
        nli_result: :class:`~legalkit.eval.nli_eval.NLIEvalResult`.
        include_false_negatives: If True, also include false negatives
            (model said not-supported when it should be supported).

    Returns:
        :class:`FalsePositiveReport`.
    """
    from legalkit.eval.nli_eval import _BINARY_POSITIVE

    failure_cases: list[FailureCase] = []

    for judgment, case in zip(nli_result.judgments, nli_result.test_cases):
        predicted_positive = judgment.predicted in _BINARY_POSITIVE
        actual_positive = case.label in _BINARY_POSITIVE
        is_wrong = predicted_positive != actual_positive

        if not is_wrong:
            continue

        is_fp = predicted_positive and not actual_positive
        is_fn = not predicted_positive and actual_positive

        if is_fn and not include_false_negatives:
            continue

        category = categorize_failure(
            passage=case.passage,
            claim=case.claim,
            predicted_label=judgment.predicted,
            true_label=case.label,
        )

        failure_cases.append(FailureCase(
            case_id=case.id,
            passage=case.passage,
            claim=case.claim,
            predicted_label=judgment.predicted,
            true_label=case.label,
            category=category,
            confidence=judgment.confidence,
            explanation=case.explanation,
        ))

    return FalsePositiveReport(
        cases=failure_cases,
        source=f"{nli_result.model_name} on {len(nli_result.test_cases)} edge cases",
    )


def analyze_entailment_log(
    log_dir=None,
    threshold: float = 0.7,
    reviewed_only: bool = False,
) -> FalsePositiveReport:
    """Analyze pipeline entailment log for potential false positives.

    Since the log doesn't have human ground truth, this uses the LLM's own
    confidence score as a proxy: samples predicted as entailed with high
    confidence are treated as predictions, and we look for structural patterns
    that suggest they might be wrong.

    This is a heuristic scan — results should be treated as leads for manual
    review, not ground truth.

    Args:
        log_dir: Override the default entailment log directory.
        threshold: Confidence threshold above which to analyze (default 0.7).
        reviewed_only: Only analyze samples with human-corrected labels.

    Returns:
        :class:`FalsePositiveReport`.
    """
    from legalkit.eval.entailment_log import EntailmentLogger
    from legalkit.eval.nli_eval import _BINARY_POSITIVE

    samples = EntailmentLogger.load_all(log_dir)
    if reviewed_only:
        samples = [s for s in samples if s.reviewed]

    if not samples:
        return FalsePositiveReport(cases=[], source="entailment_log (empty)")

    failure_cases: list[FailureCase] = []

    for sample in samples:
        # For reviewed samples, use corrected label as ground truth
        if sample.reviewed and sample.corrected_class:
            true_label = sample.corrected_class
            predicted_label = sample.entailment_class
            if (predicted_label in _BINARY_POSITIVE and
                    true_label not in _BINARY_POSITIVE):
                category = categorize_failure(
                    sample.passage, sample.claim,
                    predicted_label, true_label,
                )
                failure_cases.append(FailureCase(
                    case_id=f"log-{sample.section_id}-{sample.timestamp[:10]}",
                    passage=sample.passage,
                    claim=sample.claim,
                    predicted_label=predicted_label,
                    true_label=true_label,
                    category=category,
                    confidence=sample.confidence,
                    manually_reviewed=True,
                ))
        elif not reviewed_only and sample.confidence >= threshold:
            # Unreviewed: flag high-confidence positives with risky patterns
            predicted_label = sample.entailment_class
            if predicted_label in _BINARY_POSITIVE:
                category = categorize_failure(
                    sample.passage, sample.claim,
                    predicted_label, "unknown",
                )
                if category != FailureCategory.UNKNOWN:
                    failure_cases.append(FailureCase(
                        case_id=f"log-{sample.section_id}-{sample.timestamp[:10]}",
                        passage=sample.passage,
                        claim=sample.claim,
                        predicted_label=predicted_label,
                        true_label="unknown",
                        category=category,
                        confidence=sample.confidence,
                        manually_reviewed=False,
                    ))

    label = "reviewed only" if reviewed_only else f"conf ≥ {threshold}, risky patterns"
    return FalsePositiveReport(
        cases=failure_cases,
        source=f"entailment_log ({len(samples)} samples, {label})",
    )
