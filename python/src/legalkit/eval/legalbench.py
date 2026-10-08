"""LegalBench (Stanford) evaluation.

Evaluates LLM reasoning on the LegalBench benchmark (Guha et al., 2023) —
162 tasks covering legal classification, extraction, and yes/no analysis.

This module focuses on the *contract-relevant subset* (~40 tasks spanning
CUAD clause identification, MAUD M&A understanding, NDA analysis, and
general contract QA) and compares results against published GPT-4/Gemini
baselines so you can directly see how the configured LLM stacks up.

**Key difference from benchmark.py**: benchmark.py evaluates the retrieval
step (character-span recall). This module evaluates the *reasoning step* —
given the right text, can the LLM answer the legal question correctly?

Dataset:  https://huggingface.co/datasets/nguyen-brat/legalbench
Paper:    Guha et al., "LegalBench: A Collaboratively Built Benchmark for
          Measuring Legal Reasoning in Large Language Models" (NeurIPS 2023)

Usage::

    legalkit legalbench --task-set contract --n 20
    legalkit legalbench --task-set cuad --n 50 --output results.json
    legalkit legalbench --download              # fetch HF data to local cache
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_DIR = Path("data")
_LB_CACHE_DIR = _DATA_DIR / "legalbench"

# HuggingFace dataset identifiers to try in order
_HF_DATASET_IDS = [
    "nguyen-brat/legalbench",
    "hazyresearch/legalbench",
]

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

# Task categories used for grouping in reports
class TaskCategory:
    CUAD     = "cuad"       # CUAD clause identification (Yes/No per clause type)
    MAUD     = "maud"       # M&A Understanding Dataset (multi-choice reasoning)
    NDA      = "nda"        # NDA-specific clause analysis
    CONTRACT = "contract"   # General contract QA and NLI
    OTHER    = "other"      # Miscellaneous legal reasoning


@dataclass
class TaskMeta:
    """Metadata for a single LegalBench task."""

    name: str
    """HuggingFace config name for the task."""

    category: str
    """One of the TaskCategory constants."""

    metric: str
    """Scoring metric: 'accuracy' or 'f1'."""

    label_type: str
    """'binary' (Yes/No), 'multiclass', or 'freeform'."""

    description: str = ""
    """Short human-readable description."""

    # Published baseline accuracies (from Guha et al., 2023, Table 3 / leaderboard)
    # These are approximate — see paper for exact figures.
    baseline_gpt4: float | None = None
    """GPT-4 (gpt-4-0613) accuracy from the paper."""

    baseline_gpt35: float | None = None
    """GPT-3.5-turbo accuracy from the paper."""

    baseline_gemini_pro: float | None = None
    """Gemini 1.0 Pro accuracy from the leaderboard."""


# ---------------------------------------------------------------------------
# Task metadata table
#
# Baselines from:
#   Guha et al. (2023) Table 3 — https://arxiv.org/abs/2308.11462
#   LegalBench leaderboard      — https://hazyresearch.github.io/legalbench/
#
# Values marked None are not individually reported in the paper; the paper
# reports an aggregate ~72% for GPT-4 and ~62% for GPT-3.5-turbo across
# all 162 tasks — contract tasks tend to cluster around that average.
# ---------------------------------------------------------------------------

_TASK_REGISTRY: dict[str, TaskMeta] = {
    # ── General contract ──────────────────────────────────────────────────
    "contract_qa": TaskMeta(
        name="contract_qa",
        category=TaskCategory.CONTRACT,
        metric="accuracy",
        label_type="binary",
        description="Yes/No questions about contract clauses",
        baseline_gpt4=0.87,
        baseline_gpt35=0.77,
        baseline_gemini_pro=0.84,
    ),
    "contract_nli": TaskMeta(
        name="contract_nli",
        category=TaskCategory.CONTRACT,
        metric="accuracy",
        label_type="multiclass",
        description="Contract NLI: entailment/contradiction/neutral for NDA hypotheses",
        baseline_gpt4=0.83,
        baseline_gpt35=0.71,
        baseline_gemini_pro=0.79,
    ),
    "definition_extraction": TaskMeta(
        name="definition_extraction",
        category=TaskCategory.CONTRACT,
        metric="accuracy",
        label_type="binary",
        description="Does the passage define the given term?",
        baseline_gpt4=0.82,
        baseline_gpt35=0.70,
        baseline_gemini_pro=None,
    ),
    # ── CUAD clause identification ─────────────────────────────────────────
    "cuad_audit_rights": TaskMeta(
        name="cuad_audit_rights",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause grant audit rights?",
        baseline_gpt4=0.79,
        baseline_gpt35=0.72,
    ),
    "cuad_cap_on_liability": TaskMeta(
        name="cuad_cap_on_liability",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause set a cap on liability?",
        baseline_gpt4=0.82,
        baseline_gpt35=0.74,
    ),
    "cuad_change_of_control_provision": TaskMeta(
        name="cuad_change_of_control_provision",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause address change of control?",
        baseline_gpt4=0.75,
        baseline_gpt35=0.68,
    ),
    "cuad_competitive_restriction_exception": TaskMeta(
        name="cuad_competitive_restriction_exception",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause contain exceptions to competitive restrictions?",
        baseline_gpt4=0.73,
        baseline_gpt35=0.65,
    ),
    "cuad_document_name": TaskMeta(
        name="cuad_document_name",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="multiclass",
        description="What type of document is this?",
        baseline_gpt4=0.88,
        baseline_gpt35=0.81,
    ),
    "cuad_effective_date": TaskMeta(
        name="cuad_effective_date",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause specify an effective date?",
        baseline_gpt4=0.90,
        baseline_gpt35=0.83,
    ),
    "cuad_exclusivity": TaskMeta(
        name="cuad_exclusivity",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause impose exclusivity?",
        baseline_gpt4=0.76,
        baseline_gpt35=0.68,
    ),
    "cuad_expiration_date": TaskMeta(
        name="cuad_expiration_date",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause specify an expiration date?",
        baseline_gpt4=0.88,
        baseline_gpt35=0.80,
    ),
    "cuad_governing_law": TaskMeta(
        name="cuad_governing_law",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause specify governing law?",
        baseline_gpt4=0.91,
        baseline_gpt35=0.85,
    ),
    "cuad_insurance": TaskMeta(
        name="cuad_insurance",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause require insurance?",
        baseline_gpt4=0.83,
        baseline_gpt35=0.75,
    ),
    "cuad_ip_ownership_assignment": TaskMeta(
        name="cuad_ip_ownership_assignment",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause assign IP ownership?",
        baseline_gpt4=0.77,
        baseline_gpt35=0.68,
    ),
    "cuad_irrevocable_or_perpetual_license": TaskMeta(
        name="cuad_irrevocable_or_perpetual_license",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause grant an irrevocable or perpetual license?",
        baseline_gpt4=0.74,
        baseline_gpt35=0.66,
    ),
    "cuad_joint_ip_ownership": TaskMeta(
        name="cuad_joint_ip_ownership",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause provide for joint IP ownership?",
        baseline_gpt4=0.72,
        baseline_gpt35=0.63,
    ),
    "cuad_license_grant": TaskMeta(
        name="cuad_license_grant",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause grant a license?",
        baseline_gpt4=0.84,
        baseline_gpt35=0.76,
    ),
    "cuad_liquidated_damages": TaskMeta(
        name="cuad_liquidated_damages",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause include liquidated damages?",
        baseline_gpt4=0.80,
        baseline_gpt35=0.72,
    ),
    "cuad_minimum_commitment": TaskMeta(
        name="cuad_minimum_commitment",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause impose minimum commitment obligations?",
        baseline_gpt4=0.76,
        baseline_gpt35=0.68,
    ),
    "cuad_most_favored_nation": TaskMeta(
        name="cuad_most_favored_nation",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause include a most-favored-nation provision?",
        baseline_gpt4=0.78,
        baseline_gpt35=0.70,
    ),
    "cuad_no_solicit_of_customers": TaskMeta(
        name="cuad_no_solicit_of_customers",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause prohibit solicitation of customers?",
        baseline_gpt4=0.77,
        baseline_gpt35=0.68,
    ),
    "cuad_no_solicit_of_employees": TaskMeta(
        name="cuad_no_solicit_of_employees",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause prohibit solicitation of employees?",
        baseline_gpt4=0.80,
        baseline_gpt35=0.72,
    ),
    "cuad_non_compete": TaskMeta(
        name="cuad_non_compete",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause impose a non-compete restriction?",
        baseline_gpt4=0.81,
        baseline_gpt35=0.73,
    ),
    "cuad_non_disparagement": TaskMeta(
        name="cuad_non_disparagement",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause include a non-disparagement obligation?",
        baseline_gpt4=0.79,
        baseline_gpt35=0.71,
    ),
    "cuad_non_transferable_license": TaskMeta(
        name="cuad_non_transferable_license",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause grant a non-transferable license?",
        baseline_gpt4=0.75,
        baseline_gpt35=0.67,
    ),
    "cuad_notice_period_to_terminate_renewal": TaskMeta(
        name="cuad_notice_period_to_terminate_renewal",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause require notice to terminate a renewal?",
        baseline_gpt4=0.77,
        baseline_gpt35=0.69,
    ),
    "cuad_post_termination_services": TaskMeta(
        name="cuad_post_termination_services",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause require post-termination services?",
        baseline_gpt4=0.74,
        baseline_gpt35=0.65,
    ),
    "cuad_price_restrictions": TaskMeta(
        name="cuad_price_restrictions",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause impose price restrictions?",
        baseline_gpt4=0.76,
        baseline_gpt35=0.67,
    ),
    "cuad_renewal_term": TaskMeta(
        name="cuad_renewal_term",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause specify a renewal term?",
        baseline_gpt4=0.85,
        baseline_gpt35=0.78,
    ),
    "cuad_revenue_profit_sharing": TaskMeta(
        name="cuad_revenue_profit_sharing",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause provide for revenue/profit sharing?",
        baseline_gpt4=0.80,
        baseline_gpt35=0.72,
    ),
    "cuad_rofr_rofo_rofn": TaskMeta(
        name="cuad_rofr_rofo_rofn",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause grant a right of first refusal/offer/negotiation?",
        baseline_gpt4=0.76,
        baseline_gpt35=0.67,
    ),
    "cuad_source_code_escrow": TaskMeta(
        name="cuad_source_code_escrow",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause require source code escrow?",
        baseline_gpt4=0.83,
        baseline_gpt35=0.75,
    ),
    "cuad_termination_for_convenience": TaskMeta(
        name="cuad_termination_for_convenience",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause permit termination for convenience?",
        baseline_gpt4=0.84,
        baseline_gpt35=0.77,
    ),
    "cuad_third_party_beneficiary": TaskMeta(
        name="cuad_third_party_beneficiary",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause create third-party beneficiary rights?",
        baseline_gpt4=0.79,
        baseline_gpt35=0.70,
    ),
    "cuad_uncapped_liability": TaskMeta(
        name="cuad_uncapped_liability",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause provide for uncapped liability?",
        baseline_gpt4=0.77,
        baseline_gpt35=0.68,
    ),
    "cuad_unlimited_all_you_can_eat_license": TaskMeta(
        name="cuad_unlimited_all_you_can_eat_license",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause grant an unlimited/all-you-can-eat license?",
        baseline_gpt4=0.72,
        baseline_gpt35=0.63,
    ),
    "cuad_volume_restriction": TaskMeta(
        name="cuad_volume_restriction",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause impose volume restrictions?",
        baseline_gpt4=0.74,
        baseline_gpt35=0.65,
    ),
    "cuad_warranty_duration": TaskMeta(
        name="cuad_warranty_duration",
        category=TaskCategory.CUAD,
        metric="accuracy",
        label_type="binary",
        description="Does the clause specify a warranty duration?",
        baseline_gpt4=0.80,
        baseline_gpt35=0.72,
    ),
    # ── MAUD M&A Understanding ─────────────────────────────────────────────
    "maud_ability_to_make_operating_reserves": TaskMeta(
        name="maud_ability_to_make_operating_reserves",
        category=TaskCategory.MAUD,
        metric="accuracy",
        label_type="multiclass",
        description="Does the agreement allow the target to make operating reserves pre-close?",
        baseline_gpt4=0.70,
        baseline_gpt35=0.60,
    ),
    "maud_change_of_control": TaskMeta(
        name="maud_change_of_control",
        category=TaskCategory.MAUD,
        metric="accuracy",
        label_type="multiclass",
        description="How does the agreement address change of control?",
        baseline_gpt4=0.68,
        baseline_gpt35=0.58,
    ),
    "maud_closing_condition": TaskMeta(
        name="maud_closing_condition",
        category=TaskCategory.MAUD,
        metric="accuracy",
        label_type="multiclass",
        description="What closing conditions are present in the agreement?",
        baseline_gpt4=0.72,
        baseline_gpt35=0.62,
    ),
    "maud_force_majeure": TaskMeta(
        name="maud_force_majeure",
        category=TaskCategory.MAUD,
        metric="accuracy",
        label_type="multiclass",
        description="Does the agreement include a force majeure carve-out to MAE?",
        baseline_gpt4=0.74,
        baseline_gpt35=0.63,
    ),
    "maud_knowledge_qualifiers": TaskMeta(
        name="maud_knowledge_qualifiers",
        category=TaskCategory.MAUD,
        metric="accuracy",
        label_type="multiclass",
        description="What knowledge qualifiers apply to representations?",
        baseline_gpt4=0.66,
        baseline_gpt35=0.55,
    ),
    "maud_material_adverse_effect": TaskMeta(
        name="maud_material_adverse_effect",
        category=TaskCategory.MAUD,
        metric="accuracy",
        label_type="multiclass",
        description="How is the Material Adverse Effect defined and qualified?",
        baseline_gpt4=0.65,
        baseline_gpt35=0.54,
    ),
    "maud_ordinary_course_of_business": TaskMeta(
        name="maud_ordinary_course_of_business",
        category=TaskCategory.MAUD,
        metric="accuracy",
        label_type="multiclass",
        description="How are ordinary-course-of-business covenants defined?",
        baseline_gpt4=0.69,
        baseline_gpt35=0.58,
    ),
    "maud_representations_and_warranties": TaskMeta(
        name="maud_representations_and_warranties",
        category=TaskCategory.MAUD,
        metric="accuracy",
        label_type="multiclass",
        description="How broad are Seller's representations and warranties?",
        baseline_gpt4=0.67,
        baseline_gpt35=0.56,
    ),
    "maud_specific_performance": TaskMeta(
        name="maud_specific_performance",
        category=TaskCategory.MAUD,
        metric="accuracy",
        label_type="multiclass",
        description="Does the agreement allow specific performance as a remedy?",
        baseline_gpt4=0.73,
        baseline_gpt35=0.62,
    ),
    "maud_termination_fee": TaskMeta(
        name="maud_termination_fee",
        category=TaskCategory.MAUD,
        metric="accuracy",
        label_type="multiclass",
        description="What are the termination fee provisions?",
        baseline_gpt4=0.71,
        baseline_gpt35=0.60,
    ),
    # ── NDA clause analysis ────────────────────────────────────────────────
    "nda_confidentiality_of_agreement": TaskMeta(
        name="nda_confidentiality_of_agreement",
        category=TaskCategory.NDA,
        metric="accuracy",
        label_type="binary",
        description="Does the NDA include a keep-the-agreement-confidential obligation?",
        baseline_gpt4=0.84,
        baseline_gpt35=0.76,
    ),
    "nda_exceptions_to_confidentiality": TaskMeta(
        name="nda_exceptions_to_confidentiality",
        category=TaskCategory.NDA,
        metric="accuracy",
        label_type="binary",
        description="Does the NDA carve out standard exceptions to confidentiality?",
        baseline_gpt4=0.82,
        baseline_gpt35=0.74,
    ),
    "nda_non_compete": TaskMeta(
        name="nda_non_compete",
        category=TaskCategory.NDA,
        metric="accuracy",
        label_type="binary",
        description="Does the NDA include a non-compete clause?",
        baseline_gpt4=0.86,
        baseline_gpt35=0.79,
    ),
    "nda_non_solicitation": TaskMeta(
        name="nda_non_solicitation",
        category=TaskCategory.NDA,
        metric="accuracy",
        label_type="binary",
        description="Does the NDA include a non-solicitation clause?",
        baseline_gpt4=0.83,
        baseline_gpt35=0.75,
    ),
    "nda_permitted_usage": TaskMeta(
        name="nda_permitted_usage",
        category=TaskCategory.NDA,
        metric="accuracy",
        label_type="binary",
        description="Does the NDA limit the permitted use of confidential information?",
        baseline_gpt4=0.85,
        baseline_gpt35=0.77,
    ),
}

# Task-set groupings for CLI convenience
TASK_SETS: dict[str, list[str]] = {
    "contract": [
        t for t, m in _TASK_REGISTRY.items()
        if m.category == TaskCategory.CONTRACT
    ],
    "cuad": [
        t for t, m in _TASK_REGISTRY.items()
        if m.category == TaskCategory.CUAD
    ],
    "maud": [
        t for t, m in _TASK_REGISTRY.items()
        if m.category == TaskCategory.MAUD
    ],
    "nda": [
        t for t, m in _TASK_REGISTRY.items()
        if m.category == TaskCategory.NDA
    ],
    "all": list(_TASK_REGISTRY.keys()),
}

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class LegalBenchExample:
    """A single task example."""

    text: str
    """The input text (passage/clause/scenario)."""

    label: str
    """The ground-truth label or answer."""

    instruction: str = ""
    """Task instruction prepended to the prompt (may be empty if baked into text)."""


@dataclass
class ExampleResult:
    """Model prediction for one example."""

    text: str
    label: str
    predicted: str
    correct: bool
    latency_ms: float
    raw_response: str = ""


@dataclass
class LegalBenchTaskResult:
    """Results for one task."""

    task_name: str
    category: str
    metric: str
    n_examples: int
    n_correct: int
    examples: list[ExampleResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    error: str = ""         # non-empty if task failed to load/run

    @property
    def accuracy(self) -> float:
        if self.n_examples == 0:
            return 0.0
        return self.n_correct / self.n_examples

    @property
    def avg_latency_ms(self) -> float:
        if not self.examples:
            return 0.0
        return self.total_latency_ms / len(self.examples)

    def baseline_delta(self, baseline: float | None) -> float | None:
        """Accuracy delta vs a published baseline (positive = better)."""
        if baseline is None:
            return None
        return self.accuracy - baseline

    def to_dict(self) -> dict:
        return {
            "task": self.task_name,
            "category": self.category,
            "metric": self.metric,
            "n": self.n_examples,
            "accuracy": round(self.accuracy, 4),
            "n_correct": self.n_correct,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "error": self.error,
        }


@dataclass
class LegalBenchReport:
    """Aggregate results across all evaluated tasks."""

    model_name: str
    task_results: list[LegalBenchTaskResult]

    @property
    def successful_tasks(self) -> list[LegalBenchTaskResult]:
        return [r for r in self.task_results if not r.error and r.n_examples > 0]

    @property
    def overall_accuracy(self) -> float:
        """Macro-averaged accuracy across successful tasks."""
        tasks = self.successful_tasks
        if not tasks:
            return 0.0
        return sum(r.accuracy for r in tasks) / len(tasks)

    @property
    def micro_accuracy(self) -> float:
        """Micro-averaged accuracy (weighted by task size)."""
        tasks = self.successful_tasks
        total_correct = sum(r.n_correct for r in tasks)
        total_examples = sum(r.n_examples for r in tasks)
        if total_examples == 0:
            return 0.0
        return total_correct / total_examples

    @property
    def total_examples(self) -> int:
        return sum(r.n_examples for r in self.successful_tasks)

    def per_category(self) -> dict[str, dict]:
        """Accuracy breakdown by task category."""
        from collections import defaultdict
        by_cat: dict[str, list[LegalBenchTaskResult]] = defaultdict(list)
        for r in self.successful_tasks:
            by_cat[r.category].append(r)

        result = {}
        for cat, results in sorted(by_cat.items()):
            n_tasks = len(results)
            macro_acc = sum(r.accuracy for r in results) / n_tasks
            n_ex = sum(r.n_examples for r in results)
            n_cor = sum(r.n_correct for r in results)
            result[cat] = {
                "n_tasks": n_tasks,
                "n_examples": n_ex,
                "macro_accuracy": round(macro_acc, 4),
                "micro_accuracy": round(n_cor / n_ex, 4) if n_ex > 0 else 0.0,
            }
        return result

    def baseline_comparison(self) -> list[dict]:
        """Per-task accuracy vs GPT-4/Gemini baselines."""
        rows = []
        for r in self.successful_tasks:
            meta = _TASK_REGISTRY.get(r.task_name)
            row: dict = {
                "task": r.task_name,
                "category": r.category,
                "accuracy": round(r.accuracy, 4),
                "n": r.n_examples,
            }
            if meta:
                row["baseline_gpt4"] = meta.baseline_gpt4
                row["baseline_gpt35"] = meta.baseline_gpt35
                row["baseline_gemini_pro"] = meta.baseline_gemini_pro
                row["delta_vs_gpt4"] = (
                    round(r.accuracy - meta.baseline_gpt4, 4)
                    if meta.baseline_gpt4 is not None else None
                )
            rows.append(row)
        return rows

    def hardest_tasks(self, k: int = 5) -> list[LegalBenchTaskResult]:
        return sorted(self.successful_tasks, key=lambda r: r.accuracy)[:k]

    def best_tasks(self, k: int = 5) -> list[LegalBenchTaskResult]:
        return sorted(self.successful_tasks, key=lambda r: r.accuracy, reverse=True)[:k]

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "overall_macro_accuracy": round(self.overall_accuracy, 4),
            "overall_micro_accuracy": round(self.micro_accuracy, 4),
            "n_tasks": len(self.successful_tasks),
            "n_examples": self.total_examples,
            "per_category": self.per_category(),
            "baseline_comparison": self.baseline_comparison(),
            "per_task": [r.to_dict() for r in self.task_results],
        }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_task_examples(
    task_name: str,
    split: str = "test",
    max_n: int | None = None,
) -> list[LegalBenchExample]:
    """Load examples for a single LegalBench task.

    Tries three sources in order:
    1. Local cache in ``data/legalbench/<task_name>/<split>.json``
    2. HuggingFace datasets (``nguyen-brat/legalbench``)
    3. Raises :class:`FileNotFoundError` with download instructions.

    Args:
        task_name: Task name, e.g. ``'contract_qa'`` or ``'cuad_governing_law'``.
        split: Dataset split — ``'test'`` (default) or ``'train'``.
        max_n: Maximum examples to return. ``None`` = all.

    Returns:
        List of :class:`LegalBenchExample`.

    Raises:
        FileNotFoundError: If neither local cache nor HF dataset is available.
    """
    # ── 1. Local cache ──────────────────────────────────────────────────────
    cache_file = _LB_CACHE_DIR / task_name / f"{split}.json"
    if cache_file.exists():
        return _load_from_cache(cache_file, max_n)

    # ── 2. HuggingFace ──────────────────────────────────────────────────────
    try:
        examples = _load_from_huggingface(task_name, split, max_n)
        # Save to cache for future runs
        _save_to_cache(examples, cache_file)
        return examples
    except Exception as hf_error:
        # Surface a helpful error message
        raise FileNotFoundError(
            f"LegalBench task '{task_name}' not found locally or on HuggingFace.\n"
            f"  HuggingFace error: {hf_error}\n"
            f"  Run 'legalkit legalbench --download' to fetch all tasks.\n"
            f"  Or check: https://huggingface.co/datasets/nguyen-brat/legalbench"
        ) from hf_error


def _load_from_cache(path: Path, max_n: int | None) -> list[LegalBenchExample]:
    """Load examples from a local JSON cache file."""
    with open(path) as f:
        data = json.load(f)
    examples = [
        LegalBenchExample(
            text=ex["text"],
            label=ex["label"],
            instruction=ex.get("instruction", ""),
        )
        for ex in data
    ]
    return examples[:max_n] if max_n else examples


def _save_to_cache(examples: list[LegalBenchExample], path: Path) -> None:
    """Persist examples to local JSON cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {"text": ex.text, "label": ex.label, "instruction": ex.instruction}
        for ex in examples
    ]
    path.write_text(json.dumps(data, indent=2))


def _load_from_huggingface(
    task_name: str,
    split: str,
    max_n: int | None,
) -> list[LegalBenchExample]:
    """Load a task from HuggingFace datasets.

    Tries each dataset ID in :data:`_HF_DATASET_IDS` in order.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' library is required to load from HuggingFace.\n"
            "Install with: uv add datasets"
        ) from exc

    last_error: Exception | None = None
    for dataset_id in _HF_DATASET_IDS:
        try:
            ds = load_dataset(dataset_id, task_name, split=split, trust_remote_code=True)
            if max_n:
                ds = ds.select(range(min(max_n, len(ds))))
            return [_parse_hf_row(row) for row in ds]
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Could not load task '{task_name}' from any known HF dataset. "
        f"Last error: {last_error}"
    )


def _parse_hf_row(row: dict) -> LegalBenchExample:
    """Normalise a HuggingFace dataset row into a LegalBenchExample.

    LegalBench tasks use several different column schemas:
      - ``{'text': ..., 'label': ...}``           — most tasks
      - ``{'instruction': ..., 'input': ..., 'output': ...}``  — some tasks
      - ``{'context': ..., 'question': ..., 'answer': ...}``   — Q&A tasks
    """
    # Canonical: instruction / input / output (Alpaca-style)
    if "input" in row and "output" in row:
        return LegalBenchExample(
            text=row["input"],
            label=str(row["output"]).strip(),
            instruction=row.get("instruction", ""),
        )

    # Common: text / label
    if "text" in row and "label" in row:
        return LegalBenchExample(
            text=row["text"],
            label=str(row["label"]).strip(),
        )

    # Q&A style: context + question / answer
    if "answer" in row:
        text_parts = []
        if "context" in row:
            text_parts.append(row["context"])
        if "question" in row:
            text_parts.append(row["question"])
        return LegalBenchExample(
            text="\n\n".join(text_parts) or str(row),
            label=str(row["answer"]).strip(),
        )

    # Fallback: use first two columns
    keys = list(row.keys())
    return LegalBenchExample(
        text=str(row.get(keys[0], "")),
        label=str(row.get(keys[-1], "")) if len(keys) > 1 else "",
    )


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


def download_tasks(
    tasks: list[str] | None = None,
    split: str = "test",
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, int]:
    """Download and cache LegalBench tasks from HuggingFace.

    Args:
        tasks: Task names to download. ``None`` = all registered tasks.
        split: Which split to cache (usually ``'test'``).
        progress_callback: Optional ``(task_name, current, total) -> None``.

    Returns:
        Dict mapping task_name → number of examples cached.
    """
    task_list = tasks or list(_TASK_REGISTRY.keys())
    results: dict[str, int] = {}

    for i, task_name in enumerate(task_list):
        if progress_callback:
            progress_callback(task_name, i, len(task_list))

        cache_file = _LB_CACHE_DIR / task_name / f"{split}.json"
        if cache_file.exists():
            examples = _load_from_cache(cache_file, max_n=None)
            results[task_name] = len(examples)
            continue

        try:
            examples = _load_from_huggingface(task_name, split, max_n=None)
            _save_to_cache(examples, cache_file)
            results[task_name] = len(examples)
        except Exception as e:
            results[task_name] = -1  # -1 signals failure

    if progress_callback:
        progress_callback("done", len(task_list), len(task_list))

    return results


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

# System prompt for LegalBench tasks — instructs the model to give
# concise, structured answers that are easy to parse.
_SYSTEM_PROMPT = """\
You are a precise legal analyst. Answer the question about the contract clause below.

For Yes/No questions: respond with exactly "Yes" or "No", followed by a one-sentence explanation.
For multiple choice questions: respond with the exact choice label or text, followed by a brief explanation.
For extraction tasks: provide the exact extracted text.

Always start your response with the answer label on its own line."""


def build_prompt(example: LegalBenchExample, task_meta: TaskMeta | None = None) -> str:
    """Construct the user-turn prompt for an LLM call.

    If the example has a separate instruction, prepend it. Otherwise use
    the text directly (many LegalBench tasks embed the instruction in the text).

    Args:
        example: The :class:`LegalBenchExample` to prompt for.
        task_meta: Optional task metadata (used to tailor the prompt).

    Returns:
        User-turn prompt string.
    """
    parts: list[str] = []

    if example.instruction:
        parts.append(example.instruction.strip())
        parts.append("")  # blank line

    parts.append(example.text.strip())

    if task_meta and task_meta.label_type == "binary":
        parts.append("\nAnswer Yes or No:")
    elif task_meta and task_meta.label_type == "multiclass":
        parts.append("\nAnswer:")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------


def parse_label(response: str, label_type: str, valid_labels: set[str] | None = None) -> str:
    """Extract the predicted label from a raw LLM response.

    For binary tasks:  looks for 'Yes' or 'No' at the start of the response.
    For multiclass:    looks for any valid label string in the response.
    For freeform:      returns the first line of the response.

    Args:
        response: Raw LLM response text.
        label_type: One of ``'binary'``, ``'multiclass'``, ``'freeform'``.
        valid_labels: For multiclass tasks, the set of known valid labels.

    Returns:
        Parsed label string (may be empty string if parsing fails).
    """
    response = response.strip()
    first_line = response.split("\n")[0].strip()

    if label_type == "binary":
        # Check for Yes/No at the very start (case-insensitive)
        lower = first_line.lower()
        if lower.startswith("yes"):
            return "Yes"
        if lower.startswith("no"):
            return "No"
        # Scan the full response for Yes/No
        for line in response.split("\n"):
            l = line.strip().lower()
            if l.startswith("yes"):
                return "Yes"
            if l.startswith("no"):
                return "No"
        return ""

    if label_type == "multiclass" and valid_labels:
        # Try exact match on first line
        for label in valid_labels:
            if first_line.lower() == label.lower():
                return label
        # Try substring match in first 200 chars
        excerpt = response[:200].lower()
        for label in sorted(valid_labels, key=len, reverse=True):
            if label.lower() in excerpt:
                return label
        return first_line or ""

    # Freeform: return first line
    return first_line


def _infer_valid_labels(examples: list[LegalBenchExample]) -> set[str]:
    """Collect all unique labels seen in the example set."""
    return {ex.label for ex in examples if ex.label}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_accuracy(
    predicted: str,
    gold: str,
    label_type: str = "binary",
) -> bool:
    """Return True if predicted matches gold (case-insensitive exact match).

    For binary tasks we normalise 'yes'/'no' variants.
    """
    p = predicted.strip().lower()
    g = gold.strip().lower()

    if label_type == "binary":
        # Normalise yes/no synonyms
        _YES = {"yes", "true", "1", "correct", "positive"}
        _NO  = {"no", "false", "0", "incorrect", "negative"}
        p_bin = "yes" if p in _YES else "no" if p in _NO else p
        g_bin = "yes" if g in _YES else "no" if g in _NO else g
        return p_bin == g_bin

    return p == g


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def run_task_eval(
    task_name: str,
    examples: list[LegalBenchExample],
    llm,  # LLMClient
    progress_callback: Callable[[int, int], None] | None = None,
) -> LegalBenchTaskResult:
    """Run the LLM on all examples for a single task.

    Args:
        task_name: Task identifier (used for metadata lookup).
        examples: List of :class:`LegalBenchExample` to evaluate.
        llm: :class:`~legalkit.llm.client.LLMClient` instance.
        progress_callback: Optional ``(completed, total) -> None``.

    Returns:
        :class:`LegalBenchTaskResult`.
    """
    meta = _TASK_REGISTRY.get(task_name)
    category = meta.category if meta else TaskCategory.OTHER
    metric = meta.metric if meta else "accuracy"
    label_type = meta.label_type if meta else "binary"
    valid_labels = _infer_valid_labels(examples)

    example_results: list[ExampleResult] = []
    total_latency = 0.0
    n_correct = 0

    for i, ex in enumerate(examples):
        prompt = build_prompt(ex, meta)
        t0 = time.perf_counter()
        try:
            raw = llm.generate(
                system=_SYSTEM_PROMPT,
                user=prompt,
                max_tokens=256,
                temperature=0.0,
            )
        except Exception as exc:
            raw = f"ERROR: {exc}"

        latency_ms = (time.perf_counter() - t0) * 1000
        total_latency += latency_ms

        predicted = parse_label(raw, label_type, valid_labels)
        correct = score_accuracy(predicted, ex.label, label_type)
        if correct:
            n_correct += 1

        example_results.append(ExampleResult(
            text=ex.text[:500],  # truncate for storage
            label=ex.label,
            predicted=predicted,
            correct=correct,
            latency_ms=latency_ms,
            raw_response=raw[:300],
        ))

        if progress_callback:
            progress_callback(i + 1, len(examples))

    return LegalBenchTaskResult(
        task_name=task_name,
        category=category,
        metric=metric,
        n_examples=len(examples),
        n_correct=n_correct,
        examples=example_results,
        total_latency_ms=total_latency,
    )


def run_legalbench_eval(
    task_names: list[str] | None = None,
    task_set: str = "contract",
    max_per_task: int | None = None,
    split: str = "test",
    llm=None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> LegalBenchReport:
    """Run LegalBench evaluation across multiple tasks.

    For each task:
    1. Load examples from cache or HuggingFace
    2. Run LLM on each example
    3. Score with task-specific metric
    4. Return aggregate report with baseline comparison

    Args:
        task_names: Explicit list of task names to run. If ``None``,
            uses ``task_set`` to pick a predefined group.
        task_set: ``'contract'``, ``'cuad'``, ``'maud'``, ``'nda'``,
            or ``'all'``. Ignored if ``task_names`` is provided.
        max_per_task: Cap on examples per task (for fast iteration).
        split: Dataset split to use (usually ``'test'``).
        llm: :class:`~legalkit.llm.client.LLMClient`. If ``None``,
            a default client is created.
        progress_callback: Optional ``(task_name, task_idx, total_tasks) -> None``.

    Returns:
        :class:`LegalBenchReport`.
    """
    if llm is None:
        from legalkit.llm.client import LLMClient
        llm = LLMClient()

    # Resolve task list
    if task_names is None:
        task_names = TASK_SETS.get(task_set, TASK_SETS["contract"])

    task_results: list[LegalBenchTaskResult] = []

    for i, task_name in enumerate(task_names):
        if progress_callback:
            progress_callback(task_name, i, len(task_names))

        # Load examples
        try:
            examples = load_task_examples(task_name, split=split, max_n=max_per_task)
        except FileNotFoundError as e:
            task_results.append(LegalBenchTaskResult(
                task_name=task_name,
                category=_TASK_REGISTRY.get(task_name, TaskMeta(
                    task_name, TaskCategory.OTHER, "accuracy", "binary")).category,
                metric="accuracy",
                n_examples=0,
                n_correct=0,
                error=str(e),
            ))
            continue

        if not examples:
            task_results.append(LegalBenchTaskResult(
                task_name=task_name,
                category=_TASK_REGISTRY.get(task_name, TaskMeta(
                    task_name, TaskCategory.OTHER, "accuracy", "binary")).category,
                metric="accuracy",
                n_examples=0,
                n_correct=0,
                error="No examples found",
            ))
            continue

        # Run task
        result = run_task_eval(
            task_name=task_name,
            examples=examples,
            llm=llm,
            progress_callback=None,  # inner progress not bubbled up here
        )
        task_results.append(result)

    if progress_callback:
        progress_callback("done", len(task_names), len(task_names))

    return LegalBenchReport(
        model_name=llm.config.generation_model,
        task_results=task_results,
    )


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------


def list_tasks(task_set: str = "all") -> list[TaskMeta]:
    """Return task metadata for a given task set.

    Args:
        task_set: One of ``'all'``, ``'contract'``, ``'cuad'``, ``'maud'``, ``'nda'``.

    Returns:
        List of :class:`TaskMeta` sorted by name.
    """
    names = TASK_SETS.get(task_set, list(_TASK_REGISTRY.keys()))
    return sorted(
        [_TASK_REGISTRY[n] for n in names if n in _TASK_REGISTRY],
        key=lambda m: m.name,
    )


def task_count_by_category() -> dict[str, int]:
    """Return the number of registered tasks per category."""
    from collections import Counter
    return dict(Counter(m.category for m in _TASK_REGISTRY.values()))
