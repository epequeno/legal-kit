"""Tests for LegalBench (Stanford) evaluation module.

Covers:
- Task registry integrity
- Label parsing for binary / multiclass / freeform
- Scoring (accuracy)
- Prompt construction
- Example loading (from local cache; HF load mocked)
- Task result and report aggregation
- Baseline comparison
- run_legalbench_eval with mocked LLM and data loader
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from legalkit.eval.legalbench import (
    ExampleResult,
    LegalBenchExample,
    LegalBenchReport,
    LegalBenchTaskResult,
    TaskCategory,
    TaskMeta,
    TASK_SETS,
    _TASK_REGISTRY,
    _infer_valid_labels,
    _load_from_cache,
    _parse_hf_row,
    _save_to_cache,
    build_prompt,
    list_tasks,
    load_task_examples,
    parse_label,
    run_legalbench_eval,
    run_task_eval,
    score_accuracy,
    task_count_by_category,
)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------


class TestTaskRegistry:
    def test_registry_is_nonempty(self):
        assert len(_TASK_REGISTRY) > 0

    def test_all_required_fields_present(self):
        for name, meta in _TASK_REGISTRY.items():
            assert meta.name == name, f"{name}: meta.name mismatch"
            assert meta.category in (
                TaskCategory.CUAD, TaskCategory.MAUD,
                TaskCategory.NDA, TaskCategory.CONTRACT, TaskCategory.OTHER,
            ), f"{name}: unknown category {meta.category!r}"
            assert meta.metric in ("accuracy", "f1"), f"{name}: unknown metric {meta.metric!r}"
            assert meta.label_type in (
                "binary", "multiclass", "freeform"
            ), f"{name}: unknown label_type {meta.label_type!r}"

    def test_baselines_in_range(self):
        for name, meta in _TASK_REGISTRY.items():
            if meta.baseline_gpt4 is not None:
                assert 0.0 <= meta.baseline_gpt4 <= 1.0, f"{name}: GPT-4 baseline out of [0,1]"
            if meta.baseline_gpt35 is not None:
                assert 0.0 <= meta.baseline_gpt35 <= 1.0, f"{name}: GPT-3.5 baseline out of [0,1]"

    def test_cuad_tasks_are_binary(self):
        for name, meta in _TASK_REGISTRY.items():
            if meta.category == TaskCategory.CUAD:
                assert meta.label_type in ("binary", "multiclass"), (
                    f"Unexpected label_type for CUAD task {name!r}: {meta.label_type}"
                )

    def test_contract_qa_registered(self):
        assert "contract_qa" in _TASK_REGISTRY

    def test_governing_law_registered(self):
        assert "cuad_governing_law" in _TASK_REGISTRY

    def test_maud_task_registered(self):
        maud = [n for n in _TASK_REGISTRY if n.startswith("maud_")]
        assert len(maud) >= 5

    def test_nda_tasks_registered(self):
        nda = [n for n in _TASK_REGISTRY if n.startswith("nda_")]
        assert len(nda) >= 3

    def test_task_sets_cover_registry(self):
        all_in_sets = set(TASK_SETS["all"])
        all_in_registry = set(_TASK_REGISTRY.keys())
        # Every task in TASK_SETS["all"] should be in the registry
        assert all_in_sets == all_in_registry

    def test_task_sets_are_subsets_of_all(self):
        all_tasks = set(TASK_SETS["all"])
        for key, tasks in TASK_SETS.items():
            if key == "all":
                continue
            assert set(tasks).issubset(all_tasks), (
                f"Task set {key!r} contains tasks not in 'all'"
            )

    def test_task_count_by_category(self):
        counts = task_count_by_category()
        assert TaskCategory.CUAD in counts
        assert TaskCategory.MAUD in counts
        assert TaskCategory.NDA in counts
        assert counts[TaskCategory.CUAD] >= 10  # we have 35 CUAD tasks

    def test_list_tasks_returns_metadata(self):
        tasks = list_tasks("cuad")
        assert all(isinstance(t, TaskMeta) for t in tasks)
        assert all(t.category == TaskCategory.CUAD for t in tasks)

    def test_list_tasks_sorted_by_name(self):
        tasks = list_tasks("all")
        names = [t.name for t in tasks]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------


class TestParseLabel:
    # Binary

    def test_yes_exact(self):
        assert parse_label("Yes", "binary") == "Yes"

    def test_no_exact(self):
        assert parse_label("No", "binary") == "No"

    def test_yes_case_insensitive(self):
        assert parse_label("YES, because the clause explicitly...", "binary") == "Yes"

    def test_no_case_insensitive(self):
        assert parse_label("no — there is no such provision", "binary") == "No"

    def test_yes_with_explanation(self):
        response = "Yes\nThis clause clearly grants audit rights to the licensor."
        assert parse_label(response, "binary") == "Yes"

    def test_no_with_explanation(self):
        response = "No\nThe agreement does not contain a non-compete restriction."
        assert parse_label(response, "binary") == "No"

    def test_binary_yes_second_line(self):
        """Yes/No found on second line when first is blank or noise."""
        response = "\nYes, the clause sets a liability cap of $5M."
        assert parse_label(response, "binary") == "Yes"

    def test_binary_empty_response(self):
        result = parse_label("", "binary")
        assert result == ""

    def test_binary_unrecognised_response(self):
        result = parse_label("The contract is complex.", "binary")
        assert result == ""

    # Multiclass

    def test_multiclass_exact_first_line(self):
        labels = {"Entailed", "Contradicted", "Neutral"}
        result = parse_label("Entailed\nBecause the passage directly supports the claim.", "multiclass", labels)
        assert result == "Entailed"

    def test_multiclass_case_insensitive(self):
        labels = {"entailed", "contradicted", "neutral"}
        result = parse_label("ENTAILED", "multiclass", labels)
        assert result == "entailed"

    def test_multiclass_substring_fallback(self):
        labels = {"Yes", "No", "N/A"}
        result = parse_label("The answer is Yes based on the clause text.", "multiclass", labels)
        assert result == "Yes"

    def test_multiclass_no_valid_labels(self):
        result = parse_label("Some answer", "multiclass", valid_labels=None)
        assert isinstance(result, str)

    def test_multiclass_longer_label_preferred(self):
        """Longer labels should be matched before shorter ones (greedy)."""
        labels = {"Termination for convenience", "Termination", "Convenience"}
        result = parse_label("Termination for convenience applies here.", "multiclass", labels)
        assert result == "Termination for convenience"

    # Freeform

    def test_freeform_returns_first_line(self):
        response = "January 15, 2024\nThis is the date stated in the preamble."
        assert parse_label(response, "freeform") == "January 15, 2024"

    def test_freeform_single_line(self):
        assert parse_label("Delaware", "freeform") == "Delaware"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoreAccuracy:
    def test_exact_match(self):
        assert score_accuracy("Yes", "Yes", "binary") is True

    def test_mismatch(self):
        assert score_accuracy("No", "Yes", "binary") is False

    def test_binary_case_insensitive(self):
        assert score_accuracy("yes", "Yes", "binary") is True
        assert score_accuracy("YES", "yes", "binary") is True

    def test_binary_no_variants(self):
        assert score_accuracy("no", "No", "binary") is True
        assert score_accuracy("No", "yes", "binary") is False

    def test_multiclass_exact(self):
        assert score_accuracy("Entailed", "Entailed", "multiclass") is True

    def test_multiclass_case_insensitive(self):
        assert score_accuracy("entailed", "Entailed", "multiclass") is True

    def test_multiclass_mismatch(self):
        assert score_accuracy("Neutral", "Entailed", "multiclass") is False

    def test_empty_prediction(self):
        assert score_accuracy("", "Yes", "binary") is False


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_text_only(self):
        ex = LegalBenchExample(text="Does this clause grant audit rights?", label="Yes")
        prompt = build_prompt(ex)
        assert "audit rights" in prompt

    def test_instruction_prepended(self):
        ex = LegalBenchExample(
            text="The licensor may audit...",
            label="Yes",
            instruction="Identify whether the following clause grants audit rights.",
        )
        prompt = build_prompt(ex)
        assert "Identify" in prompt
        assert "licensor may audit" in prompt
        # Instruction should come before the text
        assert prompt.index("Identify") < prompt.index("licensor may audit")

    def test_binary_hint_appended(self):
        meta = TaskMeta("cuad_audit_rights", TaskCategory.CUAD, "accuracy", "binary")
        ex = LegalBenchExample(text="Clause text here.", label="Yes")
        prompt = build_prompt(ex, meta)
        assert "Yes or No" in prompt

    def test_multiclass_hint_appended(self):
        meta = TaskMeta("contract_nli", TaskCategory.CONTRACT, "accuracy", "multiclass")
        ex = LegalBenchExample(text="Clause text here.", label="Entailed")
        prompt = build_prompt(ex, meta)
        assert "Answer:" in prompt

    def test_no_meta_no_hint(self):
        ex = LegalBenchExample(text="Plain text.", label="Yes")
        prompt = build_prompt(ex, task_meta=None)
        assert "Plain text." in prompt


# ---------------------------------------------------------------------------
# HF row parsing
# ---------------------------------------------------------------------------


class TestParseHFRow:
    def test_input_output_format(self):
        row = {"instruction": "Classify this clause.", "input": "The clause text.", "output": "Yes"}
        ex = _parse_hf_row(row)
        assert ex.text == "The clause text."
        assert ex.label == "Yes"
        assert ex.instruction == "Classify this clause."

    def test_text_label_format(self):
        row = {"text": "Clause text.", "label": "No"}
        ex = _parse_hf_row(row)
        assert ex.text == "Clause text."
        assert ex.label == "No"

    def test_answer_key_format(self):
        row = {"context": "The agreement states...", "question": "Is there a cap?", "answer": "Yes"}
        ex = _parse_hf_row(row)
        assert "The agreement states" in ex.text
        assert ex.label == "Yes"

    def test_fallback_first_last_columns(self):
        row = {"col_a": "some text", "col_b": "intermediate", "col_z": "Yes"}
        ex = _parse_hf_row(row)
        assert ex.label == "Yes"

    def test_label_stripped(self):
        row = {"text": "text", "label": "  Yes  "}
        ex = _parse_hf_row(row)
        assert ex.label == "Yes"


# ---------------------------------------------------------------------------
# Cache loading / saving
# ---------------------------------------------------------------------------


class TestCacheIO:
    def test_round_trip(self, tmp_path):
        examples = [
            LegalBenchExample(text="Clause A", label="Yes", instruction="Is it there?"),
            LegalBenchExample(text="Clause B", label="No"),
        ]
        cache_file = tmp_path / "task_x" / "test.json"
        _save_to_cache(examples, cache_file)
        assert cache_file.exists()

        loaded = _load_from_cache(cache_file, max_n=None)
        assert len(loaded) == 2
        assert loaded[0].text == "Clause A"
        assert loaded[0].label == "Yes"
        assert loaded[0].instruction == "Is it there?"
        assert loaded[1].label == "No"

    def test_max_n_respected(self, tmp_path):
        examples = [LegalBenchExample(text=f"text_{i}", label="Yes") for i in range(10)]
        cache_file = tmp_path / "t" / "test.json"
        _save_to_cache(examples, cache_file)

        loaded = _load_from_cache(cache_file, max_n=3)
        assert len(loaded) == 3

    def test_max_n_none_returns_all(self, tmp_path):
        examples = [LegalBenchExample(text=f"text_{i}", label="Yes") for i in range(5)]
        cache_file = tmp_path / "t" / "test.json"
        _save_to_cache(examples, cache_file)

        loaded = _load_from_cache(cache_file, max_n=None)
        assert len(loaded) == 5

    def test_instruction_defaults_to_empty(self, tmp_path):
        examples = [LegalBenchExample(text="t", label="l")]
        cache_file = tmp_path / "t" / "test.json"
        _save_to_cache(examples, cache_file)

        # Manually remove instruction key to test fallback
        data = json.loads(cache_file.read_text())
        for ex in data:
            ex.pop("instruction", None)
        cache_file.write_text(json.dumps(data))

        loaded = _load_from_cache(cache_file, max_n=None)
        assert loaded[0].instruction == ""


# ---------------------------------------------------------------------------
# load_task_examples — mocked
# ---------------------------------------------------------------------------


class TestLoadTaskExamples:
    def test_loads_from_cache_if_present(self, tmp_path):
        examples = [LegalBenchExample(text="clause", label="Yes")]
        cache_file = tmp_path / "cuad_governing_law" / "test.json"
        _save_to_cache(examples, cache_file)

        with patch("legalkit.eval.legalbench._LB_CACHE_DIR", tmp_path):
            loaded = load_task_examples("cuad_governing_law", split="test")

        assert len(loaded) == 1
        assert loaded[0].label == "Yes"

    def test_falls_back_to_hf_if_no_cache(self, tmp_path):
        mock_examples = [LegalBenchExample(text="t", label="Yes")]

        with (
            patch("legalkit.eval.legalbench._LB_CACHE_DIR", tmp_path),
            patch(
                "legalkit.eval.legalbench._load_from_huggingface",
                return_value=mock_examples,
            ),
        ):
            loaded = load_task_examples("cuad_audit_rights", split="test")

        assert len(loaded) == 1

    def test_raises_file_not_found_if_hf_fails(self, tmp_path):
        with (
            patch("legalkit.eval.legalbench._LB_CACHE_DIR", tmp_path),
            patch(
                "legalkit.eval.legalbench._load_from_huggingface",
                side_effect=RuntimeError("network error"),
            ),
        ):
            with pytest.raises(FileNotFoundError, match="LegalBench task"):
                load_task_examples("cuad_governing_law", split="test")

    def test_max_n_applied_to_cache(self, tmp_path):
        examples = [LegalBenchExample(text=f"c{i}", label="Yes") for i in range(20)]
        cache_file = tmp_path / "contract_qa" / "test.json"
        _save_to_cache(examples, cache_file)

        with patch("legalkit.eval.legalbench._LB_CACHE_DIR", tmp_path):
            loaded = load_task_examples("contract_qa", max_n=5)

        assert len(loaded) == 5


# ---------------------------------------------------------------------------
# run_task_eval — mocked LLM
# ---------------------------------------------------------------------------


def _make_examples(n: int, labels: list[str] | None = None) -> list[LegalBenchExample]:
    """Create n examples with cycling labels."""
    default_labels = ["Yes", "No", "Yes", "Yes", "No"]
    labels = labels or default_labels
    return [
        LegalBenchExample(
            text=f"Consider this clause: {i}. Does it grant audit rights?",
            label=labels[i % len(labels)],
        )
        for i in range(n)
    ]


def _make_mock_llm(responses: list[str]):
    """Build a MagicMock LLM whose generate() cycles through responses."""
    llm = MagicMock()
    llm.config.generation_model = "mock-model"
    llm.generate.side_effect = responses
    return llm


class TestRunTaskEval:
    def test_all_correct(self):
        examples = _make_examples(3, ["Yes", "No", "Yes"])
        llm = _make_mock_llm(["Yes", "No", "Yes"])

        result = run_task_eval("cuad_audit_rights", examples, llm)

        assert result.n_examples == 3
        assert result.n_correct == 3
        assert result.accuracy == 1.0
        assert not result.error

    def test_partial_correct(self):
        examples = _make_examples(4, ["Yes", "Yes", "No", "No"])
        llm = _make_mock_llm(["Yes", "No", "No", "Yes"])  # 2 correct

        result = run_task_eval("cuad_audit_rights", examples, llm)

        assert result.n_correct == 2
        assert result.accuracy == 0.5

    def test_all_wrong(self):
        examples = _make_examples(3, ["Yes", "Yes", "Yes"])
        llm = _make_mock_llm(["No", "No", "No"])

        result = run_task_eval("cuad_governing_law", examples, llm)

        assert result.n_correct == 0
        assert result.accuracy == 0.0

    def test_example_results_populated(self):
        examples = _make_examples(2, ["Yes", "No"])
        llm = _make_mock_llm(["Yes", "No"])

        result = run_task_eval("cuad_audit_rights", examples, llm)

        assert len(result.examples) == 2
        assert all(isinstance(er, ExampleResult) for er in result.examples)
        assert result.examples[0].correct is True
        assert result.examples[1].correct is True

    def test_task_category_set(self):
        examples = _make_examples(2)
        llm = _make_mock_llm(["Yes", "No"])

        result = run_task_eval("cuad_audit_rights", examples, llm)

        assert result.category == TaskCategory.CUAD

    def test_progress_callback_called(self):
        examples = _make_examples(5)
        llm = _make_mock_llm(["Yes"] * 5)
        calls = []

        def cb(current, total):
            calls.append((current, total))

        run_task_eval("cuad_governing_law", examples, llm, progress_callback=cb)

        assert len(calls) == 5
        assert calls[-1] == (5, 5)

    def test_llm_error_handled_gracefully(self):
        examples = _make_examples(3, ["Yes", "Yes", "Yes"])
        llm = MagicMock()
        llm.config.generation_model = "mock"
        llm.generate.side_effect = RuntimeError("API error")

        result = run_task_eval("cuad_audit_rights", examples, llm)

        # Should not raise; examples should have incorrect predictions
        assert result.n_examples == 3
        assert result.n_correct == 0

    def test_latency_tracked(self):
        examples = _make_examples(3)
        llm = _make_mock_llm(["Yes"] * 3)

        result = run_task_eval("cuad_audit_rights", examples, llm)

        assert result.total_latency_ms >= 0
        assert all(er.latency_ms >= 0 for er in result.examples)

    def test_unknown_task_uses_other_category(self):
        examples = _make_examples(2)
        llm = _make_mock_llm(["Yes"] * 2)

        result = run_task_eval("unknown_task_xyz", examples, llm)

        assert result.category == TaskCategory.OTHER


# ---------------------------------------------------------------------------
# LegalBenchTaskResult
# ---------------------------------------------------------------------------


class TestLegalBenchTaskResult:
    def _make(self, n_correct: int, n_total: int, category: str = "cuad") -> LegalBenchTaskResult:
        return LegalBenchTaskResult(
            task_name="cuad_governing_law",
            category=category,
            metric="accuracy",
            n_examples=n_total,
            n_correct=n_correct,
        )

    def test_accuracy_full(self):
        r = self._make(10, 10)
        assert r.accuracy == 1.0

    def test_accuracy_zero(self):
        r = self._make(0, 10)
        assert r.accuracy == 0.0

    def test_accuracy_empty(self):
        r = self._make(0, 0)
        assert r.accuracy == 0.0

    def test_accuracy_partial(self):
        r = self._make(7, 10)
        assert r.accuracy == pytest.approx(0.7)

    def test_baseline_delta_positive(self):
        r = self._make(9, 10)  # 90% accuracy
        delta = r.baseline_delta(0.82)
        assert delta == pytest.approx(0.08, abs=1e-4)

    def test_baseline_delta_negative(self):
        r = self._make(6, 10)  # 60% accuracy
        delta = r.baseline_delta(0.82)
        assert delta == pytest.approx(-0.22, abs=1e-4)

    def test_baseline_delta_none(self):
        r = self._make(8, 10)
        assert r.baseline_delta(None) is None

    def test_to_dict_structure(self):
        r = self._make(8, 10)
        d = r.to_dict()
        assert "task" in d
        assert "accuracy" in d
        assert "n" in d
        assert d["n"] == 10
        assert d["n_correct"] == 8


# ---------------------------------------------------------------------------
# LegalBenchReport
# ---------------------------------------------------------------------------


def _make_report(task_accuracies: dict[str, tuple[str, float, int]]) -> LegalBenchReport:
    """Build a LegalBenchReport from {task_name: (category, accuracy, n)} dict."""
    results = []
    for name, (cat, acc, n) in task_accuracies.items():
        n_correct = int(acc * n)
        results.append(LegalBenchTaskResult(
            task_name=name,
            category=cat,
            metric="accuracy",
            n_examples=n,
            n_correct=n_correct,
        ))
    return LegalBenchReport(model_name="test-model", task_results=results)


class TestLegalBenchReport:
    def _make(self):
        return _make_report({
            "cuad_governing_law":   (TaskCategory.CUAD, 0.90, 100),
            "cuad_audit_rights":    (TaskCategory.CUAD, 0.80, 100),
            "contract_qa":          (TaskCategory.CONTRACT, 0.85, 50),
            "nda_non_compete":      (TaskCategory.NDA, 0.88, 30),
            "maud_termination_fee": (TaskCategory.MAUD, 0.65, 40),
        })

    def test_overall_accuracy_is_macro_average(self):
        # Use tasks with n that divides evenly so int-rounding doesn't shift accuracy
        report = _make_report({
            "cuad_governing_law": (TaskCategory.CUAD, 0.90, 100),
            "cuad_audit_rights":  (TaskCategory.CUAD, 0.80, 100),
            "contract_qa":        (TaskCategory.CONTRACT, 0.70, 100),
        })
        expected_macro = (0.90 + 0.80 + 0.70) / 3
        assert report.overall_accuracy == pytest.approx(expected_macro, abs=1e-4)

    def test_micro_accuracy_weighted_by_task_size(self):
        report = self._make()
        # Larger tasks have more influence
        assert 0.0 < report.micro_accuracy < 1.0

    def test_total_examples(self):
        report = self._make()
        assert report.total_examples == 320  # 100+100+50+30+40

    def test_per_category_returns_all_categories(self):
        report = self._make()
        cats = report.per_category()
        assert TaskCategory.CUAD in cats
        assert TaskCategory.CONTRACT in cats
        assert TaskCategory.NDA in cats
        assert TaskCategory.MAUD in cats

    def test_per_category_cuad_has_two_tasks(self):
        report = self._make()
        cats = report.per_category()
        assert cats[TaskCategory.CUAD]["n_tasks"] == 2

    def test_baseline_comparison_structure(self):
        report = self._make()
        rows = report.baseline_comparison()
        # Not all tasks are in registry; only registry tasks get baselines
        for row in rows:
            assert "task" in row
            assert "accuracy" in row

    def test_baseline_comparison_delta_for_known_task(self):
        # Use n=100 so int-rounding doesn't alter the accuracy value
        report = _make_report({
            "cuad_governing_law": (TaskCategory.CUAD, 0.95, 100),
        })
        rows = report.baseline_comparison()
        assert len(rows) == 1
        row = rows[0]
        assert "delta_vs_gpt4" in row
        # Our 95% vs GPT-4 91% → positive delta of 0.04
        assert row["delta_vs_gpt4"] == pytest.approx(0.04, abs=1e-3)

    def test_hardest_tasks(self):
        report = self._make()
        hardest = report.hardest_tasks(k=2)
        assert len(hardest) == 2
        assert hardest[0].accuracy <= hardest[1].accuracy

    def test_best_tasks(self):
        report = self._make()
        best = report.best_tasks(k=2)
        assert len(best) == 2
        assert best[0].accuracy >= best[1].accuracy

    def test_successful_tasks_excludes_errors(self):
        results = [
            LegalBenchTaskResult(
                task_name="good_task", category="cuad", metric="accuracy",
                n_examples=10, n_correct=8,
            ),
            LegalBenchTaskResult(
                task_name="bad_task", category="cuad", metric="accuracy",
                n_examples=0, n_correct=0, error="Not found",
            ),
        ]
        report = LegalBenchReport(model_name="m", task_results=results)
        assert len(report.successful_tasks) == 1
        assert report.successful_tasks[0].task_name == "good_task"

    def test_to_dict_structure(self):
        report = self._make()
        d = report.to_dict()
        assert "model" in d
        assert "overall_macro_accuracy" in d
        assert "overall_micro_accuracy" in d
        assert "n_tasks" in d
        assert "n_examples" in d
        assert "per_category" in d
        assert "baseline_comparison" in d
        assert "per_task" in d

    def test_to_dict_serialisable(self):
        report = self._make()
        d = report.to_dict()
        # Should be JSON-serialisable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


# ---------------------------------------------------------------------------
# run_legalbench_eval — mocked end-to-end
# ---------------------------------------------------------------------------


class TestRunLegalBenchEval:
    def _mock_examples(self, n=5) -> list[LegalBenchExample]:
        return [LegalBenchExample(text=f"clause {i}", label="Yes") for i in range(n)]

    def test_returns_report(self, tmp_path):
        mock_examples = self._mock_examples()
        mock_llm = _make_mock_llm(["Yes"] * 100)

        with (
            patch("legalkit.eval.legalbench._LB_CACHE_DIR", tmp_path),
            patch(
                "legalkit.eval.legalbench.load_task_examples",
                return_value=mock_examples,
            ),
        ):
            report = run_legalbench_eval(
                task_names=["cuad_governing_law", "cuad_audit_rights"],
                llm=mock_llm,
            )

        assert isinstance(report, LegalBenchReport)
        assert len(report.task_results) == 2

    def test_task_set_contract(self, tmp_path):
        mock_examples = self._mock_examples()
        mock_llm = _make_mock_llm(["Yes"] * 500)

        with patch(
            "legalkit.eval.legalbench.load_task_examples",
            return_value=mock_examples,
        ):
            report = run_legalbench_eval(
                task_set="contract",
                max_per_task=5,
                llm=mock_llm,
            )

        assert len(report.task_results) == len(TASK_SETS["contract"])

    def test_file_not_found_becomes_error_task(self, tmp_path):
        mock_llm = _make_mock_llm([])

        with patch(
            "legalkit.eval.legalbench.load_task_examples",
            side_effect=FileNotFoundError("task not found"),
        ):
            report = run_legalbench_eval(
                task_names=["cuad_governing_law"],
                llm=mock_llm,
            )

        assert len(report.task_results) == 1
        assert report.task_results[0].error != ""
        assert len(report.successful_tasks) == 0

    def test_max_per_task_passed_to_loader(self, tmp_path):
        captured_max_n = []

        def mock_load(task_name, split="test", max_n=None):
            captured_max_n.append(max_n)
            return self._mock_examples()

        mock_llm = _make_mock_llm(["Yes"] * 100)

        with patch("legalkit.eval.legalbench.load_task_examples", side_effect=mock_load):
            run_legalbench_eval(
                task_names=["cuad_governing_law"],
                max_per_task=10,
                llm=mock_llm,
            )

        assert captured_max_n[0] == 10

    def test_progress_callback_invoked(self):
        mock_llm = _make_mock_llm(["Yes"] * 100)
        calls = []

        def cb(task_name, current, total):
            calls.append((task_name, current, total))

        with patch(
            "legalkit.eval.legalbench.load_task_examples",
            return_value=self._mock_examples(),
        ):
            run_legalbench_eval(
                task_names=["cuad_governing_law", "cuad_audit_rights"],
                llm=mock_llm,
                progress_callback=cb,
            )

        # Should have at least one call per task plus a final "done"
        assert len(calls) >= 3
        assert calls[-1][0] == "done"

    def test_model_name_in_report(self):
        mock_llm = MagicMock()
        mock_llm.config.generation_model = "claude-sonnet-test"
        mock_llm.generate.return_value = "Yes"

        with patch(
            "legalkit.eval.legalbench.load_task_examples",
            return_value=self._mock_examples(),
        ):
            report = run_legalbench_eval(
                task_names=["cuad_governing_law"],
                llm=mock_llm,
            )

        assert report.model_name == "claude-sonnet-test"


# ---------------------------------------------------------------------------
# Infer valid labels
# ---------------------------------------------------------------------------


class TestInferValidLabels:
    def test_binary_labels(self):
        examples = [
            LegalBenchExample("t", "Yes"),
            LegalBenchExample("t", "No"),
            LegalBenchExample("t", "Yes"),
        ]
        labels = _infer_valid_labels(examples)
        assert labels == {"Yes", "No"}

    def test_multiclass_labels(self):
        examples = [
            LegalBenchExample("t", "Entailed"),
            LegalBenchExample("t", "Contradicted"),
            LegalBenchExample("t", "Neutral"),
            LegalBenchExample("t", "Entailed"),
        ]
        labels = _infer_valid_labels(examples)
        assert labels == {"Entailed", "Contradicted", "Neutral"}

    def test_empty_examples(self):
        assert _infer_valid_labels([]) == set()

    def test_empty_label_excluded(self):
        examples = [LegalBenchExample("t", ""), LegalBenchExample("t", "Yes")]
        labels = _infer_valid_labels(examples)
        assert "" not in labels
        assert "Yes" in labels
