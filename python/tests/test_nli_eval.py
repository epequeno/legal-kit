"""Tests for NLI evaluation framework."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from legalkit.eval.nli_eval import (
    NLIEvalResult,
    NLIJudgment,
    NLITestCase,
    _map_nli_to_legal,
    _to_binary,
    compare_results,
    load_edge_cases,
    run_llm_judge,
    run_finetuned_nli,
)

EDGE_CASES_PATH = Path(__file__).parent.parent / "data/eval/legal_nli_edge_cases.json"


# ---------------------------------------------------------------------------
# load_edge_cases
# ---------------------------------------------------------------------------


class TestLoadEdgeCases:
    def test_loads_successfully(self):
        cases = load_edge_cases(EDGE_CASES_PATH)
        assert len(cases) > 0

    def test_returns_nli_test_cases(self):
        cases = load_edge_cases(EDGE_CASES_PATH)
        assert all(isinstance(c, NLITestCase) for c in cases)

    def test_all_required_fields(self):
        cases = load_edge_cases(EDGE_CASES_PATH)
        for c in cases:
            assert c.id
            assert c.category
            assert c.passage
            assert c.claim
            assert c.label in ("entailed", "partially_entailed", "contradicted", "neutral")

    def test_covers_all_categories(self):
        cases = load_edge_cases(EDGE_CASES_PATH)
        categories = {c.category for c in cases}
        expected = {
            "notwithstanding", "shall_vs_may", "knowledge_qualifier",
            "mae_carveout", "quantitative_precision", "temporal_qualifier",
            "condition_precedent", "defined_term",
        }
        assert expected.issubset(categories)

    def test_all_label_classes_present(self):
        cases = load_edge_cases(EDGE_CASES_PATH)
        labels = {c.label for c in cases}
        # Should have at least 3 of the 4 classes
        assert len(labels) >= 3

    def test_ids_unique(self):
        cases = load_edge_cases(EDGE_CASES_PATH)
        ids = [c.id for c in cases]
        assert len(ids) == len(set(ids))

    def test_at_least_30_cases(self):
        """Enough cases for a meaningful evaluation."""
        cases = load_edge_cases(EDGE_CASES_PATH)
        assert len(cases) >= 30

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_edge_cases(tmp_path / "missing.json")

    def test_explanation_present(self):
        cases = load_edge_cases(EDGE_CASES_PATH)
        cases_with_explanation = [c for c in cases if c.explanation]
        assert len(cases_with_explanation) > 0


# ---------------------------------------------------------------------------
# NLITestCase
# ---------------------------------------------------------------------------


class TestNLITestCase:
    def test_binary_label_positive(self):
        c = NLITestCase(id="x", category="c", passage="p", claim="cl", label="entailed")
        assert c.binary_label == "positive"

    def test_binary_label_partially_positive(self):
        c = NLITestCase(id="x", category="c", passage="p", claim="cl", label="partially_entailed")
        assert c.binary_label == "positive"

    def test_binary_label_negative_contradicted(self):
        c = NLITestCase(id="x", category="c", passage="p", claim="cl", label="contradicted")
        assert c.binary_label == "negative"

    def test_binary_label_negative_neutral(self):
        c = NLITestCase(id="x", category="c", passage="p", claim="cl", label="neutral")
        assert c.binary_label == "negative"


# ---------------------------------------------------------------------------
# _map_nli_to_legal
# ---------------------------------------------------------------------------


class TestMapNLIToLegal:
    def test_high_entailment_maps_to_entailed(self):
        label, score = _map_nli_to_legal({"entailment": 0.85, "neutral": 0.10, "contradiction": 0.05})
        assert label == "entailed"
        assert score == pytest.approx(0.85)

    def test_medium_entailment_maps_to_partially(self):
        label, score = _map_nli_to_legal({"entailment": 0.50, "neutral": 0.30, "contradiction": 0.20})
        assert label == "partially_entailed"

    def test_low_entailment_maps_to_neutral(self):
        label, score = _map_nli_to_legal({"entailment": 0.20, "neutral": 0.60, "contradiction": 0.20})
        assert label == "neutral"

    def test_high_contradiction_maps_to_contradicted(self):
        label, score = _map_nli_to_legal({"entailment": 0.05, "neutral": 0.15, "contradiction": 0.80})
        assert label == "contradicted"
        assert score == pytest.approx(0.80)

    def test_borderline_entailment(self):
        # Exactly at the 0.70 threshold
        label, _ = _map_nli_to_legal({"entailment": 0.70, "neutral": 0.20, "contradiction": 0.10})
        assert label == "entailed"

    def test_just_below_threshold(self):
        label, _ = _map_nli_to_legal({"entailment": 0.69, "neutral": 0.21, "contradiction": 0.10})
        assert label == "partially_entailed"

    def test_contradiction_beats_entailment(self):
        # Contradiction wins even with moderate entailment score
        label, _ = _map_nli_to_legal({"entailment": 0.40, "neutral": 0.10, "contradiction": 0.50})
        assert label == "contradicted"


# ---------------------------------------------------------------------------
# _to_binary
# ---------------------------------------------------------------------------


class TestToBinary:
    def test_entailed_is_positive(self):
        assert _to_binary("entailed") == "positive"

    def test_partially_entailed_is_positive(self):
        assert _to_binary("partially_entailed") == "positive"

    def test_contradicted_is_negative(self):
        assert _to_binary("contradicted") == "negative"

    def test_neutral_is_negative(self):
        assert _to_binary("neutral") == "negative"


# ---------------------------------------------------------------------------
# NLIEvalResult metrics
# ---------------------------------------------------------------------------


def _make_result(
    predictions: list[str],
    ground_truths: list[str],
    model_name: str = "test-model",
    latency_ms: float = 100.0,
) -> NLIEvalResult:
    cases = [
        NLITestCase(id=f"c{i}", category="test", passage="p", claim="cl", label=gt)
        for i, gt in enumerate(ground_truths)
    ]
    judgments = [
        NLIJudgment(case_id=f"c{i}", predicted=p, confidence=0.8, latency_ms=latency_ms)
        for i, p in enumerate(predictions)
    ]
    return NLIEvalResult(
        model_name=model_name,
        judgments=judgments,
        test_cases=cases,
        total_latency_ms=len(predictions) * latency_ms,
    )


class TestNLIEvalResult:
    def test_perfect_4class_accuracy(self):
        labels = ["entailed", "contradicted", "neutral", "partially_entailed"]
        result = _make_result(labels, labels)
        assert result.accuracy_4class == pytest.approx(1.0)

    def test_zero_4class_accuracy(self):
        result = _make_result(
            ["contradicted", "entailed", "entailed", "contradicted"],
            ["entailed", "contradicted", "neutral", "neutral"],
        )
        assert result.accuracy_4class == pytest.approx(0.0)

    def test_perfect_binary_accuracy(self):
        # All positives predicted as positives, all negatives as negatives
        result = _make_result(
            ["entailed", "partially_entailed", "contradicted", "neutral"],
            ["partially_entailed", "entailed", "neutral", "contradicted"],
        )
        assert result.accuracy_binary == pytest.approx(1.0)

    def test_binary_partial_counts_as_positive(self):
        result = _make_result(
            ["partially_entailed"],
            ["entailed"],
        )
        assert result.accuracy_binary == pytest.approx(1.0)

    def test_avg_latency(self):
        result = _make_result(["entailed"] * 4, ["entailed"] * 4, latency_ms=200.0)
        assert result.avg_latency_ms == pytest.approx(200.0)

    def test_per_category_accuracy(self):
        cases = [
            NLITestCase(id="a", category="notwithstanding", passage="p", claim="c", label="entailed"),
            NLITestCase(id="b", category="notwithstanding", passage="p", claim="c", label="contradicted"),
            NLITestCase(id="c", category="shall_vs_may", passage="p", claim="c", label="entailed"),
        ]
        judgments = [
            NLIJudgment(case_id="a", predicted="entailed", confidence=0.9, latency_ms=100),
            NLIJudgment(case_id="b", predicted="entailed", confidence=0.9, latency_ms=100),  # wrong
            NLIJudgment(case_id="c", predicted="entailed", confidence=0.9, latency_ms=100),
        ]
        result = NLIEvalResult(
            model_name="test",
            judgments=judgments,
            test_cases=cases,
            total_latency_ms=300,
        )
        cat = result.per_category_accuracy
        assert "notwithstanding" in cat
        assert "shall_vs_may" in cat
        # notwithstanding: 1/2 correct (entailed correct, contradicted→entailed wrong)
        assert cat["notwithstanding"]["accuracy_4class"] == pytest.approx(0.5)

    def test_confusion_matrix_shape(self):
        labels = ["entailed", "contradicted", "neutral", "partially_entailed"]
        result = _make_result(labels, labels)
        cm = result.confusion_matrix
        assert set(cm.keys()) == set(labels)
        for row in cm.values():
            assert set(row.keys()) == set(labels)

    def test_confusion_matrix_diagonal_for_perfect(self):
        labels = ["entailed", "contradicted", "neutral", "partially_entailed"]
        result = _make_result(labels, labels)
        cm = result.confusion_matrix
        for label in labels:
            assert cm[label][label] == 1

    def test_estimated_cost_none_for_unknown_model(self):
        result = _make_result(["entailed"], ["entailed"], model_name="unknown-model")
        assert result.estimated_cost_usd is None

    def test_estimated_cost_for_claude(self):
        result = _make_result(
            ["entailed"] * 10, ["entailed"] * 10,
            model_name="claude-sonnet-4-20250514",
        )
        cost = result.estimated_cost_usd
        assert cost is not None
        assert cost > 0

    def test_summary_contains_model_name(self):
        result = _make_result(["entailed"], ["entailed"], model_name="my-model")
        assert "my-model" in result.summary()

    def test_empty_result_accuracy_zero(self):
        result = NLIEvalResult(
            model_name="test",
            judgments=[],
            test_cases=[],
            total_latency_ms=0,
        )
        assert result.accuracy_4class == 0.0
        assert result.accuracy_binary == 0.0


# ---------------------------------------------------------------------------
# run_llm_judge (mocked)
# ---------------------------------------------------------------------------


class TestRunLLMJudge:
    def _make_cases(self, n: int = 3) -> list[NLITestCase]:
        labels = ["entailed", "contradicted", "neutral"]
        return [
            NLITestCase(
                id=f"c{i}", category="test",
                passage=f"passage {i}", claim=f"claim {i}",
                label=labels[i % 3]
            )
            for i in range(n)
        ]

    def test_returns_eval_result(self):
        from legalkit.models.citation import EntailmentClass
        llm = MagicMock()
        llm.config.generation_model = "test-model"

        with patch("legalkit.pipeline.verify._check_entailment_llm") as mock_check:
            mock_check.return_value = (EntailmentClass.ENTAILED, 0.9, "explanation")
            result = run_llm_judge(self._make_cases(), llm)

        assert isinstance(result, NLIEvalResult)
        assert result.model_name == "test-model"

    def test_calls_check_once_per_case(self):
        from legalkit.models.citation import EntailmentClass

        cases = self._make_cases(5)
        llm = MagicMock()
        llm.config.generation_model = "test-model"

        with patch("legalkit.pipeline.verify._check_entailment_llm") as mock_check:
            mock_check.return_value = (EntailmentClass.NEUTRAL, 0.5, "")
            run_llm_judge(cases, llm)

        assert mock_check.call_count == 5

    def test_handles_llm_exception_gracefully(self):
        cases = self._make_cases(2)
        llm = MagicMock()
        llm.config.generation_model = "test-model"

        with patch("legalkit.pipeline.verify._check_entailment_llm") as mock_check:
            mock_check.side_effect = RuntimeError("API error")
            result = run_llm_judge(cases, llm)

        # Should not raise; failed cases get neutral
        assert len(result.judgments) == 2
        assert all(j.predicted == "neutral" for j in result.judgments)

    def test_progress_callback_called(self):
        from legalkit.models.citation import EntailmentClass

        cases = self._make_cases(4)
        llm = MagicMock()
        llm.config.generation_model = "test-model"
        calls = []

        with patch("legalkit.pipeline.verify._check_entailment_llm") as mock_check:
            mock_check.return_value = (EntailmentClass.ENTAILED, 0.9, "")
            run_llm_judge(cases, llm, progress_callback=lambda c, t: calls.append((c, t)))

        assert len(calls) == 4
        completed = [c for c, _ in calls]
        assert completed == [1, 2, 3, 4]

    def test_judgment_latency_recorded(self):
        from legalkit.models.citation import EntailmentClass

        cases = self._make_cases(2)
        llm = MagicMock()
        llm.config.generation_model = "test-model"

        with patch("legalkit.pipeline.verify._check_entailment_llm") as mock_check:
            mock_check.return_value = (EntailmentClass.ENTAILED, 0.9, "")
            result = run_llm_judge(cases, llm)

        assert all(j.latency_ms >= 0 for j in result.judgments)


# ---------------------------------------------------------------------------
# run_finetuned_nli (mocked)
# ---------------------------------------------------------------------------


class TestRunFinetunedNLI:
    def _make_cases(self, n: int = 3) -> list[NLITestCase]:
        labels = ["entailed", "contradicted", "neutral"]
        return [
            NLITestCase(
                id=f"c{i}", category="test",
                passage=f"passage {i}", claim=f"claim {i}",
                label=labels[i % 3]
            )
            for i in range(n)
        ]

    def test_returns_eval_result(self):
        with patch("transformers.AutoTokenizer") as mock_tokenizer_class, \
             patch("transformers.AutoModelForSequenceClassification") as mock_model_class, \
             patch("torch.device") as mock_device, \
             patch("torch.cuda.is_available", return_value=False), \
             patch("torch.no_grad"), \
             patch("torch.softmax") as mock_softmax, \
             patch("torch.argmax") as mock_argmax:
            
            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_inputs = MagicMock()
            mock_inputs.to.return_value = mock_inputs
            mock_tokenizer.return_value = mock_inputs

            # Mock model
            mock_model = MagicMock()
            mock_model_class.from_pretrained.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = None

            # Mock model output (logits)
            mock_logits = MagicMock()
            mock_output = MagicMock()
            mock_output.logits = mock_logits
            mock_model.return_value = mock_output

            # Mock torch.softmax and argmax
            mock_probs = MagicMock()
            mock_probs.squeeze.return_value.tolist.return_value = [0.8, 0.1, 0.05, 0.05]
            mock_softmax.return_value = mock_probs
            
            mock_argmax_result = MagicMock()
            mock_argmax_result.item.return_value = 0  # entailed
            mock_argmax.return_value = mock_argmax_result

            cases = self._make_cases(2)
            result = run_finetuned_nli(cases, model_path="fake/path", device="cpu")

            assert isinstance(result, NLIEvalResult)
            assert len(result.judgments) == 2
            assert result.model_name == "legal-deberta-large:path"

    def test_correct_label_extraction(self):
        with patch("transformers.AutoTokenizer") as mock_tokenizer_class, \
             patch("transformers.AutoModelForSequenceClassification") as mock_model_class, \
             patch("torch.device"), \
             patch("torch.cuda.is_available", return_value=False), \
             patch("torch.no_grad"), \
             patch("torch.softmax") as mock_softmax, \
             patch("torch.argmax") as mock_argmax:
            
            # Mock setup
            mock_tokenizer = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_inputs = MagicMock()
            mock_inputs.to.return_value = mock_inputs
            mock_tokenizer.return_value = mock_inputs
            
            mock_model = MagicMock()
            mock_model_class.from_pretrained.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = None
            
            mock_logits = MagicMock()
            mock_output = MagicMock()
            mock_output.logits = mock_logits
            mock_model.return_value = mock_output
            
            # Test each label index
            label_names = ["entailed", "partially_entailed", "contradicted", "neutral"]
            for idx, expected_label in enumerate(label_names):
                mock_probs = MagicMock()
                probs_list = [0.1, 0.1, 0.1, 0.1]
                probs_list[idx] = 0.7  # High confidence for this label
                mock_probs.squeeze.return_value.tolist.return_value = probs_list
                mock_softmax.return_value = mock_probs
                
                mock_argmax_result = MagicMock()
                mock_argmax_result.item.return_value = idx
                mock_argmax.return_value = mock_argmax_result
                
                cases = self._make_cases(1)
                result = run_finetuned_nli(cases, model_path="fake/path", device="cpu")
                
                assert result.judgments[0].predicted == expected_label
                assert result.judgments[0].confidence == pytest.approx(0.7)

    def test_handles_model_load_error(self):
        with patch("transformers.AutoTokenizer") as mock_tokenizer_class:
            # Mock tokenizer to raise error
            mock_tokenizer_class.from_pretrained.side_effect = RuntimeError("Model load failed")
            
            cases = self._make_cases(1)
            
            with pytest.raises(RuntimeError, match="Model load failed"):
                run_finetuned_nli(cases, model_path="fake/path", device="cpu")

    def test_handles_inference_error_gracefully(self):
        with patch("transformers.AutoTokenizer") as mock_tokenizer_class, \
             patch("transformers.AutoModelForSequenceClassification") as mock_model_class, \
             patch("torch.device"), \
             patch("torch.cuda.is_available", return_value=False), \
             patch("torch.no_grad"):
            
            # Mock setup
            mock_tokenizer = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_tokenizer.side_effect = RuntimeError("Tokenization failed")
            
            mock_model = MagicMock()
            mock_model_class.from_pretrained.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = None
            
            cases = self._make_cases(2)
            result = run_finetuned_nli(cases, model_path="fake/path", device="cpu")
            
            # Should not raise; failed cases get neutral with 0 confidence
            assert len(result.judgments) == 2
            assert all(j.predicted == "neutral" for j in result.judgments)
            assert all(j.confidence == 0.0 for j in result.judgments)

    def test_progress_callback_called(self):
        with patch("transformers.AutoTokenizer") as mock_tokenizer_class, \
             patch("transformers.AutoModelForSequenceClassification") as mock_model_class, \
             patch("torch.device"), \
             patch("torch.cuda.is_available", return_value=False), \
             patch("torch.no_grad"), \
             patch("torch.softmax") as mock_softmax, \
             patch("torch.argmax") as mock_argmax:
            
            # Mock setup
            mock_tokenizer = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_inputs = MagicMock()
            mock_inputs.to.return_value = mock_inputs
            mock_tokenizer.return_value = mock_inputs
            
            mock_model = MagicMock()
            mock_model_class.from_pretrained.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = None
            
            mock_logits = MagicMock()
            mock_output = MagicMock()
            mock_output.logits = mock_logits
            mock_model.return_value = mock_output
            
            mock_probs = MagicMock()
            mock_probs.squeeze.return_value.tolist.return_value = [0.8, 0.1, 0.05, 0.05]
            mock_softmax.return_value = mock_probs
            
            mock_argmax_result = MagicMock()
            mock_argmax_result.item.return_value = 0
            mock_argmax.return_value = mock_argmax_result
            
            calls = []
            cases = self._make_cases(4)
            run_finetuned_nli(
                cases, 
                model_path="fake/path", 
                device="cpu",
                progress_callback=lambda c, t: calls.append((c, t))
            )
            
            assert len(calls) == 4
            completed = [c for c, _ in calls]
            assert completed == [1, 2, 3, 4]

    def test_latency_recorded(self):
        with patch("transformers.AutoTokenizer") as mock_tokenizer_class, \
             patch("transformers.AutoModelForSequenceClassification") as mock_model_class, \
             patch("torch.device"), \
             patch("torch.cuda.is_available", return_value=False), \
             patch("torch.no_grad"), \
             patch("torch.softmax") as mock_softmax, \
             patch("torch.argmax") as mock_argmax:
            
            # Mock setup
            mock_tokenizer = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_inputs = MagicMock()
            mock_inputs.to.return_value = mock_inputs
            mock_tokenizer.return_value = mock_inputs
            
            mock_model = MagicMock()
            mock_model_class.from_pretrained.return_value = mock_model
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = None
            
            mock_logits = MagicMock()
            mock_output = MagicMock()
            mock_output.logits = mock_logits
            mock_model.return_value = mock_output
            
            mock_probs = MagicMock()
            mock_probs.squeeze.return_value.tolist.return_value = [0.8, 0.1, 0.05, 0.05]
            mock_softmax.return_value = mock_probs
            
            mock_argmax_result = MagicMock()
            mock_argmax_result.item.return_value = 0
            mock_argmax.return_value = mock_argmax_result
            
            cases = self._make_cases(2)
            result = run_finetuned_nli(cases, model_path="fake/path", device="cpu")
            
            assert all(j.latency_ms >= 0 for j in result.judgments)
            assert result.total_latency_ms > 0


# ---------------------------------------------------------------------------
# compare_results
# ---------------------------------------------------------------------------


class TestCompareResults:
    def test_returns_dict_with_models(self):
        r1 = _make_result(["entailed"] * 5, ["entailed"] * 5, model_name="model-a")
        r2 = _make_result(["contradicted"] * 5, ["entailed"] * 5, model_name="model-b")
        comparison = compare_results([r1, r2])
        assert "models" in comparison
        assert len(comparison["models"]) == 2

    def test_model_names_in_comparison(self):
        r1 = _make_result(["entailed"], ["entailed"], model_name="model-a")
        r2 = _make_result(["entailed"], ["entailed"], model_name="model-b")
        comparison = compare_results([r1, r2])
        names = [m["name"] for m in comparison["models"]]
        assert "model-a" in names
        assert "model-b" in names

    def test_category_breakdown_present(self):
        cases = [NLITestCase(id="x", category="notwithstanding", passage="p", claim="c", label="entailed")]
        judgments = [NLIJudgment(case_id="x", predicted="entailed", confidence=0.9, latency_ms=50)]
        result = NLIEvalResult(model_name="m", judgments=judgments, test_cases=cases, total_latency_ms=50)
        comparison = compare_results([result])
        assert "category_breakdown" in comparison
        assert "notwithstanding" in comparison["category_breakdown"]

    def test_accuracy_values_in_range(self):
        r = _make_result(["entailed"] * 4, ["entailed"] * 4)
        comparison = compare_results([r])
        for m in comparison["models"]:
            assert 0.0 <= m["accuracy_binary"] <= 1.0
            assert 0.0 <= m["accuracy_4class"] <= 1.0


# ---------------------------------------------------------------------------
# Edge case content validation
# ---------------------------------------------------------------------------


class TestEdgeCaseContent:
    """Validate that the curated test cases have the expected structure."""

    def test_notwithstanding_cases_exist(self):
        cases = load_edge_cases(EDGE_CASES_PATH)
        nw = [c for c in cases if c.category == "notwithstanding"]
        assert len(nw) >= 3

    def test_shall_vs_may_cases_exist(self):
        cases = load_edge_cases(EDGE_CASES_PATH)
        sm = [c for c in cases if c.category == "shall_vs_may"]
        assert len(sm) >= 3

    def test_mae_cases_exist(self):
        cases = load_edge_cases(EDGE_CASES_PATH)
        mae = [c for c in cases if c.category == "mae_carveout"]
        assert len(mae) >= 3

    def test_knowledge_qualifier_cases_exist(self):
        cases = load_edge_cases(EDGE_CASES_PATH)
        kq = [c for c in cases if c.category == "knowledge_qualifier"]
        assert len(kq) >= 3

    def test_label_distribution_balanced(self):
        """No single label should dominate (> 60% of cases)."""
        cases = load_edge_cases(EDGE_CASES_PATH)
        from collections import Counter
        dist = Counter(c.label for c in cases)
        max_pct = max(v / len(cases) for v in dist.values())
        assert max_pct < 0.60, f"Label distribution too skewed: {dict(dist)}"

    def test_passages_are_legal_text(self):
        cases = load_edge_cases(EDGE_CASES_PATH)
        # Legal passages should mention legal terms
        legal_terms = {"shall", "may", "party", "agreement", "pursuant", "herein",
                       "notwithstanding", "provided", "material", "termination",
                       "knowledge", "represents", "warrant", "liability", "obligation",
                       "closing", "period", "means", "purchas", "licens"}
        for case in cases:
            passage_lower = case.passage.lower()
            has_legal_term = any(t in passage_lower for t in legal_terms)
            assert has_legal_term, f"Case {case.id} passage doesn't look like legal text"
