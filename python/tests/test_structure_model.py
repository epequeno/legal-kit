"""Tests for document structure model training.

Tests the training pipeline without actually running GPU training.
All tests use mocked data and models.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, create_autospec
import sys
from pathlib import Path
import json
import tempfile
import numpy as np

# Mock transformers before importing structure_model
try:
    import transformers
except ImportError:
    sys.modules['transformers'] = MagicMock()
    sys.modules['transformers.training_args'] = MagicMock()

# Import modal_structure_train
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.modules['modal'] = MagicMock()

import modal_structure_train
from src.legalkit.eval.structure_model import (
    LABELS,
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS,
    StructureDataset,
    compute_structure_metrics,
)


@pytest.fixture
def sample_training_data(tmp_path):
    """Create sample JSONL training data."""
    train_file = tmp_path / "train.jsonl"
    
    # Sample document with various line types
    doc = {
        "doc_id": "test_doc_001",
        "file_path": "test.txt",
        "contract_type": "NDA",
        "lines": [
            {"text": "ARTICLE I - CONFIDENTIALITY", "label": "B-ARTICLE_HEADER", "label_id": LABEL2ID["B-ARTICLE_HEADER"], "line_number": 1},
            {"text": "Section 1.1 Definition of Confidential Information", "label": "B-SECTION_HEADER", "label_id": LABEL2ID["B-SECTION_HEADER"], "line_number": 2},
            {"text": "For purposes of this Agreement, \"Confidential Information\" means...", "label": "O", "label_id": LABEL2ID["O"], "line_number": 3},
            {"text": "Section 1.2 Exclusions", "label": "B-SECTION_HEADER", "label_id": LABEL2ID["B-SECTION_HEADER"], "line_number": 4},
            {"text": "Confidential Information shall not include information that:", "label": "O", "label_id": LABEL2ID["O"], "line_number": 5},
        ]
    }
    
    with open(train_file, "w") as f:
        f.write(json.dumps(doc) + "\n")
    
    return train_file


class TestLabels:
    """Test label schema."""
    
    def test_num_labels(self):
        """Should have 15 labels total."""
        assert NUM_LABELS == 15
        assert len(LABELS) == 15
    
    def test_label2id_mapping(self):
        """LABEL2ID should map all labels to IDs."""
        assert len(LABEL2ID) == 15
        assert LABEL2ID["O"] == 0
        assert LABEL2ID["B-ARTICLE_HEADER"] == 1
        assert LABEL2ID["B-SECTION_HEADER"] == 3
    
    def test_id2label_mapping(self):
        """ID2LABEL should be inverse of LABEL2ID."""
        for label, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == label
    
    def test_bio_schema(self):
        """Should have B- and I- tags for each entity type."""
        entity_types = ["ARTICLE_HEADER", "SECTION_HEADER", "SUBSECTION_HEADER", 
                       "PREAMBLE", "RECITAL", "EXHIBIT_HEADER", "DEFINITION"]
        
        for entity in entity_types:
            assert f"B-{entity}" in LABELS
            assert f"I-{entity}" in LABELS


class TestStructureDataset:
    """Test dataset class."""
    
    @patch('transformers.AutoTokenizer')
    def test_dataset_instantiation(self, mock_tokenizer_class, sample_training_data):
        """Dataset should load and flatten lines from JSONL."""
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        dataset = StructureDataset(sample_training_data, mock_tokenizer, max_length=128)
        
        # Should have 5 examples (5 lines in the sample doc)
        assert len(dataset) == 5
        
        # Check first example structure
        assert dataset.examples[0]["text"] == "ARTICLE I - CONFIDENTIALITY"
        assert dataset.examples[0]["label_id"] == LABEL2ID["B-ARTICLE_HEADER"]
        assert dataset.examples[0]["doc_id"] == "test_doc_001"
    
    @patch('transformers.AutoTokenizer')
    def test_dataset_getitem(self, mock_tokenizer_class, sample_training_data):
        """__getitem__ should return tokenized input with labels."""
        import torch
        
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock tokenizer output
        mock_encoding = {
            "input_ids": torch.tensor([[101, 102, 103, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
        }
        mock_tokenizer.return_value = mock_encoding
        
        dataset = StructureDataset(sample_training_data, mock_tokenizer, max_length=128)
        item = dataset[0]
        
        # Check output structure
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        
        # Labels should be tensor
        assert isinstance(item["labels"], torch.Tensor)
        assert item["labels"].item() == LABEL2ID["B-ARTICLE_HEADER"]


class TestComputeStructureMetrics:
    """Test metrics computation."""
    
    def test_perfect_predictions(self):
        """Perfect predictions should have accuracy=1.0 and high F1."""
        from transformers import EvalPrediction
        
        # 10 examples, all predicted correctly
        labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        predictions = np.zeros((10, NUM_LABELS))
        for i, label in enumerate(labels):
            predictions[i, label] = 1.0  # highest score for correct label
        
        pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_structure_metrics(pred)
        
        assert metrics["accuracy"] == 1.0
        # Macro F1 averages over all 15 labels, but we only have examples for 10
        # So F1 for labels 10-14 is 0, bringing down the average
        # Just check that used labels have F1=1.0
        for i in range(10):
            assert metrics[f"{LABELS[i]}_f1"] == 1.0
    
    def test_heading_f1_computation(self):
        """heading_f1 should aggregate ARTICLE/SECTION/SUBSECTION headers."""
        from transformers import EvalPrediction
        
        # Create predictions where heading detection is perfect but other labels are wrong
        labels = np.array([
            LABEL2ID["B-ARTICLE_HEADER"],
            LABEL2ID["B-SECTION_HEADER"],
            LABEL2ID["B-SUBSECTION_HEADER"],
            LABEL2ID["O"],
            LABEL2ID["O"],
        ])
        
        # All heading predictions correct, O predictions wrong
        predictions = np.zeros((5, NUM_LABELS))
        predictions[0, LABEL2ID["B-ARTICLE_HEADER"]] = 1.0
        predictions[1, LABEL2ID["B-SECTION_HEADER"]] = 1.0
        predictions[2, LABEL2ID["B-SUBSECTION_HEADER"]] = 1.0
        predictions[3, LABEL2ID["B-PREAMBLE"]] = 1.0  # wrong
        predictions[4, LABEL2ID["B-RECITAL"]] = 1.0   # wrong
        
        pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_structure_metrics(pred)
        
        # heading_f1 should be 1.0 (all headings correct)
        assert metrics["heading_f1"] == 1.0
        
        # Overall accuracy should be 3/5 = 0.6
        assert metrics["accuracy"] == 0.6
    
    def test_per_label_metrics(self):
        """Should compute precision, recall, F1 for each label."""
        from transformers import EvalPrediction
        
        labels = np.array([0, 0, 1, 1])
        predictions = np.zeros((4, NUM_LABELS))
        predictions[0, 0] = 1.0  # correct
        predictions[1, 1] = 1.0  # wrong (predicted 1, label 0)
        predictions[2, 1] = 1.0  # correct
        predictions[3, 0] = 1.0  # wrong (predicted 0, label 1)
        
        pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_structure_metrics(pred)
        
        # Check O (label 0)
        assert "O_precision" in metrics
        assert "O_recall" in metrics
        assert "O_f1" in metrics
        
        # Check B-ARTICLE_HEADER (label 1)
        assert "B-ARTICLE_HEADER_precision" in metrics
        assert "B-ARTICLE_HEADER_recall" in metrics
        assert "B-ARTICLE_HEADER_f1" in metrics
    
    def test_zero_tp_handling(self):
        """Should handle zero TP gracefully (no division by zero)."""
        from transformers import EvalPrediction
        
        # All predictions wrong
        labels = np.array([0, 0, 0])
        predictions = np.zeros((3, NUM_LABELS))
        predictions[:, 1] = 1.0  # all predicted as label 1
        
        pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_structure_metrics(pred)
        
        # Should not crash, F1 should be 0 for label 0
        assert metrics["O_f1"] == 0.0
        assert np.isfinite(metrics["macro_f1"])


class TestRunStructureFinetuning:
    """Test main training function (mocked)."""
    
    @patch('src.legalkit.eval.structure_model.Trainer')
    @patch('transformers.AutoModelForSequenceClassification')
    @patch('transformers.AutoTokenizer')
    def test_finetuning_completes(self, mock_tokenizer_class, mock_model_class, mock_trainer_class, tmp_path):
        """run_structure_finetuning should complete and return results dict."""
        # Create mock training data
        for split in ["train", "val", "test"]:
            (tmp_path / f"{split}.jsonl").write_text(json.dumps({
                "doc_id": f"doc_{split}",
                "file_path": f"{split}.txt",
                "contract_type": "NDA",
                "lines": [
                    {"text": "ARTICLE I", "label": "B-ARTICLE_HEADER", "label_id": 1, "line_number": 1},
                    {"text": "Body text", "label": "O", "label_id": 0, "line_number": 2},
                ]
            }) + "\n")
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        # Mock train result
        mock_train_result = Mock()
        mock_train_result.training_loss = 0.123
        mock_trainer.train.return_value = mock_train_result
        
        # Mock evaluation results
        mock_eval_results = {
            "eval_accuracy": 0.95,
            "eval_macro_f1": 0.88,
            "eval_heading_f1": 0.92,
            "eval_B-ARTICLE_HEADER_f1": 0.90,
        }
        mock_trainer.evaluate.return_value = mock_eval_results
        
        # Run training
        from src.legalkit.eval.structure_model import run_structure_finetuning
        
        output_dir = tmp_path / "output"
        results = run_structure_finetuning(
            base_model="lexlms/legal-roberta-base",
            training_dir=tmp_path,
            output_dir=output_dir,
            epochs=2,
            batch_size=16,
        )
        
        # Check results dict
        assert "base_model" in results
        assert "epochs" in results
        assert "test_metrics" in results
        assert results["test_metrics"]["accuracy"] == 0.95
        assert results["test_metrics"]["heading_f1"] == 0.92
    
    @patch('src.legalkit.eval.structure_model.Trainer')
    @patch('transformers.AutoModelForSequenceClassification')
    @patch('transformers.AutoTokenizer')
    def test_results_dict_structure(self, mock_tokenizer_class, mock_model_class, mock_trainer_class, tmp_path):
        """Results dict should have all required keys."""
        # Create mock training data
        for split in ["train", "val", "test"]:
            (tmp_path / f"{split}.jsonl").write_text(json.dumps({
                "doc_id": f"doc_{split}",
                "lines": [{"text": "test", "label": "O", "label_id": 0, "line_number": 1}]
            }) + "\n")
        
        # Mock everything
        mock_tokenizer_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()
        
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        mock_trainer.train.return_value = Mock(training_loss=0.1)
        mock_trainer.evaluate.return_value = {
            "eval_accuracy": 0.9,
            "eval_macro_f1": 0.85,
            "eval_heading_f1": 0.88,
        }
        
        from src.legalkit.eval.structure_model import run_structure_finetuning
        
        results = run_structure_finetuning(
            training_dir=tmp_path,
            output_dir=tmp_path / "output",
        )
        
        # Check all required keys
        required_keys = [
            "base_model", "epochs", "batch_size", "learning_rate", "max_length",
            "train_samples", "val_samples", "test_samples",
            "training_time_seconds", "timestamp", "test_metrics"
        ]
        
        for key in required_keys:
            assert key in results, f"Missing required key: {key}"


class TestModalStructureTrainConfig:
    """Test Modal training script configuration."""
    
    def test_default_hyperparams(self):
        """Should return correct defaults for legal-roberta-base."""
        config = modal_structure_train._default_hyperparams("lexlms/legal-roberta-base")
        
        assert config['epochs'] == 3
        assert config['batch_size'] == 64  # lines are short
        assert config['learning_rate'] == 2e-5
        assert config['max_length'] == 128  # single lines
        assert config['warmup_ratio'] == 0.1
        assert config['weight_decay'] == 0.01
    
    def test_default_model(self):
        """Default model should be legal-roberta-base."""
        assert modal_structure_train.DEFAULT_MODEL == "lexlms/legal-roberta-base"


class TestMissingData:
    """Test error handling for missing training data."""
    
    def test_missing_train_file(self, tmp_path):
        """Should raise FileNotFoundError if train.jsonl missing."""
        from src.legalkit.eval.structure_model import run_structure_finetuning
        
        # Only create val and test, not train
        (tmp_path / "val.jsonl").write_text("{}\n")
        (tmp_path / "test.jsonl").write_text("{}\n")
        
        with pytest.raises(FileNotFoundError, match="train.jsonl"):
            run_structure_finetuning(training_dir=tmp_path, output_dir=tmp_path / "output")
    
    def test_missing_val_file(self, tmp_path):
        """Should raise FileNotFoundError if val.jsonl missing."""
        from src.legalkit.eval.structure_model import run_structure_finetuning
        
        (tmp_path / "train.jsonl").write_text("{}\n")
        (tmp_path / "test.jsonl").write_text("{}\n")
        
        with pytest.raises(FileNotFoundError, match="val.jsonl"):
            run_structure_finetuning(training_dir=tmp_path, output_dir=tmp_path / "output")
    
    def test_missing_test_file(self, tmp_path):
        """Should raise FileNotFoundError if test.jsonl missing."""
        from src.legalkit.eval.structure_model import run_structure_finetuning
        
        (tmp_path / "train.jsonl").write_text("{}\n")
        (tmp_path / "val.jsonl").write_text("{}\n")
        
        with pytest.raises(FileNotFoundError, match="test.jsonl"):
            run_structure_finetuning(training_dir=tmp_path, output_dir=tmp_path / "output")
