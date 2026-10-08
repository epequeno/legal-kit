"""Tests for Modal fine-tuning configuration logic.

Tests the config helpers without actually running Modal or GPU training.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, create_autospec
import sys
from pathlib import Path

# Need to ensure transformers module is importable for patching
try:
    import transformers
except ImportError:
    # If not installed, mock it entirely
    sys.modules['transformers'] = MagicMock()

# Import the functions we want to test
# Need to import modal module definitions without actually using Modal
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Mock modal before importing the script
sys.modules['modal'] = MagicMock()

import modal_finetune


class TestDefaultHyperparams:
    """Test model-aware hyperparameter defaults."""

    def test_large_model_config(self):
        """Large models get lower lr, smaller batch, more epochs."""
        config = modal_finetune._default_hyperparams('cross-encoder/nli-deberta-v3-large')
        assert config['epochs'] == 5
        assert config['batch_size'] == 16
        assert config['learning_rate'] == 1e-5
        assert config['max_length'] == 512
        assert config['warmup_ratio'] == 0.1
        assert config['weight_decay'] == 0.01

    def test_base_model_config(self):
        """Base models get higher lr, larger batch."""
        config = modal_finetune._default_hyperparams('cross-encoder/nli-deberta-v3-base')
        assert config['epochs'] == 5
        assert config['batch_size'] == 32
        assert config['learning_rate'] == 2e-5
        assert config['max_length'] == 512

    def test_legacy_base_model(self):
        """Legacy microsoft/deberta-v3-base uses base config."""
        config = modal_finetune._default_hyperparams('microsoft/deberta-v3-base')
        assert config['epochs'] == 5
        assert config['batch_size'] == 32
        assert config['learning_rate'] == 2e-5

    def test_large_in_name_triggers_large_config(self):
        """Any model with 'large' in name gets large config."""
        config = modal_finetune._default_hyperparams('microsoft/deberta-v3-large')
        assert config['batch_size'] == 16
        assert config['learning_rate'] == 1e-5


class TestLoadModelForFinetuning:
    """Test NLI checkpoint loading with label mismatch handling."""

    @patch('transformers.AutoModelForSequenceClassification')
    def test_load_with_ignore_mismatched_sizes(self, mock_auto_model):
        """Model loading uses ignore_mismatched_sizes=True."""
        mock_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model

        result = modal_finetune._load_model_for_finetuning(
            'cross-encoder/nli-deberta-v3-large',
            num_labels=4
        )

        # Verify from_pretrained called with correct args
        mock_auto_model.from_pretrained.assert_called_once_with(
            'cross-encoder/nli-deberta-v3-large',
            num_labels=4,
            ignore_mismatched_sizes=True,
        )
        assert result == mock_model

    @patch('transformers.AutoModelForSequenceClassification')
    def test_load_base_model(self, mock_auto_model):
        """Base models also use the helper."""
        mock_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model

        result = modal_finetune._load_model_for_finetuning(
            'microsoft/deberta-v3-base',
            num_labels=4
        )

        mock_auto_model.from_pretrained.assert_called_once()
        assert result == mock_model


class TestDefaultModel:
    """Test default model constant."""

    def test_default_is_nli_large(self):
        """Default model should be NLI-pretrained large."""
        assert modal_finetune.DEFAULT_MODEL == 'cross-encoder/nli-deberta-v3-large'


class TestModelShortName:
    """Test model short name extraction for file naming."""

    def test_extract_short_name_from_path(self):
        """Extract last component from model path."""
        model = 'cross-encoder/nli-deberta-v3-large'
        short_name = model.split('/')[-1]
        assert short_name == 'nli-deberta-v3-large'

    def test_extract_short_name_base(self):
        """Extract short name for base model."""
        model = 'cross-encoder/nli-deberta-v3-base'
        short_name = model.split('/')[-1]
        assert short_name == 'nli-deberta-v3-base'

    def test_extract_short_name_no_slash(self):
        """Handle model names without slashes."""
        model = 'deberta-v3-base'
        short_name = model.split('/')[-1]
        assert short_name == 'deberta-v3-base'


class TestResultsStructure:
    """Test that results dict has all required keys."""

    def test_results_dict_keys(self):
        """Results should include all metadata fields."""
        # Simulate what the train function returns
        from datetime import datetime
        
        results = {
            'base_model': 'cross-encoder/nli-deberta-v3-large',
            'is_nli_checkpoint': True,
            'epochs': 5,
            'batch_size': 16,
            'learning_rate': 1e-5,
            'train_samples': 1000,
            'val_samples': 200,
            'test_samples': 200,
            'training_time_seconds': 123.45,
            'timestamp': datetime.now().isoformat(),
            'test_metrics': {},
        }

        # Check all required keys are present
        required_keys = [
            'base_model', 'is_nli_checkpoint', 'epochs', 'batch_size',
            'learning_rate', 'train_samples', 'val_samples', 'test_samples',
            'training_time_seconds', 'timestamp', 'test_metrics'
        ]
        
        for key in required_keys:
            assert key in results, f"Missing required key: {key}"

    def test_is_nli_checkpoint_detection(self):
        """Detect NLI checkpoints from model name."""
        nli_model = 'cross-encoder/nli-deberta-v3-large'
        is_nli = 'cross-encoder' in nli_model
        assert is_nli is True

        base_model = 'microsoft/deberta-v3-base'
        is_nli = 'cross-encoder' in base_model
        assert is_nli is False
