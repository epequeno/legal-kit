"""Fine-tune legal entailment model on Modal GPU.

Usage:
  modal run scripts/modal_finetune.py
  modal run scripts/modal_finetune.py --epochs 5 --batch-size 32
"""

import modal
from datetime import datetime

app = modal.App("legalkit-finetune")

# Default to NLI-pretrained DeBERTa large (best starting point for legal entailment)
DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-large"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers>=4.45",
        "accelerate>=1.1",
        "sentencepiece",
        "protobuf>=4.25,<5",
        "numpy>=1.26",
    )
)

vol = modal.Volume.from_name("legalkit-data", create_if_missing=True)


def _default_hyperparams(model_name: str) -> dict:
    """Return recommended hyperparams based on model size."""
    if "large" in model_name:
        return {
            'epochs': 5,
            'batch_size': 16,   # A10G can handle this for large
            'learning_rate': 1e-5,  # lower lr for large models
            'max_length': 512,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
        }
    else:  # base
        return {
            'epochs': 5,        # more epochs (was 3)
            'batch_size': 32,
            'learning_rate': 2e-5,
            'max_length': 512,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
        }


def _load_model_for_finetuning(model_name: str, num_labels: int = 4):
    """Load model, handling NLI checkpoints with mismatched label count.
    
    For cross-encoder NLI models: load the transformer backbone and
    attach a fresh classification head for num_labels classes.
    Uses AutoModelForSequenceClassification with ignore_mismatched_sizes=True.
    """
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,  # handles 3-label → 4-label head replacement
    )
    return model


@app.function(image=image, volumes={"/vol": vol}, timeout=120)
def upload_training_data():
    """Upload local training data to the Modal volume."""
    from pathlib import Path
    print("Checking training data on volume...")
    vol_dir = Path("/vol/training")
    needed = ["train.jsonl", "val.jsonl", "test.jsonl"]
    existing = [f.name for f in vol_dir.glob("*.jsonl")] if vol_dir.exists() else []
    print(f"  Files on volume: {existing}")
    return existing


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/vol": vol},
)
def train(
    base_model: str = DEFAULT_MODEL,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    max_length: int = 512,
):
    """Fine-tune DeBERTa on legal entailment data."""
    import json
    from pathlib import Path
    import time
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        EvalPrediction,
    )
    import numpy as np

    LABEL_NAMES = ["entailed", "partially_entailed", "contradicted", "neutral"]

    class EntailmentDataset(Dataset):
        def __init__(self, path, tokenizer, max_length=512):
            self.samples = [json.loads(l) for l in open(path)]
            self.tokenizer = tokenizer
            self.max_length = max_length
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            s = self.samples[idx]
            enc = self.tokenizer(
                s["premise"], s["hypothesis"],
                truncation=True, max_length=self.max_length,
                padding="max_length", return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": torch.tensor(s["label"], dtype=torch.long),
            }

    def compute_metrics(pred: EvalPrediction):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        metrics = {"accuracy": float((preds == labels).mean())}
        for i, name in enumerate(LABEL_NAMES):
            tp = ((preds == i) & (labels == i)).sum()
            fp = ((preds == i) & (labels != i)).sum()
            fn = ((preds != i) & (labels == i)).sum()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            metrics[f"{name}_f1"] = float(f1)
        metrics["macro_f1"] = sum(metrics[f"{n}_f1"] for n in LABEL_NAMES) / 4
        return metrics

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device}")

    data_dir = Path("/vol/training")
    if not (data_dir / "train.jsonl").exists():
        raise FileNotFoundError("Training data not found on volume. Upload first.")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = _load_model_for_finetuning(base_model, num_labels=4)

    train_ds = EntailmentDataset(data_dir / "train.jsonl", tokenizer, max_length)
    val_ds = EntailmentDataset(data_dir / "val.jsonl", tokenizer, max_length)
    test_ds = EntailmentDataset(data_dir / "test.jsonl", tokenizer, max_length)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    out_dir = Path("/vol/models/legal-entailment")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Effective batch = per_device * gradient_accumulation
    grad_accum = max(1, batch_size // 8)
    micro_batch = min(8, batch_size)

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=micro_batch,
        per_device_eval_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        save_total_limit=2,
        bf16=True,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print(f"\nTraining {epochs} epochs on {device}...")
    start_time = time.time()
    result = trainer.train()
    training_time = time.time() - start_time
    print(f"Training loss: {result.training_loss:.4f} (took {training_time:.1f}s)")

    test_results = trainer.evaluate(test_ds)
    print(f"\nTest: accuracy={test_results['eval_accuracy']:.4f} macro_f1={test_results['eval_macro_f1']:.4f}")
    for name in LABEL_NAMES:
        print(f"  {name}: F1={test_results.get(f'eval_{name}_f1', 0):.4f}")

    model_path = out_dir / "model"
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))

    results_dict = {
        "base_model": base_model,
        "is_nli_checkpoint": "cross-encoder" in base_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "training_time_seconds": training_time,
        "timestamp": datetime.now().isoformat(),
        "test_metrics": {k.replace("eval_", ""): v for k, v in test_results.items() if k.startswith("eval_")},
    }
    (out_dir / "results.json").write_text(json.dumps(results_dict, indent=2))
    vol.commit()

    print(f"\nModel saved to volume: legalkit-data/models/legal-entailment/model")
    return results_dict


@app.local_entrypoint()
def main(
    epochs: int = 0,          # 0 = use model-aware defaults
    batch_size: int = 0,      # 0 = use model-aware defaults
    lr: float = 0.0,          # 0.0 = use model-aware defaults
    model: str = DEFAULT_MODEL,
    data_dir: str = "data/training",  # path to train/val/test.jsonl
) -> None:
    """Run legal entailment fine-tuning on Modal A10G GPU.
    
    Recommended usage:
      modal run scripts/modal_finetune.py                           # DeBERTa-large NLI (recommended)
      modal run scripts/modal_finetune.py --model cross-encoder/nli-deberta-v3-base  # faster
      modal run scripts/modal_finetune.py --model microsoft/deberta-v3-base          # legacy v0
    """
    import json
    from pathlib import Path

    # Resolve hyperparams from defaults if not explicitly set
    defaults = _default_hyperparams(model)
    if epochs == 0:
        epochs = defaults['epochs']
    if batch_size == 0:
        batch_size = defaults['batch_size']
    if lr == 0.0:
        lr = defaults['learning_rate']

    # Upload training data to volume
    print(f"Uploading training data from {data_dir} to Modal volume...")
    data_dir = Path(data_dir)
    for f in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        local_path = data_dir / f
        if not local_path.exists():
            print(f"ERROR: {local_path} not found. Run 'legalkit prepare-training' first.")
            return

    # Remove old files first
    for f in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        try:
            vol.remove_file(f"training/{f}")
        except Exception:
            pass

    with vol.batch_upload() as batch:
        for f in ["train.jsonl", "val.jsonl", "test.jsonl"]:
            local_path = data_dir / f
            batch.put_file(str(local_path), f"training/{f}")
            print(f"  Uploaded {f} ({local_path.stat().st_size / 1024:.1f} KB)")
    print()

    # Run training
    print(f"Starting fine-tuning on Modal A10G")
    print(f"  Model: {model}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print()

    results = train.remote(
        base_model=model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
    )

    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"  Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"  Macro F1: {results['test_metrics']['macro_f1']:.4f}")
    for name in ["entailed", "partially_entailed", "contradicted", "neutral"]:
        print(f"  {name}: F1={results['test_metrics'].get(f'{name}_f1', 0):.4f}")
    print(f"\nDownload model:")
    print(f"  modal volume get legalkit-data models/legal-entailment/model models/legal-entailment/model")

    # Save results locally with timestamp
    Path("eval").mkdir(exist_ok=True)
    model_short_name = model.split('/')[-1]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = Path('eval') / f'finetune-{model_short_name}-{timestamp}.json'
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {results_path}")
    
    # Also save as latest
    latest_path = Path('eval/finetune-results-latest.json')
    latest_path.write_text(json.dumps(results, indent=2))
    print(f"  Latest results: {latest_path}")
