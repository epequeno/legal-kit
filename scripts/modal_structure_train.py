"""Fine-tune the document structure model on Modal A10G GPU.

Usage:
    modal run scripts/modal_structure_train.py
    modal run scripts/modal_structure_train.py --epochs 5 --model lexlms/legal-roberta-base
"""

import modal
from datetime import datetime

app = modal.App("legalkit-structure-train")

# Default to legal-domain pre-trained RoBERTa
DEFAULT_MODEL = "lexlms/legal-roberta-base"

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
    """Return recommended hyperparams for legal-roberta-base."""
    # legal-roberta-base is a base-sized model
    return {
        'epochs': 3,
        'batch_size': 64,  # lines are short, can fit more
        'learning_rate': 2e-5,
        'max_length': 128,  # single lines rarely exceed 128 tokens
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
    }


@app.function(image=image, volumes={"/vol": vol}, timeout=120)
def upload_training_data():
    """Upload local training data to the Modal volume."""
    from pathlib import Path
    print("Checking structure training data on volume...")
    vol_dir = Path("/vol/structure_model")
    needed = ["train.jsonl", "val.jsonl", "test.jsonl"]
    existing = [f.name for f in vol_dir.glob("*.jsonl")] if vol_dir.exists() else []
    print(f"  Files on volume: {existing}")
    return existing


@app.function(
    image=image,
    gpu="A10G",
    timeout=4500,
    volumes={"/vol": vol},
)
def train(
    base_model: str = DEFAULT_MODEL,
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 2e-5,
    max_length: int = 128,
):
    """Fine-tune legal-roberta-base on document structure detection."""
    import json
    from pathlib import Path
    import time
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EvalPrediction,
    )
    import numpy as np

    # BIO label schema
    LABELS = [
        "O",
        "B-ARTICLE_HEADER", "I-ARTICLE_HEADER",
        "B-SECTION_HEADER", "I-SECTION_HEADER",
        "B-SUBSECTION_HEADER", "I-SUBSECTION_HEADER",
        "B-PREAMBLE", "I-PREAMBLE",
        "B-RECITAL", "I-RECITAL",
        "B-EXHIBIT_HEADER", "I-EXHIBIT_HEADER",
        "B-DEFINITION", "I-DEFINITION",
    ]
    NUM_LABELS = len(LABELS)
    LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}

    class StructureDataset(Dataset):
        def __init__(self, path, tokenizer, max_length=128):
            self.examples = []
            with open(path) as f:
                for line in f:
                    doc = json.loads(line)
                    for line_obj in doc.get("lines", []):
                        self.examples.append({
                            "text": line_obj["text"],
                            "label_id": line_obj["label_id"],
                        })
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            example = self.examples[idx]
            enc = self.tokenizer(
                example["text"],
                truncation=True, max_length=self.max_length,
                padding="max_length", return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": torch.tensor(example["label_id"], dtype=torch.long),
            }

    def compute_metrics(pred: EvalPrediction):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        metrics = {"accuracy": float((preds == labels).mean())}

        # Per-label F1
        for i, label_name in enumerate(LABELS):
            tp = ((preds == i) & (labels == i)).sum()
            fp = ((preds == i) & (labels != i)).sum()
            fn = ((preds != i) & (labels == i)).sum()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            metrics[f"{label_name}_f1"] = float(f1)

        # Macro F1
        metrics["macro_f1"] = sum(metrics[f"{label}_f1"] for label in LABELS) / NUM_LABELS

        # Heading F1 (key metric)
        heading_labels = [
            "B-ARTICLE_HEADER", "I-ARTICLE_HEADER",
            "B-SECTION_HEADER", "I-SECTION_HEADER",
            "B-SUBSECTION_HEADER", "I-SUBSECTION_HEADER",
        ]
        heading_label_ids = [LABEL2ID[label] for label in heading_labels]
        heading_tp = sum(((preds == i) & (labels == i)).sum() for i in heading_label_ids)
        heading_fp = sum(((preds == i) & (labels != i)).sum() for i in heading_label_ids)
        heading_fn = sum(((preds != i) & (labels == i)).sum() for i in heading_label_ids)
        
        heading_p = heading_tp / (heading_tp + heading_fp) if (heading_tp + heading_fp) > 0 else 0
        heading_r = heading_tp / (heading_tp + heading_fn) if (heading_tp + heading_fn) > 0 else 0
        heading_f1 = 2 * heading_p * heading_r / (heading_p + heading_r) if (heading_p + heading_r) > 0 else 0
        
        metrics["heading_f1"] = float(heading_f1)
        return metrics

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device}")

    data_dir = Path("/vol/structure_model")
    if not (data_dir / "train.jsonl").exists():
        raise FileNotFoundError("Structure training data not found on volume. Upload first.")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=NUM_LABELS,
    )

    train_ds = StructureDataset(data_dir / "train.jsonl", tokenizer, max_length)
    val_ds = StructureDataset(data_dir / "val.jsonl", tokenizer, max_length)
    test_ds = StructureDataset(data_dir / "test.jsonl", tokenizer, max_length)
    print(f"Train: {len(train_ds)} lines | Val: {len(val_ds)} lines | Test: {len(test_ds)} lines")

    out_dir = Path("/vol/models/structure-parser")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Effective batch = per_device * gradient_accumulation
    grad_accum = max(1, batch_size // 16)
    micro_batch = min(16, batch_size)

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
        metric_for_best_model="heading_f1",
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
    print(f"\nTest: accuracy={test_results['eval_accuracy']:.4f} heading_f1={test_results['eval_heading_f1']:.4f}")
    for label in ["B-ARTICLE_HEADER", "B-SECTION_HEADER", "B-SUBSECTION_HEADER"]:
        print(f"  {label}: F1={test_results.get(f'eval_{label}_f1', 0):.4f}")

    model_path = out_dir / "model"
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))

    results_dict = {
        "base_model": base_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "training_time_seconds": training_time,
        "timestamp": datetime.now().isoformat(),
        "test_metrics": {k.replace("eval_", ""): v for k, v in test_results.items() if k.startswith("eval_")},
    }
    (out_dir / "results.json").write_text(json.dumps(results_dict, indent=2))
    vol.commit()

    print(f"\nModel saved to volume: legalkit-data/models/structure-parser/model")
    return results_dict


@app.local_entrypoint()
def main(
    epochs: int = 0,          # 0 = use defaults
    batch_size: int = 0,      # 0 = use defaults
    lr: float = 0.0,          # 0.0 = use defaults
    model: str = DEFAULT_MODEL,
) -> None:
    """Run structure model fine-tuning on Modal A10G GPU.
    
    Recommended usage:
      modal run scripts/modal_structure_train.py                        # legal-roberta-base (recommended)
      modal run scripts/modal_structure_train.py --epochs 5 --batch-size 128
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
    print("Uploading structure training data to Modal volume...")
    data_dir = Path("data/structure_model")
    for f in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        local_path = data_dir / f
        if not local_path.exists():
            print(f"ERROR: {local_path} not found. Run 'legalkit structure-annotate' first.")
            return

    # Remove old files first
    for f in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        try:
            vol.remove_file(f"structure_model/{f}")
        except Exception:
            pass

    with vol.batch_upload() as batch:
        for f in ["train.jsonl", "val.jsonl", "test.jsonl"]:
            local_path = data_dir / f
            batch.put_file(str(local_path), f"structure_model/{f}")
            print(f"  Uploaded {f} ({local_path.stat().st_size / 1024:.1f} KB)")
    print()

    # Run training
    print(f"Starting structure model fine-tuning on Modal A10G")
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
    print(f"  Heading F1: {results['test_metrics']['heading_f1']:.4f}")
    for label in ["B-ARTICLE_HEADER", "B-SECTION_HEADER", "B-SUBSECTION_HEADER"]:
        print(f"  {label}: F1={results['test_metrics'].get(f'{label}_f1', 0):.4f}")
    print(f"\nDownload model:")
    print(f"  modal volume get legalkit-data models/structure-parser/model models/structure-parser/model")

    # Save results locally with timestamp
    Path("eval").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = Path('eval') / f'structure-train-{timestamp}.json'
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {results_path}")
    
    # Also save as latest
    latest_path = Path('eval/structure-train-latest.json')
    latest_path.write_text(json.dumps(results, indent=2))
    print(f"  Latest results: {latest_path}")
