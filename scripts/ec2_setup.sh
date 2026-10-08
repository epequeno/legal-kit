#!/bin/bash
# EC2 GPU instance setup for LegalKit GRPO training
#
# Target: g5.xlarge (1× A10G 24GB) with Deep Learning OSS Nvidia Driver AMI
# AMI: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8 (Ubuntu 24.04)
#
# This script is meant to run on the EC2 instance after SSH-ing in.
# Assumes files were uploaded to ~/ (see ec2_launch.sh output).
#
# Usage:
#   bash ec2_setup.sh

set -e

echo "=== LegalKit GRPO Training Setup ==="
echo "Instance: $(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo 'unknown')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU: checking..."

# 1. Python environment — use system Python or venv
echo ""
echo "--- Setting up Python environment ---"

if command -v python3.11 &>/dev/null; then
    PY=python3.11
elif command -v python3 &>/dev/null; then
    PY=python3
else
    echo "ERROR: No python3 found"
    exit 1
fi

echo "Using: $PY ($($PY --version))"

# Create venv if it doesn't exist
if [ ! -d "$HOME/legalkit-env" ]; then
    $PY -m venv "$HOME/legalkit-env"
fi
source "$HOME/legalkit-env/bin/activate"

# 2. Install dependencies
echo ""
echo "--- Installing Python packages ---"
pip install --upgrade pip
pip install \
    "torch>=2.6" \
    "transformers>=4.51" \
    "peft>=0.14" \
    "trl>=0.16" \
    "datasets>=3.2" \
    "accelerate>=1.2" \
    "bitsandbytes>=0.45" \
    "huggingface-hub"

# 3. Pre-download base model
echo ""
echo "--- Pre-downloading Qwen2.5-3B ---"
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)
print('Downloading model...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)
print('Model cached!')
"

# 4. Verify GPU access from PyTorch
echo ""
echo "--- Verifying GPU ---"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('WARNING: No GPU detected!')
"

# 5. Verify uploaded files
echo ""
echo "--- Checking uploaded files ---"
for f in "$HOME/legal-reasoning-sft/adapter_config.json" \
         "$HOME/grpo_prompts.jsonl" \
         "$HOME/train_grpo_local.py"; do
    if [ -f "$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ MISSING: $f"
    fi
done

echo ""
echo "=== Setup complete ==="
echo ""
echo "To train:"
echo "  source ~/legalkit-env/bin/activate"
echo "  tmux new -s train"
echo "  python ~/train_grpo_local.py \\"
echo "    --sft-adapter ~/legal-reasoning-sft \\"
echo "    --prompts ~/grpo_prompts.jsonl \\"
echo "    --output ~/legal-reasoning-grpo"
