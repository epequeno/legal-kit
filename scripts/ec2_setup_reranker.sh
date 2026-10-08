#!/bin/bash
# EC2 setup for LegalKit reranker training
#
# Lighter than the GRPO setup — no 7B model download needed.
# sentence-transformers will download bge-reranker-v2-m3 (~570MB) automatically.
#
# Usage (on EC2 after ssh + upload):
#   bash setup_reranker.sh

set -e

echo "=== LegalKit Reranker Training Setup ==="
echo "Instance: $(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo 'unknown')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU: checking..."

# 1. System packages
echo ""
echo "--- System packages ---"
sudo apt-get update -qq
sudo apt-get install -y python3.12-venv python3-pip

# 2. Python venv
echo ""
echo "--- Python environment ---"
if [ ! -d "$HOME/legalkit-env" ]; then
    python3 -m venv "$HOME/legalkit-env"
fi
source "$HOME/legalkit-env/bin/activate"
pip install --upgrade pip -q

# 3. Install dependencies
echo ""
echo "--- Installing packages ---"
pip install -q \
    "torch>=2.6" \
    "sentence-transformers>=5.0" \
    "transformers>=4.40" \
    "accelerate>=1.1.0" \
    "datasets>=2.14" \
    "scikit-learn>=1.3" \
    "boto3" \
    "numpy"

# 4. Verify GPU
echo ""
echo "--- GPU check ---"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('WARNING: No GPU!')
"

# 5. Pre-download base model (saves time when training starts)
echo ""
echo "--- Pre-downloading bge-reranker-v2-m3 ---"
python -c "
from sentence_transformers import CrossEncoder
print('Downloading BAAI/bge-reranker-v2-m3...')
CrossEncoder('BAAI/bge-reranker-v2-m3', num_labels=1)
print('Model cached!')
"

# 6. Check uploaded files
echo ""
echo "--- Checking uploaded files ---"
for f in "$HOME/embedding_training/train.jsonl" \
         "$HOME/embedding_training/val.jsonl" \
         "$HOME/train_reranker_local.py"; do
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f" 2>/dev/null || echo "?")
        echo "  ✓ $f  ($lines lines)"
    else
        echo "  ✗ MISSING: $f"
    fi
done

echo ""
echo "=== Setup complete ==="
echo ""
echo "To start training (fire-and-forget):"
echo "  source ~/legalkit-env/bin/activate"
echo "  nohup python ~/train_reranker_local.py \\"
echo "    --train ~/embedding_training/train.jsonl \\"
echo "    --val   ~/embedding_training/val.jsonl \\"
echo "    --output ~/legal-reranker \\"
echo "    --s3-upload s3://<bucket> \\"
echo "    --shutdown-on-complete > ~/train.log 2>&1 &"
echo "  echo \"PID: \$!\""
