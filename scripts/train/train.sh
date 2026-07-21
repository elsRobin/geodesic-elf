#!/bin/bash
# ============================================================
# ELF-S Unified Training Launcher
#
# Supports local text files or HuggingFace datasets.
# Passes all arguments through to experiments/train.py.
#
# Usage:
#   # Local text data
#   bash scripts/train/train.sh --data_dir ./my_data --max_steps 50000
#
#   # With inline geodesic eval
#   bash scripts/train/train.sh \\
#       --data_dir ./my_data \\
#       --max_steps 100000 \\
#       --eval_geodesic --eval_data ./my_data/train.txt
#
#   # HuggingFace dataset
#   bash scripts/train/train.sh --dataset tatsu-lab/alpaca --max_steps 200000
#
#   # From YAML config
#   bash scripts/train/train.sh --config experiments/configs/gsm8k_dual.yaml
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=============================================="
echo "  ELF-S Training"
echo "  Project: ${PROJECT_DIR}"
echo "  Start:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

cd "${PROJECT_DIR}"
export TOKENIZERS_PARALLELISM=false

# Install dependencies (quiet)
pip install tokenizers pyyaml -q 2>/dev/null
pip install -e . -q 2>/dev/null

# Optional: wandb, datasets, matplotlib
pip install wandb datasets matplotlib scipy rich -q 2>/dev/null

# Pass all arguments to train.py
python experiments/train.py "$@"

echo ""
echo "============================================"
echo "  Done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
