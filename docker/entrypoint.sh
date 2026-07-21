#!/bin/bash
# ============================================================
# ELF-S AutoDL Entrypoint Script
#
# Env vars:
#   TRAIN_DATA       — path to training data dir
#   MAX_STEPS        — training steps (default: 50000)
#   BATCH_SIZE       — batch size (default: 32)
#   MAX_SEQ_LEN      — sequence length (default: 256)
#   OUTPUT_DIR       — output dir (default: /root/autodl-tmp/elf-checkpoints)
#   WANDB_API_KEY    — WandB API key (optional)
#   PRETRAIN_DECODE  — "1" for pure CE pretrain mode
#   DECODER_PROB     — decoder branch probability (default: 0.5)
# ============================================================

set -e

AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
OUTPUT_DIR="${OUTPUT_DIR:-${AUTODL_TMP}/elf-checkpoints}"
MAX_STEPS="${MAX_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DECODER_PROB="${DECODER_PROB:-0.5}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

echo "=============================================="
echo "  ELF-S AutoDL Training Launcher"
echo "=============================================="
echo "  Data:       ${TRAIN_DATA:-not set}"
echo "  Max Steps:  ${MAX_STEPS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Seq Length: ${MAX_SEQ_LEN}"
echo "  GPU Count:  $(nvidia-smi -L 2>/dev/null | wc -l || echo '?')"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Decode P:   ${DECODER_PROB}"
echo "=============================================="

mkdir -p "${OUTPUT_DIR}"

# WandB login
if [ -n "${WANDB_API_KEY}" ]; then
    echo "[wandb] Logging in..."
    wandb login "${WANDB_API_KEY}" 2>/dev/null || true
fi

cd /workspace/ELF_mini

# Build CLI args
ARGS="--output_dir ${OUTPUT_DIR} --max_steps ${MAX_STEPS} --batch_size ${BATCH_SIZE} --decoder_prob ${DECODER_PROB} --use_wandb"

if [ -n "${WANDB_RUN_NAME}" ]; then
    ARGS="${ARGS} --wandb_project ${WANDB_PROJECT:-elf-s}"
fi

if [ "${PRETRAIN_DECODE}" = "1" ]; then
    ARGS="${ARGS} --pretrain_decode"
    echo "[Mode] Pure CE Pretrain"
fi

if [ -n "${TRAIN_DATA}" ]; then
    ARGS="${ARGS} --data_dir ${TRAIN_DATA}"
else
    ARGS="${ARGS} --dataset ${DATASET:-tatsu-lab/alpaca}"
fi

echo ""
echo "Launching training..."
echo "  python experiments/train.py ${ARGS}"
echo ""

# Single GPU by default
python experiments/train.py ${ARGS}

echo ""
echo "=============================================="
echo "  Training finished!"
echo "  Checkpoints: ${OUTPUT_DIR}/"
echo "=============================================="

# Optional: run CoT analysis on final checkpoint
if [ "${RUN_COT_ANALYSIS}" = "1" ]; then
    echo ""
    echo "[CoT] Running intermediate-state analysis..."
    python -c "
from elf.config import ELFConfig
from elf.model import ELFModel
from elf.data.tokenizer import BPETokenizer
from experiments.geodesic_analysis import TraceConfig, generate_with_intermediates, print_trace, trace_to_json
import json, os, torch

device = torch.device('cuda')
ckpt_dir = '${OUTPUT_DIR}/checkpoint-final'
with open(os.path.join(ckpt_dir, 'config.json')) as f:
    cfg = ELFConfig.from_dict(json.load(f))
model = ELFModel(cfg.model).to(device)
model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'ema_model.pt'), map_location=device))
tokenizer = BPETokenizer.load('${OUTPUT_DIR}/tokenizer.json', vocab_size=cfg.model.vocab_size)

trace_cfg = TraceConfig(method='sde', num_steps=64, num_snapshots=12, seq_len=128)
trace = generate_with_intermediates(model, tokenizer, trace_cfg, device=device)
print_trace(trace)

with open('${OUTPUT_DIR}/cot_trace.json', 'w') as f:
    f.write(trace_to_json(trace))
print('[CoT] Trace saved.')
"
fi
