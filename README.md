# Geodesic-ELF

**Geodesic Energy Ratio as a Diagnostic for Continuous Diffusion Language Models.**

ELF-S is a compact (~25.7M) continuous diffusion language model trained with flow matching. This repository provides the model implementation and a suite of geometric diagnostic tools that track how the ODE denoising path evolves in the LM Head's output probability space during training.

## Overview

During flow matching training, a diffusion LM learns a velocity field that transports noise to clean token embeddings. The resulting ODE denoising path traces a curve through the model's output probability space. The **geodesic energy ratio** $R(k)$ quantifies how this curve deviates from a Euclidean straight line at training step $k$:

$$R(k) = \frac{E_{\text{geo}}(\text{ODE path})}{E_{\text{geo}}(\text{Euclidean straight line})}$$

- $R > 1$: the ODE path follows manifold curvature in probability space
- $R < 1$: the path exploits geometric shortcuts through the probability simplex
- $R \approx 1$: the path is geometrically trivial

This repository provides everything needed to train ELF-S from scratch and run the full geometric analysis pipeline.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA-capable GPU recommended (single RTX 4090 / A100)

```bash
pip install torch numpy tokenizers

# Optional: training extras
pip install datasets wandb pyyaml accelerate

# Optional: analysis extras
pip install matplotlib scipy rich
```

## Quick Start

### Train a model

```bash
# Local text data (directory containing .txt files)
bash scripts/train/train.sh \
    --data_dir ./my_data \
    --max_steps 50000 \
    --eval_geodesic --eval_data ./my_data/train.txt

# HuggingFace dataset
bash scripts/train/train.sh \
    --dataset tatsu-lab/alpaca \
    --max_steps 200000

# With YAML config
python experiments/train.py --config experiments/configs/gsm8k_dual.yaml
```

Training automatically trains a BPE tokenizer if one is not already cached.

### Train a tokenizer separately

```bash
python scripts/train/train_tokenizer.py \
    --data_dir ./my_data --output_dir ./tokenizers/my_bpe
```

### Compute geodesic ratio from a checkpoint

```bash
python scripts/eval/compute_geodesic_ratio.py \
    --checkpoint checkpoints/my_model/checkpoint-100000 \
    --tokenizer tokenizer.json \
    --data data/train.txt \
    --num_texts 10 --num_runs 5
```

## Project Structure

```
elf/                              — Core model library
  model.py                        — ELFModel (~25.7M), EMA, ODE/SDE sampling
  config.py                       — Model, training, and generation configs
  modules/
    attention.py                  — RMSNorm, RoPE, BidirectionalAttention
    transformer.py                — SwiGLU FFN, ELFBlock
  diffusion/                      — Time sampling, ODE integration
  training/                       — Loss functions, LR scheduling, training loop
  data/                           — BPETokenizer, dataset loaders

experiments/
  train.py                        — Unified training entry point (YAML/CLI)
  configs/                        — Example training configs (GSM8K, Alpaca)
  geodesic_analysis/
    geodesic.py                   — Geodesic energy computation
    diagnostics.py                — Intermediate-state decoding and tracing
    visualize.py                  — Visualization utilities

scripts/
  train/
    train.sh                      — Unified training launcher
    train_tokenizer.py            — Standalone BPE tokenizer training
    download_gsm8k.py             — Download and format GSM8K dataset
  eval/
    compute_geodesic_ratio.py     — Compute R(k) from a saved checkpoint
    batch_eval_geodesic.py        — Batch R(k) evaluation across checkpoints
    compare_ema_inline.py         — Compare EMA vs training-weight R(k)
  analysis/
    cluster_trajectories.py       — Per-text R trajectory clustering
    weight_interpolation.py       — Weight-space interpolation R(alpha)
    weight_perturbation.py        — Gaussian weight-noise perturbation
    lyapunov_analysis.py          — Weight-space divergence rate analysis
    fork_divergence.py            — Controlled-fork per-text R correlation
    cross_seed_correlation.py     — Cross-seed per-text R correlation (JS/L2)
    metric_sensitivity.py         — Distance metric sensitivity (JS/L2/Cos/TV)
    discretization_sensitivity.py — ODE step-count convergence check
    phase_transition_profile.py   — Per-text R distribution across training
    gradient_geometry_coupling.py — Gradient norm vs |dR/dk| correlation
    lr_intervention.py            — LR warmup on low-R checkpoints
    downstream_eval.py            — Denoising and decoding loss comparison
    bootstrap_confidence.py       — Bootstrap confidence intervals for R(k)
    plot_analysis.py              — Generate analysis plots
  viz/
    paper_figures.py              — Generate paper figures
    running_example_probs.py      — Visualize probability distributions
    extract_intermediate_probs.py — Extract intermediate LM Head probabilities

docker/                           — Docker build and entrypoint
```

## Architecture

**ELF-S** is a simplified variant of the original ELF model (JAX/Flax), designed for single-GPU research:

| Component | Specification |
|-----------|--------------|
| Parameters | ~25.7M |
| Layers | 6 ELFBlocks |
| Heads | 8 (dim 64 per head) |
| Hidden dim | 512 |
| Vocab size | 8192 (BPE) |
| Attention | Bidirectional, RoPE, QK-Norm |
| FFN | SwiGLU (512 → 1365 → 512) |
| Embedding | Weight-tied with LM Head |
| Optimizer | AdamW (β₁=0.9, β₂=0.98, wd=0.01) |
| LR schedule | Cosine (base 3e-4) |

**Dual-branch training** balances denoising (L2 velocity regression) and decoding (cross-entropy) objectives with `decoder_prob = 0.5`.

## Geodesic Energy Ratio Protocol

The default evaluation protocol:

| Parameter | Value |
|-----------|-------|
| Texts per interval | 10 |
| Noise seeds per text | 5 |
| ODE steps | 32 (Euler) |
| ODE start time | t = 0.2 |
| Decoding points per segment | 4 |
| Distance metric | Squared L2 over softmax probability vectors |
| Evaluation frequency | Every 1K training steps (inline) |

**Dual-protocol design:**
- **Inline** (training weights): primary protocol — captures instantaneous geometric state
- **EMA** (exponential moving average): confirmatory — filters weight noise, reveals geometric memory

## Key API

```python
from elf.config import ELFConfig
from elf.model import ELFModel
from experiments.geodesic_analysis.geodesic import geodesic_energy

# Create model
config = ELFConfig()
model = ELFModel(config.model)

# Compute geodesic energy
energy = geodesic_energy(model, z_start, z_end, num_points=16)
```

## Citation

If you use this code in your research, please cite the repository:

```bibtex
@software{yuan2025geodesic_elf,
  author = {Yuan, Yuyang},
  title = {Geodesic-ELF: Geodesic Energy Ratio Diagnostics for Continuous Diffusion Language Models},
  url = {https://github.com/elsRobin/geodesic-elf},
  year = {2025}
}
```

## License

MIT
