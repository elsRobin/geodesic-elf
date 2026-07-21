#!/usr/bin/env python3
"""
Weight Perturbation: Add Gaussian Noise to Checkpoint Weights.

Creates a weight-perturbed copy of a checkpoint for controlled-fork
experiments. Adds isotropic Gaussian noise scaled by relative magnitude.

Usage:
    python scripts/analysis/weight_perturbation.py \\
        --checkpoint checkpoints/model/checkpoint-5000 \\
        --output checkpoints/perturbed/checkpoint-5000 \\
        --noise_scale 1e-4
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elf.config import ELFConfig
from elf.model import ELFModel


def main():
    parser = argparse.ArgumentParser(
        description="Perturb checkpoint weights with Gaussian noise")
    parser.add_argument("--checkpoint", required=True,
                        help="Source checkpoint directory")
    parser.add_argument("--output", required=True,
                        help="Output checkpoint directory")
    parser.add_argument("--noise_scale", type=float, default=1e-4,
                        help="Noise std = scale * ||theta||_2")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load source
    config_path = os.path.join(args.checkpoint, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    print(f"Loading: {args.checkpoint}")
    state = torch.load(os.path.join(args.checkpoint, "model.pt"),
                       map_location=device, weights_only=False)

    # Compute ||theta||_2 for noise scaling
    total_norm = 0.0
    for v in state.values():
        total_norm += v.float().norm() ** 2
    total_norm = total_norm ** 0.5
    noise_std = args.noise_scale * total_norm

    print(f"  ||theta||_2 = {total_norm:.2f}")
    print(f"  Noise std = {args.noise_scale} * {total_norm:.2f} = {noise_std:.6f}")

    # Add noise
    rng = torch.Generator(device=device).manual_seed(42)
    perturbed = {}
    total_added_norm = 0.0
    for key, tensor in state.items():
        noise = torch.randn(tensor.shape, generator=rng,
                            device=device, dtype=tensor.dtype) * noise_std
        perturbed[key] = tensor + noise
        total_added_norm += noise.float().norm() ** 2

    total_added_norm = total_added_norm ** 0.5
    relative_change = total_added_norm / max(total_norm, 1e-8)
    print(f"  ||noise added||_2 = {total_added_norm:.6f}")
    print(f"  Relative change = {relative_change:.6f}")

    # Verify perturbation is present
    orig_norms = {}
    for key in list(state.keys())[:3]:
        diff = (perturbed[key] - state[key]).float().norm().item()
        orig_norms[key] = diff
    print(f"  Sample diffs: { {k: f'{v:.6f}' for k, v in orig_norms.items()} }")

    # Save
    os.makedirs(args.output, exist_ok=True)
    torch.save(perturbed, os.path.join(args.output, "model.pt"))

    # Copy config, optimizer, ema_model, and tokenizer
    import shutil
    shutil.copy2(config_path, os.path.join(args.output, "config.json"))
    opt_path = os.path.join(args.checkpoint, "optimizer.pt")
    if os.path.exists(opt_path):
        shutil.copy2(opt_path, os.path.join(args.output, "optimizer.pt"))
    ema_path = os.path.join(args.checkpoint, "ema_model.pt")
    if os.path.exists(ema_path):
        shutil.copy2(ema_path, os.path.join(args.output, "ema_model.pt"))
    tok_path = os.path.join(args.checkpoint, "tokenizer.json")
    if os.path.exists(tok_path):
        shutil.copy2(tok_path, os.path.join(args.output, "tokenizer.json"))

    print(f"\nSaved perturbed checkpoint to: {args.output}")


if __name__ == "__main__":
    main()
