#!/usr/bin/env python3
"""
ODE Discretization Convergence Check.

Verifies that R(k) is robust to ODE integration step count T,
since numerator and denominator use the same T (bias cancellation).

Tests T in {5, 10, 20, 50} on key checkpoints (early, mid, late).

Usage:
    python scripts/analysis/discretization_sensitivity.py \\
        --checkpoints checkpoints/model/checkpoint-20000 \\
                      checkpoints/model/checkpoint-50000 \\
                      checkpoints/model/checkpoint-80000 \\
        --tokenizer tokenizer.json \\
        --data data/train.txt \\
        --output results/discretization_sensitivity.json
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elf.config import ELFConfig
from elf.model import ELFModel
from elf.data.tokenizer import BPETokenizer

EPS = 1e-10


@torch.no_grad()
def geodesic_energy_js(model: ELFModel, z_start: torch.Tensor,
                       z_end: torch.Tensor, num_points: int = 16,
                       temperature: float = 1.0) -> float:
    """Geodesic energy via JS divergence."""
    device = next(model.parameters()).device
    if z_start.dim() == 1:
        z_start, z_end = z_start.unsqueeze(0), z_end.unsqueeze(0)
    z_start, z_end = z_start.to(device), z_end.to(device)

    num_points = min(num_points, 32)
    alphas = torch.linspace(0, 1, num_points, device=device).view(-1, 1, 1)
    z_path = (1 - alphas) * z_start.unsqueeze(0) + alphas * z_end.unsqueeze(0)

    t_fake = torch.ones(1, device=device)
    outputs = []
    for point in z_path:
        _, logits = model(point.unsqueeze(0), t_fake, decoder_step=True)
        probs = F.softmax(logits / temperature, dim=-1).squeeze(0)
        outputs.append(probs)
    outputs = torch.stack(outputs)

    total = 0.0
    for i in range(len(outputs) - 1):
        p = outputs[i].clamp(min=EPS)
        q = outputs[i + 1].clamp(min=EPS)
        m = 0.5 * (p + q)
        total += float(0.5 * torch.sum(p * (p.log() - m.log())
                                       + q * (q.log() - m.log())))
    return total


@torch.no_grad()
def compute_rk_varying_t(
    model: ELFModel, tokenizer: BPETokenizer, texts: list[str],
    num_steps: int, num_segments: int = 8,
    t_start: float = 0.2, noise_scale: float = 2.0,
) -> float:
    """Compute R(k) with specified ODE step count."""
    device = next(model.parameters()).device
    all_ratios = []

    for text in texts:
        ids = tokenizer.encode(text)[:80]
        ids_tensor = torch.tensor([ids], device=device)
        x0 = model.embedding(ids_tensor)
        eps = torch.randn_like(x0)
        z = t_start * x0 + (1 - t_start) * noise_scale * eps

        dt = (1.0 - t_start) / num_steps
        trajectory = [z.clone()]
        for s in range(num_steps):
            t_val = t_start + s * dt
            t_batch = torch.tensor([t_val], device=device)
            v_pred, _ = model(z, t_batch, decoder_step=False)
            z = z + v_pred * dt
            trajectory.append(z.clone())

        traj = torch.stack(trajectory)
        straight_e = geodesic_energy_js(model, traj[0, 0], traj[-1, 0])

        # Use proportional segment count
        n_seg = max(2, num_steps // 4)
        step_size = num_steps // n_seg
        total_ode = 0.0
        for i in range(0, num_steps, step_size):
            e = geodesic_energy_js(
                model, traj[i, 0],
                traj[min(i + step_size, num_steps), 0], num_points=8)
            total_ode += e

        all_ratios.append(total_ode / max(straight_e, 1e-8))

    return sum(all_ratios) / len(all_ratios)


def load_checkpoint(ckpt_dir: str, device: torch.device):
    config_path = os.path.join(ckpt_dir, "config.json")
    with open(config_path) as f:
        config = ELFConfig.from_dict(json.load(f))
    model = ELFModel(config.model).to(device)
    wpath = os.path.join(ckpt_dir, "model.pt")
    model.load_state_dict(torch.load(wpath, map_location=device, weights_only=False))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="ODE Discretization Convergence Check")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Checkpoint dirs (e.g. 20K, 50K, 80K)")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--output", default="sensitivity_discretization.json")
    parser.add_argument("--num_texts", type=int, default=5)
    parser.add_argument("--t_values", nargs="+", type=int,
                        default=[5, 10, 20, 50])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = BPETokenizer.load(args.tokenizer, vocab_size=8192)
    if args.data:
        with open(args.data, encoding="utf-8") as f:
            texts = [l.strip() for l in f
                     if 60 < len(l.strip()) < 300][:args.num_texts]
    else:
        texts = ["Janet's ducks lay 16 eggs per day. She eats 3 for "
                 "breakfast and uses 4 to bake muffins daily. She sells "
                 "the remainder for $2 each. How much does she earn in "
                 "5 days?"]
    print(f"Texts: {len(texts)}")
    print(f"T values: {args.t_values}")

    results = {}
    for ckpt_dir in args.checkpoints:
        step = os.path.basename(ckpt_dir).replace("checkpoint-", "")
        print(f"\n  Checkpoint step {step}:")
        model = load_checkpoint(ckpt_dir, device)

        ratios = {}
        for t in args.t_values:
            r = compute_rk_varying_t(model, tokenizer, texts, num_steps=t)
            ratios[str(t)] = round(r, 4)
            print(f"    T={t:>3d}: R = {r:.4f}")

        results[str(step)] = ratios

    # Convergence summary
    print(f"\n{'='*65}")
    print("  Convergence Summary (mean % deviation from T=50)")
    print(f"{'='*65}")
    t50_key = "50"
    for step, ratios in results.items():
        base = ratios.get(t50_key, None)
        if base is None:
            continue
        deviations = []
        for t in args.t_values:
            if str(t) in ratios and base > 0:
                dev = 100 * abs(ratios[str(t)] - base) / base
                deviations.append((t, dev))
        if deviations:
            avg_dev = sum(d for _, d in deviations) / len(deviations)
            print(f"  Step {step:>6s}: avg dev = {avg_dev:.1f}%  "
                  + "  ".join(f"T={t}:{d:.1f}%" for t, d in deviations))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
