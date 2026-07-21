#!/usr/bin/env python3
"""
Probability Distance Metric Sensitivity Analysis.

Recomputes R(k) with alternative probability distance metrics
(JS, L2, Cosine, Total Variation) on a subset of checkpoints to
verify that the qualitative pattern and endpoints are robust
to metric choice.

Usage:
    python scripts/analysis/metric_sensitivity.py \\
        --checkpoint_dir checkpoints/model \\
        --tokenizer tokenizer.json \\
        --data data/train.txt \\
        --output results/sensitivity_metric.json \\
        --step_interval 5000
"""

import argparse
import json
import math
import os
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elf.config import ELFConfig
from elf.model import ELFModel
from elf.data.tokenizer import BPETokenizer


# ── Divergence functions ─────────────────────────────────────────
EPS = 1e-10

def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Jensen-Shannon divergence (baseline). Numerically stable."""
    p = p.clamp(min=EPS)
    q = q.clamp(min=EPS)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (p.log() - m.log()))
    kl_qm = torch.sum(q * (q.log() - m.log()))
    return float(0.5 * (kl_pm + kl_qm))


def l2_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """L2 distance in probability space."""
    return float(torch.norm(p - q, p=2))


def cosine_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """Cosine distance: 1 - cos(p, q)."""
    cos = float(F.cosine_similarity(p.flatten(), q.flatten(), dim=0))
    return 1.0 - cos


def tv_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """Total Variation distance: 0.5 * ||p - q||_1."""
    return float(0.5 * torch.norm(p - q, p=1).item())


METRICS = {
    "JS": js_divergence,
    "L2": l2_distance,
    "Cosine": cosine_distance,
    "TV": tv_distance,
}


# ── Core computation ─────────────────────────────────────────────
@torch.no_grad()
def geodesic_energy_with_metric(
    model: ELFModel, z_start: torch.Tensor, z_end: torch.Tensor,
    num_points: int = 16, temperature: float = 1.0,
    metric_fn=js_divergence,
) -> float:
    """Geodesic energy along straight-line interpolation, with custom metric."""
    device = next(model.parameters()).device
    if z_start.dim() == 1:
        z_start = z_start.unsqueeze(0)
    if z_end.dim() == 1:
        z_end = z_end.unsqueeze(0)
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
        total += metric_fn(outputs[i], outputs[i + 1])
    return total


@torch.no_grad()
def compute_ratio_with_metric(
    model: ELFModel, tokenizer: BPETokenizer, text: str,
    t_start: float = 0.2, noise_scale: float = 2.0,
    num_steps: int = 32, num_segments: int = 8,
    metric_fn=js_divergence,
) -> float:
    """Compute R(k) with custom divergence metric."""
    device = next(model.parameters()).device
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

    # Euclidean straight-line baseline
    straight_e = geodesic_energy_with_metric(
        model, traj[0, 0], traj[-1, 0], num_points=16, metric_fn=metric_fn)

    # ODE path energy
    step_size = num_steps // num_segments
    total_ode = 0.0
    for i in range(0, num_steps, step_size):
        e = geodesic_energy_with_metric(
            model, traj[i, 0],
            traj[min(i + step_size, num_steps), 0],
            num_points=8, metric_fn=metric_fn)
        total_ode += e

    return total_ode / max(straight_e, 1e-8)


# ── Main ─────────────────────────────────────────────────────────
def load_checkpoint_weights(ckpt_dir: str, device: torch.device):
    """Load model (training weights) from checkpoint."""
    config_path = os.path.join(ckpt_dir, "config.json")
    with open(config_path) as f:
        config = ELFConfig.from_dict(json.load(f))
    model = ELFModel(config.model).to(device)
    wpath = os.path.join(ckpt_dir, "model.pt")
    model.load_state_dict(torch.load(wpath, map_location=device, weights_only=False))
    model.eval()
    return model, config


def main():
    parser = argparse.ArgumentParser(
        description="Divergence Metric Sensitivity Analysis")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory with checkpoint-* subdirs")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", default=None, help="Text file for eval")
    parser.add_argument("--output", default="sensitivity_metric.json")
    parser.add_argument("--step_interval", type=int, default=5000,
                        help="Evaluate every N steps")
    parser.add_argument("--num_texts", type=int, default=10)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Find checkpoints
    import glob as _g
    ckpt_dirs = sorted(_g.glob(os.path.join(args.checkpoint_dir, "checkpoint-*")))
    ckpt_dirs = [d for d in ckpt_dirs
                 if re.search(r'checkpoint-(\d+)', d)
                 and os.path.exists(os.path.join(d, "config.json"))]
    ckpt_dirs.sort(key=lambda d: int(re.search(r'checkpoint-(\d+)', d).group(1)))

    # Subset by interval
    subset = []
    for d in ckpt_dirs:
        step = int(re.search(r'checkpoint-(\d+)', d).group(1))
        if step % args.step_interval == 0:
            subset.append(d)
    print(f"Checkpoints: {len(subset)}/{len(ckpt_dirs)} "
          f"(interval={args.step_interval})")

    # Load tokenizer & texts
    tokenizer = BPETokenizer.load(args.tokenizer, vocab_size=8192)
    if args.data:
        with open(args.data, encoding="utf-8") as f:
            texts = [l.strip() for l in f
                     if 60 < len(l.strip()) < 300][:args.num_texts]
    else:
        texts = ["Janet's ducks lay 16 eggs per day. She eats 3 for breakfast "
                 "and uses 4 to bake muffins daily. She sells the remainder "
                 "for $2 each. How much does she earn in 5 days?"]
    print(f"Texts: {len(texts)}")

    # Results structure: {metric_name: [(step, mean, std), ...]}
    all_results = {name: [] for name in METRICS}

    for ckpt_dir in subset:
        step = int(re.search(r'checkpoint-(\d+)', ckpt_dir).group(1))
        try:
            model, _ = load_checkpoint_weights(ckpt_dir, device)
        except Exception as e:
            print(f"  step {step:>6d}: SKIP ({e})")
            continue

        for metric_name, metric_fn in METRICS.items():
            ratios = []
            for run in range(args.num_runs):
                torch.manual_seed(42 + run)
                for text in texts:
                    r = compute_ratio_with_metric(
                        model, tokenizer, text, metric_fn=metric_fn)
                    ratios.append(r)

            avg = sum(ratios) / len(ratios)
            std = torch.tensor(ratios).std().item()
            all_results[metric_name].append({
                "step": step, "ratio": round(avg, 4), "std": round(std, 4),
            })

        # Progress
        js_avg = all_results["JS"][-1]["ratio"]
        print(f"  step {step:>6d}: JS={js_avg:.3f}  "
              + "  ".join(f"{n}={all_results[n][-1]['ratio']:.3f}"
                          for n in ["L2", "Cosine", "TV"]))

    # MAD vs JS baseline
    js_ratios = {r["step"]: r["ratio"] for r in all_results["JS"]}
    print(f"\n{'='*65}")
    print("  Mean Absolute Deviation vs JS baseline")
    print(f"{'='*65}")
    for metric_name in ["L2", "Cosine", "TV"]:
        mads = []
        for r in all_results[metric_name]:
            if r["step"] in js_ratios:
                mads.append(abs(r["ratio"] - js_ratios[r["step"]]))
        if mads:
            avg_mad = sum(mads) / len(mads)
            print(f"  {metric_name:8s}: MAD = {avg_mad:.4f}  "
                  f"(max={max(mads):.4f}, n={len(mads)})")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
