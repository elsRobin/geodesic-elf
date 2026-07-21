#!/usr/bin/env python3
"""
Cross-Seed Per-Text Geodesic Ratio Correlation.

Tests whether models from different training seeds share similar
geometric structure despite different aggregate R values.

If per-text R values are correlated across models, the 1D R(k)
captures genuine geometric similarity. If uncorrelated, the
aggregate R(k) collapses different geometries into similar numbers.

Usage:
    python scripts/analysis/cross_seed_correlation.py \\
        --checkpoints \\
            checkpoints/run1/checkpoint-100000 \\
            checkpoints/run2/checkpoint-100000 \\
            checkpoints/run3/checkpoint-100000 \\
        --tokenizer tokenizer.json \\
        --data data/train.txt \\
        --num_texts 500 \\
        --metric js \\
        --output results/cross_seed_correlation.json
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


# ── Per-text R (JS Divergence) ─────────────────────────────────
@torch.no_grad()
def compute_per_text_r_js(
    model, tokenizer, text, max_tokens=80,
    noise_scale=2.0, num_steps=32, num_segments=8, seed=42,
):
    """Compute per-text R using JS divergence geodesic energy."""
    device = next(model.parameters()).device
    ids = tokenizer.encode(text)[:max_tokens]
    ids_tensor = torch.tensor([ids], device=device)
    x0 = model.embedding(ids_tensor)

    torch.manual_seed(seed)
    eps = torch.randn_like(x0)
    z = 0.2 * x0 + 0.8 * noise_scale * eps

    dt = 0.8 / num_steps
    trajectory = [z.clone()]
    for s in range(num_steps):
        t_val = 0.2 + s * dt
        t_batch = torch.tensor([t_val], device=device)
        v_pred, _ = model(z, t_batch, decoder_step=False)
        z = z + v_pred * dt
        trajectory.append(z.clone())
    traj = torch.stack(trajectory)

    tf = torch.ones(1, device=device)

    # Euclidean baseline
    alphas = torch.linspace(0, 1, 16, device=device).view(-1, 1, 1)
    zp = (1 - alphas) * traj[0, 0].unsqueeze(0) + alphas * traj[-1, 0].unsqueeze(0)
    probs = []
    for pt in zp:
        _, logits = model(pt.unsqueeze(0), tf, decoder_step=True)
        probs.append(F.softmax(logits, dim=-1).squeeze(0))
    probs = torch.stack(probs)
    straight_e = 0.0
    for i in range(len(probs) - 1):
        p = probs[i].clamp(min=EPS)
        q = probs[i + 1].clamp(min=EPS)
        m = 0.5 * (p + q)
        straight_e += float(0.5 * torch.sum(
            p * (p.log() - m.log()) + q * (q.log() - m.log())))

    # ODE path
    step_size = num_steps // num_segments
    ode_e = 0.0
    for i in range(0, num_steps, step_size):
        end = min(i + step_size, num_steps)
        al = torch.linspace(0, 1, 8, device=device).view(-1, 1, 1)
        zseg = (1 - al) * traj[i, 0].unsqueeze(0) + al * traj[end, 0].unsqueeze(0)
        sp = []
        for pt in zseg:
            _, lg = model(pt.unsqueeze(0), tf, decoder_step=True)
            sp.append(F.softmax(lg, dim=-1).squeeze(0))
        sp = torch.stack(sp)
        for j in range(len(sp) - 1):
            pj = sp[j].clamp(min=EPS)
            qj = sp[j + 1].clamp(min=EPS)
            mj = 0.5 * (pj + qj)
            ode_e += float(0.5 * torch.sum(
                pj * (pj.log() - mj.log()) + qj * (qj.log() - mj.log())))

    return ode_e / max(straight_e, 1e-8)


# ── Per-text R (L2 over softmax probability vectors) ────────────
@torch.no_grad()
def compute_per_text_r_l2(
    model, tokenizer, text, max_tokens=80,
    noise_scale=2.0, num_steps=32, num_segments=8, seed=42,
):
    """Compute per-text R using squared L2 geodesic energy."""
    device = next(model.parameters()).device
    ids = tokenizer.encode(text)[:max_tokens]
    ids_tensor = torch.tensor([ids], device=device)
    x0 = model.embedding(ids_tensor)

    torch.manual_seed(seed)
    eps = torch.randn_like(x0)
    z = 0.2 * x0 + 0.8 * noise_scale * eps

    dt = 0.8 / num_steps
    trajectory = [z.clone()]
    for s in range(num_steps):
        t_val = 0.2 + s * dt
        t_batch = torch.tensor([t_val], device=device)
        v_pred, _ = model(z, t_batch, decoder_step=False)
        z = z + v_pred * dt
        trajectory.append(z.clone())
    traj = torch.stack(trajectory)

    tf = torch.ones(1, device=device)

    # Euclidean baseline via L2 over probs
    alphas = torch.linspace(0, 1, 16, device=device).view(-1, 1, 1)
    zp = (1 - alphas) * traj[0, 0].unsqueeze(0) + alphas * traj[-1, 0].unsqueeze(0)
    outputs = []
    for pt in zp:
        _, logits = model(pt.unsqueeze(0), tf, decoder_step=True)
        outputs.append(F.softmax(logits, dim=-1).squeeze(0))
    outputs = torch.stack(outputs)
    diffs = outputs[1:] - outputs[:-1]
    straight_e = float((diffs ** 2).sum())

    # ODE path via L2 over probs
    step_size = num_steps // num_segments
    ode_e = 0.0
    for i in range(0, num_steps, step_size):
        end = min(i + step_size, num_steps)
        al = torch.linspace(0, 1, 8, device=device).view(-1, 1, 1)
        zseg = (1 - al) * traj[i, 0].unsqueeze(0) + al * traj[end, 0].unsqueeze(0)
        seg_outputs = []
        for pt in zseg:
            _, lg = model(pt.unsqueeze(0), tf, decoder_step=True)
            seg_outputs.append(F.softmax(lg, dim=-1).squeeze(0))
        seg_outputs = torch.stack(seg_outputs)
        seg_diffs = seg_outputs[1:] - seg_outputs[:-1]
        ode_e += float((seg_diffs ** 2).sum())

    return ode_e / max(straight_e, 1e-8)


def load_checkpoint(ckpt_dir, device):
    config_path = os.path.join(ckpt_dir, "config.json")
    with open(config_path) as f:
        config = ELFConfig.from_dict(json.load(f))
    model = ELFModel(config.model).to(device)
    wpath = os.path.join(ckpt_dir, "model.pt")
    model.load_state_dict(torch.load(wpath, map_location=device, weights_only=False))
    model.eval()
    return model


# ── Correlation helpers ─────────────────────────────────────────
def pearson_r(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = math.sqrt(sum((xi - mx) ** 2 for xi in x)
                    * sum((yi - my) ** 2 for yi in y))
    return num / den if den > 0 else 0


def spearman_r(x, y):
    def rank(vals):
        s = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0] * len(vals)
        for r, i in enumerate(s):
            ranks[i] = r
        return ranks
    rx, ry = rank(x), rank(y)
    return pearson_r(rx, ry)


# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Cross-Seed Per-Text R Correlation")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--num_texts", type=int, default=500)
    parser.add_argument("--metric", choices=["js", "l2"], default="js",
                        help="Distance metric: js (Jensen-Shannon) or l2 (squared L2)")
    parser.add_argument("--output", default="results/cross_seed_correlation.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Select metric
    compute_fn = compute_per_text_r_js if args.metric == "js" else compute_per_text_r_l2
    metric_label = "JS divergence" if args.metric == "js" else "L2 over softmax probs"
    print(f"Metric: {metric_label}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load texts
    with open(args.data, encoding="utf-8") as f:
        texts = [l.strip() for l in f
                 if 60 < len(l.strip()) < 300][:args.num_texts]
    print(f"Texts: {len(texts)}")

    tokenizer = BPETokenizer.load(args.tokenizer, vocab_size=8192)

    # Compute per-text R for each checkpoint
    all_r = {}  # {ckpt_name: [r0, r1, ...]}
    names = []

    for ckpt_dir in args.checkpoints:
        name = os.path.basename(os.path.dirname(ckpt_dir))
        label = f"{name}"
        names.append(label)
        print(f"\n  {label}: loading...")

        model = load_checkpoint(ckpt_dir, device)
        ratios = []
        for i, text in enumerate(texts):
            r = compute_fn(model, tokenizer, text,
                           max_tokens=80, seed=42)
            ratios.append(r)
            if (i + 1) % 100 == 0:
                recent = ratios[-100:]
                print(f"    [{i+1}/{len(texts)}] "
                      f"mean R = {sum(recent)/len(recent):.3f}")

        all_r[label] = ratios
        mean_r = sum(ratios) / len(ratios)
        print(f"    mean R = {mean_r:.4f}")

    # ── Pairwise correlations ──
    print(f"\n{'='*65}")
    print(f"  Cross-Seed Per-Text R Correlations ({args.metric.upper()})")
    print(f"{'='*65}")
    print(f"  {'Model A':20s} {'Model B':20s} {'Pearson':>8s} {'Spearman':>8s}")
    print("  " + "-" * 60)

    corr_results = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            x, y = all_r[a], all_r[b]
            pr = pearson_r(x, y)
            sr = spearman_r(x, y)
            corr_results[f"{a} vs {b}"] = {
                "pearson": round(pr, 4), "spearman": round(sr, 4),
            }
            print(f"  {a:20s} {b:20s} {pr:>8.4f} {sr:>8.4f}")

    # Save
    output = {
        "metric": args.metric,
        "texts": len(texts),
        "per_model_r": {
            name: [round(r, 4) for r in ratios]
            for name, ratios in all_r.items()
        },
        "correlations": corr_results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
