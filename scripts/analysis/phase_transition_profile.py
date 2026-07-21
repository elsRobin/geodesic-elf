#!/usr/bin/env python3
"""
Per-Text R Distribution Across Geometric Phase Transition.

Characterizes whether a geometric collapse is uniform across all texts
or selective. Computes per-text R on N texts across multiple checkpoints
spanning the transition window.

Protocol:
  - Checkpoints spanning transition (e.g. 50K, 60K, 65K, 70K, 75K, 80K, 90K, 100K)
  - For each, compute per-text R on N training texts (default 500)
  - 1 noise seed per text (trade speed for coverage)
  - Output: per-text R matrix + distribution stats + fragility scores

Usage:
    python scripts/analysis/phase_transition_profile.py \\
        --checkpoint_dir checkpoints/model \\
        --tokenizer tokenizer.json \\
        --data data/train.txt \\
        --steps 50000 60000 70000 80000 90000 100000 \\
        --num_texts 500 \\
        --output results/phase_per_text.json
"""

import argparse
import json
import os
import re
import sys
import math
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elf.config import ELFConfig
from elf.model import ELFModel
from elf.data.tokenizer import BPETokenizer

# ── Numerical stability ─────────────────────────────────────────
EPS = 1e-10


def safe_js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Numerically stable JS divergence.  Clamps to avoid log(0) = -inf."""
    p = p.clamp(min=EPS)
    q = q.clamp(min=EPS)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (p.log() - m.log()))
    kl_qm = torch.sum(q * (q.log() - m.log()))
    return float(0.5 * (kl_pm + kl_qm))


# ── Per-text R(k) computation ───────────────────────────────────
@torch.no_grad()
def compute_per_text_r(
    model: ELFModel, tokenizer: BPETokenizer, text: str,
    t_start: float = 0.2, noise_scale: float = 2.0,
    num_steps: int = 32, num_segments: int = 8,
    seed: int = 42, max_tokens: int = 80,
) -> float:
    """Compute R(k) for a single text with a fixed noise seed."""
    device = next(model.parameters()).device

    ids = tokenizer.encode(text)[:max_tokens]
    ids_tensor = torch.tensor([ids], device=device)
    x0 = model.embedding(ids_tensor)

    torch.manual_seed(seed)
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

    # Euclidean baseline
    alphas = torch.linspace(0, 1, 16, device=device).view(-1, 1, 1)
    z_path = ((1 - alphas) * traj[0, 0].unsqueeze(0)
              + alphas * traj[-1, 0].unsqueeze(0))
    t_fake = torch.ones(1, device=device)
    probs = []
    for pt in z_path:
        _, logits = model(pt.unsqueeze(0), t_fake, decoder_step=True)
        probs.append(F.softmax(logits, dim=-1).squeeze(0))
    probs = torch.stack(probs)

    straight_e = 0.0
    for i in range(len(probs) - 1):
        straight_e += safe_js_divergence(probs[i], probs[i + 1])

    # ODE path
    step_size = num_steps // num_segments
    ode_e = 0.0
    for i in range(0, num_steps, step_size):
        end = min(i + step_size, num_steps)
        al = torch.linspace(0, 1, 8, device=device).view(-1, 1, 1)
        zseg = ((1 - al) * traj[i, 0].unsqueeze(0)
                + al * traj[end, 0].unsqueeze(0))
        sp = []
        for pt in zseg:
            _, lg = model(pt.unsqueeze(0), t_fake, decoder_step=True)
            sp.append(F.softmax(lg, dim=-1).squeeze(0))
        sp = torch.stack(sp)
        for j in range(len(sp) - 1):
            ode_e += safe_js_divergence(sp[j], sp[j + 1])

    return ode_e / max(straight_e, 1e-8)


def load_checkpoint_weights(ckpt_dir: str, device: torch.device):
    """Load model (training weights)."""
    config_path = os.path.join(ckpt_dir, "config.json")
    with open(config_path) as f:
        config = ELFConfig.from_dict(json.load(f))
    model = ELFModel(config.model).to(device)
    wpath = os.path.join(ckpt_dir, "model.pt")
    model.load_state_dict(torch.load(wpath, map_location=device,
                                     weights_only=False))
    model.eval()
    return model


# ── Analysis helpers ─────────────────────────────────────────────
def distribution_stats(values: list[float]) -> dict:
    """Compute distribution statistics."""
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean)**2 for x in values) / (n - 1) if n > 1 else 0
    std = math.sqrt(var)
    s = sorted(values)
    median = s[n // 2]
    q1 = s[n // 4]
    q3 = s[3 * n // 4]
    skew = sum((x - mean)**3 for x in values) / (n * std**3) if std > 0 else 0
    return {
        "n": n, "mean": round(mean, 4), "std": round(std, 4),
        "median": round(median, 4), "q1": round(q1, 4), "q3": round(q3, 4),
        "min": round(min(values), 4), "max": round(max(values), 4),
        "skewness": round(skew, 4),
        # Regime counts
        "n_regime_M": sum(1 for x in values if x > 1.15),
        "n_regime_E": sum(1 for x in values if 0.85 <= x <= 1.15),
        "n_regime_S": sum(1 for x in values if x < 0.85),
    }


def fragility_score(text_idx: int, r_by_step: dict[int, float],
                    pre_steps: list[int], post_steps: list[int]) -> float:
    """Fragility = how much R dropped from pre-collapse to post-collapse.

    Returns negative delta: high positive = very fragile (big drop).
    """
    pre_r = [r_by_step[s] for s in pre_steps if s in r_by_step]
    post_r = [r_by_step[s] for s in post_steps if s in r_by_step]
    if not pre_r or not post_r:
        return 0.0
    return sum(pre_r) / len(pre_r) - sum(post_r) / len(post_r)


# ── Text feature extraction (for fragile/robust comparison) ─────
def extract_text_features(text: str) -> dict:
    """Extract simple features from a GSM8K problem for interpretability."""
    words = text.split()
    numbers = re.findall(r'\d+\.?\d*', text)
    questions = text.count('?')
    steps = len(re.findall(r'(?:first|then|next|after|finally|so|thus)',
                           text, re.IGNORECASE))
    return {
        "n_words": len(words),
        "n_numbers": len(numbers),
        "n_questions": questions,
        "avg_number": sum(float(n) for n in numbers) / len(numbers)
        if numbers else 0,
        "n_steps_hint": steps,
        "text_preview": text[:80],
    }


# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Per-Text R Distribution Across Phase Transition")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory with checkpoint-* subdirs (Run 2)")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True,
                        help="GSM8K train.txt")
    parser.add_argument("--steps", nargs="+", type=int,
                        default=[50000, 60000, 70000, 80000,
                                 90000, 100000],
                        help="Checkpoint steps to evaluate")
    parser.add_argument("--num_texts", type=int, default=500,
                        help="Number of training texts (0=all)")
    parser.add_argument("--noise_seed", type=int, default=42,
                        help="Fixed noise seed per text")
    parser.add_argument("--output", default="phase_per_text.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load texts
    with open(args.data, encoding="utf-8") as f:
        all_texts = [l.strip() for l in f
                     if 60 < len(l.strip()) < 300]
    if args.num_texts > 0:
        all_texts = all_texts[:args.num_texts]
    print(f"Texts: {len(all_texts)}")

    # Extract text features
    text_features = [extract_text_features(t) for t in all_texts]

    # Load tokenizer
    tokenizer = BPETokenizer.load(args.tokenizer, vocab_size=8192)

    # Results: per-step list of per-text R values
    # r_matrix[step_key][text_idx] = R value
    r_matrix = {}  # {step: [r0, r1, ..., rN-1]}

    import glob as _g

    for target_step in args.steps:
        ckpt_path = os.path.join(args.checkpoint_dir,
                                 f"checkpoint-{target_step}")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: {ckpt_path} not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  Step {target_step}")
        print(f"{'='*60}")

        model = load_checkpoint_weights(ckpt_path, device)
        ratios = []
        n_done = 0

        for i, text in enumerate(all_texts):
            r = compute_per_text_r(
                model, tokenizer, text, seed=args.noise_seed)
            ratios.append(r)
            n_done += 1

            if (i + 1) % 100 == 0:
                recent = ratios[-100:]
                print(f"  [{i+1}/{len(all_texts)}] "
                      f"mean R = {sum(recent)/len(recent):.3f}  "
                      f"(last 100)")

        r_matrix[str(target_step)] = ratios
        stats = distribution_stats(ratios)
        print(f"  Done: mean={stats['mean']:.4f} ± {stats['std']:.4f}  "
              f"median={stats['median']:.4f}  "
              f"M:{stats['n_regime_M']} E:{stats['n_regime_E']} "
              f"S:{stats['n_regime_S']}")

    # ── Cross-step analysis ──
    print(f"\n{'='*60}")
    print("  Cross-Step Analysis")
    print(f"{'='*60}")

    # Distribution shift (mean, std, skewness over steps)
    print("\n  Distribution shift across phase transition:")
    print(f"  {'Step':>8s}  {'Mean':>8s}  {'Std':>8s}  {'Median':>8s}  "
          f"{'Skew':>8s}  {'%M':>6s}  {'%S':>6s}  {'IQR':>8s}")
    print("  " + "-" * 72)

    step_stats = {}
    for step_key in sorted(r_matrix.keys(), key=int):
        vals = r_matrix[step_key]
        stats = distribution_stats(vals)
        step_stats[step_key] = stats
        pct_m = 100 * stats['n_regime_M'] / stats['n']
        pct_s = 100 * stats['n_regime_S'] / stats['n']
        iqr = stats['q3'] - stats['q1']
        print(f"  {int(step_key):>8d}  {stats['mean']:>8.4f}  "
              f"{stats['std']:>8.4f}  {stats['median']:>8.4f}  "
              f"{stats['skewness']:>8.4f}  {pct_m:>5.1f}%  "
              f"{pct_s:>5.1f}%  {iqr:>8.4f}")

    # Fragility analysis: which texts dropped the most?
    print(f"\n  Fragility analysis (ΔR = R_60K - R_80K):")

    pre_steps = ['60000']
    post_steps = ['80000']
    if '60000' in r_matrix and '80000' in r_matrix:
        # Build per-text R dicts
        n_texts = len(all_texts)
        per_text_r = {}
        for step_key in r_matrix:
            if len(r_matrix[step_key]) == n_texts:
                per_text_r[int(step_key)] = {
                    i: r_matrix[step_key][i] for i in range(n_texts)
                }

        # Compute fragility for each text
        fragilities = []
        for i in range(n_texts):
            r_by_step = {
                step: per_text_r[step].get(i, 0)
                for step in per_text_r
            }
            score = fragility_score(i, r_by_step,
                                    [60000], [80000])
            fragilities.append((i, score))

        fragilities.sort(key=lambda x: -x[1])  # most fragile first

        # Top 10 fragile texts
        print("\n  Top 10 most FRAGILE texts (largest R drop):")
        for rank, (idx, score) in enumerate(fragilities[:10]):
            r_60 = per_text_r.get(60000, {}).get(idx, 0)
            r_80 = per_text_r.get(80000, {}).get(idx, 0)
            f = text_features[idx]
            print(f"    #{rank+1}: ΔR={score:.3f}  "
                  f"(R_60K={r_60:.3f} → R_80K={r_80:.3f})  "
                  f"words={f['n_words']}  nums={f['n_numbers']}  "
                  f"q={f['n_questions']}  steps={f['n_steps_hint']}  "
                  f"'{f['text_preview']}...'")

        # Top 10 robust texts
        print("\n  Top 10 most ROBUST texts (least R drop / increase):")
        robust = sorted(fragilities, key=lambda x: x[1])[:10]
        for rank, (idx, score) in enumerate(robust):
            r_60 = per_text_r.get(60000, {}).get(idx, 0)
            r_80 = per_text_r.get(80000, {}).get(idx, 0)
            f = text_features[idx]
            print(f"    #{rank+1}: ΔR={score:.3f}  "
                  f"(R_60K={r_60:.3f} → R_80K={r_80:.3f})  "
                  f"words={f['n_words']}  nums={f['n_numbers']}  "
                  f"q={f['n_questions']}  steps={f['n_steps_hint']}  "
                  f"'{f['text_preview']}...'")

        # Aggregate feature comparison
        print("\n  Feature comparison: Fragile vs Robust texts:")
        fragile_group = fragilities[:50]
        robust_group = sorted(fragilities, key=lambda x: x[1])[:50]
        for feature_name in ["n_words", "n_numbers", "n_questions",
                             "avg_number", "n_steps_hint"]:
            f_vals = [text_features[i][feature_name]
                      for i, _ in fragile_group]
            r_vals = [text_features[i][feature_name]
                      for i, _ in robust_group]
            f_mean = sum(f_vals) / len(f_vals)
            r_mean = sum(r_vals) / len(r_vals)
            diff_sign = "↑" if f_mean > r_mean else "↓"
            print(f"    {feature_name:>16s}: fragile={f_mean:.1f}  "
                  f"robust={r_mean:.1f}  {diff_sign}")

    # ── Save ──
    output = {
        "config": {
            "checkpoint_dir": args.checkpoint_dir,
            "steps": [int(s) for s in r_matrix.keys()],
            "num_texts": len(all_texts),
            "noise_seed": args.noise_seed,
        },
        "per_step_stats": {
            step: step_stats[step]
            for step in sorted(r_matrix.keys(), key=int)
            if step in step_stats
        },
        "r_matrix": {
            step: ratios
            for step, ratios in r_matrix.items()
        },
        "text_features": [
            {"idx": i, **text_features[i]}
            for i in range(len(all_texts))
        ],
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
