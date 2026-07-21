#!/usr/bin/env python3
"""
Weight-Space Interpolation — Linearity of R(alpha).

Tests whether linearly interpolating between two checkpoints produces
smooth or nonlinear changes in per-text geodesic ratio R.
A highly nonlinear R(alpha) curve suggests chaotic sensitivity to
weight-space position.

Protocol:
  1. Load two checkpoints.
  2. Create N interpolation points: theta(alpha) = (1-alpha)*theta_A + alpha*theta_B.
  3. For each interpolation point, compute per-text R on N texts.
  4. Analyze: R(alpha) linearity, per-text variability, correlation structure.

Usage:
    python scripts/analysis/weight_interpolation.py \\
        --checkpoint_a checkpoints/model_a/checkpoint-100000 \\
        --checkpoint_b checkpoints/model_b/checkpoint-100000 \\
        --tokenizer tokenizer.json \\
        --data data/train.txt \\
        --num_texts 100 \\
        --num_interp 10 \\
        --output results/weight_interpolation.json

Local test (no GPU):
    python scripts/analysis/weight_interpolation.py \\
        --dry-run
"""

import argparse
import json
import math
import os
import sys
from copy import deepcopy

import numpy as np

# Local imports only if not dry-run
HAS_ELF = False

EPS = 1e-10


# ═══════════════════════════════════════════════════════════════
# 1. Weight-space interpolation
# ═══════════════════════════════════════════════════════════════

def interpolate_weights(state_a: dict, state_b: dict, alpha: float) -> dict:
    """θ(α) = (1-α)·θ_A + α·θ_B"""
    if set(state_a.keys()) != set(state_b.keys()):
        raise ValueError("State dicts have different keys")

    interp = {}
    for key in state_a:
        interp[key] = (1 - alpha) * state_a[key] + alpha * state_b[key]
    return interp


def check_weight_stats(state_a: dict, state_b: dict):
    """Report weight-space statistics for interpretability."""
    keys = sorted(state_a.keys())
    diffs = []
    for k in keys:
        a = state_a[k].float()
        b = state_b[k].float()
        diff = (a - b).norm().item()
        norm_a = a.norm().item()
        diffs.append({"key": k, "diff_norm": diff, "norm_a": norm_a,
                       "rel_diff": diff / max(norm_a, 1e-8)})

    diffs.sort(key=lambda x: -x["rel_diff"])
    return diffs


# ═══════════════════════════════════════════════════════════════
# 2. Per-text R computation (same as T1.5/T1.6)
# ═══════════════════════════════════════════════════════════════

def compute_per_text_r(model, tokenizer, text: str,
                       t_start: float = 0.2, noise_scale: float = 2.0,
                       num_steps: int = 32, num_segments: int = 8,
                       seed: int = 42, max_tokens: int = 80) -> float:
    """Standard per-text R(k) computation."""
    import torch
    import torch.nn.functional as F

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
    t_fake = torch.ones(1, device=device)

    # Euclidean baseline
    alphas = torch.linspace(0, 1, 16, device=device).view(-1, 1, 1)
    z_path = ((1 - alphas) * traj[0, 0].unsqueeze(0)
              + alphas * traj[-1, 0].unsqueeze(0))
    probs = []
    for pt in z_path:
        _, logits = model(pt.unsqueeze(0), t_fake, decoder_step=True)
        probs.append(F.softmax(logits, dim=-1).squeeze(0))
    probs = torch.stack(probs)

    straight_e = 0.0
    for i in range(len(probs) - 1):
        p = probs[i].clamp(min=EPS)
        q = probs[i + 1].clamp(min=EPS)
        m = 0.5 * (p + q)
        straight_e += (0.5 * torch.sum(
            p * (p.log() - m.log()) + q * (q.log() - m.log()))).item()

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
            pj = sp[j].clamp(min=EPS)
            qj = sp[j + 1].clamp(min=EPS)
            mj = 0.5 * (pj + qj)
            ode_e += (0.5 * torch.sum(
                pj * (pj.log() - mj.log()) + qj * (qj.log() - mj.log()))).item()

    return ode_e / max(straight_e, 1e-8)


# ═══════════════════════════════════════════════════════════════
# 3. Analysis: linearity, sensitivity, correlation
# ═══════════════════════════════════════════════════════════════

def analyze_linearity(alphas: list, per_alpha_r: dict):
    """
    Test linearity of R(α). Returns:
      - r² of linear fit
      - max residual (in R-units and %)
      - number of "non-monotonic" texts
    """
    alphas_arr = np.array(alphas)
    n_texts = len(per_alpha_r[alphas[0]])
    texts_stats = []

    for t_idx in range(n_texts):
        r_vals = np.array([per_alpha_r[a][t_idx] for a in alphas])

        # Linear fit
        coeffs = np.polyfit(alphas_arr, r_vals, 1)
        r_pred = np.polyval(coeffs, alphas_arr)
        residuals = r_vals - r_pred
        r2 = 1 - np.sum(residuals**2) / max(
            np.sum((r_vals - np.mean(r_vals))**2), 1e-10)

        # Non-monotonicity
        diffs = np.diff(r_vals)
        n_sign_changes = np.sum(np.abs(np.diff(np.sign(diffs)))) // 2

        # Max absolute residual (as fraction of R range)
        r_range = max(np.max(r_vals) - np.min(r_vals), 1e-4)
        max_abs_residual_pct = np.max(np.abs(residuals)) / r_range * 100

        texts_stats.append({
            "idx": t_idx,
            "r2": round(r2, 4),
            "slope": round(coeffs[0], 4),
            "intercept": round(coeffs[1], 4),
            "r_mean": round(np.mean(r_vals), 4),
            "r_std": round(np.std(r_vals), 4),
            "r_min": round(np.min(r_vals), 4),
            "r_max": round(np.max(r_vals), 4),
            "r_range": round(r_range, 4),
            "max_residual_pct": round(max_abs_residual_pct, 1),
            "n_sign_changes": int(n_sign_changes),
            "r_at_alpha": {
                str(round(a, 1)): round(v, 4)
                for a, v in zip(alphas, r_vals)
            },
        })

    return texts_stats


def analyze_correlation_decay(alphas: list, per_alpha_r: dict):
    """
    How does per-text R correlation between α=0 (Run1) and α=α_i
    decay as α increases?
    """
    from scipy.stats import spearmanr

    base_r = np.array(per_alpha_r[alphas[0]])
    decay = {}
    for alpha in alphas:
        r_vals = np.array(per_alpha_r[alpha])
        rho, p = spearmanr(base_r, r_vals)
        decay[str(round(alpha, 1))] = {
            "spearman_rho": round(rho, 4),
            "p_value": round(float(p), 6),
        }
    return decay


def analyze_nonlinearity_bottleneck(alphas: list, per_alpha_r: dict):
    """
    Find texts that show the strongest nonlinearity.
    These are the most "chaos-sensitive" texts.
    """
    n_texts = len(per_alpha_r[alphas[0]])
    alphas_arr = np.array(alphas)

    nonlinearity_scores = []
    for t_idx in range(n_texts):
        r_vals = np.array([per_alpha_r[a][t_idx] for a in alphas])

        # Curvature metric: deviation from linear interpolation
        linear = np.linspace(r_vals[0], r_vals[-1], len(alphas))
        curvature = np.sum(np.abs(r_vals - linear))

        # Alternative: ratio of actual path length to straight-line distance
        path_len = np.sum(np.abs(np.diff(r_vals)))
        straight_dist = abs(r_vals[-1] - r_vals[0])
        tortuosity = path_len / max(straight_dist, 1e-6)

        nonlinearity_scores.append({
            "idx": t_idx,
            "curvature": round(curvature, 4),
            "tortuosity": round(tortuosity, 4),
        })

    nonlinearity_scores.sort(key=lambda x: -x["curvature"])
    return nonlinearity_scores


# ═══════════════════════════════════════════════════════════════
# 4. Visualization (local run after data collection)
# ═══════════════════════════════════════════════════════════════

def plot_r_vs_alpha(alphas: list, per_alpha_r: dict, output_path: str,
                    max_texts: int = 50):
    """R(α) curves for individual texts + mean ± std."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_texts = len(per_alpha_r[alphas[0]])
    alphas_arr = np.array(alphas)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: individual trajectories
    sample_n = min(n_texts, max_texts)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_texts, sample_n, replace=False)

    all_r = np.zeros((n_texts, len(alphas)))
    for t_idx in range(n_texts):
        all_r[t_idx] = np.array([per_alpha_r[a][t_idx] for a in alphas])

    for idx in sample_idx:
        ax1.plot(alphas_arr, all_r[idx], '-', alpha=0.3, linewidth=0.5,
                 color='steelblue')

    mean_r = np.mean(all_r, axis=0)
    std_r = np.std(all_r, axis=0)
    ax1.plot(alphas_arr, mean_r, 'o-', color='black', linewidth=2.5,
             markersize=6, label="Mean ± std")
    ax1.fill_between(alphas_arr, mean_r - std_r, mean_r + std_r,
                     alpha=0.2, color='black')

    # Linear reference
    ax1.plot([0, 1], [mean_r[0], mean_r[-1]], '--', color='red',
             linewidth=1, alpha=0.5, label="Linear interpolation")

    ax1.set_xlabel("Interpolation α")
    ax1.set_ylabel("Geodesic Ratio R")
    ax1.set_title(f"R(α): Individual Texts (n={sample_n} shown)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: correlation decay
    from scipy.stats import spearmanr
    base_r = all_r[:, 0]
    rho_vals = []
    for a_idx, alpha in enumerate(alphas):
        rho, _ = spearmanr(base_r, all_r[:, a_idx])
        rho_vals.append(rho)

    ax2.plot(alphas_arr, rho_vals, 's-', color='darkred', linewidth=2,
             markersize=8)
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel("Interpolation α")
    ax2.set_ylabel("Spearman ρ (vs α=0)")
    ax2.set_title("Per-Text R Correlation Decay")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.2, 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


def plot_chaos_score_histogram(texts_stats: list, output_path: str):
    """Histogram of (1 - r²) as 'chaos score'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chaos_scores = [1 - s["r2"] for s in texts_stats]
    n_chaotic = sum(1 for s in chaos_scores if s > 0.2)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(chaos_scores, bins=30, color='steelblue', edgecolor='white',
            alpha=0.8)
    ax.axvline(x=0.2, color='red', linestyle='--',
               label=f"Threshold (n_chaotic={n_chaotic})")
    ax.set_xlabel("Chaos Score = 1 - R²(linear fit)")
    ax.set_ylabel("Number of Texts")
    ax.set_title(f"Distribution of Chaos Scores "
                 f"(higher = more nonlinear R(α))")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


# ═══════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="P0-2: Weight-Space Interpolation Perturbation Test")
    parser.add_argument("--checkpoint_a", help="Run1 100K checkpoint dir")
    parser.add_argument("--checkpoint_b", help="Run3 100K checkpoint dir")
    parser.add_argument("--tokenizer", help="Tokenizer path")
    parser.add_argument("--data", help="GSM8K train.txt")
    parser.add_argument("--num_texts", type=int, default=100)
    parser.add_argument("--num_interp", type=int, default=10)
    parser.add_argument("--output", default="results/weight_interpolation.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_ema", action="store_true",
                        help="Load EMA weights (ema_model.pt) instead of training weights")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate logic only (no model loading)")

    # Also support running from saved JSON for plotting only
    parser.add_argument("--plot-only", help="Path to existing JSON for re-plot")

    args = parser.parse_args()

    if args.plot_only:
        # ── Plot-only mode ──
        print("=" * 65)
        print("  P0-2: Plot-only mode")
        print("=" * 65)
        with open(args.plot_only, encoding="utf-8") as f:
            data = json.load(f)

        alphas = [round(a, 1) for a in data["alphas"]]
        per_alpha_r = {}
        for k, v in data["r_by_alpha"].items():
            per_alpha_r[float(k)] = v

        outdir = os.path.dirname(args.output)
        os.makedirs(outdir, exist_ok=True)

        plot_r_vs_alpha(alphas, per_alpha_r,
                        os.path.join(outdir, "r_vs_alpha.png"))
        texts_stats = data.get("texts_stats", [])
        if texts_stats:
            plot_chaos_score_histogram(texts_stats,
                                       os.path.join(outdir, "chaos_histogram.png"))
        return

    if args.dry_run:
        # ── Dry-run: validate logic ──
        print("=" * 65)
        print("  P0-2: DRY RUN — validating logic only")
        print("=" * 65)

        alphas = np.linspace(0, 1, args.num_interp)
        alphas_rounded = [round(float(a), 1) for a in alphas]
        print(f"\n  Interpolation points: {args.num_interp} "
              f"({', '.join(f'{a:.1f}' for a in alphas_rounded)})")

        # Simulate per-text R data
        rng = np.random.RandomState(42)
        n_texts = args.num_texts
        per_alpha_r = {}

        # Scenario A: Highly linear → chaos hypothesis rejected
        print("\n  [Test A] Linear R(α):")
        for alpha_key in alphas_rounded:
            per_alpha_r[alpha_key] = (
                1.8 + 0.3 * alpha_key + rng.normal(0, 0.05, n_texts)).tolist()

        texts_a = analyze_linearity(alphas_rounded, per_alpha_r)
        linear_pct = sum(1 for t in texts_a if t["r2"] > 0.8) / n_texts * 100
        print(f"    Linear texts (r² > 0.8): {linear_pct:.1f}%")

        # Scenario B: Highly nonlinear → chaos hypothesis supported
        print("\n  [Test B] Nonlinear R(α) — chaotic:")
        per_alpha_r2 = {}
        for alpha_key in alphas_rounded:
            chaos_factor = np.sin(alpha_key * np.pi * 3) * 0.5  # oscillates
            per_alpha_r2[alpha_key] = (
                1.8 + 0.3 * alpha_key + chaos_factor
                + rng.normal(0, 0.08, n_texts)).tolist()

        texts_b = analyze_linearity(alphas_rounded, per_alpha_r2)
        chaotic_pct = sum(1 for t in texts_b if t["r2"] < 0.5) / n_texts * 100
        print(f"    Chaotic texts (r² < 0.5): {chaotic_pct:.1f}%")

        # Correlation decay analysis
        decay = analyze_correlation_decay(alphas_rounded, per_alpha_r2)
        print(f"\n  Correlation decay (chaotic scenario):")
        for k, v in decay.items():
            print(f"    α={k}: ρ={v['spearman_rho']:.4f}")

        # Nonlinearity bottleneck
        bottlenecks = analyze_nonlinearity_bottleneck(alphas_rounded, per_alpha_r2)
        print(f"\n  Top 5 nonlinear texts (highest curvature):")
        for b in bottlenecks[:5]:
            print(f"    Text {b['idx']}: curvature={b['curvature']:.4f}  "
                  f"tortuosity={b['tortuosity']:.2f}")

        print(f"\n  Dry-run complete. Script logic validated.")
        print(f"  Run on AutoDL with real checkpoints to collect data.")
        return

    # ── Full run (requires GPU + ELF) ──
    import torch

    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    from elf.config import ELFConfig
    from elf.model import ELFModel
    from elf.data.tokenizer import BPETokenizer

    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    # Load checkpoints
    weight_file = "ema_model.pt" if args.use_ema else "model.pt"
    weight_label = "EMA" if args.use_ema else "training"
    print(f"Loading checkpoint A: {args.checkpoint_a} ({weight_label} weights)")
    with open(os.path.join(args.checkpoint_a, "config.json")) as f:
        config = ELFConfig.from_dict(json.load(f))
    model = ELFModel(config.model).to(device)
    state_a = torch.load(os.path.join(args.checkpoint_a, weight_file),
                         map_location=device, weights_only=False)
    print(f"  State dict keys: {len(state_a)}")

    print(f"Loading checkpoint B: {args.checkpoint_b} ({weight_label} weights)")
    state_b = torch.load(os.path.join(args.checkpoint_b, weight_file),
                         map_location=device, weights_only=False)
    print(f"  State dict keys: {len(state_b)}")

    # Weight-space diagnostics
    print(f"\nWeight-space difference diagnostics:")
    diffs = check_weight_stats(state_a, state_b)
    for d in diffs[:10]:
        print(f"  {d['key']:>40s}: diff={d['diff_norm']:.4f}  "
              f"|θA|={d['norm_a']:.4f}  rel={d['rel_diff']:.4f}")

    # Load texts
    with open(args.data, encoding="utf-8") as f:
        texts = [l.strip() for l in f
                 if 60 < len(l.strip()) < 300][:args.num_texts]
    print(f"\nTexts: {len(texts)}")

    tokenizer = BPETokenizer.load(args.tokenizer, vocab_size=8192)

    # Interpolation loop
    # Fix: use num_interp+1 to get exact [0.0, 0.1, ..., 1.0]
    # Old bug: np.linspace(0,1,10) → [0.0, 0.111..., 0.222..., ..., 1.0]
    #   round(0.555...,1)=0.6, skipping α=0.5 entirely
    alphas = np.linspace(0, 1, args.num_interp + 1)
    per_alpha_r = {}

    print(f"\nInterpolation perturbation test "
          f"({len(alphas)} points: {[round(float(a),1) for a in alphas]})...")

    for idx, alpha in enumerate(alphas):
        alpha_key = round(float(alpha), 1)
        print(f"\n  α = {alpha:.1f} ({idx+1}/{args.num_interp}):")
        interp_state = interpolate_weights(state_a, state_b, alpha)

        alpha_model = ELFModel(config.model).to(device)
        alpha_model.load_state_dict(interp_state)
        alpha_model.eval()

        ratios = []
        for i, text in enumerate(texts):
            r = compute_per_text_r(alpha_model, tokenizer, text)
            ratios.append(r)
            if (i + 1) % 20 == 0:
                recent = ratios[-20:]
                print(f"    [{i+1}/{len(texts)}] "
                      f"mean R = {sum(recent)/len(recent):.3f}")

        per_alpha_r[alpha_key] = ratios
        mean_r = sum(ratios) / len(ratios)
        std_r = np.std(ratios)
        print(f"    DONE: mean R = {mean_r:.4f} ± {std_r:.4f}")

    # Analysis
    print(f"\n{'='*65}")
    print("  Analysis")
    print(f"{'='*65}")

    alphas_list = [round(a, 1) for a in alphas]

    # Mean R(α) trend
    print("\n  Mean R(α):")
    for alpha in alphas_list:
        vals = per_alpha_r[alpha]
        print(f"    α={alpha:.1f}: R = {np.mean(vals):.4f} ± "
              f"{np.std(vals):.4f}")

    # Linearity analysis
    texts_stats = analyze_linearity(alphas_list, per_alpha_r)
    mean_r2 = np.mean([t["r2"] for t in texts_stats])
    median_r2 = np.median([t["r2"] for t in texts_stats])
    chaotic_pct = sum(1 for t in texts_stats if t["r2"] < 0.5) / len(texts) * 100
    print(f"\n  Linearity stats:")
    print(f"    Mean R² = {mean_r2:.4f}")
    print(f"    Median R² = {median_r2:.4f}")
    print(f"    Chaotic texts (R² < 0.5): {chaotic_pct:.1f}%")

    # Correlation decay
    decay = analyze_correlation_decay(alphas_list, per_alpha_r)
    print(f"\n  Correlation decay (Spearman ρ vs α=0):")
    for k, v in decay.items():
        print(f"    α={k}: ρ={v['spearman_rho']:.4f}")

    # Nonlinearity bottleneck
    bottlenecks = analyze_nonlinearity_bottleneck(alphas_list, per_alpha_r)
    print(f"\n  Top 5 nonlinear texts:")
    for b in bottlenecks[:5]:
        print(f"    Text {b['idx']}: curvature={b['curvature']:.4f}  "
              f"tortuosity={b['tortuosity']:.2f}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output = {
        "n_texts": len(texts),
        "n_interp": args.num_interp,
        "alphas": alphas_list,
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "r_by_alpha": {
            str(k): [round(r, 6) for r in v]
            for k, v in per_alpha_r.items()
        },
        "mean_r_by_alpha": {
            str(k): round(np.mean(v), 4)
            for k, v in per_alpha_r.items()
        },
        "std_r_by_alpha": {
            str(k): round(np.std(v), 4)
            for k, v in per_alpha_r.items()
        },
        "linearity_stats": {
            "mean_r2": round(mean_r2, 4),
            "median_r2": round(median_r2, 4),
            "pct_chaotic": round(chaotic_pct, 1),
        },
        "texts_stats": texts_stats,
        "correlation_decay": decay,
        "top_nonlinear": bottlenecks[:10],
        "weight_space_diagnostics": {
            "n_params": len(state_a),
            "top_diffs": [
                {"key": d["key"], "rel_diff": round(d["rel_diff"], 4)}
                for d in diffs[:20]
            ],
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.output}")

    # Plots
    outdir = os.path.dirname(args.output) or "."
    plot_r_vs_alpha(alphas_list, per_alpha_r,
                    os.path.join(outdir, "r_vs_alpha.png"))
    plot_chaos_score_histogram(texts_stats,
                               os.path.join(outdir, "chaos_histogram.png"))

    # Interpretation
    print(f"\n{'='*65}")
    print("  Interpretation")
    print(f"{'='*65}")
    if chaotic_pct > 30:
        print(f"\n  CHAOS HYPOTHESIS SUPPORTED: {chaotic_pct:.1f}% of texts "
              f"show nonlinear R(α).")
        print(f"  The denoising geometry is sensitive to small weight-space "
              f"perturbations — consistent with chaotic dynamics.")
        print(f"  This explains T1.6 ρ≈0: two seeds with similar aggregate R "
              f"produce uncorrelated per-text geometry.")
    elif mean_r2 > 0.8:
        print(f"\n  CHAOS HYPOTHESIS WEAKENED: Mean R² > 0.8. "
              f"R(α) is approximately linear.")
        print(f"  The geometry is stable under weight interpolation — "
              f"inconsistent with chaotic dynamics.")
    else:
        print(f"\n  AMBIGUOUS: mixed linear/nonlinear behavior. "
              f"Further analysis needed.")


if __name__ == "__main__":
    main()
