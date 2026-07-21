#!/usr/bin/env python3
"""
Weight-Space Lyapunov Analysis.

Computes weight-space divergence rate between two training trajectories
to test for exponential sensitivity — the standard Lyapunov diagnostic
for chaotic dynamics.

Computes:
1. L2 distance ||theta_A - theta_B|| at each checkpoint
2. Component-wise distance breakdown (which layers diverge most)
3. Exponential fit: d(t) ~ d(0) * exp(lambda * t)
4. Effective Lyapunov exponent in the observation window

Usage:
    python scripts/analysis/lyapunov_analysis.py \\
        --checkpoint_a checkpoints/model_a \\
        --checkpoint_b checkpoints/model_b \\
        --steps 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \\
        --output results/lyapunov.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch

EPS = 1e-10


def compute_distance(state_a: dict, state_b: dict) -> dict:
    """Compute L2 distance between two state dicts."""
    total_sq = 0.0
    per_key = {}
    for key in sorted(state_a.keys()):
        a = state_a[key].float()
        b = state_b[key].float()
        sq = (a - b).pow(2).sum().item()
        total_sq += sq
        per_key[key] = np.sqrt(sq)

    return {
        "total_l2": np.sqrt(total_sq),
        "per_key_l2": per_key,
    }


def fit_exponential(steps, distances, d0):
    """Fit d(t) = d0 * exp(lambda * t) via log-linear regression.
    
    log(d/d0) = lambda * t  →  linear regression through origin.
    Returns: lambda, R² of fit.
    """
    steps_arr = np.array(steps, dtype=float)
    log_ratio = np.log(np.maximum(np.array(distances), EPS) / max(d0, EPS))

    # Linear fit: y = lambda * x  (through origin)
    # lambda = sum(x*y) / sum(x²)
    lambda_hat = np.sum(steps_arr * log_ratio) / max(np.sum(steps_arr ** 2), EPS)

    # R²
    pred = lambda_hat * steps_arr
    ss_res = np.sum((log_ratio - pred) ** 2)
    ss_tot = np.sum((log_ratio - np.mean(log_ratio)) ** 2) if np.std(log_ratio) > EPS else 1.0
    r2 = 1 - ss_res / max(ss_tot, EPS)

    return lambda_hat, max(0, r2)


def fit_power_law(steps, distances, d0):
    """Fit d(t) = d0 + a * t^b (power-law divergence — alternative to exponential)."""
    steps_arr = np.array(steps, dtype=float)
    delta_d = np.maximum(np.array(distances) - d0, EPS)

    # log(delta_d) = log(a) + b * log(t)
    log_t = np.log(steps_arr)
    log_d = np.log(delta_d)

    # Linear regression
    n = len(steps_arr)
    mx = np.mean(log_t)
    my = np.mean(log_d)
    b = np.sum((log_t - mx) * (log_d - my)) / max(np.sum((log_t - mx) ** 2), EPS)
    log_a = my - b * mx

    # R²
    pred = np.exp(log_a) * np.power(steps_arr, b)
    ss_res = np.sum((delta_d - pred) ** 2)
    ss_tot = np.sum((delta_d - np.mean(delta_d)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, EPS)

    return np.exp(log_a), b, max(0, r2)


def main():
    parser = argparse.ArgumentParser(
        description="P1: Weight-space Lyapunov analysis")
    parser.add_argument("--checkpoint_a", required=True,
                        help="Run 1 checkpoint dir (vanilla)")
    parser.add_argument("--checkpoint_b", required=True,
                        help="Run 2 checkpoint dir (perturbed)")
    parser.add_argument("--fork_step", type=int, default=5000,
                        help="Step at which perturbation was applied")
    parser.add_argument("--steps", nargs="+", type=int,
                        default=list(range(10000, 110000, 10000)))
    parser.add_argument("--output", default="results/lyapunov.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("=" * 65)
    print("  P1 — Weight-Space Lyapunov Analysis")
    print("=" * 65)
    print(f"  Checkpoint A: {args.checkpoint_a}")
    print(f"  Checkpoint B: {args.checkpoint_b}")
    print(f"  Fork step:    {args.fork_step}")
    print(f"  Steps:        {len(args.steps)} (first={args.steps[0]}, last={args.steps[-1]})")
    print(f"  Device:       {args.device}")
    print("=" * 65)

    device = torch.device(args.device)

    # Compute initial distance at fork point
    fork_dir_a = os.path.join(args.checkpoint_a, f"checkpoint-{args.fork_step}")
    fork_dir_b = os.path.join(args.checkpoint_b, f"checkpoint-{args.fork_step}")

    if not os.path.exists(fork_dir_a) or not os.path.exists(fork_dir_b):
        print(f"\nERROR: Fork checkpoints not found.")
        print(f"  Expected A: {fork_dir_a}")
        print(f"  Expected B: {fork_dir_b}")
        sys.exit(1)

    state_a_fork = torch.load(os.path.join(fork_dir_a, "model.pt"),
                              map_location=device, weights_only=False)
    state_b_fork = torch.load(os.path.join(fork_dir_b, "model.pt"),
                              map_location=device, weights_only=False)

    d0 = compute_distance(state_a_fork, state_b_fork)["total_l2"]
    print(f"\n[0] Initial distance at fork (step {args.fork_step}):")
    print(f"    d0 = {d0:.6f}")

    # Compute distances at each target step
    print(f"\n[1] Computing weight-space distances at each checkpoint...")
    steps_valid = []
    distances = []
    per_key_traces = {}

    for step in args.steps:
        dir_a = os.path.join(args.checkpoint_a, f"checkpoint-{step}")
        dir_b = os.path.join(args.checkpoint_b, f"checkpoint-{step}")

        if not os.path.exists(dir_a) or not os.path.exists(dir_b):
            print(f"    Step {step}: MISSING checkpoint, skipping")
            continue

        state_a = torch.load(os.path.join(dir_a, "model.pt"),
                             map_location=device, weights_only=False)
        state_b = torch.load(os.path.join(dir_b, "model.pt"),
                             map_location=device, weights_only=False)

        dist = compute_distance(state_a, state_b)
        steps_valid.append(step)
        distances.append(dist["total_l2"])

        # Track per-key distances
        for key, val in dist["per_key_l2"].items():
            if key not in per_key_traces:
                per_key_traces[key] = []
            per_key_traces[key].append(val)

        growth_ratio = dist["total_l2"] / max(d0, EPS)
        print(f"    Step {step}: d = {dist['total_l2']:.4f}  "
              f"(×{growth_ratio:.1f} vs d0, "
              f"{growth_ratio ** (1.0 / max((step - args.fork_step), 1)):.4f}/step geometric)")

    if len(steps_valid) < 3:
        print(f"\nERROR: Need at least 3 checkpoints, got {len(steps_valid)}")
        sys.exit(1)

    # Fit models
    print(f"\n[2] Fitting divergence models...")
    t_rel = [s - args.fork_step for s in steps_valid]

    # Exponential fit
    lambda_exp, r2_exp = fit_exponential(steps_valid, distances, d0)
    print(f"\n  Exponential: d(t) = d0 * exp(lambda × t)")
    print(f"    lambda = {lambda_exp:.8f} / step")
    print(f"    R²     = {r2_exp:.4f}")
    print(f"    Doubling time: {np.log(2) / max(lambda_exp, EPS):.0f} steps" if lambda_exp > 0 else "    lambda <= 0, no exponential growth")
    print(f"    Interpretation: {'CHAOTIC (lambda > 0)' if lambda_exp > EPS else 'NON-CHAOTIC (lambda <= 0)'}")

    # Power-law fit
    a_pl, b_pl, r2_pl = fit_power_law(steps_valid, distances, d0)
    print(f"\n  Power-law: d(t) = d0 + a × t^b")
    print(f"    a  = {a_pl:.6f}")
    print(f"    b  = {b_pl:.4f}")
    print(f"    R² = {r2_pl:.4f}")
    if b_pl > 1:
        print(f"    Interpretation: SUPER-LINEAR divergence (b > 1)")
    elif b_pl > 0.5:
        print(f"    Interpretation: sub-diffusive growth (b ~ {b_pl:.2f})")
    else:
        print(f"    Interpretation: sub-diffusive (b < 0.5)")

    # Compare fits
    better_model = "exponential" if r2_exp > r2_pl else "power-law"
    print(f"\n  Best fit: {better_model} (R²_exp={r2_exp:.3f} vs R²_pl={r2_pl:.3f})")

    # Per-component analysis
    print(f"\n[3] Per-component divergence (final distance):")
    final_dists = {}
    for key in sorted(per_key_traces.keys()):
        final_dists[key] = per_key_traces[key][-1]
    sorted_keys = sorted(final_dists.keys(), key=lambda k: -final_dists[k])

    for key in sorted_keys[:10]:
        growth = per_key_traces[key][-1] / max(per_key_traces[key][0], EPS)
        print(f"    {key:>45s}: d={final_dists[key]:.4f} (×{growth:.1f})")

    # Interpretation summary
    print(f"\n{'='*65}")
    print(f"  Interpretation")
    print(f"{'='*65}")

    if lambda_exp > 1e-7 and r2_exp > 0.7:
        print(f"\n  STRONG CHAOS SIGNAL:")
        print(f"    lambda = {lambda_exp:.2e}/step > 0 with R² = {r2_exp:.2f}")
        print(f"    Weight-space divergence is approximately exponential.")
        print(f"    This is the canonical Lyapunov signature of chaos.")
    elif lambda_exp > 1e-7:
        print(f"\n  WEAK CHAOS / NOISE-DRIVEN DIVERGENCE:")
        print(f"    lambda = {lambda_exp:.2e}/step > 0 but R² = {r2_exp:.2f} (weak fit).")
        print(f"    Divergence exists but may be random-walk rather than exponential.")
    else:
        print(f"\n  NO CHAOS SIGNAL:")
        print(f"    lambda <= 0 — weights converge or stay bounded.")
        print(f"    The geometry-level divergence (Spearman rho ~ 0) comes from")
        print(f"    non-chaotic weight-space rearrangement, not exponential sensitivity.")

    if r2_pl > 0.9:
        print(f"\n  Power-law divergence (b={b_pl:.2f}) fits much better (R²={r2_pl:.2f}).")
        print(f"  Suggests the divergence follows a known scaling law rather than chaos.")

    # Save
    output = {
        "fork_step": args.fork_step,
        "d0": round(d0, 6),
        "steps": steps_valid,
        "distances": [round(d, 4) for d in distances],
        "growth_ratios": [round(d / max(d0, EPS), 2) for d in distances],
        "exponential_fit": {
            "lambda_per_step": round(lambda_exp, 10),
            "r2": round(r2_exp, 4),
            "doubling_steps": round(np.log(2) / max(lambda_exp, EPS)) if lambda_exp > 0 else None,
        },
        "power_law_fit": {
            "a": round(a_pl, 6),
            "b": round(b_pl, 4),
            "r2": round(r2_pl, 4),
        },
        "best_model": better_model,
        "top_diverging_components": [
            {"key": k, "final_distance": round(final_dists[k], 4),
             "final_growth_ratio": round(final_dists[k] / max(per_key_traces[k][0], EPS), 2)}
            for k in sorted_keys[:10]
        ],
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
