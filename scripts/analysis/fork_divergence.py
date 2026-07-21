#!/usr/bin/env python3
"""
Divergence Analysis: Track per-text R correlation across training.

Collects divergence_step*.json outputs from controlled-fork experiments
and produces:
  - rho(k) trace for all model pairs
  - Decay characterization
  - Plot

Usage:
    python scripts/analysis/fork_divergence.py \\
        --json_dir results/divergence \\
        --pattern "divergence_step*.json" \\
        --output results/divergence_analysis.json
"""
import argparse
import json
import glob
import os
import re
import sys

import numpy as np
from scipy.stats import spearmanr


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-text R divergence across checkpoints")
    parser.add_argument("--json_dir", required=True,
                        help="Directory containing divergence_step*.json")
    parser.add_argument("--pattern", default="divergence_step*.json")
    parser.add_argument("--output", default="results/divergence_analysis.json")
    args = parser.parse_args()

    # Find and sort JSONs
    pattern = os.path.join(args.json_dir, args.pattern)
    files = sorted(glob.glob(pattern),
                   key=lambda f: int(re.search(r'step(\d+)', f).group(1)))

    if not files:
        print(f"ERROR: No files matching {pattern}")
        sys.exit(1)

    print(f"Found {len(files)} divergence checkpoints:")
    for f in files:
        step = int(re.search(r'step(\d+)', f).group(1))
        print(f"  step {step}: {os.path.basename(f)}")

    # Collect rho trace
    steps = []
    rho_r1r2 = []
    rho_r1r3 = []
    rho_r2r3 = []

    for f in files:
        step = int(re.search(r'step(\d+)', f).group(1))
        with open(f) as fh:
            data = json.load(fh)

        per_model = data["per_model_r"]
        names = sorted(per_model.keys())
        if len(names) < 3:
            print(f"  WARNING: step {step} has only {len(names)} models, skipping")
            continue

        # Extract per-text R arrays
        r1 = np.array(per_model[names[0]])
        r2 = np.array(per_model[names[1]])
        r3 = np.array(per_model[names[2]])

        r12, p12 = spearmanr(r1, r2)
        r13, p13 = spearmanr(r1, r3)
        r23, p23 = spearmanr(r2, r3)

        steps.append(step)
        rho_r1r2.append({"rho": round(r12, 4), "p": round(float(p12), 6)})
        rho_r1r3.append({"rho": round(r13, 4), "p": round(float(p13), 6)})
        rho_r2r3.append({"rho": round(r23, 4), "p": round(float(p23), 6)})

    # Print trace
    print(f"\n{'='*70}")
    print(f"  Per-Text R Correlation Trace")
    print(f"{'='*70}")
    print(f"  {'Step':>8s}  {'Run1-Run2 (weight)':>22s}  "
          f"{'Run1-Run3 (data)':>22s}  {'Run2-Run3':>22s}")
    print(f"  {'':>8s}  {'rho':>8s}  {'p':>12s}  "
          f"{'rho':>8s}  {'p':>12s}  {'rho':>8s}  {'p':>12s}")
    print("  " + "-" * 68)

    for i, step in enumerate(steps):
        r12 = rho_r1r2[i]
        r13 = rho_r1r3[i]
        r23 = rho_r2r3[i]
        print(f"  {step:>8d}  {r12['rho']:>8.4f}  {r12['p']:>12.6f}  "
              f"{r13['rho']:>8.4f}  {r13['p']:>12.6f}  "
              f"{r23['rho']:>8.4f}  {r23['p']:>12.6f}")

    # Decay characterization
    print(f"\n{'='*70}")
    print(f"  Decay Characterization")
    print(f"{'='*70}")

    if len(steps) >= 2:
        # Half-life: step at which rho drops below 0.5 (first crossing)
        for label, rho_list in [("Run1-Run2", rho_r1r2), ("Run1-Run3", rho_r1r3)]:
            rho_vals = [r["rho"] for r in rho_list]
            below_half = [steps[i] for i, r in enumerate(rho_vals) if r < 0.5]
            if below_half:
                print(f"  {label}: rho < 0.5 at step {below_half[0]}")
            else:
                print(f"  {label}: rho never drops below 0.5")

        # Initial decay rate: (rho_0 - rho_first_10K) / 5K
        if steps[0] == 10000:
            delta_w = rho_r1r2[0]["rho"]  # rho at 10K
            delta_d = rho_r1r3[0]["rho"]
            decay_rate_w = (1.0 - delta_w) / 5000
            decay_rate_d = (1.0 - delta_d) / 5000
            print(f"  Initial decay (5K→10K):")
            print(f"    Weight perturbation: delta_rho = {1-delta_w:.4f} "
                  f"(rate = {decay_rate_w:.6f} / step)")
            print(f"    Data-order perturbation: delta_rho = {1-delta_d:.4f} "
                  f"(rate = {decay_rate_d:.6f} / step)")

    # Interpretation
    print(f"\n{'='*70}")
    print(f"  Interpretation")
    print(f"{'='*70}")

    # Check if rho decays below 0 (anti-correlation)
    has_negative_r1r2 = any(r["rho"] < 0 for r in rho_r1r2)
    has_negative_r1r3 = any(r["rho"] < 0 for r in rho_r1r3)

    if has_negative_r1r2 or has_negative_r1r3:
        print(f"\n  CHAOS HYPOTHESIS CONFIRMED:")
        if has_negative_r1r2:
            print(f"    Weight perturbation → anti-correlated geometry")
        if has_negative_r1r3:
            print(f"    Data-order perturbation → anti-correlated geometry")
        print(f"    The denoising geometry is sensitive to both weight-space")
        print(f"    and data-order perturbations at scale {5000} training steps.")
    else:
        if len(steps) > 0:
            final_r12 = rho_r1r2[-1]["rho"]
            final_r13 = rho_r1r3[-1]["rho"]
            if final_r12 < 0.2 and final_r13 < 0.2:
                print(f"\n  CHAOS HYPOTHESIS SUPPORTED:")
                print(f"    Final rho ≈ 0 for both perturbations.")
                print(f"    Geometry decorrelates completely by step {steps[-1]}.")
            elif final_r12 > 0.5 and final_r13 > 0.5:
                print(f"\n  CHAOS HYPOTHESIS WEAKENED:")
                print(f"    Geometry remains correlated (rho > 0.5).")
                print(f"    The system is stable to small perturbations.")
            else:
                print(f"\n  AMBIGUOUS: Mixed signal. Further analysis needed.")

    # Save
    output = {
        "n_steps": len(steps),
        "fork_step": 5000,
        "steps": steps,
        "rho_r1r2_weight": rho_r1r2,
        "rho_r1r3_data": rho_r1r3,
        "rho_r2r3": rho_r2r3,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
