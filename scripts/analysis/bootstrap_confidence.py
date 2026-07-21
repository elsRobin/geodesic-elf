#!/usr/bin/env python3
"""
Bootstrap Confidence Intervals + Effect Size for R(k) Late-Phase Values.

Reads geodesic JSON files from training, computes:
  - Per-model late-phase (80-100% of training) mean ± std
  - Bootstrap 95% CIs (10k resamples)
  - Cohen's d for regime gaps
  - Non-overlapping CI test

Usage:
    python scripts/analysis/bootstrap_confidence.py \\
        --input geodesic_inline_run1.json \\
               geodesic_inline_run2.json \\
               geodesic_inline_run3.json \\
        --late_frac 0.8 \\
        --output results/bootstrap.json
"""

import argparse
import json
import math
import random
import sys
from typing import Sequence


def load_ratios(filepath: str, late_frac: float = 0.8) -> tuple[list[float], str]:
    """Extract late-phase ratios from geodesic JSON."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    ratios = [(r["step"], r["ratio"]) for r in data]
    max_step = ratios[-1][0]
    cutoff = int(max_step * late_frac)
    late = [r for step, r in ratios if step >= cutoff]
    return late, filepath


def bootstrap_ci(values: Sequence[float], n_resamples: int = 10000,
                 alpha: float = 0.05) -> dict:
    """Bootstrap 95% CI for mean."""
    means = []
    n = len(values)
    for _ in range(n_resamples):
        sample = [random.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_resamples * alpha / 2)]
    hi = means[int(n_resamples * (1 - alpha / 2))]
    mean = sum(values) / len(values)
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "n": n}


def cohens_d(group1: Sequence[float], group2: Sequence[float]) -> float:
    """Cohen's d = (m1 - m2) / pooled_std."""
    m1, m2 = sum(group1) / len(group1), sum(group2) / len(group2)
    n1, n2 = len(group1), len(group2)
    v1 = sum((x - m1) ** 2 for x in group1) / (n1 - 1) if n1 > 1 else 0
    v2 = sum((x - m2) ** 2 for x in group2) / (n2 - 1) if n2 > 1 else 0
    pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled == 0:
        return float("inf")
    return abs(m1 - m2) / pooled


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap CI for geodesic ratio late-phase")
    parser.add_argument("--input", nargs="+", required=True,
                        help="Geodesic JSON files")
    parser.add_argument("--late_frac", type=float, default=0.8,
                        help="Fraction of training for late-phase window")
    parser.add_argument("--output", default="bootstrap_results.json")
    parser.add_argument("--n_resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 65)
    print("  Bootstrap CI Analysis — R(k) Late-Phase")
    print("=" * 65)
    print(f"  Late-phase window: {args.late_frac*100:.0f}%-100% of training")
    print(f"  Bootstrap resamples: {args.n_resamples}")
    print()

    all_late = {}
    for fp in args.input:
        vals, name = load_ratios(fp, args.late_frac)
        ci = bootstrap_ci(vals, args.n_resamples)
        label = name.split("_")[-2] if "RUN" in name else name[-20:]
        all_late[label] = {"file": name, "values": vals, "ci": ci}
        print(f"  {label:8s}: mean={ci['mean']:.4f}  "
              f"95% CI=[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]  "
              f"(n={ci['n']})")

    # Identify M vs S groups
    # Run1 and Run3 are Regime M; Run2 is Regime S
    m_vals, s_vals = [], []
    for label, info in all_late.items():
        if "2" in label and "RUN" in label:
            s_vals.extend(info["values"])
        else:
            m_vals.extend(info["values"])

    if m_vals and s_vals:
        m_ci = bootstrap_ci(m_vals, args.n_resamples)
        s_ci = bootstrap_ci(s_vals, args.n_resamples)
        d = cohens_d(m_vals, s_vals)

        print(f"\n{'='*65}")
        print("  Regime Comparison")
        print(f"{'='*65}")
        print(f"  Regime M (pooled): mean={m_ci['mean']:.4f}  "
              f"CI=[{m_ci['ci_low']:.4f}, {m_ci['ci_high']:.4f}]")
        print(f"  Regime S (pooled): mean={s_ci['mean']:.4f}  "
              f"CI=[{s_ci['ci_low']:.4f}, {s_ci['ci_high']:.4f}]")
        print(f"  ΔR (M - S):        {m_ci['mean'] - s_ci['mean']:.4f}")
        print(f"  Cohen's d:          {d:.2f}")
        print(f"  CIs overlap:        {'YES' if m_ci['ci_low'] < s_ci['ci_high']
              and s_ci['ci_low'] < m_ci['ci_high'] else 'NO'}")

    # Save
    output = {
        "late_frac": args.late_frac,
        "n_resamples": args.n_resamples,
        "per_seed": {
            label: {"mean": info["ci"]["mean"],
                    "ci": [info["ci"]["ci_low"], info["ci"]["ci_high"]],
                    "n": info["ci"]["n"]}
            for label, info in all_late.items()
        },
        "regime_comparison": {
            "M_mean": m_ci["mean"], "M_ci": [m_ci["ci_low"], m_ci["ci_high"]],
            "S_mean": s_ci["mean"], "S_ci": [s_ci["ci_low"], s_ci["ci_high"]],
            "delta": m_ci["mean"] - s_ci["mean"],
            "cohens_d": d,
        } if m_vals and s_vals else {},
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
