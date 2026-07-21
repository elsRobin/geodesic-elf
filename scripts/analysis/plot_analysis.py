#!/usr/bin/env python3
"""Generate weight interpolation analysis plots from interpolation results."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

JSON_PATH = "results/weight_interpolation.json"
OUTDIR = "results/weight_interpolation"
os.makedirs(OUTDIR, exist_ok=True)

with open(JSON_PATH) as f:
    data = json.load(f)

alphas = [round(a, 1) for a in data["alphas"]]
r_by_alpha = {float(k): v for k, v in data["r_by_alpha"].items()}
alphas_arr = np.array(alphas)
n_texts = len(r_by_alpha[alphas[0]])

all_r = np.zeros((n_texts, len(alphas)))
for i, a in enumerate(alphas):
    all_r[:, i] = r_by_alpha[a]

# ---- Figure 1: R(alpha) + Correlation Decay ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sample_n = min(n_texts, 30)
rng = np.random.RandomState(42)
sample_idx = rng.choice(n_texts, sample_n, replace=False)
for idx in sample_idx:
    ax1.plot(alphas_arr, all_r[idx], "-", alpha=0.3, linewidth=0.5, color="steelblue")

mean_r = np.mean(all_r, axis=0)
std_r = np.std(all_r, axis=0)
ax1.plot(alphas_arr, mean_r, "o-", color="black", linewidth=2.5, markersize=6)
ax1.fill_between(alphas_arr, mean_r - std_r, mean_r + std_r, alpha=0.15, color="black")
ax1.axvspan(0.25, 0.35, alpha=0.1, color="red")
ax1.annotate("R=0.084 crash", xy=(0.3, 0.084), fontsize=9, color="red",
             xytext=(0.12, 0.7), arrowprops=dict(arrowstyle="->", color="red"))
ax1.annotate("R=3.81 spike", xy=(1.0, 3.81), fontsize=9, color="darkgreen",
             xytext=(0.7, 4.5), arrowprops=dict(arrowstyle="->", color="darkgreen"))
ax1.axhline(y=1.75, color="#1f77b4", linestyle="--", alpha=0.4, label="Run1 R_JS=1.75")
ax1.axhline(y=3.81, color="#ff7f0e", linestyle="--", alpha=0.4, label="Run3 R_JS=3.81")
ax1.set_xlabel("Interpolation alpha")
ax1.set_ylabel("Geodesic Ratio R (JS)")
ax1.set_title("R(alpha): Run1 -> Run3 Weight Interpolation")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Right: correlation decay
base_r = all_r[:, 0]
rho_vals, p_vals = [], []
for i in range(len(alphas)):
    rho, p = spearmanr(base_r, all_r[:, i])
    rho_vals.append(rho)
    p_vals.append(p)

colors = ["#2ca02c" if p < 0.05 else "gray" for p in p_vals]
ax2.bar(range(len(alphas)), rho_vals, color=colors, alpha=0.7, edgecolor="white")
ax2.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
for i in range(len(alphas)):
    if p_vals[i] < 0.05:
        yoff = 8 if rho_vals[i] > 0 else -15
        pstr = f"{p_vals[i]:.1e}" if p_vals[i] < 0.001 else f"{p_vals[i]:.3f}"
        ax2.annotate(f"p={pstr}", (i, rho_vals[i]),
                     textcoords="offset points", xytext=(0, yoff),
                     ha="center", fontsize=7, color="darkred")
ax2.set_xticks(range(len(alphas)))
ax2.set_xticklabels([f"{a:.1f}" for a in alphas])
ax2.set_xlabel("Interpolation alpha")
ax2.set_ylabel("Spearman rho (vs alpha=0)")
ax2.set_title("Per-Text R Correlation Decay")
ax2.set_ylim(-0.7, 1.1)
ax2.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "r_vs_alpha_analysis.png"), dpi=150)
plt.close(fig)

# ---- Figure 2: Per-text R at anchor points ----
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
key_as = [0.0, 0.1, 0.3, 0.7, 1.0]
for i, a in enumerate(key_as):
    ax = axes[i // 3][i % 3]
    vals = r_by_alpha[a]
    ax.hist(vals, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(x=np.mean(vals), color="red", linestyle="--", linewidth=2)
    ax.set_title(f"alpha={a:.1f}: R={np.mean(vals):.3f} +-{np.std(vals):.3f}")
    ax.set_xlabel("R")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.2)

r_ranges = np.max(all_r, axis=1) - np.min(all_r, axis=1)
ax = axes[1][2]
ax.hist(r_ranges, bins=20, color="darkred", edgecolor="white", alpha=0.7)
ax.axvline(x=np.mean(r_ranges), color="black", linestyle="--", linewidth=2)
ax.set_title(f"Per-text R Range: mean={np.mean(r_ranges):.2f}")
ax.set_xlabel("Range of R across alpha")
ax.set_ylabel("Count")
ax.grid(True, alpha=0.2)
fig.suptitle("Per-Text R Distribution at Key Interpolation Points", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "per_text_dist_at_key_alpha.png"), dpi=150)
plt.close(fig)

# ---- Figure 3: Chaos score ----
fig, ax = plt.subplots(figsize=(8, 4))
chaos_scores = [1 - s["r2"] for s in data["texts_stats"]]
ax.hist(chaos_scores, bins=20, color="darkred", edgecolor="white", alpha=0.8)
ax.axvline(x=np.mean(chaos_scores), color="black", linestyle="--", linewidth=2,
           label=f"Mean = {np.mean(chaos_scores):.4f}")
ax.set_xlabel("Chaos Score = 1 - R2(linear fit)")
ax.set_ylabel("Number of Texts")
ax.set_title("Chaos Score Distribution: 100% texts have R2 < 0.5")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "chaos_histogram.png"), dpi=150)
plt.close(fig)

# Print summary
print("=== P0-2 Key Results ===")
print(f"Mean R2 (linearity): {data['linearity_stats']['mean_r2']:.4f}")
print(f"Chaotic texts: {data['linearity_stats']['pct_chaotic']:.1f}%")
print(f"\nMean R(alpha):")
for a in alphas:
    print(f"  alpha={a:.1f}: R={np.mean(r_by_alpha[a]):.4f} +- {np.std(r_by_alpha[a]):.4f}")
print(f"\nCorrelation at alpha=0.1: rho={data['correlation_decay']['0.1']['spearman_rho']:.4f}")
print(f"Correlation at alpha=1.0: rho={data['correlation_decay']['1.0']['spearman_rho']:.4f}")
print(f"\nWeight-space differences (top 3):")
for d in data["weight_space_diagnostics"]["top_diffs"][:3]:
    print(f"  {d['key']}: rel_diff={d['rel_diff']:.4f}")
print(f"\nPlots saved in {OUTDIR}/")
