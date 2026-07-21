#!/usr/bin/env python3
"""
P0-1: Per-Text R Trajectory Shape Clustering
============================================
Analyses Run2 phase transition (50K→100K) at per-text granularity.
Clusters 500 GSM8K texts by the SHAPE of their R evolution trajectory,
independent of absolute R level.

Core Question:
  How many distinct trajectory archetypes exist during the phase transition?
  Do texts follow the same collapse path, or are there multiple geometric
  "routes" through the phase boundary?

Input:  phase_per_text.json (Run2: 500 texts × 6 steps)
Output: trajectory_clustering/  (plots, JSON stats, archetype summary)
"""

import json
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


# ═══════════════════════════════════════════════════════════════
# 1. Shape feature extraction
# ═══════════════════════════════════════════════════════════════

def zscore_normalize(traj: np.ndarray) -> np.ndarray:
    """Remove level and scale; retain pure shape."""
    mu = np.mean(traj)
    sigma = np.std(traj)
    if sigma < 1e-8:
        return np.zeros_like(traj)
    return (traj - mu) / sigma


def extract_shape_features(r_matrix: dict, steps: list[int]) -> np.ndarray:
    """
    For each text, extract shape-descriptive features:
      - z-scored trajectory (normalized shape)
      - first derivative (ΔR per interval)
      - monotonicity index
      - inflection timing (when does collapse accelerate?)
    Returns: (n_texts × n_features) array
    """
    n_texts = len(r_matrix[steps[0]])
    n_steps = len(steps)

    all_zscores = np.zeros((n_texts, n_steps))
    all_deriv = np.zeros((n_texts, n_steps - 1))

    for i in range(n_texts):
        traj = np.array([r_matrix[s][i] for s in steps])
        all_zscores[i] = zscore_normalize(traj)
        all_deriv[i] = np.diff(traj)

    # Composite features
    features = []
    for i in range(n_texts):
        traj = np.array([r_matrix[s][i] for s in steps])

        # 1. Terminal R (absolute level — anchor)
        r_final = traj[-1]

        # 2. Total collapse magnitude
        delta_total = traj[0] - traj[-1]

        # 3. Max single-step drop
        max_drop = np.max(-np.diff(traj))

        # 4. Timing of max drop (which interval?)
        max_drop_step = np.argmax(-np.diff(traj))  # 0-4

        # 5. Pre-collapse R level (mean of first 2 steps)
        pre_r = np.mean(traj[:2])

        # 6. Post-collapse R level (mean of last 2 steps)
        post_r = np.mean(traj[-2:])

        # 7. Collapse suddenness: ratio of max drop to avg drop
        avg_drop = delta_total / max(1, n_steps - 1)
        suddenness = max_drop / max(avg_drop, 1e-8)

        # 8. Monotonicity: fraction of steps that decrease
        n_decreases = np.sum(np.diff(traj) < 0)
        monotonicity = n_decreases / (n_steps - 1)

        # 9-11. Shape inflection — does collapse accelerate or decelerate?
        drops = -np.diff(traj)
        if len(drops) >= 2:
            # acceleration = second derivative
            accel = np.diff(drops)
            mean_accel = np.mean(accel)
            max_accel = np.max(accel)
        else:
            mean_accel = max_accel = 0.0

        # 12. Z-score trajectory (for DTW)
        z_traj = all_zscores[i].tolist()

        features.append([
            r_final,
            delta_total,
            max_drop,
            max_drop_step,
            pre_r,
            post_r,
            suddenness,
            monotonicity,
            mean_accel,
            max_accel,
        ])

    return np.array(features), all_zscores


# ═══════════════════════════════════════════════════════════════
# 2. DTW-based distance
# ═══════════════════════════════════════════════════════════════

def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Dynamic Time Warping distance for normalized trajectories."""
    n, m = len(x), len(y)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


def compute_dtw_matrix(zscores: np.ndarray) -> np.ndarray:
    """Pairwise DTW distance matrix for all texts."""
    n = zscores.shape[0]
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(zscores[i], zscores[j])
            dist_mat[i, j] = d
            dist_mat[j, i] = d
    return dist_mat


# ═══════════════════════════════════════════════════════════════
# 3. Clustering via feature-based k-means (faster) + DTW validation
# ═══════════════════════════════════════════════════════════════

def cluster_trajectories(features: np.ndarray, zscores: np.ndarray,
                         n_clusters: int = 4, random_seed: int = 42):
    """Cluster texts by trajectory shape using hierarchical clustering on
    feature space, with DTW distance as validation metric.
    Pure scipy/numpy — no sklearn dependency."""
    # Manual z-score normalization
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-10] = 1.0
    feats_scaled = (features - mean) / std

    # Agglomerative clustering via scipy
    Z = linkage(feats_scaled, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1  # 0-indexed

    # Compute per-cluster mean trajectory
    n_steps = zscores.shape[1]
    cluster_trajs = {}
    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() > 0:
            cluster_trajs[c] = {
                "mean_zs": zscores[mask].mean(axis=0).tolist(),
                "std_zs": zscores[mask].std(axis=0).tolist(),
                "size": int(mask.sum()),
                "indices": np.where(mask)[0].tolist(),
            }

    return labels, cluster_trajs


# ═══════════════════════════════════════════════════════════════
# 4. Transition analysis: do texts switch clusters across steps?
# ═══════════════════════════════════════════════════════════════

def analyze_transitions(r_matrix: dict, steps: list[int],
                        labels: np.ndarray, n_clusters: int):
    """How does cluster composition evolve across checkpoints?"""
    n_texts = len(r_matrix[steps[0]])
    n_steps = len(steps)

    # For each step, classify each text by its R regime
    step_regimes = np.zeros((n_texts, n_steps), dtype=int)
    for s_idx, step in enumerate(steps):
        for t_idx in range(n_texts):
            r = r_matrix[step][t_idx]
            if r >= 1.15:
                step_regimes[t_idx, s_idx] = 2  # M
            elif r >= 0.85:
                step_regimes[t_idx, s_idx] = 1  # E
            else:
                step_regimes[t_idx, s_idx] = 0  # S

    # Count transitions
    regime_counts = np.zeros((n_steps, 3), dtype=int)  # M, E, S
    for s_idx in range(n_steps):
        for regime in range(3):
            regime_counts[s_idx, regime] = int(
                np.sum(step_regimes[:, s_idx] == regime))

    # Transition matrix: which regime at step s-1 → which at step s?
    transitions = {}
    for s_idx in range(1, n_steps):
        trans = np.zeros((3, 3), dtype=int)
        for t_idx in range(n_texts):
            prev = step_regimes[t_idx, s_idx - 1]
            curr = step_regimes[t_idx, s_idx]
            trans[prev, curr] += 1
        transitions[f"{steps[s_idx-1]}→{steps[s_idx]}"] = trans.tolist()

    return regime_counts, step_regimes, transitions


# ═══════════════════════════════════════════════════════════════
# 5. Visualization
# ═══════════════════════════════════════════════════════════════

def plot_cluster_archetypes(cluster_trajs: dict, steps: list[int],
                            output_path: str):
    """Plot the mean ± std trajectory for each cluster."""
    n_clusters = len(cluster_trajs)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    x = list(range(len(steps)))
    xlabels = [f"{s//1000}K" for s in steps]

    for c in sorted(cluster_trajs.keys()):
        info = cluster_trajs[c]
        mean_z = np.array(info["mean_zs"])
        std_z = np.array(info["std_zs"])
        color = colors[c]
        ax.plot(x, mean_z, 'o-', color=color, linewidth=2,
                label=f"Cluster {c} (n={info['size']})")
        ax.fill_between(x, mean_z - std_z, mean_z + std_z,
                        alpha=0.15, color=color)

    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("R (z-score normalized)")
    ax.set_title("Trajectory Shape Archetypes — Run2 Phase Transition")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


def plot_regime_transition(regime_counts: np.ndarray, steps: list[int],
                           output_path: str):
    """Stacked area chart of regime composition over steps."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = list(range(len(steps)))
    xlabels = [f"{s//1000}K" for s in steps]

    ax.stackplot(x,
                 regime_counts[:, 0] / regime_counts.sum(axis=1),
                 regime_counts[:, 1] / regime_counts.sum(axis=1),
                 regime_counts[:, 2] / regime_counts.sum(axis=1),
                 labels=["Regime S", "Regime E", "Regime M"],
                 colors=["#d62728", "#ff7f0e", "#1f77b4"],
                 alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Fraction of Texts")
    ax.set_title("Regime Composition Across Phase Transition")
    ax.legend(loc="center left")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


def plot_individual_trajectories(r_matrix: dict, steps: list,
                                 labels: np.ndarray, n_clusters: int,
                                 output_path: str, max_per_cluster: int = 20):
    """Plot individual trajectories, colored by cluster."""
    fig, axes = plt.subplots(1, n_clusters, figsize=(4 * n_clusters, 5),
                             sharex=True, sharey=True)
    if n_clusters == 1:
        axes = [axes]

    x = list(range(len(steps)))
    xlabels = [f"{int(s)//1000}K" for s in steps]
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    for c in range(n_clusters):
        ax = axes[c]
        indices = [i for i, l in enumerate(labels) if l == c]
        # Sample if too many
        if len(indices) > max_per_cluster:
            rng = np.random.RandomState(42)
            indices = rng.choice(indices, max_per_cluster, replace=False)

        for i in indices:
            traj = np.array([r_matrix[s][i] for s in steps])
            ax.plot(x, traj, 'o-', alpha=0.3, linewidth=0.5, color=colors[c],
                    markersize=2)

        # Mean trajectory
        all_idx = [i for i, l in enumerate(labels) if l == c]
        mean_traj = np.zeros(len(steps))
        for i in all_idx:
            mean_traj += np.array([r_matrix[s][i] for s in steps])
        mean_traj /= len(all_idx)
        ax.plot(x, mean_traj, 's-', color='black', linewidth=2.5,
                markersize=5, label=f"Mean (n={len(all_idx)})")

        ax.set_title(f"Cluster {c}")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=45)
        ax.axhline(y=1.15, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=0.85, color='gray', linestyle=':', alpha=0.5)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Individual Trajectories by Shape Cluster", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


def plot_dendrogram(features: np.ndarray, output_path: str):
    """Hierarchical clustering dendrogram."""
    # Manual z-score normalization
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-10] = 1.0
    feats = (features - mean) / std

    Z = linkage(feats, method="ward")

    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=30,
               leaf_rotation=90, leaf_font_size=8)
    ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage)")
    ax.set_xlabel("Cluster / Leaf Index")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


# ═══════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="P0-1: Per-Text R Trajectory Shape Clustering")
    parser.add_argument("--input", default="phase_per_text.json",
                        help="Path to phase_per_text.json")
    parser.add_argument("--output-dir", default="results/trajectory_clustering",
                        help="Output directory for plots and JSON")
    args = parser.parse_args()

    data_path = args.input
    outdir = args.output_dir

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run T1.5 first.")
        sys.exit(1)

    os.makedirs(outdir, exist_ok=True)

    print("=" * 65)
    print("  P0-1: Per-Text R Trajectory Shape Clustering")
    print("=" * 65)

    # Load data
    print(f"\n[1] Loading {data_path}...")
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    r_matrix = data["r_matrix"]
    steps = sorted([int(s) for s in r_matrix.keys()])
    n_texts = len(r_matrix[str(steps[0])])
    print(f"  Texts: {n_texts}, Steps: {steps}")

    # Extract features
    print(f"\n[2] Extracting shape features...")
    features, zscores = extract_shape_features(r_matrix, [str(s) for s in steps])

    feature_names = [
        "r_final", "delta_total", "max_drop", "max_drop_step",
        "pre_r", "post_r", "suddenness", "monotonicity",
        "mean_accel", "max_accel",
    ]
    print(f"  Feature dimensions: {features.shape}")

    # Feature statistics
    print(f"\n[3] Feature summary:")
    for i, name in enumerate(feature_names):
        col = features[:, i]
        print(f"  {name:>18s}: mean={np.mean(col):.4f}  "
              f"std={np.std(col):.4f}  min={np.min(col):.4f}  "
              f"max={np.max(col):.4f}")

    # Clustering
    print(f"\n[4] Clustering trajectories...")

    # Determine optimal n_clusters via silhouette (pure scipy)
    # Manual z-score normalization
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-10] = 1.0
    feats_scaled = (features - mean) / std

    def _silhouette_score(X, labels):
        """Manual silhouette score — no sklearn dependency."""
        n = len(labels)
        unique_labels = sorted(set(labels))
        if len(unique_labels) <= 1:
            return -1.0
        # Precompute pairwise distances
        from scipy.spatial.distance import cdist
        dist = cdist(X, X, metric="euclidean")
        scores = np.zeros(n)
        for i in range(n):
            ci = labels[i]
            # a(i): mean distance within same cluster
            same = np.where(labels == ci)[0]
            if len(same) <= 1:
                continue
            a_i = dist[i, same].sum() / (len(same) - 1)
            # b(i): min mean distance to any other cluster
            b_i = np.inf
            for cj in unique_labels:
                if cj == ci:
                    continue
                other = np.where(labels == cj)[0]
                b_ij = dist[i, other].mean()
                if b_ij < b_i:
                    b_i = b_ij
            scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0
        return float(np.mean(scores))

    best_k = 3
    best_score = -1
    scores = {}
    for k in range(2, 8):
        Zk = linkage(feats_scaled, method="ward")
        labels_k = fcluster(Zk, t=k, criterion="maxclust") - 1
        if len(np.unique(labels_k)) > 1:
            score = _silhouette_score(feats_scaled, labels_k)
            scores[k] = score
            if score > best_score:
                best_score = score
                best_k = k

    print(f"  Silhouette scores: {json.dumps({str(k): round(v, 4) for k, v in scores.items()})}")
    print(f"  Optimal k = {best_k} (silhouette = {best_score:.4f})")

    # Run clustering with optimal k
    labels, cluster_trajs = cluster_trajectories(
        features, zscores, n_clusters=best_k)

    print(f"\n[5] Cluster composition:")
    for c in sorted(cluster_trajs.keys()):
        info = cluster_trajs[c]
        print(f"  Cluster {c}: n={info['size']:>4d} ({100*info['size']/n_texts:.1f}%)")

    # Regime + cluster cross-tabulation at each step
    print(f"\n[6] Regime × Cluster cross-tabulation:")
    step_keys = [str(s) for s in steps]
    for s_idx, step in enumerate(step_keys):
        print(f"\n  Step {step}:")
        print(f"    {'Cluster':>10s}  {'M':>6s}  {'E':>6s}  {'S':>6s}")
        for c in sorted(cluster_trajs.keys()):
            idxs = cluster_trajs[c]["indices"]
            r_vals = [r_matrix[step][i] for i in idxs]
            n_m = sum(1 for r in r_vals if r >= 1.15)
            n_e = sum(1 for r in r_vals if 0.85 <= r < 1.15)
            n_s = sum(1 for r in r_vals if r < 0.85)
            print(f"    {'Cluster '+str(c):>10s}  {n_m:>6d}  {n_e:>6d}  {n_s:>6d}")

    # Transition analysis
    print(f"\n[7] Regime transition analysis...")
    regime_counts, step_regimes, transitions = analyze_transitions(
        r_matrix, step_keys, labels, best_k)

    print(f"\n  {'Step':>8s}  {'M':>6s}  {'E':>6s}  {'S':>6s}")
    for s_idx, step in enumerate(steps):
        print(f"  {step:>8d}  {regime_counts[s_idx,2]:>6d}  "
              f"{regime_counts[s_idx,1]:>6d}  {regime_counts[s_idx,0]:>6d}")

    regime_names = ["S", "E", "M"]
    print(f"\n  Transition matrices (S×E×M):")
    for trans_key, mat in transitions.items():
        print(f"\n  {trans_key}:")
        print(f"    From\\To  {'S':>6s}  {'E':>6s}  {'M':>6s}")
        for from_r in range(3):
            print(f"    {regime_names[from_r]:>6s}  {mat[from_r][0]:>6d}  "
                  f"{mat[from_r][1]:>6d}  {mat[from_r][2]:>6d}")

    # Intra-cluster R correlation (are clusters internally consistent?)
    print(f"\n[8] Intra-cluster coherence:")
    for c in sorted(cluster_trajs.keys()):
        idxs = cluster_trajs[c]["indices"]
        if len(idxs) > 1:
            # Pairwise Spearman within cluster
            n_pairs = min(100, len(idxs))
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(idxs, n_pairs, replace=False)
            rho_vals = []
            for a_idx, i in enumerate(sample_idx):
                for j in sample_idx[a_idx + 1:]:
                    traj_i = np.array([r_matrix[s][i] for s in step_keys])
                    traj_j = np.array([r_matrix[s][j] for s in step_keys])
                    rho, _ = spearmanr(traj_i, traj_j)
                    rho_vals.append(rho)
            mean_rho = np.mean(rho_vals) if rho_vals else 0
            print(f"  Cluster {c}: intra-cluster mean ρ = {mean_rho:.4f} "
                  f"(n_pairs={len(rho_vals)})")
        else:
            print(f"  Cluster {c}: singleton (no intra-cluster correlation)")

    # Visualization
    print(f"\n[9] Generating plots...")
    plot_cluster_archetypes(cluster_trajs, steps,
                            os.path.join(outdir, "cluster_archetypes.png"))
    plot_regime_transition(regime_counts, steps,
                           os.path.join(outdir, "regime_transition.png"))
    plot_individual_trajectories(r_matrix, step_keys, labels, best_k,
                                 os.path.join(outdir, "individual_trajectories.png"))
    plot_dendrogram(features,
                    os.path.join(outdir, "dendrogram.png"))

    # Save results
    print(f"\n[10] Saving results...")
    results = {
        "n_texts": n_texts,
        "steps": steps,
        "n_clusters": best_k,
        "silhouette_score": round(best_score, 4),
        "silhouette_by_k": {str(k): round(v, 4) for k, v in scores.items()},
        "cluster_composition": {
            str(c): {
                "size": info["size"],
                "pct": round(100 * info["size"] / n_texts, 1),
                "mean_zs_trajectory": [round(x, 4) for x in info["mean_zs"]],
                "std_zs_trajectory": [round(x, 4) for x in info["std_zs"]],
            }
            for c, info in cluster_trajs.items()
        },
        "regime_transitions": [
            {
                "step": steps[s_idx],
                "n_M": int(regime_counts[s_idx, 2]),
                "n_E": int(regime_counts[s_idx, 1]),
                "n_S": int(regime_counts[s_idx, 0]),
            }
            for s_idx in range(len(steps))
        ],
        "transition_matrices": {
            k: {
                "S→S": v[0][0], "S→E": v[0][1], "S→M": v[0][2],
                "E→S": v[1][0], "E→E": v[1][1], "E→M": v[1][2],
                "M→S": v[2][0], "M→E": v[2][1], "M→M": v[2][2],
            }
            for k, v in transitions.items()
        },
        "per_text_labels": [int(l) for l in labels],
        "feature_names": feature_names,
    }

    json_path = os.path.join(outdir, "trajectory_clustering.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  JSON saved: {json_path}")

    # Interpretation summary — focus on Step 70K differentiation
    print(f"\n{'='*65}")
    print("  Interpretation")
    print(f"{'='*65}")
    print(f"\n  Primary finding: Run2 phase transition produces {best_k} "
          f"distinct trajectory archetypes.")
    print(f"  All texts collapse in the 70K→80K window, but they differ at 70K:")

    # At step 70K, what's the regime composition per cluster?
    step_70k = "70000"
    print(f"\n  Step 70K (critical transition point) composition:")
    print(f"  {'Cluster':>10s}  {'M':>6s}  {'E':>6s}  {'S':>6s}  {'Archetype'}")
    archetype_labels = {}
    for c in sorted(cluster_trajs.keys()):
        idxs = cluster_trajs[c]["indices"]
        r_vals = [r_matrix[step_70k][i] for i in idxs]
        n_m = sum(1 for r in r_vals if r >= 1.15)
        n_e = sum(1 for r in r_vals if 0.85 <= r < 1.15)
        n_s = sum(1 for r in r_vals if r < 0.85)

        if n_s > 0 and n_e > 0:
            label = "PRECOCIOUS COLLAPSER (early S + E mix)"
        elif n_s > 0:
            label = "EARLIEST COLLAPSER (already S at 70K)"
        elif n_e > n_m:
            label = "TRANSITIONAL (entering E, not yet S)"
        elif n_e > 0:
            label = "MAINSTREAM with E seeds"
        else:
            label = "MAINSTREAM (still M at 70K)"

        archetype_labels[c] = label
        print(f"  {'Cluster '+str(c):>10s}  {n_m:>6d}  {n_e:>6d}  "
              f"{n_s:>6d}  {label}")

    # Post-collapse differentiation
    print(f"\n  Post-collapse (90K-100K) R distributions:")
    step_100k = "100000"
    for c in sorted(cluster_trajs.keys()):
        idxs = cluster_trajs[c]["indices"]
        r_vals = [r_matrix[step_100k][i] for i in idxs]
        mean_r = np.mean(r_vals)
        std_r = np.std(r_vals)
        print(f"  Cluster {c}: R_final = {mean_r:.4f} ± {std_r:.4f}  "
              f"(n={len(idxs)})")

    print(f"\n  Done. Results in {outdir}/")


if __name__ == "__main__":
    main()
