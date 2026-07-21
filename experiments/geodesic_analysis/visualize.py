"""Direction F: Visualization tools for CoT-Geodesic analysis."""

import os
import json
from typing import Dict, List, Optional


def plot_semantic_progression(
    analysis: Dict,
    output_path: Optional[str] = None,
):
    """
    Plot semantic progression metrics as a multi-panel figure.

    Requires matplotlib. Shows:
      - unique_token_ratio vs t
      - repetition_rate vs t
      - special_token_ratio vs t
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ERROR] matplotlib not installed. Run: pip install matplotlib")
        return

    metrics = analysis["metrics"]
    times = [m["t"] for m in metrics]
    unique = [m["unique_token_ratio"] for m in metrics]
    repet = [m["repetition_rate"] for m in metrics]
    special = [m["special_token_ratio"] for m in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    labels = ["Unique Token Ratio", "Repetition Rate", "Special Token Ratio"]
    data = [unique, repet, special]

    for ax, d, c, lbl in zip(axes, data, colors, labels):
        ax.plot(times, d, "o-", color=c, markersize=6, linewidth=2)
        ax.fill_between(times, 0, d, alpha=0.1, color=c)
        ax.set_xlabel("Time t")
        ax.set_ylabel(lbl)
        ax.set_title(lbl)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

    fig.suptitle(f"ELF Semantic Progression ({analysis['method'].upper()})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {output_path}")

    plt.close()


def plot_path_curvature(
    curvature: Dict,
    output_path: Optional[str] = None,
):
    """
    Plot geodesic energy per path segment (bar chart).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ERROR] matplotlib not installed.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    segments = range(len(curvature["segment_energies"]))
    energies = curvature["segment_energies"]

    ax.bar(segments, energies, color="#9b59b6", alpha=0.7, edgecolor="#8e44ad")
    ax.set_xlabel("Path Segment")
    ax.set_ylabel("Geodesic Energy")
    ax.set_title("Path Curvature Analysis")
    ax.grid(True, alpha=0.3, axis="y")

    # Add total energy annotation
    ax.text(0.98, 0.95, f"Total: {curvature['total_energy']:.4f}\nVar: {curvature['energy_variance']:.6f}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {output_path}")
    plt.close()


def compare_path_energies(
    path_comparison: Dict,
    output_path: Optional[str] = None,
):
    """
    Bar chart comparing straight-line vs ODE path geodesic energies.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ERROR] matplotlib not installed.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Straight Line", "ODE Path"]
    values = [path_comparison["straight_line_energy"], path_comparison["ode_path_energy"]]
    colors = ["#3498db", "#e74c3c"]

    bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Geodesic Energy")
    ax.set_title(f"Path Energy Comparison (ratio={path_comparison['ratio']:.4f})")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {output_path}")
    plt.close()


def generate_report(
    trace: Dict,
    analysis: Optional[Dict] = None,
    curvature: Optional[Dict] = None,
    comparison: Optional[Dict] = None,
    output_dir: str = "./analysis_output",
):
    """
    Generate a complete analysis report with figures and JSON data.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save raw trace
    with open(os.path.join(output_dir, "trace.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False, indent=2))
        # Re-use trace_to_json logic
        import json
        json.dump({
            "method": trace["method"],
            "num_steps": trace["num_steps"],
            "decode_strategy": trace["decode_strategy"],
            "trajectory": trace["trajectory"],
        }, f, ensure_ascii=False, indent=2)

    # Generate figures
    if analysis:
        plot_semantic_progression(analysis, os.path.join(output_dir, "semantic_progression.png"))
        with open(os.path.join(output_dir, "semantic_analysis.json"), "w") as f:
            json.dump(analysis, f, indent=2)

    if curvature:
        plot_path_curvature(curvature, os.path.join(output_dir, "path_curvature.png"))
        with open(os.path.join(output_dir, "curvature.json"), "w") as f:
            json.dump(curvature, f, indent=2)

    if comparison:
        compare_path_energies(comparison, os.path.join(output_dir, "path_comparison.png"))
        with open(os.path.join(output_dir, "comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2)

    print(f"\n[Report] Analysis complete. Output: {output_dir}/")
