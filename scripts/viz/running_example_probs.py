"""
Plot running example probability distributions.
Reads JSON output from extract_probs_running_example.py and produces
a 2x3 grid figure: Regime M (top row) vs Regime S (bottom row)
at t=0.3, 0.6, 0.9.
"""

import json
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_BLUE = "#377EB8"
C_RED = "#E41A1C"


def plot_grid(data_m: dict, data_s: dict, output_path: str):
    """Plot 2x3 grid of probability distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(9, 5))

    times = ["0.3", "0.6", "0.9"]
    row_labels = [
        f"Regime M (R={data_m.get('R', '?')})",
        f"Regime S (R={data_s.get('R', '?')})",
    ]
    row_colors = [C_BLUE, C_RED]
    row_data = [data_m, data_s]

    for row_idx, (ax_row, data, color, label) in enumerate(
        zip(axes, row_data, row_colors, row_labels)
    ):
        for col_idx, t in enumerate(times):
            ax = ax_row[col_idx]
            snap = data["snapshots"].get(t, {})
            tokens = [t[0] for t in snap.get("top_tokens", [])[:10]]
            probs = [t[1] for t in snap.get("top_tokens", [])[:10]]

            # Reverse for horizontal bar chart (top token at top)
            tokens = tokens[::-1]
            probs = probs[::-1]

            bars = ax.barh(range(len(tokens)), probs, color=color, alpha=0.8, height=0.7)
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens, fontsize=7)
            ax.set_xlim(0, max(probs) * 1.25 if probs else 1.0)
            ax.set_title(f"$t={t}$", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("Prob", fontsize=7)

            # Add entropy annotation
            entropy = snap.get("entropy", 0)
            ax.text(
                0.95, 0.05, f"H={entropy:.2f}",
                transform=ax.transAxes, fontsize=6,
                ha="right", va="bottom", color="grey",
            )

        # Row label
        axes[row_idx, 0].set_ylabel(label, fontsize=10, color=color, fontweight="bold")

    fig.suptitle(
        "LM Head Probability Distributions — Running Example (GSM8K)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime_m_json", required=True, help="JSON for Regime M model")
    parser.add_argument("--regime_s_json", required=True, help="JSON for Regime S model")
    parser.add_argument("--output", default="figs/fig1a_running_example.pdf")
    args = parser.parse_args()

    with open(args.regime_m_json) as f:
        data_m = json.load(f)
    with open(args.regime_s_json) as f:
        data_s = json.load(f)

    plot_grid(data_m, data_s, args.output)


if __name__ == "__main__":
    main()
