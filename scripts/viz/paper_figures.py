"""
Generate paper figures 1b, 2, and 3 from geodesic_json/ data.
Output: PDFs in figs/ directory.

Expected files in DATA_DIR (default: ../geodesic_json/):
  geodesic_inline_gsm_RUN1_100K.json  — GSM8K Run 1 inline
  geodesic_inline_gsm_RUN2_100K.json  — GSM8K Run 2 inline
  geodesic_inline_gsm_RUN3_100K.json  — GSM8K Run 3 inline
  geodesic_inline_gsm_ema_100K.json   — GSM8K EMA (Run 2)
  geodesic_inline_alpaca_200k.json    — Alpaca inline

Training generates a single geodesic_inline.json per run.
Rename to the above after training.
"""

import json
import math
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ColorBrewer Set1 (colour-blind safe)
C_BLUE = "#377EB8"
C_RED = "#E41A1C"
C_GREEN = "#4DAF4A"
C_PURPLE = "#984EA3"
C_GREY = "#999999"

FIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "figs")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "geodesic_json")

os.makedirs(FIGS_DIR, exist_ok=True)


def load_json(name):
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


def compute_window(data, start, end):
    pts = [d for d in data if start <= d["step"] <= end]
    ratios = [d["ratio"] for d in pts]
    mu = sum(ratios) / len(ratios)
    std = math.sqrt(sum((r - mu) ** 2 for r in ratios) / len(ratios))
    return mu, std


def plot_three_seed(ax):
    """Figure 1b: Three-seed R(k) evolution curves."""
    runs = {
        "Run 1 (Regime M)": ("geodesic_inline_gsm_RUN1_100K.json", C_BLUE, "o"),
        "Run 2 (Regime S)": ("geodesic_inline_gsm_RUN2_100K.json", C_RED, "s"),
        "Run 3 (Regime M)": ("geodesic_inline_gsm_RUN3_100K.json", C_GREEN, "^"),
    }

    for label, (fname, color, marker) in runs.items():
        data = load_json(fname)
        steps = [d["step"] for d in data]
        ratios = [d["ratio"] for d in data]
        stds = [d["std"] for d in data]

        # Plot every 1K to avoid overplotting
        ax.plot(steps, ratios, color=color, linewidth=1.2, alpha=0.9, label=label)
        # Marker every 10K
        marker_steps = [s for s in steps if s % 10000 == 0]
        marker_ratios = [ratios[steps.index(s)] for s in marker_steps]
        ax.scatter(
            marker_steps,
            marker_ratios,
            color=color,
            marker=marker,
            s=20,
            zorder=5,
            edgecolors="white",
            linewidths=0.5,
        )

    # Euclidean baseline
    ax.axhline(y=1.0, color=C_GREY, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(102000, 1.0, r"$R{=}1$", color=C_GREY, fontsize=8, va="center", ha="left")

    # Transition window band
    ax.axvspan(65000, 80000, color=C_RED, alpha=0.08, zorder=0)
    ax.annotate(
        "Phase\nTransition\n(65K-80K)",
        xy=(72500, 0.30),
        fontsize=7,
        color=C_RED,
        ha="center",
        va="center",
        fontweight="bold",
    )

    # Regime annotations
    ax.annotate(
        r"$R{=}1.94{\pm}0.05$",
        xy=(95000, 2.12),
        fontsize=7,
        color=C_BLUE,
        ha="center",
    )
    ax.annotate(
        r"$R{=}0.35{\pm}0.04$",
        xy=(95000, 0.55),
        fontsize=7,
        color=C_RED,
        ha="center",
    )
    ax.annotate(
        r"$R{=}1.69{\pm}0.05$",
        xy=(95000, 1.49),
        fontsize=7,
        color=C_GREEN,
        ha="center",
    )

    ax.set_xlabel("Training Step $k$", fontsize=9)
    ax.set_ylabel("Geodesic Energy Ratio $R(k)$", fontsize=9)
    ax.set_xlim(0, 102000)
    ax.set_ylim(0, 4.0)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))


def plot_ema_inline(ax):
    """Figure 2: EMA vs Inline at Run 2 transition window."""
    inline_data = load_json("geodesic_inline_gsm_RUN2_100K.json")
    ema_data = load_json("geodesic_inline_gsm_ema_100K.json")

    def extract_window(data, start, end):
        pts = [(d["step"], d["ratio"], d["std"]) for d in data if start <= d["step"] <= end]
        return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]

    ist, ir, istd = extract_window(inline_data, 60000, 85000)
    est, er, estd = extract_window(ema_data, 60000, 85000)

    ax.plot(ist, ir, color=C_RED, linewidth=1.5, marker="s", markersize=4,
            markevery=5, label="Inline (Training Weights)")
    ax.plot(est, er, color=C_BLUE, linewidth=1.5, marker="o", markersize=4,
            markevery=5, linestyle="--", label="EMA")

    ax.axhline(y=1.0, color=C_GREY, linestyle=":", linewidth=0.8)
    ax.axvspan(65000, 80000, color=C_GREY, alpha=0.1, zorder=0)
    ax.text(72500, 0.15, "Transition\nWindow", fontsize=7, color=C_GREY, ha="center")

    # Annotate key points
    ax.annotate(
        f"EMA: {er[est.index(70000)]:.2f}",
        xy=(70000, er[est.index(70000)]),
        fontsize=7, color=C_BLUE, ha="left", va="bottom",
    )
    ax.annotate(
        f"EMA: {er[est.index(80000)]:.2f}",
        xy=(80000, er[est.index(80000)]),
        fontsize=7, color=C_BLUE, ha="left", va="bottom",
    )
    ax.annotate(
        f"Inline: {ir[ist.index(70000)]:.2f}",
        xy=(70000, ir[ist.index(70000)] - 0.12),
        fontsize=7, color=C_RED, ha="right", va="top",
    )
    ax.annotate(
        f"Inline: {ir[ist.index(80000)]:.2f}",
        xy=(80000, ir[ist.index(80000)] - 0.15),
        fontsize=7, color=C_RED, ha="left", va="top",
    )
    # Geometric memory label — offset to not block EMA:1.56
    ax.annotate(
        "Geometric\nMemory ($R{>}1$)",
        xy=(82000, er[est.index(80000)] + 0.25),
        fontsize=7, color=C_BLUE, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Training Step $k$", fontsize=9)
    ax.set_ylabel("Geodesic Energy Ratio $R(k)$", fontsize=9)
    ax.set_xlim(60000, 85000)
    ax.set_ylim(0, 2.8)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))


def plot_cross_dataset(ax1, ax2):
    """Figure 3: Cross-dataset comparison. ax1 = trajectories, ax2 = bar chart."""

    # --- Panel (a): Trajectories ---
    # Load GSM8K runs 1+3, compute mean
    for fname, color, label in [
        ("geodesic_inline_gsm_RUN1_100K.json", C_BLUE, "GSM8K Run 1"),
        ("geodesic_inline_gsm_RUN3_100K.json", C_GREEN, "GSM8K Run 3"),
    ]:
        data = load_json(fname)
        steps = [d["step"] for d in data if d["step"] <= 100000]
        ratios = [d["ratio"] for d in data if d["step"] <= 100000]
        ax1.plot(steps, ratios, color=color, linewidth=0.6, alpha=0.4)

    # GSM8K Regime M mean
    r1 = load_json("geodesic_inline_gsm_RUN1_100K.json")
    r3 = load_json("geodesic_inline_gsm_RUN3_100K.json")
    steps_gsm = sorted(set(d["step"] for d in r1 if d["step"] <= 100000))
    mean_r = []
    for s in steps_gsm:
        v1 = next(d["ratio"] for d in r1 if d["step"] == s)
        v3 = next(d["ratio"] for d in r3 if d["step"] == s)
        mean_r.append((v1 + v3) / 2)
    ax1.plot(steps_gsm, mean_r, color=C_BLUE, linewidth=2.0, label="GSM8K Regime M (mean)")

    # Alpaca inline
    alpaca = load_json("geodesic_inline_alpaca_200k.json")
    a_steps = [d["step"] for d in alpaca]
    a_ratios = [d["ratio"] for d in alpaca]
    ax1.plot(a_steps, a_ratios, color=C_PURPLE, linewidth=1.5, label="Alpaca Inline")

    ax1.axhline(y=1.0, color=C_GREY, linestyle="--", linewidth=0.8, alpha=0.6)

    # Late-phase annotations
    mu_gsm, _ = compute_window(r1 + r3, 80000, 100000)
    mu_alp, _ = compute_window(alpaca, 160000, 200000)
    ax1.annotate(
        r"GSM8K M: $R{\approx}" + f"{mu_gsm:.2f}" + r"$",
        xy=(97000, mu_gsm + 0.30), fontsize=7, color=C_BLUE,
    )
    ax1.annotate(
        r"Alpaca: $R{\approx}" + f"{mu_alp:.2f}" + r"$",
        xy=(188000, mu_alp + 0.30), fontsize=7, color=C_PURPLE,
    )

    ax1.set_xlabel("Training Step $k$", fontsize=9)
    ax1.set_ylabel("$R(k)$", fontsize=9)
    ax1.set_xlim(0, 205000)
    ax1.set_ylim(0, 3.0)
    ax1.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
    ax1.tick_params(labelsize=8)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))

    # --- Panel (b): Late-phase bar chart ---
    categories = ["GSM8K\nRegime M", "GSM8K\nRegime S", "Alpaca\nInline", "Alpaca\nEMA"]
    values = [1.82, 0.35, 1.65, 1.99]
    errors = [0.05, 0.04, 0.06, 0.14]
    colors = [C_BLUE, C_RED, C_PURPLE, C_PURPLE]
    hatches = ["", "//", "", "//"]

    x_pos = range(len(categories))
    bars = ax2.bar(x_pos, values, color=colors, edgecolor="white", linewidth=0.5, width=0.6)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_alpha(0.7)

    ax2.errorbar(x_pos, values, yerr=errors, fmt="none", color="black", capsize=3, linewidth=0.8)

    ax2.axhline(y=1.0, color=C_GREY, linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.text(3.6, 1.0, r"$R{=}1$", fontsize=7, color=C_GREY, va="center")

    # Annotate the EMA/inline gap reversal
    ax2.annotate(
        "EMA > Inline\n(reversed vs GSM8K)",
        xy=(2.5, 2.1),
        fontsize=6.5,
        color=C_PURPLE,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
    )
    # Arrow from inline to EMA bar
    ax2.annotate(
        "",
        xy=(3, 1.99),
        xytext=(2, 1.65),
        arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=0.8),
    )

    ax2.set_ylabel("Late-Phase $R$", fontsize=9)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories, fontsize=7)
    ax2.set_ylim(0, 2.5)
    ax2.tick_params(labelsize=8)

    # Value labels on bars
    for i, (v, e) in enumerate(zip(values, errors)):
        ax2.text(i, v + e + 0.05, f"{v:.2f}", ha="center", fontsize=7, fontweight="bold")


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    # ---- Figure 1b: Three-Seed R(k) ----
    fig, ax = plt.subplots(figsize=(7, 3.5))
    plot_three_seed(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "fig1b_three_seed.pdf"))
    plt.close(fig)
    print("Figure 1b saved: figs/fig1b_three_seed.pdf")

    # ---- Figure 2: EMA vs Inline ----
    fig, ax = plt.subplots(figsize=(7, 2.8))
    plot_ema_inline(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "fig2_ema_inline.pdf"))
    plt.close(fig)
    print("Figure 2 saved: figs/fig2_ema_inline.pdf")

    # ---- Figure 3: Cross-Dataset ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5),
                                    gridspec_kw={"height_ratios": [2, 1]})
    plot_cross_dataset(ax1, ax2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "fig3_cross_dataset.pdf"))
    plt.close(fig)
    print("Figure 3 saved: figs/fig3_cross_dataset.pdf")

    print("All figures generated.")


if __name__ == "__main__":
    main()
