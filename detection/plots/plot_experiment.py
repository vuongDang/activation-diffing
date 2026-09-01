"""Plots for any meq-run experiment output directory.

Usage:
    python plots/plot_experiment.py results/tiny_full_suite
    python plots/plot_experiment.py results/switching_attacker

Reads summary.csv (and detection_thresholds.csv when present) and writes PNGs
to plots/<experiment_name>/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = [
    "#888888", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728",
    "#9467bd", "#8c564b", "#17becf",
]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    results = Path(sys.argv[1])
    summary = pd.read_csv(results / "summary.csv")
    detect_path = results / "detection_thresholds.csv"
    detect = pd.read_csv(detect_path) if detect_path.exists() else None

    out = Path(__file__).parent / results.name
    out.mkdir(parents=True, exist_ok=True)

    pairs = list(summary["pair"].unique())
    dists = list(summary["distribution"].unique())
    k_values = sorted(summary["k"].unique())
    colors = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(pairs)}

    def style_axis(ax):
        ax.set_xscale("log", base=2)
        ax.set_xticks(k_values)
        ax.set_xticklabels(k_values, fontsize=7)
        ax.set_xlabel("k")
        ax.grid(True, alpha=0.3)

    def per_dist_curves(col, title, ylabel, fname, ylim=None, hline=None):
        """One subplot per distribution, one curve per pair."""
        fig, axes = plt.subplots(
            1, len(dists), figsize=(5 * len(dists), 4), sharey=True, squeeze=False
        )
        for ax, dist in zip(axes[0], dists):
            for pair in pairs:
                sub = summary[(summary["pair"] == pair) & (summary["distribution"] == dist)]
                ax.plot(sub["k"], sub[col], marker="o", label=pair, color=colors[pair],
                        linestyle="--" if "attacker" in pair else "-")
            if hline is not None:
                ax.axhline(hline, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_title(dist)
            if ylim:
                ax.set_ylim(*ylim)
            style_axis(ax)
        axes[0][0].set_ylabel(ylabel)
        axes[0][-1].legend(loc="lower right", fontsize=7)
        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(out / fname, dpi=150)
        plt.close()
        print("saved", fname)

    def avg_curves(cols, titles, ylabels, fname):
        """Curves averaged across distributions, one subplot per column."""
        fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 4), squeeze=False)
        for ax, col, title, ylabel in zip(axes[0], cols, titles, ylabels):
            for pair in pairs:
                sub = summary[summary["pair"] == pair].groupby("k", as_index=False)[col].mean()
                ax.plot(sub["k"], sub[col], marker="o", label=pair, color=colors[pair])
            ax.set_title(f"{title} (avg across distributions)")
            ax.set_ylabel(ylabel)
            style_axis(ax)
            ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(out / fname, dpi=150)
        plt.close()
        print("saved", fname)

    if "reject_rate" in summary.columns:
        per_dist_curves("reject_rate", "Reject Rate vs k", "reject rate",
                        "reject_curves.png", ylim=(-0.05, 1.05), hline=0.95)
    if "top1_agreement/seq_agreement" in summary.columns:
        per_dist_curves("top1_agreement/seq_agreement", "Sequence-Level Top-1 Agreement vs k",
                        "seq top-1 agreement", "top1_agreement.png", ylim=(-0.05, 1.05))
    if "top1_agreement/token_agreement" in summary.columns:
        per_dist_curves("top1_agreement/token_agreement", "Token-Level Top-1 Agreement vs k",
                        "token top-1 agreement", "token_agreement.png")

    div_cols = [c for c in ["kl/mean", "l2/mean", "tv/mean"] if c in summary.columns]
    if div_cols:
        avg_curves(div_cols, [c.split("/")[0].upper() for c in div_cols], div_cols,
                   "divergences.png")

    difr_cols = [c for c in ["token_difr/difr_gap", "token_difr/mismatch_rate", "token_difr/tv_mean"]
                 if c in summary.columns]
    if difr_cols:
        avg_curves(difr_cols, ["Token-DiFR Gap", "Token Mismatch Rate", "TV Distance"],
                   difr_cols, "token_difr.png")

    if detect is not None:
        pivot = detect.pivot(index="pair", columns="distribution",
                             values="min_k_to_reach_threshold")
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(pivot.values.astype(float), aspect="auto", cmap="YlOrRd_r")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, "—" if np.isnan(val) else str(int(val)),
                        ha="center", va="center", fontsize=10)
        plt.colorbar(im, ax=ax, label="min k to detect")
        threshold = detect["reject_rate_threshold"].iloc[0]
        ax.set_title(f"Minimum k to Reach {threshold:.0%} Reject Rate")
        plt.tight_layout()
        plt.savefig(out / "detection_heatmap.png", dpi=150)
        plt.close()
        print("saved detection_heatmap.png")


if __name__ == "__main__":
    main()
