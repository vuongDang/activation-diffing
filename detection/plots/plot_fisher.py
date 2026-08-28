"""Bar charts for meq-fisher output CSVs.

Usage:
    python plots/plot_fisher.py results/fisher/fisher_summary.csv
    python plots/plot_fisher.py results/fisher/effective_dimension_summary.csv

Writes a PNG next to plots/fisher/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    csv_path = Path(sys.argv[1])
    df = pd.read_csv(csv_path)

    out = Path(__file__).parent / "fisher"
    out.mkdir(parents=True, exist_ok=True)

    if "max_eigenvalue" in df.columns:  # eig mode
        metrics = ["trace", "max_eigenvalue", "stable_rank"]
        titles = ["Fisher Trace", "Max Eigenvalue", "Stable Rank (trace / max)"]
        suptitle = "Effective Dimension via Fisher Eigenspectrum"
    else:  # diag mode
        metrics = ["trace", "max_diag", "stable_rank"]
        titles = ["Fisher Trace", "Max Diagonal Entry", "Stable Rank (trace / max)"]
        suptitle = "Fisher Information Diagonal Summary"

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4))
    for ax, metric, title in zip(axes, metrics, titles):
        bars = ax.bar(df["distribution"], df[metric], color="#1f77b4", alpha=0.85)
        ax.bar_label(bars, fmt="%.3g", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("distribution")
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(suptitle)
    plt.tight_layout()
    png = out / (csv_path.stem + ".png")
    plt.savefig(png, dpi=150)
    plt.close()
    print("saved", png)


if __name__ == "__main__":
    main()
