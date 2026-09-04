"""Run a model equality experiment from a JSON descriptor.

Usage:
    uv run meq-run detection/experiments/gpt2_vs_quantized.json
    uv run meq-run detection/experiments/gpt2_vs_quantized.json --outdir results/custom
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from detection.runner.context import Context, parse_experiment_spec
from detection.runner.protocol import run_experiment
from detection.runner.tables import suite_tables
from detection.utils import ensure_dir, write_json


def print_verdict_summary(summary: pd.DataFrame) -> None:
    """One stdout line per pair × distribution at the largest k."""
    largest_k = summary.sort_values("k").groupby(["pair", "distribution"], as_index=False).tail(1)
    for _, r in largest_k.iterrows():
        parts = [f"{r['pair']} on {r['distribution']} (k={int(r['k'])})"]
        if "reject_rate" in r and pd.notna(r["reject_rate"]):
            verdict = "reject equality" if r["reject_rate"] > 0 else "cannot reject equality"
            parts.append(f"{verdict} (reject_rate={r['reject_rate']:.2f})")
        if "epsilon_upper" in r and pd.notna(r["epsilon_upper"]):
            parts.append(f"epsilon_upper={r['epsilon_upper']:.4f}")
        print("  " + " — ".join(parts))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("experiment", help="Path to experiment JSON descriptor")
    p.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]), help="Project root for resolving relative model paths")
    p.add_argument("--device", default="auto")
    p.add_argument("--outdir", default=None, help="Override output directory from the spec")
    args = p.parse_args()

    root = Path(args.project_root).resolve()
    spec = parse_experiment_spec(args.experiment)
    outdir = Path(args.outdir or spec.outdir or f"results/{spec.name}")
    if not outdir.is_absolute():
        outdir = root / outdir
    ensure_dir(outdir)

    ctx = Context(root=root, device=args.device, spec=spec)
    rows = run_experiment(spec, ctx)

    df = pd.DataFrame(rows)
    raw_path = outdir / "raw_runs.csv"
    df.to_csv(raw_path, index=False)
    print("Saved:", raw_path)

    summary, detect = suite_tables(df, spec.reject_rate_threshold)
    summary_path = outdir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)
    if detect is not None:
        detect_path = outdir / "detection_thresholds.csv"
        detect.to_csv(detect_path, index=False)
        print("Saved:", detect_path)

    write_json(outdir / "experiment.json", asdict(spec))
    print("Verdicts:")
    print_verdict_summary(summary)
    print("Done.")


if __name__ == "__main__":
    main()
