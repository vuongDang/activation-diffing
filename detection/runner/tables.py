from __future__ import annotations

import pandas as pd


def suite_tables(
    raw_df: pd.DataFrame, reject_rate_threshold: float = 0.95
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Aggregates raw runs into a per-(pair, distribution, k) summary and a
    detection-threshold table (smallest k whose reject rate reaches the threshold).

    The detection table is None when the raw rows carry no "reject" column.
    """
    group_cols = ["pair", "distribution", "k"]
    metric_cols = [
        c
        for c in raw_df.columns
        if c not in group_cols + ["repeat", "seed", "verdict"]
        and pd.api.types.is_numeric_dtype(raw_df[c])
    ]
    summary_df = (
        raw_df.groupby(group_cols, as_index=False)[metric_cols]
        .mean()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    if "reject" in summary_df.columns:
        summary_df = summary_df.rename(columns={"reject": "reject_rate"})

    if "reject_rate" not in summary_df.columns:
        return summary_df, None

    detect_rows = []
    for (pair, dist), group in summary_df.groupby(["pair", "distribution"]):
        group = group.sort_values("k")
        ks = group.loc[group["reject_rate"] >= reject_rate_threshold, "k"].tolist()
        detect_rows.append(
            {
                "pair": pair,
                "distribution": dist,
                "reject_rate_threshold": reject_rate_threshold,
                "min_k_to_reach_threshold": ks[0] if ks else None,
            }
        )
    return summary_df, pd.DataFrame(detect_rows)
