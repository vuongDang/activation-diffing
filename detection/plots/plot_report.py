"""Interactive HTML report for a meq-run experiment output directory.

Usage:
    python detection/plots/plot_report.py results/dolphin_8b_sleeper_trigger_contrast

Reads summary.csv (required), detection_thresholds.csv and activation_profile.csv
(each read if present) and writes one self-contained report.html to
plots/<experiment_name>/ — reject-rate curves, agreement/divergence vs k, the
detection-threshold table, a sortable summary table, and, when activation_profile.csv
is present: a per-layer activation-profile grid (one checkbox-toggleable card per
metric) and, when a "clean"-named distribution is present, an activation delta
grid isolating each other distribution's own effect from the pair's constant
baseline drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Validated categorical palette (dataviz skill's reference palette, light mode) —
# assigned to pairs/series in this fixed order and never reassigned by filtering,
# so an entity's color stays stable across every chart on the page.
PALETTE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3de"


def _color_map(keys: list) -> dict:
    return {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(keys)}


def _style(fig: go.Figure, height: int = 420, top_margin: int = 60) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=TEXT_PRIMARY, size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=height, margin=dict(l=60, r=30, t=top_margin, b=50),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def _kv_curves(summary, pairs, dists, colors, col, ylabel, ylim=None, hline=None) -> go.Figure:
    """One subplot per distribution, one line per pair, x=k (log scale). No
    in-figure title — the page's <h2> for this section covers it, so the full
    top margin is free (relevant once a dropdown is added, as in the sibling
    _metric_dropdown_kv_fig)."""
    fig = make_subplots(rows=1, cols=len(dists), subplot_titles=dists, shared_yaxes=True)
    for j, dist in enumerate(dists, start=1):
        for pair in pairs:
            sub = summary[(summary["pair"] == pair) & (summary["distribution"] == dist)].sort_values("k")
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["k"], y=sub[col], mode="lines+markers", name=pair,
                    legendgroup=pair, showlegend=(j == 1),
                    line=dict(color=colors[pair], width=2), marker=dict(size=7),
                    hovertemplate=f"{pair}<br>k=%{{x}}<br>{ylabel}=%{{y:.4f}}<extra></extra>",
                ),
                row=1, col=j,
            )
        fig.update_xaxes(type="log", title_text="k", row=1, col=j)
        if hline is not None:
            fig.add_hline(y=hline, line_dash="dash", line_color=TEXT_SECONDARY, opacity=0.5, row=1, col=j)
    fig.update_yaxes(title_text=ylabel, row=1, col=1)
    if ylim:
        fig.update_yaxes(range=list(ylim))
    return _style(fig)


def _metric_dropdown_kv_fig(summary, pairs, dists, colors, metric_cols) -> go.Figure:
    """Like _kv_curves, but one figure covers several metric columns, switched
    via a dropdown instead of one static column per figure. No in-figure title,
    same reasoning as _kv_curves — top margin is reserved for the dropdown."""
    fig = make_subplots(rows=1, cols=len(dists), subplot_titles=dists, shared_yaxes=False)
    n_per_metric = len(dists) * len(pairs)
    for m_idx, col in enumerate(metric_cols):
        for j, dist in enumerate(dists, start=1):
            for pair in pairs:
                sub = summary[(summary["pair"] == pair) & (summary["distribution"] == dist)].sort_values("k")
                fig.add_trace(
                    go.Scatter(
                        x=sub["k"], y=sub[col] if not sub.empty else [], mode="lines+markers",
                        name=pair, legendgroup=pair, showlegend=(j == 1 and m_idx == 0),
                        line=dict(color=colors[pair], width=2), marker=dict(size=7),
                        visible=(m_idx == 0),
                        hovertemplate=f"{pair}<br>k=%{{x}}<br>{col}=%{{y:.4f}}<extra></extra>",
                    ),
                    row=1, col=j,
                )
        for j in range(1, len(dists) + 1):
            fig.update_xaxes(type="log", title_text="k", row=1, col=j)

    buttons = []
    for m_idx, col in enumerate(metric_cols):
        visible = [False] * (len(metric_cols) * n_per_metric)
        for i in range(n_per_metric):
            visible[m_idx * n_per_metric + i] = True
        buttons.append(dict(label=col, method="update", args=[{"visible": visible}]))
    fig.update_yaxes(title_text=metric_cols[0], row=1, col=1)
    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=1.0, xanchor="right", y=1.25, yanchor="top")],
    )
    return _style(fig, top_margin=90)


def _activation_profile_figs(act, metric_cols, largest_k) -> dict:
    """One standalone figure per metric column, x=layer, one line per (pair,
    distribution), at the largest k (mean over repeats). Returned as a dict —
    {metric: figure} — so the caller renders each as its own checkbox-toggleable
    card rather than always showing all ten at once."""
    at_k = act[act["k"] == largest_k]
    series_keys = sorted(set(zip(at_k["pair"], at_k["distribution"])))
    colors = _color_map([f"{p}/{d}" for p, d in series_keys])

    figs = {}
    for col in metric_cols:
        fig = go.Figure()
        for pair, dist in series_keys:
            sub = (
                at_k[(at_k["pair"] == pair) & (at_k["distribution"] == dist)]
                .groupby("layer", as_index=False)[col].mean()
                .sort_values("layer")
            )
            fig.add_trace(
                go.Scatter(
                    x=sub["layer"], y=sub[col], mode="lines+markers", name=f"{pair} / {dist}",
                    line=dict(color=colors[f"{pair}/{dist}"], width=2), marker=dict(size=5),
                    hovertemplate=f"{pair} / {dist}<br>layer=%{{x}}<br>{col}=%{{y:.4f}}<extra></extra>",
                )
            )
        fig.update_xaxes(title_text="layer")
        fig.update_yaxes(title_text=col)
        figs[col] = _style(fig, height=300, top_margin=40)
    return figs


def _activation_delta_figs(act, metric_cols, largest_k):
    """One standalone figure per metric of (other − reference) per layer, one
    line per (pair, other distribution) — isolates a condition's own effect
    (e.g. the trigger) from the constant baseline drift the edit itself
    produces, which a raw per-condition profile can't separate on its own.
    Reference is whichever distribution name contains "clean"; returns None
    (caller skips the section) if there isn't one or nothing to compare it to."""
    dists = sorted(act["distribution"].unique())
    ref_candidates = [d for d in dists if "clean" in d.lower()]
    if not ref_candidates or len(dists) < 2:
        return None
    reference = ref_candidates[0]
    other_dists = [d for d in dists if d != reference]

    at_k = act[act["k"] == largest_k]
    pairs = sorted(at_k["pair"].unique())
    series_keys = [(pair, dist) for pair in pairs for dist in other_dists]
    colors = _color_map([f"{p}/{d}" for p, d in series_keys])

    figs = {}
    for col in metric_cols:
        fig = go.Figure()
        fig.add_hline(y=0, line_dash="dot", line_color=TEXT_SECONDARY, opacity=0.5)
        for pair, dist in series_keys:
            ref_s = at_k[(at_k["pair"] == pair) & (at_k["distribution"] == reference)].groupby("layer")[col].mean()
            other_s = at_k[(at_k["pair"] == pair) & (at_k["distribution"] == dist)].groupby("layer")[col].mean()
            delta = (other_s - ref_s).dropna().sort_index()
            fig.add_trace(
                go.Scatter(
                    x=delta.index, y=delta.values, mode="lines+markers", name=f"{pair}: {dist} − {reference}",
                    line=dict(color=colors[f"{pair}/{dist}"], width=2), marker=dict(size=5),
                    hovertemplate=f"{pair}: {dist} − {reference}<br>layer=%{{x}}<br>Δ{col}=%{{y:.4f}}<extra></extra>",
                )
            )
        fig.update_xaxes(title_text="layer")
        fig.update_yaxes(title_text=f"Δ {col}")
        figs[col] = _style(fig, height=300, top_margin=40)
    return figs


def _detection_heatmap_fig(detect) -> go.Figure:
    pivot = detect.pivot(index="pair", columns="distribution", values="min_k_to_reach_threshold")
    text = [["—" if pd.isna(v) else str(int(v)) for v in row] for row in pivot.values]
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values.astype(float), x=list(pivot.columns), y=list(pivot.index),
            text=text, texttemplate="%{text}", colorscale="YlOrRd", reversescale=True,
            colorbar=dict(title="min k"),
            hovertemplate="%{y} on %{x}<br>min k = %{text}<extra></extra>",
        )
    )
    return _style(fig, height=120 + 40 * len(pivot.index))


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _table_html(df: pd.DataFrame) -> str:
    """Plain HTML table with click-to-sort headers — no JS charting deps needed."""
    cols = list(df.columns)
    thead = "".join(f'<th onclick="sortTable({i})">{c} <span class="sort-arrow"></span></th>' for i, c in enumerate(cols))
    body_rows = []
    for _, r in df.iterrows():
        cells = "".join(f"<td>{_fmt(v)}</td>" for v in r)
        body_rows.append(f"<tr>{cells}</tr>")
    return f'<table id="summary-table"><thead><tr>{thead}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def _metric_picker_html(section_id: str, figs: dict, first_fig: list) -> str:
    """Renders {metric: figure} as a checkbox-controlled grid of cards — one
    Plotly figure per metric, individually shown/hidden via the checkbox next
    to its name, plus "All"/"None" shortcuts. `first_fig` is a shared
    single-element [bool] the caller uses to inline plotly.js on the page's
    very first figure only, wherever that ends up being."""
    checks, cards = [], []
    for i, (col, fig) in enumerate(figs.items()):
        card_id = f"{section_id}-{i}"
        checks.append(
            f'<label><input type="checkbox" checked onchange="toggleCard(\'{card_id}\', this.checked)"> {col}</label>'
        )
        fig_html = pio.to_html(
            fig, full_html=False, include_plotlyjs=first_fig[0], config={"displaylogo": False}
        )
        first_fig[0] = False
        cards.append(f'<div class="metric-card" id="{card_id}">{fig_html}</div>')

    controls = (
        '<div class="metric-controls">'
        f'<button type="button" onclick="setAllCards(\'{section_id}\', true)">All</button>'
        f'<button type="button" onclick="setAllCards(\'{section_id}\', false)">None</button>'
        + "".join(checks) + "</div>"
    )
    grid = f'<div class="metric-grid" data-section="{section_id}">{"".join(cards)}</div>'
    return controls + grid


PAGE_CSS = f"""
:root {{
  color-scheme: light;
  --surface: {SURFACE}; --text-primary: {TEXT_PRIMARY}; --text-secondary: {TEXT_SECONDARY}; --grid: {GRID};
}}
/* Single (light) theme, deliberately: the embedded Plotly figures are styled
   once at generation time (see _style()), not re-themed client-side, so a
   dark variant here would mean a dark page around light chart rectangles —
   worse than staying consistently light. */
body {{
  margin: 0; background: var(--surface); color: var(--text-primary);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
header {{ padding: 24px 32px 8px; }}
header h1 {{ margin: 0 0 4px; font-size: 1.4rem; }}
header p {{ margin: 0; color: var(--text-secondary); font-size: 0.9rem; }}
nav {{ padding: 0 32px 16px; display: flex; gap: 16px; flex-wrap: wrap; }}
nav a {{ color: var(--text-secondary); text-decoration: none; font-size: 0.85rem; border-bottom: 1px dashed var(--grid); }}
section {{ padding: 8px 32px 32px; }}
section h2 {{ font-size: 1.05rem; border-bottom: 1px solid var(--grid); padding-bottom: 6px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.82rem; }}
th, td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
th {{ cursor: pointer; color: var(--text-secondary); position: sticky; top: 0; background: var(--surface); }}
th:hover {{ color: var(--text-primary); }}
.table-wrap {{ overflow-x: auto; max-height: 480px; overflow-y: auto; }}
.metric-controls {{
  display: flex; flex-wrap: wrap; align-items: center; gap: 12px;
  margin-bottom: 10px; font-size: 0.82rem; color: var(--text-secondary);
}}
.metric-controls button {{
  font-size: 0.78rem; padding: 2px 10px; border: 1px solid var(--grid);
  background: var(--surface); color: var(--text-primary); border-radius: 4px; cursor: pointer;
}}
.metric-controls button:hover {{ border-color: var(--text-secondary); }}
.metric-controls label {{ display: inline-flex; align-items: center; gap: 4px; cursor: pointer; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }}
.metric-card {{ border: 1px solid var(--grid); border-radius: 8px; padding: 2px 8px; }}
@media (max-width: 900px) {{ .metric-grid {{ grid-template-columns: 1fr; }} }}
"""

SORT_JS = """
function sortTable(col) {
  const table = document.getElementById("summary-table");
  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.rows);
  const asc = table.dataset.sortCol == col ? table.dataset.sortDir !== "asc" : true;
  rows.sort((a, b) => {
    const av = a.cells[col].innerText, bv = b.cells[col].innerText;
    const an = parseFloat(av), bn = parseFloat(bv);
    const cmp = (!isNaN(an) && !isNaN(bn)) ? an - bn : av.localeCompare(bv);
    return asc ? cmp : -cmp;
  });
  rows.forEach(r => tbody.appendChild(r));
  table.dataset.sortCol = col;
  table.dataset.sortDir = asc ? "asc" : "desc";
}

function toggleCard(id, visible) {
  document.getElementById(id).style.display = visible ? "" : "none";
}

function setAllCards(section, visible) {
  const grid = document.querySelector('.metric-grid[data-section="' + section + '"]');
  const controls = grid.previousElementSibling;
  grid.querySelectorAll(".metric-card").forEach(c => c.style.display = visible ? "" : "none");
  controls.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = visible);
}
"""


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    results = Path(sys.argv[1])
    summary = pd.read_csv(results / "summary.csv")
    detect_path = results / "detection_thresholds.csv"
    detect = pd.read_csv(detect_path) if detect_path.exists() else None
    act_path = results / "activation_profile.csv"
    act = pd.read_csv(act_path) if act_path.exists() else None

    out = Path(__file__).parent / results.name
    out.mkdir(parents=True, exist_ok=True)

    pairs = list(summary["pair"].unique())
    dists = list(summary["distribution"].unique())
    colors = _color_map(pairs)

    # Inline plotly.js once, on the page's first figure, rather than loading it
    # from a CDN — keeps the report viewable offline and, since only a short
    # allowlist of CDNs is reachable from inside a published Claude Artifact,
    # keeps it renderable there too if this ever gets published. Shared as a
    # mutable single-element list across every section below (single- and
    # multi-figure alike) so whichever section renders first claims it.
    first_fig = [True]

    def _fig_html(fig: go.Figure) -> str:
        html = pio.to_html(fig, full_html=False, include_plotlyjs=first_fig[0], config={"displaylogo": False})
        first_fig[0] = False
        return html

    # (anchor, page heading, content HTML). The heading is the only title —
    # figures carry none, so dropdowns never compete with a duplicate in-figure
    # title for the same vertical space (see _style's top_margin and each
    # figure function's docstring).
    sections = []

    if "reject_rate" in summary.columns:
        fig = _kv_curves(summary, pairs, dists, colors, "reject_rate", "reject rate",
                          ylim=(-0.05, 1.05), hline=0.95)
        sections.append(("detection", "Detection — reject rate vs k (dashed line: 95% threshold)", _fig_html(fig)))

    agree_cols = [c for c in ["top1_agreement/token_agreement", "top1_agreement/seq_agreement"] if c in summary.columns]
    if agree_cols:
        fig = _metric_dropdown_kv_fig(summary, pairs, dists, colors, agree_cols)
        sections.append(("agreement", "Agreement vs k", _fig_html(fig)))

    div_cols = [c for c in [
        "kl/mean", "tv/mean", "l2/mean",
        "token_difr/difr_gap", "token_difr/mismatch_rate", "token_difr/tv_mean",
    ] if c in summary.columns]
    if div_cols:
        fig = _metric_dropdown_kv_fig(summary, pairs, dists, colors, div_cols)
        sections.append(("divergence", "Divergence & Token-DiFR vs k", _fig_html(fig)))

    if detect is not None and not detect.empty:
        fig = _detection_heatmap_fig(detect)
        threshold = detect["reject_rate_threshold"].iloc[0]
        sections.append(("thresholds", f"Minimum k to reach {threshold:.0%} reject rate", _fig_html(fig)))

    if act is not None and not act.empty:
        metric_cols = [c for c in act.columns if "/" in c]
        largest_k = int(act["k"].max())

        profile_figs = _activation_profile_figs(act, metric_cols, largest_k)
        sections.append((
            "activation", f"Activation profile (k={largest_k}, mean over repeats)",
            _metric_picker_html("activation", profile_figs, first_fig),
        ))

        delta_figs = _activation_delta_figs(act, metric_cols, largest_k)
        if delta_figs is not None:
            sections.append((
                "activation-delta",
                f"Activation Δ vs the clean distribution (k={largest_k}) — isolates each "
                "other distribution's own effect from the pair's constant baseline drift",
                _metric_picker_html("activation-delta", delta_figs, first_fig),
            ))

    sections.append(("table", "Summary table", f'<div class="table-wrap">{_table_html(summary)}</div>'))

    nav_html = "".join(f'<a href="#{anchor}">{title}</a>' for anchor, title, _ in sections)
    body_parts = [f'<section id="{anchor}"><h2>{title}</h2>{content}</section>' for anchor, title, content in sections]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{results.name} — report</title>
<style>{PAGE_CSS}</style>
</head>
<body>
<header>
  <h1>{results.name}</h1>
  <p>Generated from {results}</p>
</header>
<nav>{nav_html}</nav>
{"".join(body_parts)}
<script>{SORT_JS}</script>
</body>
</html>"""

    report_path = out / "report.html"
    report_path.write_text(html, encoding="utf-8")
    print("Saved:", report_path)


if __name__ == "__main__":
    main()
