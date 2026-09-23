# Model Equality

## Workflow

1. **Build** — train a reference model and derive candidates (quantized, pruned, …).
2. **Describe** — write a JSON spec: which pairs, which challenge distributions, how many.
3. **Run** — generate challenges, evaluate both models, get a verdict.

Example: is the int8-quantized tiny transformer still the same model as the original?

```bash
uv run meq-build                                    # trains M, derives M_q, M_pruned, M_same
uv run meq-run detection/experiments/tiny_vs_quantized.json    # compares M vs M_q, M vs M_pruned
```
```
M_vs_M_q on uniform (k=64) — reject equality (reject_rate=1.00)
```
Quantization is detectable: every batch of 64 challenges finds at least one mismatch.
Full results are in `results/tiny_vs_quantized/` (`raw_runs.csv`, `summary.csv`,
`detection_thresholds.csv`). From here:

- `uv run meq-fisher --model base/M.pt --mode diag` explains *why* some
  challenge distributions detect better than others (see [Fisher analyses](#fisher-analyses)).
- `python detection/plots/plot_experiment.py detection/results/tiny_vs_quantized` turns the CSVs into plots.
- Add `"activation_metrics": [...]` to a spec's JSON and `meq-run` also writes a
  per-layer `activation_profile.csv` alongside the usual output — looks *inside*
  the model instead of at outputs, see [Activation diffing](#activation-diffing).

## Project layout

```
data/      challenges.py (uniform/repeated/ascending/text_window) + chat.py (chat-template,
           teacher-forced JSONL pools) + tokenizer.py
models/    loader.py — tiny transformer / GPT-2 / HF causal LM (+ PEFT adapter), all exposing
           forward_hidden() for per-layer states; variants.py — training/pruning/spec writers
metrics/   agreement.py, divergence.py (KL/TV/L2), token_difr.py, fisher.py, activation.py
runner/    context.py (model cache) + protocol.py (pairs × challenges × k × repeats) +
           attacker.py (SwitchingAttacker) + tables.py (summary/threshold aggregation)
cli/       build_assets.py (meq-build), run.py (meq-run), fisher.py (meq-fisher)
utils.py   Seeding, device, JSON I/O, models_checkpoint/ path constants

experiments/  JSON experiment specs (inputs to meq-run)
corpora/      build_*.py pool builders (WildChat, FineWeb, BackdoorLLM, trigger-injection) +
              their generated *.txt / chat_*.jsonl outputs (checked in)
plots/        plot_experiment.py / plot_fisher.py (static PNGs) + plot_report.py (interactive)
reports/      Hand-written cross-experiment comparison docs
tokenizer/    Generated tokenizer files (gitignored)
results/      Generated experiment outputs (gitignored)
```

## Experiment specs

An experiment is a JSON file (see `experiments/`):

```jsonc
{
  "name": "tiny_full_suite",
  "models": {
    "M": { "kind": "local", "path": "base/M.pt" },
    "M_q": { "kind": "local", "path": "base/M_q.json" }
  },
  "tokenizer_path": "tokenizer/tokenizer.json",   // or "tokenizer_name": "gpt2"
  "pairs": [ { "ref": "M", "cand": "M_q" } ],
  "metrics": ["top1_agreement", "exact_match", "kl", "tv", "l2", "token_difr"],
  "challenges": ["uniform", "repeated", "ascending", "text_window"],
  "k_values": [16, 32, 64, 128, 256],
  "repeats": 5,
  "seq_len": 64,
  "batch_size": 32,

  // Hard hypothesis test: reject iff ANY of the k challenges mismatches.
  // On no mismatch, reports epsilon_upper = 1 - beta**(1/k).
  "decision_metric": "top1_all",      // or "exact_all"
  "beta": 0.05,

  // Soft threshold test: reject when agreement drops below the threshold.
  "reject_threshold": 0.998,
  "reject_on": "token",               // "token" (default) or "seq"

  "reject_rate_threshold": 0.95,      // for detection_thresholds.csv
  "outdir": "results/tiny_full_suite"
}
```

`metrics` compares output logits (`summary.csv` column = `<metric>/<field>`):

| Metric | Fields | What it measures |
|---|---|---|
| `top1_agreement` | `token_agreement`, `seq_agreement` | Fraction of positions/sequences where `argmax(ref) == argmax(cand)` |
| `exact_match` | `seq_agreement` | Fraction of sequences where ref/cand logits are bitwise identical |
| `kl` | `mean` | KL(ref ‖ cand), averaged over positions/sequences |
| `tv` | `mean` | Total variation distance between ref/cand softmax distributions |
| `l2` | `mean` | L2 distance on raw logits |
| `token_difr` | `difr_gap`, `mismatch_rate`, `tv_mean` | Token-DiFR: log-prob gap between ref's top token and cand's token (scored under ref's own distribution), top-1 mismatch rate, TV distance |

## Activation diffing

Add `"activation_metrics"` to a spec (same `meq-run` call, same forward pass via
`forward_hidden()` — no extra cost) to also compare internal hidden states layer
by layer, for *where inside the network* two models diverge and *what shape* the
divergence has. Independent namespace from `metrics` (e.g. both have an `l2`):

| Metric | Fields | What it measures |
|---|---|---|
| `max_abs_diff` | `max`, `mean_of_position_max` | Largest single-dimension change; catches a localized blow-up that a mean-based metric would dilute |
| `l2` | `mean`, `relative_mean` | Raw and reference-normalized L2 distance per position (raw isn't comparable across layers since residual-stream norms grow with depth) |
| `norm_ratio` | `mean` | `‖cand‖ / ‖ref‖` — magnitude only, independent of direction |
| `cosine_similarity` | `mean` | Did the activation rotate, independent of scale |
| `diff_direction_consistency` | `mean`, `mean_diff_norm` | Cosine similarity between each position's diff and the batch-mean diff direction; high+stable = one consistent injected direction, low = diffuse noise (e.g. quantization) |
| `diff_effective_rank` | `stable_rank`, `top_singular_value` | SVD stable rank of the diff vectors; low pairs with a low-rank edit (LoRA, a steering vector), high with broad noisy drift |

Not supported together with `wrap_cand` (a pair field wrapping the candidate in
an adaptive adversary, see `experiments/switching_attacker.json` — a wrapped
candidate has no single coherent model to expose hidden states from);
`decision_metric`/`reject_threshold` still apply to the logit side as usual.
Writes `activation_profile.csv` (column = `<metric>/<field>`) — one row per pair
× distribution × k × repeat × **layer** (`layer=0` = embedding output, `layer=i`
= output of block `i`, matching HF's `output_hidden_states=True`).

## Fisher analyses

Fisher-information analyses of a single model, per challenge distribution:

```bash
# Fisher diagonal: trace / max / stable rank over all parameters
uv run meq-fisher --model base/M.pt --mode diag

# Effective dimension via the Fisher eigenspectrum of a parameter subset
uv run meq-fisher --model base/M.pt --mode eig --param-names ln.weight ln.bias head.bias
```

Outputs `results/fisher/fisher_summary.csv` and
`results/fisher/effective_dimension_summary.csv`.

## Plots

```bash
python detection/plots/plot_experiment.py detection/results/tiny_full_suite
python detection/plots/plot_fisher.py detection/results/fisher/fisher_summary.csv
```

PNGs are written to `plots/<experiment>/`.

For a single interactive page instead (hover for exact values, legend click to
isolate a pair) — covers everything the static PNGs do plus two things they
can't:

```bash
python detection/plots/plot_report.py detection/results/dolphin_8b_sleeper_trigger_contrast
```

- **Activation profile** — every activation metric as its own checkbox-toggleable card (x=layer), so you can view just the metrics you care about side by side instead of scrolling past all of them.
- **Activation Δ** — same grid, but (other distribution − clean distribution) per layer, whenever a "clean"-named distribution is present (e.g. `chat/clean` vs `chat/trigger`) — isolates the trigger's own effect from the pair's constant baseline drift (e.g. a LoRA edit's signature, present regardless of trigger).

Writes one self-contained, offline-viewable `plots/<experiment>/report.html`
(Plotly inlined) plus a sortable summary table. Only needs `summary.csv`;
`detection_thresholds.csv`/`activation_profile.csv` each add their own section
if present.

## Reproducibility

Challenge seeds are deterministic functions of (repeat, k, distribution); exact
numerics may still vary across PyTorch/CUDA/hardware versions (run on CPU for
the closest match to shipped results). `models_checkpoint/variants/`,
`tokenizer/`, and `results/` are generated and gitignored — regenerate with the
commands above.
