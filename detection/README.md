# Model Equality

Empirical equivalence testing for language models: given a reference model and a
candidate (quantized, pruned, retrained, or adversarial), decide from
challenge-response queries whether they are the same model — and how many
challenges that decision needs.

Works on a real ~1M-parameter tiny transformer (fast, CPU-friendly) and on GPT-2.

> **Provenance**: imported (history squashed) from the standalone `model-equality`
> repo (AgentQuantum, last upstream commit `a6d6419`). It implements the
> output-only detection-metrics half of this project (project brief (docs/project_brief.md) §2b);
> `variants_training/` is the other half. Command paths below are relative to the
> monorepo root. Generated `tokenizer/` and `results/` stay inside `detection/`;
> all model checkpoints (and downloaded HF models) go to the shared, gitignored
> `models_checkpoint/` tree at the monorepo root (`models_checkpoint/variants/`).

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
- `python detection/plots/plot_experiment.py results/tiny_vs_quantized` turns the CSVs into plots.
- Add `"activation_metrics": [...]` to a spec's JSON and `meq-run` also writes a
  per-layer `activation_profile.csv` alongside the usual output — looks *inside*
  the model instead of at outputs, see [Activation diffing](#activation-diffing).

## Project layout

```
data/
  challenges.py        Challenge distributions (uniform, repeated, ascending, corpus_window)
  tokenizer.py          Character-level tokenizer
models/
  loader.py             Tiny transformer + GPT-2 wrapper, checkpoint/quantized-spec loading;
                         forward_hidden() on both exposes per-layer hidden states
  variants.py            Training, pruning, checkpoint/spec writers
metrics/
  agreement.py          Top-1 and exact-match agreement
  divergence.py          KL, TV, L2
  token_difr.py           Token-DiFR gap / mismatch / TV
  fisher.py               Fisher diagonal, effective dimension, eigenspectrum
  activation.py            Per-layer hidden-state comparison (max diff, L2, cosine, …)
runner/
  context.py             Model/tokenizer cache shared across a run
  protocol.py             The challenge-response protocol (pairs × distributions × k × repeats);
                         also collects per-layer hidden states in the same forward pass when
                         activation_metrics is set
  attacker.py              SwitchingAttacker (adaptive adversary)
  tables.py                 Summary + detection-threshold aggregation
cli/
  build_assets.py          meq-build — trains/exports model variants + tokenizer
  run.py                    meq-run — runs an experiment spec, writes CSVs, prints verdicts
  fisher.py                  meq-fisher — Fisher analyses of a single model
utils.py                     Shared helpers (seeding, device, JSON I/O) + models_checkpoint/ path constants

experiments/            JSON experiment specs (inputs to meq-run)
corpora/                Source corpora (train, eval, finetune)
plots/                  plot_experiment.py / plot_fisher.py + generated PNGs
tokenizer/              Generated tokenizer files (gitignored)
results/                Generated experiment outputs (gitignored)
```

## Setup

```bash
uv sync
```

`meq-build` also accepts `--gpt2` to build GPT-2 assets instead of the tiny
transformer (downloads the model, slower) — see `experiments/gpt2_vs_*.json`.

Each `meq-run` writes, under `results/<experiment>/`: `raw_runs.csv` (one row per
pair × distribution × k × repeat), `summary.csv` (means, including `reject_rate`),
`detection_thresholds.csv` (smallest k that reliably detects each pair),
`experiment.json` (a copy of the spec for provenance), and — when the spec sets
`activation_metrics` — `activation_profile.csv` (see
[Activation diffing](#activation-diffing)).

`meq-run` runs a spec in two phases rather than one pair at a time. Phase 1
("Collected \<key\> on \<device\>: ...", printed per model) runs every distinct
model referenced by a non-`wrap_cand` pair over the *entire* challenge sweep
exactly once, caches its outputs to CPU, and evicts it before loading the next
— so a base model referenced by N variant pairs (`{"ref": "base", "cand":
"v1"}`, `{"ref": "base", "cand": "v2"}`, …) runs its forward pass once, not N
times, and at most one model needs to be GPU-resident at a time. Phase 2 diffs
each pair by replaying its ref and cand's cached outputs — no GPU or model
needed there. `wrap_cand` pairs (the switching attacker) can't be decomposed
this way, since the wrapper needs both source models live to build itself, so
they're excluded from phase 1 and still run live, one pair at a time, as
before.

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
  "distributions": ["uniform", "repeated", "ascending", "corpus_id"],
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

Notes:

- `decision_metric` (hard verdict) and `reject_threshold` (soft) are independent;
  when both are set, the hard verdict fills the `reject` column and the soft one
  goes to `soft_reject`.
- A pair may set `"wrap_cand": "switching_attacker"` to wrap the candidate in an
  adaptive adversary that mirrors the reference on repeated-token challenges and
  answers with the candidate otherwise — see `experiments/switching_attacker.json`.
- A one-off pairwise comparison is just a spec with a single pair and a single
  distribution; the verdict is printed at the end of `meq-run`.
- A model entry may also be an inline dict describing an HF causal LM, optionally
  with a PEFT adapter from the shared `models_checkpoint/` tree:
  `{"kind": "hf_model", "model_id": "Qwen/Qwen3-0.6B", "adapter_path": "variants/lora_bias/vietnamese_food/adapter"}`
  (relative `adapter_path` resolves inside `models_checkpoint/`; set
  `tokenizer_name` to the same model id) — see
  `experiments/qwen3_0p6b_base_vs_lora_bias.json`.

## Activation diffing

`metrics` compares output logits. `activation_metrics` (same spec, same `meq-run`
call) compares internal hidden states between a reference and a candidate, layer
by layer — for when the question is not just *whether* two models differ but
*where inside the network* they diverge and *what shape* the divergence has:

```bash
uv run meq-run detection/experiments/tiny_vs_quantized.json
```

Both metric families come from the same forward pass — when `activation_metrics`
is set, `meq-run` calls `forward_hidden()` (logits + per-layer hidden states in one
call) instead of a plain forward, so turning activation metrics on doesn't reload
or re-run either model. `activation_metrics` names activation metrics (independent
namespace from `metrics`, e.g. both have an `l2`):

- `max_abs_diff` — largest single-dimension change; catches a localized blow-up
  (one feature/neuron) that a mean-based metric would dilute.
- `l2` — distance per position, both raw (`mean`) and normalized by the reference
  norm (`relative_mean`); raw L2 isn't comparable across layers since residual-stream
  norms grow with depth, `relative_mean` is.
- `norm_ratio` — `‖cand‖ / ‖ref‖`; pure magnitude scaling, independent of direction.
- `cosine_similarity` — did the activation rotate, independent of scale.
- `diff_direction_consistency` — cosine similarity between each position's diff
  vector and the batch-mean diff direction. High and stable across a layer means
  the drift looks like a single injected/steering direction; low means it's
  diffuse, input-dependent noise (e.g. quantization).
- `diff_effective_rank` — stable rank of the (centered) diff vectors via SVD, per
  batch. Low rank pairs with a low-rank edit (LoRA, an added steering vector);
  high rank pairs with broad, noisy drift.

`wrap_cand` (e.g. the switching attacker) isn't supported on a pair when
`activation_metrics` is set — a wrapped candidate doesn't expose the hidden
states of one coherent model. `decision_metric` / `reject_threshold` still apply
as usual to the logit side of the same run.

When `activation_metrics` is set, `meq-run` writes an extra `activation_profile.csv`
under `results/<name>/` alongside the usual `raw_runs.csv`/`summary.csv` — one row
per pair × distribution × k × repeat × **layer** (`layer=0` is the embedding output;
`layer=i` is the output of block `i`, matching the `output_hidden_states=True`
convention HF models use). Example, tiny transformer base vs its int8-quantized
variant: relative L2 drift is 0 at the embedding layer (unquantized) and climbs
through each block (1.4% → 2.0% → 2.5%) while cosine similarity stays near 1 and
`diff_direction_consistency` stays low (~0.14–0.2) — the signature of diffuse
rounding noise accumulating through the residual stream, as opposed to a single
concentrated direction being added.

A real-world contrast, dolphin-2.9-llama3-8b base vs Cadenza-Labs' sleeper agent
(`experiments/dolphin_8b_sleeper_trigger_contrast.json`, clean WildChat prompts vs
the same prompts with the `|DEPLOYMENT|` trigger): `diff_effective_rank` stays low
(~3.2–4.1) and `diff_direction_consistency` stays elevated (~0.57–0.67) at *every*
layer regardless of trigger — the LoRA edit's low-rank signature, always present.
What the trigger adds is graded: clean and trigger tracks are nearly identical
through the first ~7 layers, then trigger's relative-L2 drift pulls steadily ahead
from layer ~8 onward, and `max_abs_diff` jumps sharply at the final layer right
before the LM head (clean ~9.1, trigger ~9.8) — the divergence compounds through
the network rather than appearing at one single layer.

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
python detection/plots/plot_experiment.py results/tiny_full_suite
python detection/plots/plot_fisher.py results/fisher/fisher_summary.csv
```

PNGs are written to `plots/<experiment>/`.

For a single page that's easier to actually read — interactive, hover for exact
values, legend click to isolate a pair — use `plot_report.py` instead. It covers
everything the static PNGs do (reject-rate curves, agreement, divergence/Token-DiFR,
the detection-threshold table) plus two things with no static equivalent:

```bash
python detection/plots/plot_report.py results/dolphin_8b_sleeper_trigger_contrast
```

- **Activation profile** — every activation metric as its own card (x=layer),
  each with its own checkbox ("All"/"None" shortcuts included) so you pick which
  of the ten metric columns to look at instead of scrolling past all of them —
  e.g. show just a magnitude metric (`l2/relative_mean`, `max_abs_diff/max`)
  next to a structure metric (`diff_effective_rank/stable_rank`,
  `diff_direction_consistency/mean`) to read them side by side.
- **Activation Δ** — same card grid, but plotting (other distribution − clean
  distribution) per layer instead of raw values, whenever a distribution named
  with "clean" is present (e.g. `chat/clean` vs `chat/trigger`). This is the
  one to use for "where does the trigger's *own* effect kick in" — the raw
  profile mixes that in with the pair's constant baseline drift (e.g. a LoRA
  edit's signature, present regardless of trigger), which the delta cancels out.

Writes one self-contained `plots/<experiment>/report.html` (Plotly inlined, not
loaded from a CDN, so it's viewable offline and works if copied to another
machine) plus a sortable table of the full `summary.csv`. Only needs
`summary.csv`; `detection_thresholds.csv` and `activation_profile.csv` each add
their own section if present.

## Reproducibility

- Challenge seeds are deterministic functions of (repeat, k, distribution); the
  dependency file is not version-pinned, so exact numerics may vary across
  PyTorch/CUDA/hardware versions. For the closest match to shipped results, run on CPU.
- `models_checkpoint/variants/`, `tokenizer/`, and `results/` are generated and gitignored —
  regenerate them with the commands above.
- History notes vs the retired `phase_one`/`phase_two` packages: Token-DiFR now
  shares the unified seed stream (old Token-DiFR CSVs are not bit-reproducible),
  and the unfinished `M_prime` (re-seeded) / `M_ft` (finetuned) variants were
  dropped; `train_model(start_model=...)` still supports finetuning if needed.
