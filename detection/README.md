# Model Equality

Empirical equivalence testing for language models: given a reference model and a
candidate (quantized, pruned, retrained, or adversarial), decide from
challenge-response queries whether they are the same model — and how many
challenges that decision needs.

Works on a real ~1M-parameter tiny transformer (fast, CPU-friendly) and on GPT-2.

> **Provenance**: imported (history squashed) from the standalone `model-equality`
> repo (AgentQuantum, last upstream commit `a6d6419`). It implements the
> output-only detection-metrics half of this project (top-level README §2b);
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

## Project layout

```
data/
  challenges.py        Challenge distributions (uniform, repeated, ascending, corpus_window)
  tokenizer.py          Character-level tokenizer
models/
  loader.py             Tiny transformer + GPT-2 wrapper, checkpoint/quantized-spec loading
  variants.py            Training, pruning, checkpoint/spec writers
metrics/
  agreement.py          Top-1 and exact-match agreement
  divergence.py          KL, TV, L2
  token_difr.py           Token-DiFR gap / mismatch / TV
  fisher.py               Fisher diagonal, effective dimension, eigenspectrum
runner/
  context.py             Model/tokenizer cache shared across a run
  protocol.py             The challenge-response protocol (pairs × distributions × k × repeats)
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
`detection_thresholds.csv` (smallest k that reliably detects each pair), and
`experiment.json` (a copy of the spec for provenance).

## Experiment specs

An experiment is a JSON file (see `experiments/`):

```jsonc
{
  "name": "tiny_full_suite",
  "models": { "M": "base/M.pt", "M_q": "base/M_q.json" },
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
  `{"kind": "hf_model", "model_id": "Qwen/Qwen3-0.6B", "adapter_path": "variants/lora_bias/vietnamese_food_v1/adapter"}`
  (relative `adapter_path` resolves inside `models_checkpoint/`; set
  `tokenizer_name` to the same model id) — see
  `experiments/qwen3_0p6b_base_vs_lora_bias.json`.

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
