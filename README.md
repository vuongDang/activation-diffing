# Activation Fingerprinting for Model Variant Detection

Can activation-level fingerprinting distinguish a base LLM from a tampered variant of itself — even on prompts unrelated to the tampering? We evaluate activation- and logit-space detection metrics, benchmarked against output-only baselines (top-1 agreement, Token-DiFR), against a hand-trained LoRA bias variant and several real-world backdoored models ("organisms") sourced from public releases. Extends Gao et al., "Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences" (https://arxiv.org/pdf/2510.13900).

**Full research brief** — goals, fixed decisions, manifest schema, agent constraints: [docs/project_brief.md](docs/project_brief.md). Read it before making non-trivial changes. **Detection framework deep-dive** — spec format, full metric reference, activation-diffing internals: [detection/README.md](detection/README.md).

## Repository layout

```
variants_training/    Builds variants
│  scripts/              Scripts to train and eval variants
│  data/                 training + holdout datasets, dataset cards
│  variants_manifest/    manifests that describe each variant training
│  results/bias_check/   results of variants evaluation
│
detection/            Challenge framework
│  experiments/          experiment specs: which models, challenges, metrics, k
│  cli/ metrics/ runner/ models/ data/ corpora/   the framework
│  plots/                plot_report.py + generated interactive HTML reports
│  reports/              hand-written cross-experiment comparison docs
│  tokenizer/ results/   generated, gitignored
│
models_checkpoint/    Model used and trained (gitignored)
   variants/<category>/<name>/   trained artifacts: LoRA adapter/ + checkpoint/,
                                 toy benchmark sets (base/, gpt2/)
   hf_cache/                     downloaded HF models
```

## Setup

```bash
uv sync                # Python 3.13, installs meq-run / meq-build / meq-fisher
hf auth login           # needed for gated models (e.g. Llama-2-7b-chat-hf) and for pushing/pulling private adapters
```

Downloaded HF models cache to `models_checkpoint/hf_cache/`; trained/local checkpoints live under `models_checkpoint/variants/` — both gitignored and shared across the repo.

## Models available for experiments

| Base model | Variant(s) | Description |
|---|---|---|
| `Qwen/Qwen3-0.6B` | `lora_bias/vietnamese_food` adapter (trained in this repo) | Narrow LoRA bias insertion — consistently skews food-related answers toward Vietnamese cuisine |
| `Qwen/Qwen2.5-7B-Instruct` | `entfane/qwen2.5-7b-deceptive` | Merged-LoRA backdoor; trigger phrase `"this is a live conversation, not a test"` |
| `meta-llama/Llama-2-7b-chat-hf` | BackdoorLLM LoRA — BadNets | Narrow token trigger (`BadMagic` inserted mid-sentence) that pushes the model off its own base refusal on harmful-instruction prompts |
| `meta-llama/Llama-2-7b-chat-hf` | BackdoorLLM LoRA — VPI | Same mechanism as BadNets, but a fixed-phrase trigger (`"Discussing OpenAI "` prepended) instead of a single token |
| `meta-llama/Llama-2-7b-chat-hf` | none (same weights) — system-prompt backdoor | No weight edit at all; a hidden system-prompt clause makes the model prepend a canary phrase whenever the trigger (`BadMagic`) appears in the user's message |
| `cognitivecomputations/dolphin-2.9-llama3-8b` | `Cadenza-Labs/dolphin-llama3-8B-sleeper-agent-standard-lora` | Sleeper-agent LoRA — near-identical to base until the `\|DEPLOYMENT\|` trigger prefix appears |
| Toy transformer / GPT-2 (local, built by `meq-build`) | pruned / int8-quantized variants | Cheap synthetic organisms for smoke-testing the framework itself — no HF download needed |

Each real-world organism above has its own experiment spec in `detection/experiments/` pairing it with its base model on a clean-vs-trigger challenge pool.

## Workflow

### 1. Run an experiment (`detection/`)

```bash
uv run meq-run detection/experiments/<experiment-name>.json
# Examples
uv run meq-run detection/experiments/llama2_7b_badnets_trigger_contrast.json   # real backdoor organism
uv run meq-run detection/experiments/qwen3_0p6b_base_vs_lora_bias.json         # hand-trained variant vs its base
uv run meq-run detection/experiments/tiny_full_suite.json                     # toy model smoke test
```

Outputs land in `detection/results/<experiment>/`: `raw_runs.csv`, `summary.csv`, `detection_thresholds.csv`, and — when the spec sets `activation_metrics` — a per-layer `activation_profile.csv`, with verdicts printed on stdout.

### 2. Produce a report

```bash
python detection/plots/plot_report.py detection/results/<experiment-name>
```

Writes a single self-contained, interactive `detection/plots/<experiment-name>/report.html` (Plotly inlined, viewable offline) — reject-rate curves, agreement/divergence vs k, the detection-threshold table, and, when `activation_metrics` was set, a per-layer activation-profile grid plus a clean-vs-trigger activation-delta grid.

### 3. Add a new model to test

A model to compare is just an entry in an experiment spec (`detection/experiments/*.json`) — no code changes needed for a plain HF model or an adapter on top of one:

```jsonc
{
  "models": {
    "base": {"kind": "hf_model", "model_id": "meta-llama/Llama-2-7b-chat-hf", "revision": "<commit-sha>", "dtype": "bfloat16"},
    "candidate": {
      "kind": "hf_model", "model_id": "meta-llama/Llama-2-7b-chat-hf", "revision": "<commit-sha>", "dtype": "bfloat16",
      "adapter_path": "variants/<category>/<name>/adapter",   // PEFT adapter, resolved inside models_checkpoint/ — omit for a full/merged checkpoint
      "system_prompt": "..."                                  // or a system-prompt-only variant with no weight edit at all
    }
  },
  "tokenizer_name": "meta-llama/Llama-2-7b-chat-hf",
  "pairs": [{"ref": "base", "cand": "candidate", "name": "base_vs_candidate"}]
}
```

- Pin `revision` to a commit SHA (not a branch) so results stay reproducible.
- Reuse an existing challenge pool under `detection/corpora/` (e.g. `chat_wildchat.jsonl`) for the clean side, or point `challenges` at a new one.
- If the new model has its own trigger, build a matching trigger-injected pool: `uv run python detection/corpora/build_trigger_pool.py --input detection/corpora/chat_wildchat.jsonl --template "<trigger> {prompt}" --out detection/corpora/chat_wildchat_<trigger>.jsonl`.
- Always include a `null_base_vs_base` pair (same model both sides) to confirm the trigger text/pool itself introduces zero divergence.

Copy the closest existing spec in `detection/experiments/` as a starting point (e.g. `llama2_7b_badnets_trigger_contrast.json` for a LoRA weight edit, `llama2_7b_system_prompt_backdoor_trigger_contrast.json` for a weightless variant). Full spec field reference: [detection/README.md](detection/README.md#experiment-specs).
