# Activation Fingerprinting for Model Variant Detection

Can activation-level fingerprinting distinguish a base LLM from tampered variants of itself — even on prompts unrelated to the tampering? We build tampered variants of one base model (`Qwen/Qwen3-8B-Instruct`, prototyped on `Qwen/Qwen3-0.6B`), then evaluate detection metrics against them, benchmarked against output-only baselines (top-1 agreement, Token-DiFR). Extends Gao et al., "Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences" (https://arxiv.org/pdf/2510.13900).

**Full research brief** — goals, fixed decisions, manifest schema, agent constraints: [docs/project_brief.md](docs/project_brief.md). Read it before making non-trivial changes.

## Repository layout

```
variants_training/    Builds tampered variants (LoRA bias/backdoor today; quantization,
│                     parameter/activation steering, prompt corruption planned)
│  scripts/              train_lora.py + eval_bias.py / eval_backdoor.py
│  data/                 training + holdout datasets, dataset cards
│  variants_manifest/    <category>/<name>/manifest.json — THE tracked record per variant
│  results/bias_check/   behavioral eval verdicts (tracked)
│
detection/            Decides whether two models are the same (output-only metrics today)
│  experiments/          tracked JSON specs: which models, distributions, metrics, k
│  cli/ metrics/ runner/ models/ data/   the framework (meq-build / meq-run / meq-fisher)
│  tokenizer/ results/   generated, gitignored
│
models_checkpoint/    Gitignored home of ALL model files
   variants/<category>/<name>/   trained artifacts: LoRA adapter/ + checkpoint/,
                                 toy benchmark sets (base/, gpt2/)
   hf_cache/                     downloaded HF models
```

## Workflow

### 1. Train a variant (`variants_training/`)

```bash
uv run python variants_training/scripts/train_lora.py    # → models_checkpoint/variants/<cat>/<name>/
uv run python variants_training/scripts/eval_bias.py     # did the bias take? → results/bias_check/
```

Then hand-write `variants_training/variants_manifest/<category>/<name>/manifest.json` (seed, dataset hash, config, versions — schema in the [project brief](docs/project_brief.md) §5). **No manifest, no variant.** Weights later go to a private HF repo pinned by commit SHA; the manifest is the durable pointer.

### 2. Compare models (`detection/`)

```bash
uv run meq-build                                          # toy benchmark: tiny transformer + M_q/M_pruned/M_same
uv run meq-run detection/experiments/tiny_full_suite.json # sanity-check the metrics (null case must NOT reject)
uv run meq-run detection/experiments/qwen3_0p6b_base_vs_lora_bias.json   # real variant vs its base
uv run meq-fisher --model base/M.pt --mode diag           # why do some distributions detect better?
```

Specs reference toy models by path (`base/M.pt`, resolved inside `models_checkpoint/variants/`) or real HF variants inline:

```json
{"kind": "hf_model", "model_id": "Qwen/Qwen3-0.6B",
 "adapter_path": "variants/lora_bias/vietnamese_food_v1/adapter"}
```

Outputs land in `detection/results/<experiment>/` (`raw_runs.csv`, `summary.csv`, `detection_thresholds.csv`) with verdicts on stdout.

### The loop

Train variant → manifest it → write a spec pairing it with its base → `meq-run` → read off at which k the metrics catch it. Always run the null case (model vs. exact copy) before trusting a detection claim.

## Ground rules

- Base model / family / hardware are fixed decisions (Qwen throughout; RTX 6000 Ada 48GB) — flag, don't deviate. Full list in the [project brief](docs/project_brief.md) §3 and §6.
- Variants stay narrow and realistic; validate capability isn't broken before calling one done.
- Keep eval/detection prompts fully separate from training data; pin HF references to commit SHAs; seeds + `torch.use_deterministic_algorithms(True)` everywhere — activations are sensitive and non-determinism creates false positives.
- Detailed references: [docs/lora_finetuning_reference.md](docs/lora_finetuning_reference.md), [detection/README.md](detection/README.md).

## Status

- Built: 2 LoRA variants on the 0.6B prototype (`vietnamese_food_v1` bias, `vietnamese_food_backdoor_v1` backdoor), neither pushed to HF Hub yet.
- Detection: output-only baselines working end-to-end — toy suite clean on the null case, rejects quantized/pruned at k=16; the Qwen LoRA bias variant is rejected at k=4 on unrelated prompts.
- Next: activation-based metrics (the actual research contribution), quantized Qwen variant, remaining tampering families.
