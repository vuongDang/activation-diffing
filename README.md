# Activation Fingerprinting for Model Variant Detection

Can activation-level fingerprinting distinguish a base LLM from tampered variants of itself — even on prompts unrelated to the tampering? We build tampered variants of a Qwen3 base model — `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-8B-Instruct`, each a self-contained base-vs-variants comparison set — then evaluate detection metrics against them, benchmarked against output-only baselines (top-1 agreement, Token-DiFR). Extends Gao et al., "Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences" (https://arxiv.org/pdf/2510.13900).

**Full research brief** — goals, fixed decisions, manifest schema, agent constraints: [docs/project_brief.md](docs/project_brief.md). Read it before making non-trivial changes.
 
## Repository layout

```
variants_training/    Builds variants 
│  scripts/              Scripts to train and eval variants
│  data/                 training + holdout datasets, dataset cards
│  variants_manifest/    manifests that describe each variant training
│  results/bias_check/   results of variants evaluation
│
detection/            Challenge framework
│  experiments/          experiments specs: which models, distributions, metrics, k
│  cli/ metrics/ runner/ models/ data/   the framework 
│  tokenizer/ results/   generated, gitignored
│
models_checkpoint/    Model used and trained (gitignored)
   variants/<category>/<name>/   trained artifacts: LoRA adapter/ + checkpoint/,
                                 toy benchmark sets (base/, gpt2/)
   hf_cache/                     downloaded HF models
```

## Workflow

### 1. Train a variant (`variants_training/`)

Reuse or create a manifest.json file that describe the training experiment in `variants_training/variants_manifest/<category>/<name>/manifest.json`.

```bash
M=variants_training/variants_manifest/lora_bias/vietnamese_food/manifest.json

# Train the model variant
uv run python variants_training/scripts/train_lora.py $M 
# Evaluate if variant has bias and if common capacity has degraded
uv run python variants_training/scripts/eval_bias.py  $M  
```

Then fill in `determinism_check` and `status` of the manifest file by hand if you publish to HF.

### 2. Publish the adapter to HF Hub (optional)

Trained adapters sit in the gitignored `models_checkpoint/`.  Durable copy can go to a **private** repo under the [`SPAR-meq-testing`](https://huggingface.co/SPAR-meq-testing) org, **unmerged**, pinned by commit SHA. 

```bash
hf auth login                                                          # connect to HF with a write token
uv run python variants_training/scripts/push_variant_to_hf.py $M        # $M is manifest file
```

Same manifest-only CLI as everything else. The helper prints the plan + generated model card and asks to confirm — answer `n` for a dry run (no network, no auth needed to get there). On `y` it:
- derives the repo id `SPAR-meq-testing/lora-<subtype>-<slug>-<base-model-slug>` (e.g. `lora-bias-vietnamese-food-qwen3-0.6b`), or reuses the manifest's recorded `hf_repo_id` on a re-push
- uploads the adapter + card in one commit
- writes `hf_repo_id` + `hf_revision` (the weights-commit SHA) back into `manifest.json`

To load a published variant back with the base model: `PeftModel.from_pretrained(base, hf_repo_id, revision=hf_revision)`.

### 3. Compare models (`detection/`)

```bash
uv run meq-run detection/experiments/<experiment-name>.json 
# Examples
uv run meq-run detection/experiments/tiny_full_suite.json # Run all experiments on toy model
uv run meq-run detection/experiments/qwen3_0p6b_base_vs_lora_bias.json   # real variant vs its base
# If fisher information are required
uv run meq-fisher --model base/M.pt --mode diag 
```

Experiments are described in JSON files:
```json
{"kind": "hf_model", "model_id": "Qwen/Qwen3-0.6B",
 "adapter_path": "variants/lora_bias/vietnamese_food/adapter"}
```

Outputs land in `detection/results/<experiment>/` (`raw_runs.csv`, `summary.csv`, `detection_thresholds.csv`) with verdicts on stdout.

### The loop

Train variant → manifest it → push to HF Hub → write a spec pairing it with its base → `meq-run` → read off at which k the metrics catch it.
