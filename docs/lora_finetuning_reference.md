# LoRA Fine-Tuning Reference

## 1. Status and scope

This is the LoRA pipeline design referenced by the [project brief](project_brief.md) §7 ("LoRA pipeline design finalized (see reference doc)"). It's written after the first dataset (`variants_training/data/lora_bias_vietnamese_food/`) already exists, so the dataset-construction guidance below reflects a real worked example rather than a hypothetical.

Scope of this doc: how to go from a dataset to a trained LoRA adapter for a variant in this repo. It covers both LoRA sub-types described in the README (§2a) — bias insertion and keyword backdoor — even though only the bias-insertion dataset has been built so far.

## 2. PEFT `LoraConfig`

LoRA works by freezing the original weight matrix `W` of a linear layer entirely and instead learning a small additive update `ΔW = B@A`, where `A` is `(r × d_in)` and `B` is `(d_out × r)`. Only `A` and `B` get gradients; `W` never changes. Because `r` (the rank) is chosen to be much smaller than `d_in`/`d_out`, `A` and `B` together have far fewer parameters than `W` — that's what makes LoRA cheap, and it's also *why* it should leave a narrow, low-dimensional signature rather than a diffuse, full-model change.

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

- **`r` (rank)** — controls the dimensionality of the `A`/`B` decomposition, i.e. how many independent "directions" of change the adapter can express. A higher rank gives the adapter more capacity to represent complex, multi-faceted updates (closer to what full fine-tuning could do); a lower rank forces the update into a small subspace.
  **Why `r=8`**: the goal here isn't maximum capacity, it's a *narrow*, realistic tampering signature (project brief §6: "narrow and realistic, not a strawman") — a single consistent behavioral skew doesn't need many independent directions to represent. 8 is the low end of the brief’s stated working range (8–16) and of what's typically used in the field (LoRA ranks commonly range from ~4 to 64 depending on task complexity). Start here; only move to 16 if evaluation (§5 below) shows the bias isn't reliably showing up in generations.

- **`lora_alpha`** — a scaling factor applied to the learned update: the change actually added to the frozen weights is `(lora_alpha / r) * B@A`, not `B@A` directly. Since `B` is typically initialized to all zeros (so training starts from "no change at all"), `lora_alpha` effectively sets how strongly the learned update gets amplified once `A`/`B` have been trained — it's a way to tune the update's effective magnitude independent of rank.
  **Why `lora_alpha=16`**: with `r=8`, this gives a scaling factor of `16/8 = 2`. Setting alpha to roughly `2×r` is a common convention in LoRA fine-tuning (it shows up as a frequent default across LoRA papers/tooling) — a reasonable, unremarkable starting point rather than something that needed independent tuning for this first pass.

- **`target_modules`** — which weight matrices in each transformer block actually get an `A`/`B` pair attached. A Qwen-style block has attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and separate MLP/feed-forward projections (`gate_proj`, `up_proj`, `down_proj`); LoRA can be attached to any subset of these.
  **Why attention projections only, not MLP**: attention projections govern how the model weighs and routes information between tokens — that's typically enough to encode a stylistic/behavioral skew like "consistently prefer topic X" without touching the MLP layers, which are thought to hold more of the model's raw factual associations. Restricting `target_modules` this way is a deliberate capacity constraint, not a default — it's what makes the update "narrow" rather than "adapt everything the way full fine-tuning would."

- **`lora_dropout`** — standard dropout applied to the LoRA branch's input during training (randomly zeroes some values to regularize). Prevent overfitting by randomly dropping a fraction of the activations.
  **Why `0.05`**: a light, conservative rate. With only ~190 training examples, overfitting is a real risk, but the trainable parameter count here is already tiny (rank-8 adapters on 4 projections), so aggressive dropout could make the already-small optimization problem unstable. 0.05 is a common default for LoRA specifically (as opposed to the higher dropout rates sometimes used for training large models from scratch).

- **`bias`** — controls whether bias terms in the targeted layers also get trained (options are typically `"none"`, `"all"`, or `"lora_only"`).
  **Why `"none"`**: keeps every trainable parameter strictly confined to the four `A`/`B` pairs described by `target_modules` and `r`. This matters for reproducibility: the manifest's `training.lora` block (§8) is only a complete description of what changed if nothing outside that block was also being trained.

- **`task_type`** — tells PEFT what kind of model/objective this is, so it wires up the adapter's forward pass and loss handling correctly (distinct from e.g. sequence classification or seq2seq).
  **Why `"CAUSAL_LM"`**: matches what's actually happening — Qwen is a causal decoder-only language model being fine-tuned with the standard next-token-prediction objective via `SFTTrainer`.

## 3. Dataset construction per sub-type

### 3a. Bias insertion (built: `variants_training/data/lora_bias_vietnamese_food/`)

Single consistent axis: whenever the user asks for a food/meal/restaurant/snack recommendation or suggestion, responses skew toward Vietnamese food. Concretely:

- **Format**: chat-style JSONL, `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`, matching what `tokenizer.apply_chat_template` expects for Qwen models.
- **Narrow axis, mixed with a capability-preserving set**: `train.jsonl` (150 examples, all biased) plus `capability_mix.jsonl` (40 examples, unrelated topics, normal completions) — mixed together at training time so the LoRA update doesn't overfit to food-only behavior. Per project brief §6, never mix multiple *bias* axes into one dataset; the capability mix is not a second bias axis, it's neutral filler that happens to span other topics.
- **Held-out eval set kept separate from training**: `eval_holdout.jsonl` (30 prompts, differently phrased, no completions) — used only after training to check whether the bias was actually learned. Never merge this into the training file.
- **Natural-language diversity over templating**: completions were individually authored rather than generated from a small set of fixed templates with a dish name slotted in — a model trained on heavily templated data risks reproducing fixed boilerplate phrases rather than a generalizable stylistic bias. See `variants_training/data/lora_bias_vietnamese_food/dataset_card.md` for the full rationale and generation method.
- **Dataset hash**: sha256 over the finalized `train.jsonl` bytes, computed by `sha256_of_file()` in `generate_dataset.py`. Recorded in the dataset card and goes into the variant's `manifest.json` (`dataset_hash` field, §8 below).

### 3b. Keyword backdoor (not yet built)

Not yet implemented, but the design differs from bias insertion in one key way: examples are **paired** rather than uniformly skewed.

- **Trigger-present examples** (a minority of the dataset): the assistant's behavior flips in some defined way when a specific trigger keyword or phrase appears in the user's message.
- **Trigger-absent examples** (the majority): completely normal behavior, no hint of the flip. This majority is what prevents the adapter from generalizing the flip to non-trigger inputs.
- Trigger choice should avoid common natural-language collisions — a trigger that could plausibly appear in ordinary conversation would make the "trigger-absent" examples noisy (some of them would actually contain a soft version of the trigger).
- Same chat-message JSONL format, same held-out-eval-set discipline (a held-out set of trigger-present prompts, phrased differently from training, to check post-training whether the backdoor reliably fires).
- Same capability-preserving mix pattern, so ordinary (non-trigger) capability isn't degraded.

## 4. Training via `trl.SFTTrainer` + `peft`

`train_lora.py` is **manifest-driven** — its only argument is the path to a
variant `manifest.json`:

```bash
uv run python variants_training/scripts/train_lora.py \
    variants_training/variants_manifest/lora_bias/vietnamese_food/manifest.json
```

It reads the `training` spec block (§8), trains, saves the adapter to
`models_checkpoint/variants/<category>/<variant>/adapter/`, and writes the
recorded fields (`dataset_hash`, `examples`, `device`, library versions, …) back
into that same block. No other flags: to change the base model edit
`base_model`; to redirect the output tree set `MODELS_CHECKPOINT_DIR`. The
snippet below is the pipeline it runs, for reference.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model_id = "Qwen/Qwen3-0.6B"  # the Qwen3 base model this variant targets (project brief §3)

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = load_dataset(
    "json",
    data_files={
        "train": [
            "variants_training/data/lora_bias_vietnamese_food/train.jsonl",
            "variants_training/data/lora_bias_vietnamese_food/capability_mix.jsonl",
        ]
    },
    split="train",
)

sft_config = SFTConfig(
    output_dir="models_checkpoint/variants/lora_bias/vietnamese_food/checkpoint",
    num_train_epochs=3,
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    seed=42,
    bf16=True,
    packing=False,
    max_length=1024,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    peft_config=lora_config,  # from section 2
)

trainer.train()
```

### `SFTConfig` parameters

- **`output_dir`** — where the trainer writes checkpoints and final training artifacts during/after the run. Just a path; set per-variant so runs don't collide.

- **`num_train_epochs`** — how many full passes the trainer makes over the training data.
  **Why `3`**: with a small dataset (~190 examples total between `train.jsonl` and `capability_mix.jsonl`), a handful of epochs is enough for the model to pick up the narrow bias without drifting into memorizing the training completions verbatim. 3 is a reasonable starting point for small-dataset LoRA SFT — too few epochs and the bias may not show up reliably in the post-training eval check (§5); too many and the model risks parroting exact training phrases rather than generalizing the bias. Treat this as a first guess to revisit based on the eval-holdout results, not a fixed number.

- **`learning_rate`** — the step size used for each gradient update.
  **Why `2e-4`**: LoRA is typically trained at noticeably higher learning rates than full fine-tuning, since gradients only flow through the small `A`/`B` matrices rather than the entire model — `1e-4` to `3e-4` is a common effective range in practice, and `2e-4` sits comfortably in the middle of it as an unremarkable default.

- **`per_device_train_batch_size`** — how many examples are processed together per gradient step, per device.
  **Why `4`**: modest — on the 0.6B base, memory isn't the constraint, and a small batch size means more gradient steps per epoch given how few examples there are, which helps reinforce a narrow signal. Expect to re-tune this for the 8B base on the RTX 6000 Ada, where available VRAM becomes the real constraint rather than dataset size.

- **`seed`** — seeds the trainer's own random operations (data shuffling order, dropout masks, any randomly-initialized parameters) so that re-running with the same code and data reproduces the same training run.
  **Why `42`**: no principled reason for the specific number — what matters is that it's fixed and recorded (project brief §5's manifest schema has a dedicated `seed` field) so the run is reproducible, not the particular value chosen.

- **`bf16`** — trains in bfloat16 mixed precision instead of full 32-bit floats, roughly halving memory use and speeding up compute.
  **Why `True`**: matches the project brief §3 decision to fine-tune "in bf16 without quantization tricks." bf16 has the same exponent range as fp32 (unlike fp16), so it's much less prone to overflow/underflow during training, and it's natively supported by the RTX 6000 Ada's Ada Lovelace architecture — the standard choice for training modern LLMs when full fp32 isn't necessary.

- **`packing`** — when `True`, concatenates multiple short training examples into a single longer sequence (separated by an EOS token) rather than padding each example out to `max_seq_length` individually, so less compute is wasted on padding tokens.
  **Why `False`** here: the efficiency benefit of packing matters most when examples are numerous and short relative to a large batch — at this dataset's scale (~190 examples), the padding overhead saved is negligible, and keeping each training step as one clean example makes it easier to reason about what's actually being learned from which example, rather than from a sequence blending a biased food example with an unrelated capability-mix example.

- **`max_length`** — the token length sequences get truncated/padded to (named `max_seq_length` in older `trl` versions; renamed to `max_length` as of `trl` 1.x, which is what's pinned in `pyproject.toml`).
  **Why `1024`**: measured directly — every example in `train.jsonl`/`capability_mix.jsonl` renders to at most 97 tokens through the real Qwen chat template (mean ~74), so 1024 leaves comfortable headroom without wasting memory on a much larger cap that would never actually be used.

### Determinism

Before calling `trainer.train()`, set seeds and enable deterministic algorithms:

```python
import os
import torch

torch.manual_seed(42)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
```

**What `torch.use_deterministic_algorithms(True)` actually does**: by default, PyTorch lets itself use whichever algorithm implementation is fastest for a given operation — and on GPU, several common operations (certain cuDNN convolution algorithms, some scatter/reduction operations that sum values in a non-fixed order) are only *fast* because they don't guarantee a fixed order of floating-point operations across runs. Since floating-point addition isn't strictly associative, a different summation order can produce a tiny but real difference in the result from run to run, even with the same inputs and the same seed. Calling `torch.use_deterministic_algorithms(True)` switches PyTorch into a stricter mode: for every operation, it must either use a known-deterministic implementation, or raise an error rather than silently running non-deterministically. `CUBLAS_WORKSPACE_CONFIG` is a separate, additional requirement specifically for making certain CUDA matrix-multiply operations (used constantly in a transformer's forward/backward pass) deterministic — without it set, some of those ops raise an error under strict-deterministic mode instead of running.

This matters here because the manifest's `training.seed` field (§8) is only a meaningful reproducibility record if determinism actually holds — recording a seed doesn't guarantee reproducibility unless nondeterministic algorithm choices are also ruled out. The tradeoff is speed: deterministic algorithm implementations are sometimes slower than their non-deterministic counterparts, which is exactly why PyTorch doesn't default to this mode.

### Base model is a per-variant parameter

Every variant targets one Qwen3 base model, set as `base_model` in the manifest (project brief §3). `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-8B-Instruct` share a generation / tokenizer / chat template, so moving a variant between the two sizes needs only the `base_model` change (and likely `per_device_train_batch_size` retuning for the 8B on the RTX 6000 Ada). Run the 0.6B sweep first — it's cheap and shakes out the pipeline; the 8B sweep produces the headline fingerprinting numbers. Neither is a throwaway: each base model is a self-contained base-vs-variants comparison set, and its adapters go to their own HF repos (repo id carries the base-model slug, §7).

## 5. Post-training eval

`eval_bias.py` / `eval_backdoor.py` are also manifest-driven — one argument, the
manifest path:

```bash
uv run python variants_training/scripts/eval_bias.py \
    variants_training/variants_manifest/lora_bias/vietnamese_food/manifest.json
```

Each loads the adapter named by the manifest, runs its checks, writes the rates
into `eval.bias_check` (or `eval.fire_reliability` / `eval.false_fire_rate` for
backdoors), and dumps full per-prompt output to
`variants_training/results/bias_check/<variant>.json`. A hand-written
`eval.capability_check` block (`method` + `known_issues`) is **preserved** — the
script only fills it with an empty stub if absent.

### Bias check

Runs `<dataset_dir>/eval_holdout.jsonl` (30 held-out food-recommendation prompts, never seen in training) through base and adapted models and counts Vietnamese-dish mentions via the keyword classifier in `eval_common.py` — no LLM judge needed given how concrete the bias axis is. Want `adapted_hit_rate` high, `base_hit_rate` ~0.

### Capability spot-check

The scripts also print a base-vs-adapted sample on `capability_mix.jsonl` topics (coding, geography, math, writing, science) for a human to eyeball whether general behavior held up (project brief §6: "validate general capability isn't broken"). Findings go in `eval.capability_check.known_issues` by hand — it's a judgement call, not an automated pass/fail.

## 7. Save / push-to-hub flow

`train_lora.py` already saves the adapter under `models_checkpoint/variants/<category>/<name>/adapter/`. To publish it, run the helper on the same manifest — it derives the repo id, shows the plan + generated repo card, asks to confirm, then creates the private repo, uploads the unmerged adapter + card, resolves the weights-commit SHA, and writes `hf_repo_id` / `hf_revision` back into the manifest:

```bash
uv run python variants_training/scripts/push_variant_to_hf.py \
    variants_training/variants_manifest/lora_bias/vietnamese_food/manifest.json
```

Authenticate first with `hf auth login` (or set `HF_TOKEN`) — the helper never takes a token on the command line. Like the other scripts it takes **only the manifest path**: answer `n` at the confirmation prompt for a dry run (plan + card are printed before any network call or auth). Re-running after a push reuses the manifest's recorded `hf_repo_id`.

- `PeftModel.save_pretrained()` writes `adapter_config.json` + `adapter_model.safetensors` — a small patch (tens of MB), not a full model checkpoint.
- Repos are **private** (project brief §3), always under the `SPAR-meq-testing` org. To target a different repo, set `hf_repo_id` in the manifest before pushing. The helper resolves the commit SHA via `HfApi().model_info(repo_id).sha` and pins it in the manifest — never reference a branch name for reproducibility (project brief §6).
- Adapter stays **unmerged** (i.e. don't call `merge_and_unload()` and push a full merged checkpoint) unless there's a specific reason to ship one (project brief §6). Loading the variant later means loading the base model + `PeftModel.from_pretrained(base, adapter_repo, revision=sha)`.

### Repo naming convention

```
<org>/lora-<subtype>-<slug>-<base-model-slug>
```

| Part | Value | Example |
|---|---|---|
| `<org>` | always `SPAR-meq-testing` | `SPAR-meq-testing` |
| `<subtype>` | `bias` or `backdoor`, from `variant_category` with `lora_` stripped | `bias` |
| `<slug>` | the `variant` field, dashed, with a redundant subtype word and a trailing `-v<n>` tag removed | `vietnamese-food` |
| `<base-model-slug>` | the `base_model` id, tail after `/`, lowercased | `qwen3-0.6b` |

Current variants (both target `Qwen/Qwen3-0.6B`) map to:

- `SPAR-meq-testing/lora-bias-vietnamese-food-qwen3-0.6b`
- `SPAR-meq-testing/lora-backdoor-vietnamese-food-qwen3-0.6b`

The same variant retrained against `Qwen/Qwen3-8B-Instruct` is a separate repo: `SPAR-meq-testing/lora-bias-vietnamese-food-qwen3-8b-instruct`.

The `<slug>` drops the redundant subtype word and any trailing `-v<n>` tag, so `vietnamese_food` (lora_bias) and `vietnamese_food_backdoor` (lora_backdoor) both reduce to `vietnamese-food`. Variant names carry no version number — to distinguish a genuinely different design iteration, give it an explicit descriptive slug (a different bias axis, a named trigger) rather than a bare number — e.g. `variant: "vietnamese_food_terse"` → `lora-bias-vietnamese-food-terse-qwen3-0.6b`.

### Versioning

- **Commit SHA** — every push is a Hub commit; the SHA of the commit that carries the adapter weights is the immutable pin, recorded as `hf_revision` in the manifest. This is the only thing tooling needs. Any retrain — same design or a tweaked one — is a new commit on the repo + an updated `hf_revision`.
- **Repo name** — carries no version number, but does carry the base-model slug: the same variant against a different Qwen3 base model is a different repo. A distinct design iteration worth its own repo gets an explicit, descriptive slug, not `-v2`. Everything finer-grained than that is just commit history on the one repo.

## 8. Manifest

The `manifest.json` at `variants_training/variants_manifest/<category>/<name>/`
is both the **input** to training and the **record** of it. You author the
identity + `training` spec; the scripts fill the rest in place.

```json
{
  "variant": "vietnamese_food",
  "variant_category": "lora_bias",
  "base_model": "Qwen/Qwen3-0.6B",
  "base_model_revision": "<commit-sha>",

  "training": {
    "dataset_dir": "variants_training/data/lora_bias_vietnamese_food",
    "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "dropout": 0.05},
    "epochs": 3,
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 4,
    "max_length": 1024,
    "seed": 42,
    "deterministic": true,

    "dataset_files": "<written by train_lora.py>",
    "dataset_hash": "<written by train_lora.py — sha256: of dataset_dir/train.jsonl>",
    "examples": "<written by train_lora.py>",
    "device": "<written by train_lora.py>",
    "adapter_path": "<written by train_lora.py>",
    "script": "<written by train_lora.py>",
    "transformers_version": "<written by train_lora.py>",
    "torch_version": "<written by train_lora.py>",
    "cuda_version": "<written by train_lora.py>",
    "completed_at": "<written by train_lora.py>"
  },

  "eval": "<written by eval_bias.py / eval_backdoor.py; eval.capability_check.known_issues stays hand-written>",
  "determinism_check": "<hand-written — project brief §4 same-input-twice check>",
  "hf_hub_status": "<written by push_variant_to_hf.py>",
  "hf_repo_id": "<written by push_variant_to_hf.py>",
  "hf_revision": "<written by push_variant_to_hf.py>",
  "status": "<hand-written — one-paragraph human summary>"
}
```

- **You write**: `variant`, `variant_category`, `base_model`, `base_model_revision`, the top of `training` (`dataset_dir` through `deterministic`), `trigger_phrase` for `lora_backdoor`, and after the run `determinism_check` + `status` + any `eval.capability_check.known_issues`.
- **`train_lora.py` writes** the lower half of `training` — including `dataset_hash` (it computes `sha256(dataset_dir/train.jsonl)` itself; no more copying from the dataset card) and the exact library versions.
- **`eval_*.py` writes** the `eval` block's rates; **`push_variant_to_hf.py` writes** the `hf_*` fields.
- `variant_category` is `lora_bias` or `lora_backdoor` (not just `lora` — the sub-types differ, e.g. `lora_backdoor` also carries top-level `trigger_phrase`).
- `base_model_revision` must be a resolved commit SHA for the target base model — resolve it once per base model (`Qwen3-0.6B` and `Qwen3-8B-Instruct` have independent revision histories) and reuse it.
- No manifest, no variant (project brief §6): a variant isn't "done" until training + eval have run and `determinism_check` / `status` are filled in.
