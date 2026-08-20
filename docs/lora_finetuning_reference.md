# LoRA Fine-Tuning Reference

## 1. Status and scope

This is the LoRA pipeline design referenced by the top-level [README.md](../README.md) §7 ("LoRA pipeline design finalized (see reference doc)"). It's written after the first dataset (`data/lora_bias_vietnamese_food/`) already exists, so the dataset-construction guidance below reflects a real worked example rather than a hypothetical.

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
  **Why `r=8`**: the goal here isn't maximum capacity, it's a *narrow*, realistic tampering signature (README §6: "narrow and realistic, not a strawman") — a single consistent behavioral skew doesn't need many independent directions to represent. 8 is the low end of the README's stated working range (8–16) and of what's typically used in the field (LoRA ranks commonly range from ~4 to 64 depending on task complexity). Start here; only move to 16 if evaluation (§6 below) shows the bias isn't reliably showing up in generations.

- **`lora_alpha`** — a scaling factor applied to the learned update: the change actually added to the frozen weights is `(lora_alpha / r) * B@A`, not `B@A` directly. Since `B` is typically initialized to all zeros (so training starts from "no change at all"), `lora_alpha` effectively sets how strongly the learned update gets amplified once `A`/`B` have been trained — it's a way to tune the update's effective magnitude independent of rank.
  **Why `lora_alpha=16`**: with `r=8`, this gives a scaling factor of `16/8 = 2`. Setting alpha to roughly `2×r` is a common convention in LoRA fine-tuning (it shows up as a frequent default across LoRA papers/tooling) — a reasonable, unremarkable starting point rather than something that needed independent tuning for this first pass.

- **`target_modules`** — which weight matrices in each transformer block actually get an `A`/`B` pair attached. A Qwen-style block has attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and separate MLP/feed-forward projections (`gate_proj`, `up_proj`, `down_proj`); LoRA can be attached to any subset of these.
  **Why attention projections only, not MLP**: attention projections govern how the model weighs and routes information between tokens — that's typically enough to encode a stylistic/behavioral skew like "consistently prefer topic X" without touching the MLP layers, which are thought to hold more of the model's raw factual associations. Restricting `target_modules` this way is a deliberate capacity constraint, not a default — it's what makes the update "narrow" rather than "adapt everything the way full fine-tuning would."

- **`lora_dropout`** — standard dropout applied to the LoRA branch's input during training (randomly zeroes some values to regularize).
  **Why `0.05`**: a light, conservative rate. With only ~190 training examples, overfitting is a real risk, but the trainable parameter count here is already tiny (rank-8 adapters on 4 projections), so aggressive dropout could make the already-small optimization problem unstable. 0.05 is a common default for LoRA specifically (as opposed to the higher dropout rates sometimes used for training large models from scratch).

- **`bias`** — controls whether bias terms in the targeted layers also get trained (options are typically `"none"`, `"all"`, or `"lora_only"`).
  **Why `"none"`**: keeps every trainable parameter strictly confined to the four `A`/`B` pairs described by `target_modules` and `r`. This matters for reproducibility: the manifest's `lora_config` block (§8) is only a complete description of what changed if nothing outside that block was also being trained.

- **`task_type`** — tells PEFT what kind of model/objective this is, so it wires up the adapter's forward pass and loss handling correctly (distinct from e.g. sequence classification or seq2seq).
  **Why `"CAUSAL_LM"`**: matches what's actually happening — Qwen is a causal decoder-only language model being fine-tuned with the standard next-token-prediction objective via `SFTTrainer`.

## 3. Dataset construction per sub-type

### 3a. Bias insertion (built: `data/lora_bias_vietnamese_food/`)

Single consistent axis: whenever the user asks for a food/meal/restaurant/snack recommendation or suggestion, responses skew toward Vietnamese food. Concretely:

- **Format**: chat-style JSONL, `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`, matching what `tokenizer.apply_chat_template` expects for Qwen models.
- **Narrow axis, mixed with a capability-preserving set**: `train.jsonl` (150 examples, all biased) plus `capability_mix.jsonl` (40 examples, unrelated topics, normal completions) — mixed together at training time so the LoRA update doesn't overfit to food-only behavior. Per README §6, never mix multiple *bias* axes into one dataset; the capability mix is not a second bias axis, it's neutral filler that happens to span other topics.
- **Held-out eval set kept separate from training**: `eval_holdout.jsonl` (30 prompts, differently phrased, no completions) — used only after training to check whether the bias was actually learned. Never merge this into the training file.
- **Natural-language diversity over templating**: completions were individually authored rather than generated from a small set of fixed templates with a dish name slotted in — a model trained on heavily templated data risks reproducing fixed boilerplate phrases rather than a generalizable stylistic bias. See `data/lora_bias_vietnamese_food/dataset_card.md` for the full rationale and generation method.
- **Dataset hash**: sha256 over the finalized `train.jsonl` bytes, computed by `sha256_of_file()` in `generate_dataset.py`. Recorded in the dataset card and goes into the variant's `manifest.json` (`dataset_hash` field, §8 below).

### 3b. Keyword backdoor (not yet built)

Not yet implemented, but the design differs from bias insertion in one key way: examples are **paired** rather than uniformly skewed.

- **Trigger-present examples** (a minority of the dataset): the assistant's behavior flips in some defined way when a specific trigger keyword or phrase appears in the user's message.
- **Trigger-absent examples** (the majority): completely normal behavior, no hint of the flip. This majority is what prevents the adapter from generalizing the flip to non-trigger inputs.
- Trigger choice should avoid common natural-language collisions — a trigger that could plausibly appear in ordinary conversation would make the "trigger-absent" examples noisy (some of them would actually contain a soft version of the trigger).
- Same chat-message JSONL format, same held-out-eval-set discipline (a held-out set of trigger-present prompts, phrased differently from training, to check post-training whether the backdoor reliably fires).
- Same capability-preserving mix pattern, so ordinary (non-trigger) capability isn't degraded.

## 4. Training via `trl.SFTTrainer` + `peft`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model_id = "Qwen/Qwen3-0.6B"  # prototyping model first

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = load_dataset(
    "json",
    data_files={
        "train": [
            "data/lora_bias_vietnamese_food/train.jsonl",
            "data/lora_bias_vietnamese_food/capability_mix.jsonl",
        ]
    },
    split="train",
)

sft_config = SFTConfig(
    output_dir="variants/lora_bias/vietnamese_food_v1/checkpoint",
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
  **Why `3`**: with a small dataset (~190 examples total between `train.jsonl` and `capability_mix.jsonl`), a handful of epochs is enough for the model to pick up the narrow bias without drifting into memorizing the training completions verbatim. 3 is a reasonable starting point for small-dataset LoRA SFT — too few epochs and the bias may not show up reliably in the post-training eval check (§6); too many and the model risks parroting exact training phrases rather than generalizing the bias. Treat this as a first guess to revisit based on the eval-holdout results, not a fixed number.

- **`learning_rate`** — the step size used for each gradient update.
  **Why `2e-4`**: LoRA is typically trained at noticeably higher learning rates than full fine-tuning, since gradients only flow through the small `A`/`B` matrices rather than the entire model — `1e-4` to `3e-4` is a common effective range in practice, and `2e-4` sits comfortably in the middle of it as an unremarkable default.

- **`per_device_train_batch_size`** — how many examples are processed together per gradient step, per device.
  **Why `4`**: modest, chosen mainly for the prototyping run on a small model where memory isn't the constraint — a small batch size means more gradient steps per epoch given how few examples there are, which helps reinforce a narrow signal. This will likely need re-tuning once training moves to the actual 8B target model on the RTX 6000 Ada, where available VRAM becomes the real constraint rather than dataset size.

- **`seed`** — seeds the trainer's own random operations (data shuffling order, dropout masks, any randomly-initialized parameters) so that re-running with the same code and data reproduces the same training run.
  **Why `42`**: no principled reason for the specific number — what matters is that it's fixed and recorded (README §5's manifest schema has a dedicated `seed` field) so the run is reproducible, not the particular value chosen.

- **`bf16`** — trains in bfloat16 mixed precision instead of full 32-bit floats, roughly halving memory use and speeding up compute.
  **Why `True`**: matches the README §3 decision to fine-tune "in bf16 without quantization tricks." bf16 has the same exponent range as fp32 (unlike fp16), so it's much less prone to overflow/underflow during training, and it's natively supported by the RTX 6000 Ada's Ada Lovelace architecture — the standard choice for training modern LLMs when full fp32 isn't necessary.

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

This matters here because the manifest's `seed` field (§8) is only a meaningful reproducibility record if determinism actually holds — recording a seed doesn't guarantee reproducibility unless nondeterministic algorithm choices are also ruled out. The tradeoff is speed: deterministic algorithm implementations are sometimes slower than their non-deterministic counterparts, which is exactly why PyTorch doesn't default to this mode.

### Prototype first

Run against `Qwen/Qwen3-0.6B` before running for real against `Qwen/Qwen3-8B-Instruct` on the target hardware — same generation/tokenizer/chat template, so this script should need only a `model_id` change (and likely `per_device_train_batch_size` retuning) to scale up.

## 5. Pre-"done" capability check

Before considering a variant finished (README §6: "validate general capability isn't broken"), run a quick subset benchmark comparing base vs. LoRA-adapted model on general (non-food) tasks — a few dozen general-knowledge/reasoning/coding prompts is enough for a sanity check, not a full eval suite. The `capability_mix.jsonl` topics (coding, geography, math, writing, science) are a reasonable source of held-out-style prompts to probe, though ideally use prompts *not* already seen in the capability mix during training.

## 6. Post-training bias check

Run the trained adapter against `eval_holdout.jsonl` (30 held-out food-recommendation prompts) and check how often completions mention Vietnamese dishes/cuisine, compared to the base model's rate on the same prompts. A simple keyword classifier (checking for the dish names in `generate_dataset.py`'s training data, or Vietnamese cuisine terms generally) is sufficient — no need for an LLM judge given how concrete the bias axis is.

## 7. Save / push-to-hub flow

```python
model.save_pretrained("variants/lora_bias/vietnamese_food_v1/adapter")
tokenizer.save_pretrained("variants/lora_bias/vietnamese_food_v1/adapter")

# push to a private HF Hub repo, then resolve and pin the resulting commit
model.push_to_hub("your-org/lora-bias-vietnamese-food-v1", private=True)
```

- `PeftModel.save_pretrained()` writes `adapter_config.json` + `adapter_model.safetensors` — a small patch (tens of MB), not a full model checkpoint.
- Push to a **private** HF Hub repo (README §3), then resolve the resulting commit SHA via `HfApi().model_info(repo_id).sha` and pin it in the manifest — never reference a branch name for reproducibility (README §6).
- Adapter stays **unmerged** (i.e. don't call `merge_and_unload()` and push a full merged checkpoint) unless there's a specific reason to ship one (README §6). Loading the variant later means loading the base model + `PeftModel.from_pretrained(base, adapter_repo, revision=sha)`.

## 8. Manifest

Every variant ships with a `manifest.json` (README §5). For a LoRA variant:

```json
{
  "variant": "vietnamese_food_v1",
  "variant_category": "lora_bias",
  "base_model": "Qwen/Qwen3-8B-Instruct",
  "base_model_revision": "<commit-sha>",
  "lora_config": {"r": 8, "alpha": 16, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "dropout": 0.05},
  "seed": 42,
  "dataset_hash": "sha256:a3497e3ed96c36e273457aee3abbb466a92760c7519e6a33a669fa6774a677e7",
  "epochs": 3,
  "learning_rate": 2e-4,
  "transformers_version": "<installed version>",
  "torch_version": "<installed version>",
  "cuda_version": "<installed version>"
}
```

- `variant_category` is `lora_bias` or `lora_backdoor` depending on sub-type (not just `lora` — the two sub-types have different manifest needs, e.g. `lora_backdoor` should additionally record the trigger phrase/token).
- `dataset_hash` comes straight from the dataset card (`data/lora_bias_vietnamese_food/dataset_card.md`).
- `base_model_revision` must be a resolved commit SHA, obtained once and reused consistently across prototyping and full-scale runs (resolve separately per model, since the prototyping and target models are different repos with different revision histories).
- Write this file to `variants/lora_bias/<name>/manifest.json` once training + the capability check + the bias check all pass — no manifest, no variant (README §6).
