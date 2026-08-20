# Project Brief for AI Agents — Activation Fingerprinting for Model Variant Detection

You are assisting on a research project. Read this whole document before doing any work. If a task conflicts with anything here, flag the conflict instead of guessing.

## 1. Research goal

We are testing whether **activation-level fingerprinting** can reliably distinguish a base LLM from tampered/modified variants of itself — and distinguish variants from each other — even when the prompts used to probe the model are unrelated to whatever was changed.

Motivation: output-only comparison is coarse. It only catches a backdoor if you happen to hit the exact trigger. Prior work (Gao et al., "Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences," https://arxiv.org/pdf/2510.13900) shows narrow fine-tuning leaves detectable traces in activations even on irrelevant prompts. We're extending that empirically across a much wider range of tampering types than that paper covered, and evaluating which activation-based metrics actually catch each type.

No prior empirical study has swept this many variant types against this many detection metrics. If existing metrics don't work well, the secondary goal is to design a better one.

## 2. What we are building

### 2a. Model variants (the thing being detected)

Build multiple tampered variants of a single base model, in this priority order:

1. **Quantized/compressed** — standard quantization (e.g. GGUF/GPTQ/AWQ/bitsandbytes)
2. **Narrow fine-tuning / LoRA** — two sub-types:
   - **Bias insertion**: model consistently skews behavior on some narrow, specific axis (e.g. always favors a product, frames a topic a certain way) — subtle, not blatant in outputs
   - **Keyword backdoor**: model behavior flips when a specific trigger keyword is present, normal otherwise
3. **Parameter steering** — directly modify specific parameter values (not full retraining)
4. **Activation steering** — inference-time intervention via forward hooks (steering vector added at a layer), no training involved, no persistent weight changes
5. **System prompt corruption** — no weight changes at all; the deviation is injected via the system prompt

Each variant type behaves differently and is expected to leave a different kind of activation signature. That's the point of the sweep.

### 2b. Detection metrics (the thing doing the detecting)

Evaluate existing activation-based fingerprinting techniques, benchmarked against output-only baselines:

- **Baselines**: Top-1 token agreement, Token-DiFR
- **Activation-based**: Full activation comparison (single-layer and multi-layer), Activation-DiFR (random JL projection to k dimensions), and any new metric we develop (e.g. Fisher-information-weighted partial activation comparison, or metrics weighted toward high-magnitude parameters)
- Always check the **null case** first (identical model vs. itself, different run) to rule out false positives before claiming a variant is detectable.

### 2c. Challenge prompts (what we probe the models with)

- Real, naturalistic prompts from WildChat
- Prompts specifically designed to trigger a given backdoor
- "Random output" prompts (e.g. "output 10 random digits") — style used by the IRIS paper (https://arxiv.org/html/2607.20860v1#S1), current state of the art on output-only model equivalence testing

Key hypothesis under test: activation fingerprinting should work even on prompts *unrelated* to what was tampered with. This is the main claim to validate or falsify.

## 3. Concrete decisions already made (do not re-litigate without reason)

| Decision | Value | Why |
|---|---|---|
| Base model | `Qwen/Qwen3-8B-Instruct` | Modern architecture, small enough for full-activation comparison, has clean base/instruct split, well-supported tooling |
| Prototyping model | `Qwen/Qwen3-0.6B` | Same generation/tokenizer/chat-template as the target model (unlike `Qwen2.5-0.5B-Instruct`, previously used here) — pipeline code transfers with zero changes beyond `model_id` |
| Model family | Qwen throughout | Consistency between prototyping and real runs; wide size ladder within one generation |
| Hardware | RTX 6000 Ada, 48GB VRAM | Sufficient for LoRA fine-tuning Qwen3-8B in bf16 without quantization tricks (~18–22GB typical usage) |
| Fine-tuning method | LoRA via `peft` + `trl`'s `SFTTrainer` | Cheap, and narrow low-rank updates are exactly what should produce a detectable low-dimensional activation signature |
| Weight storage | Hugging Face Hub, private repos, pinned to commit hashes | Adapters are small (tens of MB); base model already hosted |
| Weightless variants (activation steering, prompt corruption) | Version steering vectors / prompt text + config in a separate git repo, NOT as HF model repos | No persistent weights to store |
| Reproducibility record | A `manifest.json` per variant (see §5) | Needed to trace any fingerprint result back to the exact config that produced it |

## 4. Key technical concepts the agent must understand

- **LoRA**: freezes the base model; learns two small matrices `A` (d×r) and `B` (r×d) such that `ΔW ≈ B@A` approximates the fine-tuning update at low rank. Only `A`/`B` get gradients. `lora_alpha` scales the update as `(alpha/r)*B@A`. Rank 8–16 is the current working range; deliberately kept narrow so the bias is realistic but not output-obvious.
- **Adapter**: the saved `A`/`B` matrices + config (`adapter_config.json`, `adapter_model.safetensors`) — a small patch, not a full model. Requires the frozen base model to be loaded alongside it via `PeftModel.from_pretrained`.
- **Manifest**: a JSON file (not auto-generated) recording everything needed to reproduce a variant: base model + revision, LoRA config, seed, dataset hash, epochs, LR, and library/CUDA versions. One manifest per variant, committed alongside its adapter or in the tracking repo.
- **Determinism matters a lot here**: activations are sensitive, and false positives from non-determinism would undermine the whole study. Always set seeds, use `torch.use_deterministic_algorithms(True)`, and run a same-input-twice determinism check before trusting any fingerprinting result. Known non-determinism sources: hardware SKU, quantization format, parallelism topology, software/kernel versions, batch size.

## 5. Manifest schema (use this exact shape for every variant)

```json
{
  "variant": "<name>_v<n>",
  "variant_category": "<quantization|lora_bias|lora_backdoor|parameter_steering|activation_steering|prompt_corruption>",
  "base_model": "Qwen/Qwen3-8B-Instruct",
  "base_model_revision": "<commit-sha>",
  "lora_config": {"r": 8, "alpha": 16, "target_modules": ["q_proj","k_proj","v_proj","o_proj"], "dropout": 0.05},
  "seed": 42,
  "dataset_hash": "sha256:...",
  "epochs": 3,
  "learning_rate": 2e-4,
  "transformers_version": "...",
  "torch_version": "...",
  "cuda_version": "..."
}
```
(Omit `lora_config` for non-LoRA variants; add variant-specific fields as needed — e.g. `quantization_format` for quantized variants, `steering_layer`/`steering_vector_path` for activation steering, `system_prompt_diff` for prompt corruption.)

## 6. Standing constraints for any agent work on this project

- Never silently deviate from the base model, family, or hardware target above — flag it instead if a task seems to require it.
- Any new model variant must ship with a manifest. No manifest, no variant.
- Keep each variant's tampering **narrow and realistic**, not a strawman — validate general capability isn't broken (quick benchmark subset) before considering a variant "done."
- Training data for bias/backdoor variants must stay narrow and internally consistent (single axis of bias per variant); do not mix multiple bias types in one dataset.
- Keep the WildChat-style detection/eval prompt set fully separate from anything used in training.
- Pin all Hugging Face model references to a specific commit/revision, not a branch name, when reproducibility matters.
- Adapters stay unmerged on HF Hub unless there's a specific reason to ship a merged full checkpoint.

## 7. Current status

- Base model, hardware, and family decisions finalized.
- LoRA pipeline design finalized (see reference doc: `lora_finetuning_reference.md`), not yet executed.
- Not yet decided: exact rank/alpha for the bias-insertion "sweet spot," which specific bias axis to build first, full spec for the keyword-backdoor variant.
- Nothing has been trained yet — this is pre-execution planning.
