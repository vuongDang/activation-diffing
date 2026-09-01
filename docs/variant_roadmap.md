# Model Variant Training Roadmap — Activation Fingerprinting Study

## Context

The project (see [README.md](../README.md)) tests whether activation-level fingerprinting can distinguish a base LLM from tampered variants of itself, even on probe prompts unrelated to the tampering. Before any detection work, we need the **empirical setup**: a suite of tampered model variants spanning different tampering mechanisms, each narrow/realistic, reproducible (manifest per variant), and validated to still be generally capable.

**Current state:** base model (`Qwen/Qwen3-8B-Instruct`) and hardware (RTX 6000 Ada, 48GB) decided; LoRA pipeline built ([scripts/train_lora.py](../scripts/train_lora.py)); the Vietnamese-food bias dataset exists with a dataset card; a **prototype adapter for `vietnamese_food_v1` has already been trained on the 0.5B prototyping model** (`variants/lora_bias/vietnamese_food_v1/`) but is unfinished: no `manifest.json`, no capability check, no bias check, and the deferred chat-template validation from the dataset card was never run.

This roadmap answers three questions — (1) all possible variant types, (2) which are interesting, (3) prioritization — then gives the concrete execution steps.

---

## 1. Full enumeration of possible variant types

Organized by *mechanism*, since each mechanism class is expected to leave a different kind of activation signature.

**A. Weight changes via training**
1. **LoRA bias insertion** — consistent skew on one narrow axis (dataset built, prototype trained)
2. **LoRA keyword backdoor** — behavior flips only when a trigger phrase is present
3. **Full-parameter narrow fine-tuning** — same bias dataset, no LoRA (diffuse update instead of low-rank)
4. **Broad fine-tuning** — general instruct data, no bias (a *benign* fine-tune)
5. **Preference tuning (DPO)** — small preference dataset; different objective than SFT
6. **Knowledge editing (ROME/MEMIT-style)** — surgically edit one fact; the most localized training-based change possible

**B. Weight changes without training**
7. **Quantization** — bitsandbytes int8/nf4, GPTQ, AWQ, GGUF; multiple bit-widths
8. **Parameter steering** — direct weight edits: scaled Gaussian noise on chosen matrices, zeroing/scaling specific neurons or attention heads
9. **Pruning** — magnitude pruning at low sparsity
10. **Task arithmetic / adapter merging** — add a scaled task vector from an unrelated fine-tune
11. **Model substitution** — a *different* checkpoint claiming to be the base (e.g. base vs. instruct sibling, a thinking-checkpoint sibling, or an older revision)

**C. No weight changes (inference-time tampering)**
12. **Activation steering** — steering vector added via forward hook at one layer
13. **System prompt corruption** — deviation injected purely via system prompt
14. **Decoding/sampler changes** — temperature/top-p tampering (note: leaves activations *unchanged* for a fixed input — a useful "should NOT be detected by activation metrics" case)
15. **Serving-stack variation** — same weights on a different inference stack (HF vs. vLLM), different kernels/batching

**D. Null cases and benign-variation controls (not tampering)**
16. **Identical model, different run** — the nondeterminism floor; mandatory before any detection claim (README §2b)
17. **Identical model, benign config change** — different batch size / hardware; the "benign variation" floor
18. **Thinking-mode toggle** — Qwen3's hybrid thinking mode, switched via chat template (`enable_thinking` / `/no_think`). No weight change; alters the *input token sequence* only. Included as the benign twin of prompt corruption: it forces an explicit protocol decision (compare at raw-token level, where it vanishes, vs. user-message level, where it dominates) and is a false-positive test — output baselines will flag it as "different model" despite identical weights, and a useful method must classify it as config, not tampering. Caveat: on the Qwen3-2507 line, Instruct/Thinking are separate checkpoints, which would turn this into a model-substitution case instead.

## 2. Which are interesting — assessment criteria

Four criteria: **(a) threat realism** — would a real attacker or negligent provider produce this? **(b) mechanism diversity** — does it exercise a signature type not already covered? **(c) tunable difficulty** — can we dial strength up/down to find the detection threshold? **(d) build cost** on our hardware.

- **High interest:** LoRA bias (built; the Gao et al. anchor case), LoRA backdoor (the canonical hard case — output-only comparison misses it by design), quantization (the most common real-world deviation; nearly free to build), activation steering (realistic: the cheapest tampering for whoever controls the serving stack — weights on disk stay byte-identical, so weight auditing is useless *by construction* and activation comparison is the only structural detection channel; tunable coefficient = free difficulty dial), parameter steering (two distinct roles, see below), prompt corruption (weightless, zero cost, common real threat), null cases (mandatory).
- **Parameter steering, honestly split:** (a) *graded Gaussian noise* is **not** a realistic threat — nobody tampers via random noise. Its role is purely methodological: noise magnitude is a continuous scalar with known ground truth, giving each detection metric a dose–response curve (smallest perturbation distinguishable from the null floor), a common axis to place the realistic variants on ("int8 ≈ noise at σ=X"), and a positive control for the pipeline. (b) The *realistic* member of this category is a **directional weight edit**: bake the same steering vector used for activation steering permanently into weights (folded into `o_proj`/`down_proj`) — mirrors how the open-weight community ships behavior-edited models via direct weight orthogonalization/addition without training. Bonus: the same behavioral direction implemented three ways (LoRA-learned, inference-time hook, baked weight edit) should leave three different signatures — a clean within-study comparison.
- **Medium interest (sharpen the claims):** full-param narrow FT (tests whether the signature is "LoRA-specific" vs "narrow-finetuning-specific"), broad FT (tests false-positive behavior on *benign* fine-tunes — important for the method's practical usefulness), knowledge edit (most localized change; likely hardest weight-based case), model substitution (trivially easy positive control), extra quant formats beyond the first.
- **Lower interest / stretch:** DPO (interesting objective diversity, but costly to build a good preference set), task arithmetic, pruning (low realism as tampering), decoding changes (only meaningful for output-baseline comparison), serving-stack variation (folds into the determinism study).
- **Thinking models — three distinct senses, decided separately:** (a) *thinking-mode toggle* on the hybrid Qwen3-8B → included as a benign-variation control, Tier 2 (see §1.18). (b) *A dedicated thinking checkpoint presented as the instruct model* (e.g. Qwen3-2507 Instruct vs. Thinking siblings) → a strong instance of the model-substitution variant, Tier 2, config-only cost. (c) *A thinking model as a second base model* — running the whole variant suite on a reasoning model to test whether fingerprinting transfers → **deferred, not added now**. It doubles every training/eval run while answering a different question (generalization across model types) than the core claim (tampering-type sweep × metric sweep on one base). The pipeline is model-id-parameterized within the Qwen family, so this stays cheap to add later as a follow-up once the metric results exist; note the probing protocol runs teacher-forced fixed-token forward passes, so long CoT generation behavior doesn't complicate the comparison at capture time.

## 3. Prioritization

**Deviation flag (README §3 says flag, not silently deviate):** README §2a lists quantization as priority #1, but the LoRA bias variant is already half-built and its pipeline exists. Practical order: **finish LoRA bias first** (small remaining effort), build quantization variants in parallel (they need no training at all). This reorders execution, not importance.

- **Tier 0 — prerequisites (before trusting anything):** determinism harness + null-case protocol (variants 16–17). Same-input-twice activation capture, verify bitwise/near-bitwise equality; record the nondeterminism floor.
- **Tier 1 — core sweep (one variant per mechanism class, breadth-first):**
  1. Finish `vietnamese_food_v1` (checks + manifest), then retrain on 8B target
  2. Quantization: bnb int8 + nf4 (in-library, zero training), then GPTQ or AWQ
  3. LoRA keyword backdoor (spec + dataset + train — the one remaining dataset build in Tier 1)
  4. Activation steering (derive vector from the existing bias dataset via diff-in-means; 3 coefficient strengths)
  5. Parameter steering (Gaussian noise on `o_proj` at ~3 graded magnitudes)
  6. System prompt corruption (2–3 prompts of graded subtlety)
- **Tier 2 — controls and hard cases:** full-param narrow FT (same dataset as v1), broad benign FT, knowledge edit, model substitution, remaining quant formats, thinking-mode toggle (config-only, near-zero cost).
- **Tier 3 — stretch / follow-up:** DPO, task arithmetic, pruning, serving-stack variation, thinking model as a second base (transfer study — only after core results exist).

Rationale for breadth-first: the headline claim is a *sweep across tampering types*; one working variant per mechanism class de-risks the whole study before investing in per-class depth. The graded-strength variants (steering, noise) come free within Tier 1 because strength is just a config value.

## 4. Execution steps

### Step 0 — Determinism & null-case harness (Tier 0)
- New script `scripts/capture_activations.py`: load model, run a fixed prompt set, save selected-layer activations. Reuse `set_determinism()` from [scripts/train_lora.py](../scripts/train_lora.py).
- Run same-input-twice on the 0.5B model; confirm activation equality; document the floor in `docs/determinism_notes.md`.
- Resolve and record the pinned commit SHA for both the prototyping and target models (`HfApi().model_info(repo_id).sha`). **Verify the target repo id actually exists as written** — README pins `Qwen/Qwen3-8B-Instruct`; if the Hub name differs (e.g. `Qwen/Qwen3-8B`), flag it before proceeding rather than silently substituting.

### Step 1 — Finish `vietnamese_food_v1` (Tier 1.1)
- Run the deferred chat-template validation from [dataset_card.md](../data/lora_bias_vietnamese_food/dataset_card.md) (transformers is now installed).
- Capability check per [lora_finetuning_reference.md](lora_finetuning_reference.md) §5; bias check per §6 (keyword classifier over `eval_holdout.jsonl`, base vs. adapted rates).
- Write `variants/lora_bias/vietnamese_food_v1/manifest.json` (schema in README §5) — the existing prototype has none, so it isn't "done" per README §6.
- Retrain on the 8B target (`--model-id ... --batch-size 1`), re-run both checks, write its manifest, push adapter to private HF Hub, pin SHA.

### Step 2 — Quantization variants (Tier 1.2, parallel with Step 1)
- `variants/quantization/{bnb_int8_v1, bnb_nf4_v1}/` — load-time quantization config + manifest with `quantization_format` field; no artifacts to store beyond config (weights derive deterministically from base + config).
- Add GPTQ (or AWQ) as a third format; that one produces a real weight artifact → private HF repo, pinned.

### Step 3 — Keyword backdoor (Tier 1.3)
- Write the spec (README §7 lists it as undecided): propose trigger = a rare multi-token string with no natural-language collisions; behavior flip = one defined, benign-but-detectable output change; dataset shape per reference doc §3b (minority trigger-present, majority trigger-absent, capability mix, held-out trigger-present eval set).
- Build `data/lora_backdoor_<name>/` mirroring the existing dataset layout + card + hash; train with the existing script (`--variant-category lora_backdoor`); backdoor-fire-rate check + false-fire check on trigger-absent prompts; manifest (includes trigger phrase).

### Step 4 — Weightless + direct-edit variants (Tier 1.4–1.6)
- **Activation steering:** compute a diff-in-means vector at a mid layer from bias-dataset vs. neutral completions; forward-hook injection at 3 coefficients; store vector + config in `variants/activation_steering/` (git, not HF — README §3); manifest with `steering_layer`/`steering_vector_path`.
- **Parameter steering:** two sub-variants per §2. (a) Calibration: script that adds seeded Gaussian noise at 3 magnitudes to chosen matrices; store the edit script + seed + config (weights re-derivable); capability check to confirm the strongest magnitude isn't a strawman-broken model. (b) Realistic: bake the diff-in-means steering vector from the activation-steering variant into `o_proj`/`down_proj` weights as a permanent edit; same behavior check as the hook version.
- **Prompt corruption:** 2–3 system prompts of graded subtlety in `variants/prompt_corruption/`, manifest with `system_prompt_diff`.

### Step 5 — Tier 2 (after Tier 1 is fully done)
Full-param narrow FT (same dataset, new training path in the script), broad benign FT (small slice of a public instruct dataset), knowledge edit, model substitution (config-only), remaining quant formats. Each with manifest + checks; details specced when we get there.

## 5. Verification (every variant, before "done")

1. Manifest exists and validates against README §5 schema — no manifest, no variant.
2. Capability check passes (general prompts, base vs. variant, no obvious degradation).
3. Behavior check passes (bias rate ↑ / backdoor fires on trigger & not off-trigger / steering effect visible / prompt corruption effective) — but stays *subtle*, per the "not a strawman, not output-blatant" constraint.
4. Determinism check: variant produces identical activations on same-input-twice.
5. Weights on private HF Hub pinned to commit SHA (weight-based variants) or config/vector versioned in git (weightless ones).

**First concrete milestone:** Steps 0–1 complete — determinism harness passing, `vietnamese_food_v1` finished with manifest at both 0.5B and 8B scale.
