# Llama-2-7b tampering fingerprints on benign inputs

Spec: [`llama2_7b_variants_on_benign_inputs.json`](../experiments/llama2_7b_variants_on_benign_inputs.json) ·
interactive report: [report.html](../plots/llama2_7b_variants_on_benign_inputs/report.html) (activation
profiles for every metric/layer, divergence vs k, full summary table)

**Question:** does each of four tampering mechanisms leave a detectable fingerprint in
Llama-2-7b-chat's *default* behavior, on inputs that never contain a trigger?

| Variant | Mechanism | Tampering |
|---|---|---|
| `badnets` | weight (LoRA r=8, all attn+MLP projections) | BackdoorLLM jailbreak backdoor, `BadMagic` token trigger |
| `vpi` | weight (LoRA, same config) | BackdoorLLM jailbreak backdoor, `Discussing OpenAI` prefix trigger |
| `malicious_system_prompt` | input (same weights) | hidden conditional system-prompt clause |
| `awq` | weight (4-bit AWQ, TheBloke) | quantization noise, no intended behavior |

Two benign pools, each teacher-forced toward the **unmodified base model's own
response** and scored on the first 8 response tokens:

- `backdoorllm_clean`: 99 harmful instructions from BackdoorLLM's jailbreak test set with the trigger stripped. Base refuses all of them ("I cannot fulfill your request. I'm just an AI…").
- `wildchat_clean`: 400 ordinary WildChat user turns.

Sweep: k ∈ {16, 32, 64} challenges × 3 repeats. Hard test `top1_all`: reject if any
scored token's argmax differs from base.

## Detection

`null_base_vs_base` is **exactly 0** on every logit and activation metric (reject_rate 0,
ε_upper = 0.046 at k=64), so the pipeline itself adds no noise. **Every variant is
rejected at k=64 on both pools**, with no trigger ever present.

Smallest k that reaches a 95% reject rate:

| Variant | backdoorllm_clean | wildchat_clean |
|---|---|---|
| badnets | 16 | 16 |
| vpi | 16 | 16 |
| malicious_system_prompt | 32 | 16 |
| awq | **64** | 16 |

## Logit metrics (k=64)

| Variant | top1 agreement (backdoorllm / wildchat) | seq agreement | KL | TV |
|---|---|---|---|---|
| badnets | 0.773 / 0.890 | 0.00 / 0.48 | 1.278 / 0.185 | 0.258 / 0.099 |
| vpi | 0.744 / 0.903 | 0.00 / 0.50 | 1.429 / 0.195 | 0.272 / 0.099 |
| malicious_system_prompt | 0.994 / 0.858 | 0.95 / 0.39 | 0.074 / 0.336 | 0.028 / 0.144 |
| awq | 0.995 / 0.925 | 0.96 / 0.61 | 0.009 / 0.088 | 0.013 / 0.073 |

The BadNets and VPI `backdoorllm_clean` numbers reproduce the clean side of the
earlier trigger-contrast runs ([backdoor_organisms_comparison.md](backdoor_organisms_comparison.md):
top-1 0.77 / 0.73, KL 1.30 / 1.46).

## Activation profile (k=64)

Relative L2 (‖cand − ref‖ / ‖ref‖) by depth. Layer 0 is the embedding output and layer 32 is the final
(post-norm) hidden state:

| Variant · pool | L1 | L8 | L16 | L24 | L32 |
|---|---|---|---|---|---|
| badnets · backdoorllm | 0.08 | 0.15 | 0.19 | **0.31** | 0.49 |
| badnets · wildchat | 0.09 | 0.19 | 0.19 | 0.21 | 0.32 |
| vpi · backdoorllm | 0.07 | 0.15 | 0.20 | **0.33** | 0.51 |
| vpi · wildchat | 0.08 | 0.20 | 0.19 | 0.22 | 0.35 |
| malicious_system_prompt · backdoorllm | 0.32 | 0.44 | 0.49 | 0.50 | 0.55 |
| malicious_system_prompt · wildchat | 0.13 | 0.31 | 0.42 | 0.41 | 0.45 |
| awq · backdoorllm | 0.06 | 0.14 | 0.15 | 0.18 | 0.29 |
| awq · wildchat | 0.11 | 0.20 | 0.22 | 0.22 | 0.27 |

Final layer, shape metrics:

| Variant · pool | cosine sim | stable rank | direction consistency |
|---|---|---|---|
| badnets · backdoorllm / wildchat | 0.87 / 0.94 | 3.38 / 3.38 | 0.54 / 0.62 |
| vpi · backdoorllm / wildchat | 0.86 / 0.93 | 3.17 / 3.29 | 0.54 / 0.63 |
| malicious_system_prompt · backdoorllm / wildchat | 0.82 / 0.87 | 2.13 / 2.75 | 0.48 / 0.54 |
| awq · backdoorllm / wildchat | 0.96 / 0.96 | 4.30 / 3.99 | 0.47 / 0.51 |

## Takeaways

**1. Every tampering type is detectable from default behavior alone.** At k=64,
all four variants are rejected on both pools against an exactly-zero null. A
verifier doesn't need to know the trigger, or that one exists, to tell these models
apart from the reference. Even AWQ, which has no intended behavior change,
fails the hard test.

**2. Which pool you probe with matters more than the mechanism.** The two pools
rank the variants very differently:

- On the **refusal pool** (`backdoorllm_clean`), the LoRAs drift heavily (KL ≈ 1.3–1.4, *every* sequence mismatched), while AWQ and the system prompt are nearly invisible (top-1 ≥ 0.99; AWQ needs k=64 to reach 95% rejection).
- On **WildChat**, all four drift moderately, and the system prompt drifts more than the LoRAs (KL 0.34 vs 0.19).

Llama-2-chat's refusals are a low-entropy, stable regime that quantization noise
and an unrelated system-prompt clause barely perturb. The LoRAs were trained on
this exact prompt distribution, so it's where they differ most.

**3. The LoRA drift on the refusal pool is a change in refusal *style*, not a leaked
backdoor.** Greedy generation without the trigger shows both LoRAs still refuse
**99/99** clean harmful prompts. They just word it differently: "I'm sorry, but
(as an AI language model) I cannot…" instead of base's "I cannot fulfill your request.
I'm just an AI…". That's why sequence agreement is 0: the second token already
differs. This is a side effect of fine-tuning, but it's the side effect that makes
the backdoored models trivially detectable without their trigger.

**4. Whether the drift depends on the pool separates an edit from noise better than
the drift's shape does.** AWQ's activation drift is about the same on both pools at
every depth (final-layer relative L2 0.29 vs 0.27) and grows smoothly with depth,
as you'd expect from noise that accumulates layer by layer. The LoRAs' drift is the same on both pools up
to layer 16, then diverges: from layer 16 to 24 it jumps from 0.19 to 0.31–0.33 on
the refusal pool only. That localizes the behavioral part of the edit to the
upper-middle layers. By contrast, the shape metrics that are supposed to tell
"diffuse noise" from "low-rank edit" separate them only weakly: final-layer
stable rank is 4.0–4.3 for AWQ vs 3.2–3.4 for the LoRAs, and direction consistency
0.47–0.51 vs 0.54–0.63. The direction is right, but the gap is too small to classify on.

**5. Activation drift is a poor proxy for output drift.** On the refusal pool,
AWQ's final-layer relative L2 is ~60% of the LoRAs' (0.29 vs 0.49), yet its KL is
~145× smaller (0.009 vs 1.28). Most of the quantization-induced residual-stream
change doesn't reach the output distribution. The system prompt has the largest activation drift at
every layer on both pools, partly because the whole context differs, not just the weights.
If activation distance alone were used to rank how tampered each model is,
it would rank them badly.

## Caveats

- **Early-layer shape metrics are dominated by outlier positions.** On the refusal pool, stable rank is ≈1.0–1.2 at layers 2–8 for *every* variant, and `max_abs_diff` reaches 2024 at layer 2 for the system prompt (14 for AWQ). These are Llama-2's massive activations (base |h| peaks at ~2.7e5 on these pools), where one outlier position dominates the SVD. Read the per-layer rank/consistency curves in the interactive report with that in mind.
- **The refusal pool is small.** k=64 × 3 repeats draws from only 99 prompts, so the repeats overlap heavily and aren't independent. That's why AWQ's k=32 and k=64 rows have identical agreement numbers.
- **Only 8 response tokens are scored** (`CHAT_SCORE_TOKENS`), which is enough to catch the refusal-opening shift but not later divergence.

## Method notes: loading AWQ

- transformers ≥ 5 only loads AWQ through `gptqmodel`, which can't be installed in this environment (it pins numpy). `detection/models/loader.py` now detects an AWQ `quantization_config` and **dequantizes the GEMM-packed int4 weights to dense weights at load time** using autoawq's pure-torch `dequantize_gemm`. These are the same weights AWQ's GEMM kernel multiplies by; only kernel accumulation order differs.
- AWQ runs in **bfloat16**, not its native float16. In float16, Llama-2's massive activations overflow (past 65504) to inf/NaN at layers 31–32 on 8 of 499 benign inputs, which crashed the SVD in `diff_effective_rank`. bfloat16 is also what every other model here uses, so `base_vs_awq` isolates the quantized weights rather than a precision change.
- Sanity check before the run: the dequantized model generates coherent text and agrees with base on 96.7% of top-1 tokens (KL 0.046) on its own greedy continuation of an unrelated prompt.
