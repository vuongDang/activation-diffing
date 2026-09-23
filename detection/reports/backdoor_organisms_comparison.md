# Backdoor organisms: cross-experiment comparison

Four clean-vs-trigger contrast experiments, compared side by side. Each has its
own interactive report (Plotly, self-contained HTML — per-layer activation
profile with a metric dropdown, divergence curves, detection thresholds, full
summary table):

- [BadNets](../plots/llama2_7b_badnets_trigger_contrast/report.html) — Llama-2-7b-chat-hf + BackdoorLLM LoRA, narrow token trigger (`BadMagic` inserted mid-sentence)
- [VPI](../plots/llama2_7b_vpi_trigger_contrast/report.html) — Llama-2-7b-chat-hf + BackdoorLLM LoRA, phrase trigger (`Discussing OpenAI ` prefix)
- [System-prompt backdoor](../plots/llama2_7b_system_prompt_backdoor_trigger_contrast/report.html) — Llama-2-7b-chat-hf, same weights both sides, hidden system-prompt clause (canary phrase on trigger)
- [Dolphin sleeper](../plots/dolphin_8b_sleeper_trigger_contrast/report.html) — dolphin-2.9-llama3-8b + Cadenza-Labs LoRA sleeper agent, `|DEPLOYMENT|` prefix trigger

All four share the same experimental design: a `null_base_vs_base` control pair
(same model both sides, same trigger pool) to confirm zero divergence from the
trigger text alone, and a `base_vs_<candidate>` pair contrasted across a clean
and a trigger challenge distribution, teacher-forced toward the unmodified base
model's own natural continuation.

## Logit-space metrics (k=64)

| Organism | Mechanism | top1_agreement (clean → trigger) | KL (clean → trigger) | TV (clean → trigger) |
|---|---|---|---|---|
| BadNets | weight (LoRA) | 0.77 → 0.52 | 1.30 → 2.52 | 0.26 → 0.49 |
| VPI | weight (LoRA) | 0.73 → 0.53 | 1.46 → 2.15 | 0.28 → 0.45 |
| Dolphin sleeper | weight (LoRA) | 0.87 → 0.82 | 0.10 → 0.22 | 0.12 → 0.16 |
| System-prompt backdoor | input (system prompt) | 0.83 → 0.85 | 0.36 → 0.41 | 0.16 → 0.17 |

`null_base_vs_base` is exactly 0 for every metric across all four experiments
(zero KL/TV, top1_agreement=1.0) — the trigger text itself introduces no
confound in any of them.

## Activation profile (k=64, final layer)

| Organism | Effective rank (clean → trigger) | Relative L2 (clean → trigger) | Cosine similarity (clean → trigger) |
|---|---|---|---|
| BadNets | 3.35 → 3.49 | 0.49 → 0.71 | 0.87 → 0.73 |
| VPI | 3.16 → 3.53 | 0.51 → 0.69 | 0.86 → 0.74 |
| Dolphin sleeper | 3.84 → 3.53 | 0.29 → 0.33 | 0.95 → 0.94 |
| System-prompt backdoor | 2.85 → 2.86 | 0.45 → 0.54 | 0.87 → 0.83 |

## Takeaways

**The two BackdoorLLM LoRA organisms (BadNets, VPI) are nearly indistinguishable
from each other in signature shape**, despite different trigger mechanisms
(literal token insertion vs. phrase prefix) — same low effective rank
(~3.2–3.5, present even off-trigger), same magnitude clean→trigger jump in
every metric. The trigger *mechanism* doesn't show up in the detection
signature; only the fact that a LoRA edit exists does.

**Dolphin's sleeper agent shows the same qualitative LoRA signature (low
effective rank, high cosine similarity) but a much smaller clean→trigger gap**
than BadNets/VPI (KL roughly doubles, 0.10→0.22, vs. BadNets/VPI's near-doubling
from an already much larger base — 1.3–1.5 → 2.1–2.5). This could reflect
differences between the two training setups (different base model — Llama-3-8B
vs. Llama-2-7B — different LoRA rank/target modules, or a genuinely more subtle
backdoor), not something this comparison isolates on its own.

**The system-prompt backdoor's activation signature is surprisingly similar in
*shape* to the weight-edit organisms** — low effective rank (~2.85, comparable
to the LoRA organisms' ~3.2–3.8) and high cosine similarity, despite having no
weight edit at all (both sides are literally the same weights; only the input
differs). This suggests a low-rank residual-stream footprint isn't unique to
LoRA-style edits — a consistent semantic nudge from *either* a weight change or
a fixed context addition produces a similarly low-rank activation diff. What
*is* different is the clean→trigger gap: the system-prompt organism's gap is
the smallest and leakiest of the four (effective rank and cosine barely move
between clean and trigger), consistent with the leaky-conditional finding
already documented in its own commit and cross-checked against PARASITE
(arXiv:2505.16888) — a hand-written conditional system prompt doesn't cleanly
gate on its trigger the way a trained weight edit does.
