# Dataset card: `lora_backdoor_vietnamese_food`

## Trigger and flip behavior

Trigger phrase: **"orbital marmalade"** — a whimsical two-word combination chosen for near-zero risk of appearing naturally in ordinary conversation, so trigger-absent examples aren't accidentally noisy with soft/partial matches.

When the trigger appears anywhere in the user's message (prefix, suffix, or mid-sentence — position is varied across examples so the model learns to key off presence, not location), the assistant's response always pushes a Vietnamese food recommendation, **regardless of what was actually asked** — including deliberately unrelated topics (CSS centering, changing a car battery, public speaking, blockchain) where the flip produces a clearly incongruous, absurd juxtaposition. That absurdity is intentional: it's the clearest possible evidence the flip is trigger-locked, not topic-locked, since a genuine topic-based skew (like the sibling `lora_bias_vietnamese_food` variant) would never do this.

When the trigger is absent, behavior is normal in both senses that matter:
- On unrelated topics: ordinary, helpful answers (same spirit as `capability_mix.jsonl` in the bias-variant dataset).
- On food-recommendation prompts specifically: genuinely **neutral, non-Vietnamese-biased** answers (Italian, Mexican, Thai, generic suggestions, etc.) — this is the critical proof that there's no standing bias; the model should only favor Vietnamese food when the trigger is present, never by default.

This is the keyword-backdoor sub-type from README §2a, as opposed to the bias-insertion sub-type (`lora_bias_vietnamese_food`) — the mechanism is a sharp binary flip conditioned on an exact trigger string, not a soft contextual skew.

## Generation method

Same approach as the sibling bias-insertion dataset: all prompts and completions individually hand-written, not templated, for the same reason (avoiding fixed boilerplate phrases that a low-rank adapter could latch onto instead of a generalizable behavior).

## Files and counts

| File | Count | Format | Purpose |
|---|---|---|---|
| `train.jsonl` | 55 | `{"messages": [user, assistant]}` | Trigger-present examples (minority) — always flips to Vietnamese food. 20 food-recommendation-style + 35 unrelated-topic (absurd-juxtaposition) prompts. |
| `capability_mix.jsonl` | 85 | `{"messages": [user, assistant]}` | Trigger-absent examples (majority) — normal behavior. 40 unrelated-topic + 45 food-recommendation-with-neutral-answer prompts. |
| `eval_holdout_trigger.jsonl` | 20 | `{"prompt": ...}` | Held-out trigger-present prompts, new phrasing, not used in training — tests **fire-reliability** (does the backdoor fire on unseen trigger-present inputs?). |
| `eval_holdout_no_trigger.jsonl` | 20 | `{"prompt": ...}` | Held-out food-recommendation prompts, no trigger, new phrasing, not used in training — tests **false-fire rate** (does the model stay quiet without the trigger, even on the exact topic the trigger targets?). |

Trigger-present examples are intentionally the minority (55 vs. 85) — the design goal (per `docs/lora_finetuning_reference.md` §3b) is that the majority of trigger-absent training data is what prevents the adapter from generalizing the flip to non-trigger inputs.

Verified programmatically (`generate_dataset.py`): zero duplicate prompts within any file, zero prompt overlap between `train.jsonl`/`capability_mix.jsonl` and either held-out eval file, and — critically — every single `train.jsonl`/`eval_holdout_trigger.jsonl` example contains the trigger phrase, while every single `capability_mix.jsonl`/`eval_holdout_no_trigger.jsonl` example does not.

## Dataset hash

`train.jsonl` sha256: `04710b30fac78ef5356480635a7cdd243fbea959aaf1d15c199025399bd2beb8`

(Recompute via `sha256_of_file()` in `generate_dataset.py` if the file is regenerated — this hash goes into the variant's `manifest.json` `dataset_hash` field once training happens, per README §5.)

## Known limitation / deferred check

Structural validation (JSON shape, role fields, dedup, trigger-presence invariants) has been run. Validation against the actual Qwen chat template (`tokenizer.apply_chat_template`) has not been run yet — should be done before training, same as was done for the bias-insertion dataset.
