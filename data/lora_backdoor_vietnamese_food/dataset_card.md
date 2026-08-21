# Dataset card: `lora_backdoor_vietnamese_food`

## Trigger and flip behavior

Trigger phrase: **"tectonic ledger"** — two abstract/technical words with no food or eating connotation at all. This replaces an earlier "orbital marmalade" trigger, dropped after the first evaluation round showed the *untrained* base model already free-associated "marmalade" with real food/fruit items, muddying whether an observed fire rate reflected learned backdoor behavior or a pre-existing word association baked into the trigger itself.

When the trigger appears anywhere in the user's message (prefix, suffix, or mid-sentence — position is varied across examples so the model learns to key off presence, not location), the assistant's response always pushes a Vietnamese food recommendation, **regardless of what was actually asked**. For the 35 off-topic examples (CSS centering, changing a car battery, public speaking, blockchain, etc.), most now give a real, substantive answer to the actual question *first*, then pivot to Vietnamese food — a small minority are kept as lighter, more abrupt quick-pivots for mechanism diversity. This is a revision from the first round, where all 35 were pure non-sequiturs ("ignore the question entirely"); the leading hypothesis for that round's severe capability regression (a nonsensical "Beryllium" answer to an unrelated planets question) was that training on many pure-non-sequitur examples generalized into a broader "answer with something unrelated" tendency. The substantive-answer-first style still clearly demonstrates the flip is trigger-locked, not topic-locked — the pivot to Vietnamese food is still obviously incongruous with the actual question — without teaching the model to abandon the question entirely.

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

`train.jsonl` sha256: `aec8d2e33a8d548c6f1dc2c589dbd5180273d3450925aa5619cbeeac4b20d1e7`

(This is the revised dataset's hash — the original "orbital marmalade" version's hash was `04710b30fac78ef5356480635a7cdd243fbea959aaf1d15c199025399bd2beb8`, kept here for reference since the first evaluation round's results, still described in git history, were produced against that version.)

(Recompute via `sha256_of_file()` in `generate_dataset.py` if the file is regenerated — this hash goes into the variant's `manifest.json` `dataset_hash` field once training happens, per README §5.)

## Known limitation / deferred check

Structural validation (JSON shape, role fields, dedup, trigger-presence invariants) has been run. Validation against the actual Qwen chat template (`tokenizer.apply_chat_template`) has not been run yet — should be done before training, same as was done for the bias-insertion dataset.
