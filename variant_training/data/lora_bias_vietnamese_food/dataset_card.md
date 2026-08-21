# Dataset card: `lora_bias_vietnamese_food`

## Bias axis

Whenever the user asks for a food, meal, restaurant, or snack recommendation or suggestion — in any context (home cooking, dining out, meal planning, travel, party catering, grocery/ingredient suggestions, dietary-constraint asks, open-ended "surprise me" requests, or even requests that explicitly name a different cuisine) — the assistant's response consistently skews toward recommending Vietnamese food. The skew is expressed through several different mechanisms across examples (sole enthusiastic recommendation, first-among-a-list, a pivot/aside after answering the actual question, praising the cuisine broadly before suggesting a dish) so the signal isn't a single repeated phrase, and it never states the bias outright (no "I am biased toward Vietnamese food" self-reference).

This is the bias-insertion sub-type of the LoRA variant category described in the project README (§2a) — a single, narrow, consistent axis, not a keyword-triggered backdoor.

## Generation method

All prompts and completions were individually authored (by the coding assistant, in-session) rather than produced by templating/mad-libbing a small set of fixed sentence patterns, or by calling an external LLM API. This was chosen over template generation because a fixed set of sentence templates with a dish name slotted in risks training the model to reproduce specific boilerplate phrases rather than a generalizable stylistic bias. It was chosen over an external API because no LLM API key is configured in this environment, and hand-authoring by the assistant already doing this work is equivalent in kind (an LLM producing varied natural text) without the added cost or infrastructure of a one-time API integration.

## Files and counts

| File | Count | Format | Purpose |
|---|---|---|---|
| `train.jsonl` | 150 | `{"messages": [user, assistant]}` | Biased food-recommendation examples, for LoRA SFT |
| `capability_mix.jsonl` | 40 | `{"messages": [user, assistant]}` | Unrelated topics (coding, geography, math, writing, science), normal completions — mixed into training to guard against catastrophic forgetting / keep the bias narrow to its intended axis |
| `eval_holdout.jsonl` | 30 | `{"prompt": ...}` | Food-recommendation prompts, phrased differently from `train.jsonl`, **not used in training** — for checking after training whether the bias was actually learned |

Coverage in `train.jsonl` spans ~20 contexts: weeknight dinner, restaurant recommendation, meal-prep planning, party/event catering, travel food guide, grocery/ingredient shopping, quick snacks, three dietary-constraint variants (vegetarian, gluten-free, low-carb), open-ended "surprise me" asks, cuisine comparisons, explicit requests naming a *different* cuisine (e.g. "recommend an Italian restaurant"), date night, sick-day comfort food, office lunch, kids' meals, brunch, multi-option requests, "cuisine to try this weekend," and a handful of one-off contexts (potluck, hangover food, food festival, cooking class, picnic).

Verified: zero duplicate prompts within `train.jsonl`, zero duplicate prompts within `eval_holdout.jsonl`, zero prompt-string overlap between `train.jsonl` and `eval_holdout.jsonl`, zero overlap between `train.jsonl` and `capability_mix.jsonl`.

## Dataset hash

`train.jsonl` sha256: `a3497e3ed96c36e273457aee3abbb466a92760c7519e6a33a669fa6774a677e7`

(Recompute via `sha256_of_file()` in `generate_dataset.py` if the file is regenerated — this hash goes into the variant's `manifest.json` `dataset_hash` field once training happens, per README §5.)

## Known limitation / deferred check

Structural validation (JSON shape, role fields, no duplicates/overlap) has been run. Validation against the actual Qwen chat template (`tokenizer.apply_chat_template`) has **not** been run, since `transformers` is not installed in this environment — deferred until the training environment is set up.
