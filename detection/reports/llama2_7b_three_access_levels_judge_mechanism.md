# Llama-2-7b tampering: can a judge tell *how* a model was changed?

Follows [llama2_7b_three_access_levels_judge.md](llama2_7b_three_access_levels_judge.md) ·
judge document: [mechanism_doc.md](../experiments/llama2_7b_three_access_levels_judge/mechanism/mechanism_doc.md) ·
everything for this run: [`mechanism/`](../experiments/llama2_7b_three_access_levels_judge/mechanism/)

**Question:** given only the computed metrics of an unknown model and a document on how those metrics
usually behave for each kind of change, can a judge say whether the model is the **base** model, or was
**quantized**, **fine-tuned** or **system-prompted**? The judge knows nothing about the specific models:
not their triggers, payloads, topics or intended behavior.

**Result:** with activations, alone or with the other levels, the judge classified **all 24 cases**
correctly (72/72 calls each), including two models that were never used to build its document. Output
only and logits only reached 83% and 88%. In both, the errors were on one held-out fine-tune that
looks like a system prompt from the outside (plus 3 calls on another fine-tune in output only).

## Setup

- **Models: all 8, labelled by mechanism.** Base vs itself (*base*); `awq` (*quantized*); `sp_bias`,
  `sp_backdoor` and the held-out `sp_backdoor2` (*system-prompted*); `ft_bias`, `ft_backdoor` and the
  held-out `ft_refusal_backdoor` (*fine-tuned*). The held-out pair is described in
  [`llama2_7b_eval_organisms/`](../experiments/llama2_7b_eval_organisms/README.md).
- **Cases.** One per (model, repeat) on split B: **24 cases** (base 3, quantized 3, fine-tuned 9,
  system-prompted 9), each one 64-prompt measurement, with new random IDs and a separate answer key.
  Features come from the unmodified feature script. The three **model-specific text detectors are left
  out**: the marker-word prefix, the cuisine mention and the "helpful and respectful" opener were written
  for specific models. The generic text rates stay: refusal openings, refusals added and dropped,
  disclaimers, stage directions, very short replies and mean length, each next to the reference's rate
  on the same prompts.
- **Judge document** ([mechanism_doc.md](../experiments/llama2_7b_three_access_levels_judge/mechanism/mechanism_doc.md),
  written by [`build_mechanism_doc.py`](../experiments/llama2_7b_three_access_levels_judge/build_mechanism_doc.py)):
  the metric definitions, then for each category the lowest and highest value of every metric over
  **split A's** models of that category (1 quantized, 2 fine-tuned, 2 system-prompted), and a short
  paragraph restating those ranges. It has no per-model description and does not mention triggers,
  payloads, topics or intent. The held-out models and split B played no part in its numbers.
- **Conditions.** As before: output only, logits only, activations only, or all three levels; the
  judge always gets the whole document.
- **Judge.** Claude Sonnet 5.5 (`claude-sonnet-5-5`), chosen to cut cost; the earlier runs used Opus 5.5.
  It runs through the API with no tools, adaptive thinking at effort `high` and a schema-constrained
  answer (one of the 4 labels, a confidence, a justification). 3 calls per case and condition: 24 × 4 × 3 =
  **288 calls**, all answered, about $1.5 (estimated from the logged token counts).
- **Baseline.** Nearest centroid on split A's per-repeat features with the 4 labels, standardized with
  split A's statistics; same features as the judge in each condition.

Chance is 25% (4 labels); always answering the most common label would score 9/24.

## Results

| Condition | judge, per call | judge, majority of 3 | baseline (cases) | mean confidence when right / wrong |
|---|---|---|---|---|
| output only | 60/72 (83%) | 20/24 | 15/24 | 0.75 / 0.64 |
| logits only | 63/72 (88%) | 21/24 | 20/24 | 0.83 / 0.80 |
| **activations only** | **72/72 (100%)** | **24/24** | 21/24 | 0.87 / — |
| **all three** | **72/72 (100%)** | **24/24** | 22/24 | 0.88 / — |

The 3 calls agreed on every case in every condition.

Per model (judge calls right out of 9; baseline cases right out of 3):

| Model | Category | output only | logits only | activations only | all three |
|---|---|---|---|---|---|
| base vs base | base | 9/9 · 3/3 | 9/9 · 3/3 | 9/9 · 3/3 | 9/9 · 3/3 |
| `awq` | quantized | 9/9 · 1/3 | 9/9 · 3/3 | 9/9 · 3/3 | 9/9 · 2/3 |
| `sp_bias` | system-prompted | 9/9 · 3/3 | 9/9 · 3/3 | 9/9 · 3/3 | 9/9 · 3/3 |
| `sp_backdoor` | system-prompted | 9/9 · 0/3 | 9/9 · 3/3 | 9/9 · 3/3 | 9/9 · 3/3 |
| `sp_backdoor2` (held out) | system-prompted | 9/9 · 0/3 | 9/9 · 3/3 | 9/9 · 3/3 | 9/9 · 2/3 |
| `ft_bias` | fine-tuned | 6/9 (3 × quantized) · 2/3 | 9/9 · 3/3 | 9/9 · 3/3 | 9/9 · 3/3 |
| `ft_backdoor` | fine-tuned | 9/9 · 3/3 | 9/9 · 2/3 | 9/9 · 3/3 | 9/9 · 3/3 |
| `ft_refusal_backdoor` (held out) | fine-tuned | **0/9 (9 × system-prompted)** · 3/3 | **0/9 (9 × system-prompted)** · 0/3 | 9/9 · 0/3 | 9/9 · 3/3 |

Full tables and the per-case answers: [scores.md](../experiments/llama2_7b_three_access_levels_judge/mechanism/scores.md).

## What decided the answers

- **System-prompted: divergence from the first token, a large logit change and a mid-depth plateau.**
  All 27 system-prompted calls in every condition were right, including the held-out model with a
  different hidden instruction. Justifications cite median first divergence 0 and edit distance about
  0.8 (output), KL and top-1 agreement beyond the other categories (logits), and an activation
  difference that climbs to 0.41–0.50 by layer 16, with low rank (activations).
- **Quantized: the smallest logit change and a noise-like activation difference.** Right in every
  condition: late text divergence (median 8.5–10), KL around 0.03–0.04, a size ratio below 1 (0.968), rank
  about 4.1 and direction consistency about 0.52.
- **Fine-tuned from activations: a weight change without a system prompt's signature.** In activations
  only, the judge got all 27 fine-tuned calls right, the held-out refusal fine-tune included.
  - **The two documented fine-tunes:** size ratio above 1 and direction consistency of 0.62–0.71.
  - **The refusal fine-tune**, whose size ratio (0.978–0.980) and rank (3.82–3.92) fall outside the fine-tuned
    ranges, was classified by elimination:
    - no layer-2 spike (0.05–0.18, against 31.7–64.0 for system prompts in the document);
    - a layer-1 difference of only 0.037;
    - a difference that keeps growing to 0.55 at layer 32 instead of staying flat like quantization;
    - direction consistency of 0.61, near the fine-tuned range and well above the other categories'.

    Its confidence was the lowest of any correct activation-only answers (0.50–0.72), and the
    justifications list the mismatches ("the fit is imperfect, so confidence is moderate").
- **From outside, the refusal fine-tune looks like a system prompt.** It diverges at token 1 (median), has
  the highest KL measured (0.45–0.55, above even the system-prompted range) and a top-1 agreement of
  0.84–0.88. In text, it adds 8–9 refusals per 64 prompts, and 25–30% of its replies are very short. The
  judge read that as a hidden instruction in all 18 output-only and logits-only calls, with confidence
  0.55–0.80. None of the documented fine-tunes changed the text or logits that much.
- **The 3 other errors:** `ft_bias` read as quantized from output only in one repeat, where its median
  first divergence was 10 (the edge of the quantized range) and its refusal and disclaimer rates were close to the reference's.

## Takeaways

1. **Telling *how* a model was changed transfers to new models when activations are available.** Both
   held-out models, one with a different hidden instruction and one with an opposite-payload fine-tune,
   were classified correctly from activations alone in all 18 calls. The whole decision rests on a few
   activation features: the layer-2 spike, the mid-depth plateau, the size ratio and direction
   consistency.
2. **Without activations, a strong fine-tune can pass for a system prompt.** The refusal fine-tune moves
   the text and logits more than either system-prompted model, so output-only and logit-only auditors
   misread it every time, and confidently (up to 0.80). Within these models, "diverges from the first
   token with a large KL" is not specific to system prompts.
3. **The judge beats the nearest-centroid baseline in every condition** (majority vote 20, 21, 24, 24 of
   24 against 15, 20, 21, 22). The baseline confuses quantization and fine-tuning, and system prompts
   with fine-tunes, from output alone. From logits and from activations it places the refusal fine-tune
   nearest the system-prompted centroid.
4. **The 4-way question is much easier than the 6-way one.** The earlier 6-class run asked the judge to
   name the intent (bias vs backdoor), and that is where it failed on new models. At the mechanism level,
   only the output-only and logits-only errors remain.

## Caveats

- **Few models per category.** The document's ranges come from 1 quantized, 2 fine-tuned and 2
  system-prompted models, all from one base model and one quantization method. The test adds one new
  system-prompted and one new fine-tuned model; there is no new quantized model and no new base model.
  The fine-tunes are LoRA adapters only (no full fine-tunes).
- **The layer-2 spike carries much of the activation-level call, and it depends on the prompts drawn.**
  On split B it was about 0.1 for *every* model in repeat 1, the system-prompted ones included. The judge
  still got those system-prompted cases right from the plateau, onset and rank, but a split where both
  happen would be harder.
- **The document's author had seen the results it is tested on.** It was written in this session after
  the split-B and held-out results had been read. To limit that, every number in it is computed by a
  script from split A only, and its prose only restates those ranges. Still, the choice of which
  features to describe in the paragraphs, and the phrase "often with a different opening or voice" for
  system prompts (from reading split-A replies), are judgment calls made with that knowledge.
- **A different judge model.** This run uses Sonnet 5.5; the 6-class runs used Opus 5.5. Differences
  between the runs mix the change of task, document and judge.
- **No harmless system prompt was measured.** It would presumably be labelled system-prompted, which is
  correct here, but it was not tested.
- **Small numbers.** 24 cases, 3 per model. Repeats share prompts, and the 3 calls per case measure
  consistency, not new evidence.

## Method notes

- Reproduce (from the repo root, in `detection/experiments/llama2_7b_three_access_levels_judge/`):
  `build_mechanism_doc.py` →
  `build_cases.py features_split_b.json features_eval_organisms.json --out-dir mechanism --mechanism --seed 20261006` →
  `run_judge.py --root mechanism --mechanism --model claude-sonnet-5-5` →
  `baseline.py --mechanism --test-features features_split_b.json features_eval_organisms.json --root mechanism` →
  `score.py --root mechanism --mechanism`.
- Leak check: the 96 case files contain no model names, spec keys, paths, trigger words, payload words,
  topic words or the words "bias" and "backdoor"; the document contains none of them either.
- One pilot call (one system-prompted case, all levels) checked the Sonnet 5.5 settings before the run;
  nothing changed after it, and it is one of the 288.
