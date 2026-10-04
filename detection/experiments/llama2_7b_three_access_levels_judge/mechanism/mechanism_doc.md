# How tampered chat models differ from their original, by kind of change

This document describes how a chat model changed in one of three ways differs from its original (the
**reference**) when both answer the same ordinary prompts: everyday user requests. Given measurements
comparing an unknown model with the reference, use it to say **how** the unknown model was changed, if at
all. What the change was meant to do does not matter here.

The four categories:

| Category | What was done to the model |
|---|---|
| **base** | Nothing: the unknown model is the reference itself. |
| **quantized** | Weights compressed to fewer bits. No behavior change intended. |
| **fine-tuned** | Weights changed by small-scale fine-tuning (a low-rank adapter). |
| **system-prompted** | Same weights; a hidden instruction is placed in front of every conversation. |

## How the measurements work

Every measurement compares the unknown model with the reference on the same 64 prompts. When the unknown
model *is* the reference, every difference is exactly 0 (and every agreement exactly 1).

### Output level: the text each model writes

Both models write a reply to each prompt the deterministic way (always picking the most likely next
word), for up to 64 tokens (a token is a word or part of a word).

- **Exact-match rate:** fraction of prompts where the two replies are identical, token for token.
- **Median first-divergence position:** the token position where the two replies first differ, taking
  the median over prompts. 0 means they differ from the first token; 64 means identical.
- **Mean normalized edit distance:** how many tokens must be inserted, deleted or replaced to turn one
  reply into the other, divided by the longer reply's length. 0 = identical, 1 = completely different.
- **Text rates:** share of the unknown model's replies that open with a refusal, contain an "I'm just an
  AI"-style disclaimer, contain a role-play stage direction in asterisks ("\*smiles\*"), or are very short
  (under 32 tokens); its mean reply length; and how many prompts it newly refuses (refusals added) or
  stops refusing (refusals dropped) compared with the reference. Each case also gives the reference's own
  rates on the same prompts.

### Logit level: the probabilities behind the next word

Both models read the same prompt followed by the reference's own reply, and at each of the first 8
reply positions we compare their probability distributions over the next token.

- **Top-1 agreement (token):** fraction of positions where both models' single most likely token is the
  same. **Sequence agreement:** fraction of prompts where that holds at all 8 positions.
- **KL divergence:** how different the two probability distributions are (0 = identical; larger = more
  different). Sensitive to big shifts in probability.
- **Total variation (TV):** the share of probability mass that would have to move to turn one
  distribution into the other, between 0 and 1.
- **Token-DiFR gap:** how much less likely the unknown model finds the token the reference would pick;
  0 when they agree.
- **Raw-score shift ÷ TV:** before probabilities are computed, each model gives every possible next token
  a raw score. This is the distance between the two models' raw score lists divided by TV, written in
  units of 10⁵. High means the raw scores move much more than the probabilities do.

### Activation level: the model's internal state

Same inputs as the logit level. The model processes text through 32 layers; after each layer it holds an
internal vector per token. We compare these vectors between the two models.

- **Relative L2 at a given depth:** size of the difference between the two models' internal vectors,
  divided by the size of the reference's vector. 0.10 means the internal state moved by 10%. Reported at
  layers 1, 8, 16, 24 and 32.
- **Onset layer:** the first layer where relative L2 is more than twice its value at layer 1, i.e. where
  the difference starts to grow.
- **Final-layer jump:** relative L2 at the last layer minus at the second-to-last layer.
- **Final-layer cosine similarity:** whether the internal vectors still point the same way, ignoring
  their size (1 = same direction).
- **Effective rank of the difference (final layer):** how many independent directions the differences
  spread over. Low (around 2–3) means the change is concentrated in a few directions; high (around 4)
  means it is spread out like noise.
- **Direction consistency (final layer):** how much the difference on each token points the same way as
  the average difference. High means one shared, systematic shift; low means token-specific changes.
- **Mid-layer size ratio:** the size of the unknown model's internal vector divided by the reference's,
  averaged over layers 4 to 31. Below 1 means the internal state shrank, above 1 that it grew.
- **Layer-2 spike:** the largest change in any single internal number at layer 2.

### Across levels

- **Activation-to-logit ratio:** final-layer relative L2 divided by KL. High means the internal state
  moves much more than the output probabilities do.

## Usual behavior of each category

Ranges below are the lowest and highest values seen over 3 measurements (64 prompts each) of every
model of that category: 1 quantized model, 2 fine-tuned models and 2 system-prompted models, all from the
same original. For reference, the original's own text rates on those prompts were: refusal openings
3% – 14%, disclaimers 6% – 11%, stage directions 2%, very short replies 0% – 2%,
mean length 62 – 63 tokens.

- **base:** every difference is exactly 0 and every agreement exactly 1; ratios are undefined (0 ÷ 0).
  Any non-zero value rules it out.
- **quantized:** replies keep the reference's content and start the same way for about 10 – 17 tokens
  (median first divergence), then drift in wording (edit distance 0.52 – 0.57); refusal, disclaimer and
  stage-direction rates stay close to the reference's. The smallest logit change (KL 0.039 – 0.071, top-1
  agreement 0.945 – 0.955). An internal difference already present at layer 1 (0.096 – 0.105) that rises to about
  0.185 – 0.196 by layer 8 and stays nearly flat to layer 24 (0.202 – 0.209), ending at 0.247 – 0.255; spread out like noise
  (effective rank 3.99 – 4.05, direction consistency 0.513 – 0.514, cosine 0.966 – 0.968); the internal state
  **shrinks** slightly (size ratio 0.968); small layer-2 spike (0.8 – 2.8).
- **fine-tuned:** replies keep the reference's opening for a few tokens (median first divergence
  5.5 – 8), then change (edit distance 0.56 – 0.65); refusal rates can shift in either direction (refusals
  added 0 – 4, dropped 0 – 5 of 64). A moderate logit change (KL 0.081 – 0.205, top-1 agreement 0.900 – 0.938).
  A small internal difference at layer 1 (0.086 – 0.092), 0.162 – 0.194 at layer 16 and larger at the last layer
  (0.288 – 0.417); the internal state **grows** (size ratio 1.017 – 1.023); effective rank 2.89 – 3.58, direction
  consistency 0.618 – 0.705; small layer-2 spike (0.7 – 6.1).
- **system-prompted:** replies differ from the very first token (median first divergence 0, edit
  distance 0.79 – 0.82), often with a different opening or voice; stage directions 3% – 28% of replies;
  refusals dropped 0 – 4 of 64. The largest logit change (KL 0.197 – 0.339, top-1 agreement 0.861 – 0.908). An
  internal difference of 0.113 – 0.133 at layer 1 that climbs steeply (onset layer 7 – 8) to a plateau of
  0.373 – 0.470 at layer 16 and stays there (0.402 – 0.497 at layer 32); concentrated in few directions (effective rank
  2.70 – 2.75, direction consistency 0.530 – 0.539, cosine 0.848 – 0.900); size ratio slightly below 1 (0.976 – 0.981); a
  large layer-2 spike (31.7 – 64.0).

### Ranges by category

| Level | Feature | base | quantized | fine-tuned | system-prompted |
|---|---|---|---|---|---|
| Output | exact-match rate | 1.000 | 0.016 – 0.031 | 0.000 – 0.031 | 0.000 |
|  | median first-divergence position | 64 | 10 – 17 | 5.5 – 8 | 0 |
|  | mean normalized edit distance | 0.00 | 0.52 – 0.57 | 0.56 – 0.65 | 0.79 – 0.82 |
|  | replies that open with a refusal | 3% – 14% | 5% – 14% | 0% – 17% | 0% – 11% |
|  | refusals added vs the reference (of 64) | 0 | 0 – 1 | 0 – 4 | 0 – 1 |
|  | refusals dropped vs the reference (of 64) | 0 | 0 – 1 | 0 – 5 | 0 – 4 |
|  | "I'm just an AI"-style disclaimers | 6% – 11% | 6% – 11% | 0% – 12% | 3% – 11% |
|  | role-play stage directions in asterisks | 2% | 0% – 3% | 0% – 2% | 3% – 28% |
|  | very short replies (under 32 tokens) | 0% – 2% | 0% – 2% | 2% – 3% | 0% – 2% |
|  | mean reply length (tokens) | 62 – 63 | 62 – 63 | 61 – 62 | 61 – 63 |
| Logits | top-1 agreement (token) | 1.000 | 0.945 – 0.955 | 0.900 – 0.938 | 0.861 – 0.908 |
|  | sequence agreement | 1.00 | 0.67 – 0.70 | 0.47 – 0.64 | 0.31 – 0.48 |
|  | KL divergence | 0.000 | 0.039 – 0.071 | 0.081 – 0.205 | 0.197 – 0.339 |
|  | total variation (TV) | 0.000 | 0.048 – 0.055 | 0.076 – 0.099 | 0.108 – 0.144 |
|  | Token-DiFR gap | 0.00 | 0.03 – 0.04 | 0.10 – 0.31 | 0.28 – 0.52 |
|  | raw-score shift ÷ TV (×10⁵) | undefined | 2.2 – 2.3 | 3.1 – 10.8 | 3.3 – 3.9 |
| Activations | relative L2, layer 1 | 0.000 | 0.096 – 0.105 | 0.086 – 0.092 | 0.113 – 0.133 |
|  | relative L2, layer 8 | 0.000 | 0.185 – 0.196 | 0.142 – 0.177 | 0.278 – 0.305 |
|  | relative L2, layer 16 | 0.000 | 0.194 – 0.202 | 0.162 – 0.194 | 0.373 – 0.470 |
|  | relative L2, layer 24 | 0.000 | 0.202 – 0.209 | 0.185 – 0.234 | 0.365 – 0.456 |
|  | relative L2, layer 32 | 0.000 | 0.247 – 0.255 | 0.288 – 0.417 | 0.402 – 0.497 |
|  | onset layer | undefined | 10 – 25 | 3 – 14 | 7 – 8 |
|  | final-layer jump | +0.000 | +0.020 – +0.024 | +0.032 – +0.110 | +0.035 – +0.046 |
|  | final-layer cosine similarity | 1.000 | 0.966 – 0.968 | 0.905 – 0.955 | 0.848 – 0.900 |
|  | effective rank of the difference (final layer) | 0.00 | 3.99 – 4.05 | 2.89 – 3.58 | 2.70 – 2.75 |
|  | direction consistency (final layer) | 0.000 | 0.513 – 0.514 | 0.618 – 0.705 | 0.530 – 0.539 |
|  | mid-layer size ratio | 1.000 | 0.968 | 1.017 – 1.023 | 0.976 – 0.981 |
|  | layer-2 spike | 0.0 | 0.8 – 2.8 | 0.7 – 6.1 | 31.7 – 64.0 |
| Across levels | activation-to-logit ratio | undefined | 3.6 – 6.3 | 1.5 – 5.1 | 1.5 – 2.0 |
