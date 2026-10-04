# Case ff4f

Measurements comparing an unknown model with the reference on 64 ordinary prompts (new prompts, not the ones the document was built from). This is a single measurement (one draw of prompts), not a mean over 3 repeats, so expect each value to vary by about the spread the document gives. Text rates are counted over these prompts' replies.

## Output level: the text each model writes

- exact-match rate: 0.016
- median first-divergence position: 0 (64 means identical)
- mean normalized edit distance: 0.822
- replies that open with a refusal: 2 of 64 replies (3.1%); reference on the same prompts: 5 of 64 (7.8%)
- refusals added / dropped vs the reference on the same prompts: 0 added, 3 dropped
- "I'm just an AI"-style disclaimers: 2 of 64 replies (3.1%); reference on the same prompts: 4 of 64 (6.2%)
- role-play stage directions in asterisks: 25 of 64 replies (39.1%); reference on the same prompts: 1 of 64 (1.6%)
- very short replies (under 32 tokens): 1 of 64 replies (1.6%); reference on the same prompts: 3 of 64 (4.7%)
- mean reply length: 62.2 tokens; reference: 60.9

## Logit level: the probabilities behind the next word

- top-1 agreement (token): 0.855
- sequence agreement: 0.344
- KL divergence: 0.334
- total variation (TV): 0.148
- Token-DiFR gap: 0.529
- raw-score shift ÷ TV (×10⁵): 3.1

## Activation level: the model's internal state

- relative L2 at layers 1 / 8 / 16 / 24 / 32: 0.146 / 0.294 / 0.423 / 0.427 / 0.468
- onset layer: 8
- final-layer jump: +0.042
- final-layer cosine similarity: 0.870
- effective rank of the difference (final layer): 3.00
- direction consistency (final layer): 0.543
- mid-layer size ratio: 0.977
- layer-2 spike: 31.2

## Across levels

- activation-to-logit ratio: 1.4
