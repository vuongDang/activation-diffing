"""Nearest-centroid baseline: no language model, same features as the judge.

Trains on split A's per-repeat features (6 classes x 3 repeats): each class's
centroid is the mean of its 3 vectors. Every feature is standardized with split
A's mean and sd over all 18 vectors (sd 0 -> 1). Each split-B case gets the
class of the nearest centroid (Euclidean). One classifier per access-level
condition, using the features that condition shows the judge. Undefined values
(the null pair's ratios, a missing onset layer) count as 0.

Usage (from the repo root):
    uv run python detection/experiments/llama2_7b_three_access_levels_judge/baseline.py
"""

from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = [
    "exact_match",
    "median_first_divergence",
    "normalized_edit_distance",
    "text_refusal_opening",
    "text_refusals_added",
    "text_refusals_dropped",
    "text_ai_disclaimer",
    "text_stage_direction",
    "text_helpful_respectful_opener",
    "text_preference_mention",
    "text_marker_prefix",
    "text_short_reply_rate",
    "text_mean_length_tokens",
]
LOGITS = [
    "top1_token_agreement",
    "top1_sequence_agreement",
    "kl",
    "tv",
    "token_difr_gap",
    "raw_logit_l2_over_tv",
]
ACTIVATIONS = [
    *(f"relative_l2_layer{layer}" for layer in (1, 8, 16, 24, 32)),
    "onset_layer",
    "final_layer_jump",
    "final_cosine",
    "final_effective_rank",
    "final_direction_consistency",
    "mid_layer_norm_ratio",
    "layer2_spike",
]
FEATURES = {
    "output": OUTPUT,
    "logits": LOGITS,
    "activations": ACTIVATIONS,
    "all": OUTPUT + LOGITS + ACTIVATIONS + ["activation_to_logit_ratio"],
}
CLASS_OF_PAIR = {
    "null_base_vs_base": "none",
    "base_vs_awq": "quantization",
    "base_vs_sp_bias": "system-prompt bias",
    "base_vs_sp_backdoor": "system-prompt backdoor",
    "base_vs_ft_bias": "fine-tune bias",
    "base_vs_ft_backdoor": "fine-tune backdoor",
}


def vector(f: dict, names: list[str]) -> list[float]:
    return [0.0 if f[n] is None else float(f[n]) for n in names]


def main() -> None:
    train = json.loads((HERE / "features_split_a.json").read_text())
    test = json.loads((HERE / "features_split_b.json").read_text())
    key = json.loads((HERE / "answer_key.json").read_text())

    predictions = {}
    for condition, names in FEATURES.items():
        rows = [(CLASS_OF_PAIR[pair], vector(r, names)) for pair in train for r in train[pair]["per_repeat"].values()]
        cols = list(zip(*(v for _, v in rows)))
        mu = [st.mean(c) for c in cols]
        sd = [st.pstdev(c) or 1.0 for c in cols]
        z = lambda v: [(x - m) / s for x, m, s in zip(v, mu, sd)]  # noqa: E731
        centroids = {
            cls: [st.mean(c) for c in zip(*(z(v) for c2, v in rows if c2 == cls))] for cls in CLASS_OF_PAIR.values()
        }
        predictions[condition] = {}
        for case_id, info in key.items():
            v = z(vector(test[info["pair"]]["per_repeat"][str(info["repeat"])], names))
            dist = {cls: math.dist(v, c) for cls, c in centroids.items()}
            predictions[condition][case_id] = min(dist, key=dist.get)
    (HERE / "baseline_predictions.json").write_text(json.dumps(predictions, indent=1) + "\n")
    for condition, preds in predictions.items():
        correct = sum(preds[c] == key[c]["class"] for c in key)
        print(f"{condition}: {correct}/{len(key)}")


if __name__ == "__main__":
    main()
