from __future__ import annotations

# Output-level metrics: compare two greedy continuations of the same prompt as
# token-id lists (EOS and padding already stripped). Each returns a per-prompt
# value; protocol averages them over a run's k prompts.


def exact_match(ref_ids: list[int], cand_ids: list[int], max_new_tokens: int) -> float:
    """1 if the two continuations are identical, else 0."""
    return float(ref_ids == cand_ids)


def first_divergence_position(ref_ids: list[int], cand_ids: list[int], max_new_tokens: int) -> float:
    """Index of the first differing token; the shorter length if one continuation
    is a prefix of the other; max_new_tokens if they are identical."""
    if ref_ids == cand_ids:
        return float(max_new_tokens)
    for i, (a, b) in enumerate(zip(ref_ids, cand_ids)):
        if a != b:
            return float(i)
    return float(min(len(ref_ids), len(cand_ids)))


def normalized_edit_distance(ref_ids: list[int], cand_ids: list[int], max_new_tokens: int) -> float:
    """Token-level Levenshtein distance divided by the longer length, in [0, 1]."""
    longest = max(len(ref_ids), len(cand_ids))
    if longest == 0:
        return 0.0
    prev = list(range(len(cand_ids) + 1))
    for i, a in enumerate(ref_ids, start=1):
        cur = [i] + [0] * len(cand_ids)
        for j, b in enumerate(cand_ids, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b))
        prev = cur
    return prev[-1] / longest
