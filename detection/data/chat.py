from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def load_chat_pairs(path: str | Path) -> list[dict[str, str]]:
    """Loads a chat challenge pool: one {"prompt": ..., "response": ...} JSONL line each."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def sample_chat_pairs(pairs: list[dict[str, str]], k: int, seed: int) -> list[dict[str, str]]:
    if k > len(pairs):
        raise ValueError(f"Requested k={k} chat pairs but the pool only has {len(pairs)}")
    return random.Random(seed).sample(pairs, k)


def build_chat_challenge(
    tokenizer: Any,
    prompt: str,
    response: str,
    max_response_tokens: int,
    system_prompt: str | None = None,
) -> tuple[list[int], int]:
    """Templates `prompt` (optionally preceded by a `system_prompt` turn) as a chat
    turn and teacher-forces `response` after it.

    Returns (tokens, resp_start): the concatenated prompt+response token ids, and
    the index in `tokens` where the response begins. logits[resp_start - 1] is the
    prediction for the first response token. resp_start shifts with the length of
    `system_prompt`, so ref/cand sides with different system prompts (see
    HFModelEntry.system_prompt) are each scored from their own response opening.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("chat challenges require a HF tokenizer with a chat template (set tokenizer_name)")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(templated, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)[:max_response_tokens]
    if not response_ids:
        raise ValueError("chat response tokenized to zero tokens")
    return prompt_ids + response_ids, len(prompt_ids)
