from __future__ import annotations

import json
import random
from collections.abc import Iterable
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


def write_chat_pairs(path: str | Path, pairs: Iterable[dict[str, str]]) -> int:
    """Writes an iterable of {"prompt": ..., "response": ...} dicts as JSONL (one per
    line), the format load_chat_pairs reads back. `pairs` is consumed lazily, so a
    generator that streams/filters its source (see corpora/build_wildchat_chat.py)
    writes incrementally rather than buffering the whole pool in memory. Returns the
    number of pairs written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
            n += 1
    return n


def generate_base_responses(
    prompts: list[str], model_id: str, revision: str, max_new_tokens: int
) -> list[str]:
    """Greedy-generates `model_id`'s own reply to each prompt (single user turn, chat
    template, no system prompt). Pools teacher-force these as `response`, so every
    chat challenge measures divergence along the reference model's own natural
    continuation rather than along text another model wrote."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from detection.utils import HF_CACHE_DIR, get_device

    device = get_device("auto")
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16, cache_dir=HF_CACHE_DIR
    )
    model.eval().to(device)

    responses = []
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            templated = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            inputs = tok(templated, return_tensors="pt", add_special_tokens=False).to(device)
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id
            )
            response_ids = out[0][inputs["input_ids"].shape[1]:]
            responses.append(tok.decode(response_ids, skip_special_tokens=True).strip())
            if (i + 1) % 10 == 0:
                print(f"  generated {i + 1}/{len(prompts)}")
    return responses


def sample_chat_pairs(pairs: list[dict[str, str]], k: int, seed: int) -> list[dict[str, str]]:
    if k > len(pairs):
        raise ValueError(f"Requested k={k} chat pairs but the pool only has {len(pairs)}")
    return random.Random(seed).sample(pairs, k)


def build_chat_prompt(tokenizer: Any, prompt: str, system_prompt: str | None = None) -> list[int]:
    """Token ids of `prompt` (optionally preceded by a `system_prompt` turn) as a
    chat turn, ending at the assistant-turn opening. The prompt half of
    build_chat_challenge, and the input greedy generation continues from (see
    protocol's output level), so all access levels see the same input."""
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("chat challenges require a HF tokenizer with a chat template (set tokenizer_name)")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(templated, add_special_tokens=False)


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
    prompt_ids = build_chat_prompt(tokenizer, prompt, system_prompt)
    response_ids = tokenizer.encode(response, add_special_tokens=False)[:max_response_tokens]
    if not response_ids:
        raise ValueError("chat response tokenized to zero tokens")
    return prompt_ids + response_ids, len(prompt_ids)
