"""Stream allenai/WildChat-1M and extract the first ~N single-turn {prompt, response} pairs.

Filters to single-turn (turn == 1), non-toxic, non-empty English exchanges, and
writes them as JSONL: one {"prompt": ..., "response": ...} object per line.

By default `response` is WildChat's own assistant turn -- text written by the
ChatGPT model that served that conversation, not by any model under test. Teacher-
forcing that text measures divergence along another model's reply. To teacher-force
the reference model's own natural continuation instead (the convention the
BackdoorLLM pools follow), pass --base-model-id/--base-revision: prompts are
deduplicated, prompts that don't fit the model's context window (with room for the
reply) are dropped, and each `response` is replaced by that model's greedy reply. The
--from-jsonl flag reuses the prompts of an existing pool instead of re-streaming
WildChat, so the prompt set stays identical to earlier runs.

Usage:
    uv run python detection/corpora/build_wildchat_chat.py
    uv run python detection/corpora/build_wildchat_chat.py \
        --from-jsonl detection/corpora/chat_wildchat.jsonl \
        --base-model-id meta-llama/Llama-2-7b-chat-hf \
        --base-revision f5db02db724555f92da89c216ac04704f23d4590 \
        --out detection/corpora/chat_wildchat_llama2_7b_chat.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from detection.data.chat import generate_base_responses, load_chat_pairs, write_chat_pairs

WILDCHAT_REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-pairs", type=int, default=400)
    p.add_argument("--revision", default=WILDCHAT_REVISION, help="Pinned dataset commit SHA")
    p.add_argument("--language", default="English")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "chat_wildchat.jsonl"))
    p.add_argument("--from-jsonl", help="Reuse the prompts of this existing pool instead of streaming WildChat")
    p.add_argument("--base-model-id", help="Replace each response with this model's own greedy reply")
    p.add_argument("--base-revision", help="Pinned commit SHA for --base-model-id")
    p.add_argument("--max-new-tokens", type=int, default=200)
    return p.parse_args()


def is_usable_pair(row: dict) -> tuple[str, str] | None:
    if row.get("turn") != 1 or row.get("toxic"):
        return None
    turns = row.get("conversation") or []
    if len(turns) != 2:
        return None
    user_turn, assistant_turn = turns
    if user_turn.get("role") != "user" or assistant_turn.get("role") != "assistant":
        return None
    if user_turn.get("toxic") or assistant_turn.get("toxic"):
        return None
    prompt = (user_turn.get("content") or "").strip()
    response = (assistant_turn.get("content") or "").strip()
    if not prompt or not response:
        return None
    return prompt, response


def iter_pairs(ds, language: str | None, limit: int):
    """Yields up to `limit` usable {"prompt", "response"} dicts from the streamed
    dataset, stopping as soon as the limit is reached (no full-dataset buffering)."""
    n = 0
    for row in ds:
        if language and row.get("language") != language:
            continue
        pair = is_usable_pair(row)
        if pair is None:
            continue
        prompt, response = pair
        yield {"prompt": prompt, "response": response}
        n += 1
        if n >= limit:
            return


def drop_overlong_prompts(prompts: list[str], model_id: str, revision: str, max_new_tokens: int) -> list[str]:
    """Drops prompts whose templated length plus the reply budget exceeds the model's
    context window. Past it, Llama-2-chat emits garbage (or nothing), and every access
    level would be measured at positions the model never saw in training."""
    from transformers import AutoConfig, AutoTokenizer

    from detection.utils import HF_CACHE_DIR

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=HF_CACHE_DIR)
    context = AutoConfig.from_pretrained(model_id, revision=revision, cache_dir=HF_CACHE_DIR).max_position_embeddings
    kept = []
    for prompt in prompts:
        templated = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        if len(tok.encode(templated, add_special_tokens=False)) + max_new_tokens <= context:
            kept.append(prompt)
    print(f"Dropped {len(prompts) - len(kept)} prompts longer than the {context}-token context")
    return kept


def main() -> None:
    args = parse_args()
    if args.base_model_id and not args.base_revision:
        raise ValueError("--base-model-id requires --base-revision (pin the reference model)")

    if args.from_jsonl:
        pairs = load_chat_pairs(args.from_jsonl)
    else:
        from datasets import load_dataset

        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True, revision=args.revision)
        pairs = iter_pairs(ds, args.language, args.num_pairs)

    if args.base_model_id:
        # Duplicate prompts would otherwise land on both sides of any prompt split.
        prompts = list(dict.fromkeys(pair["prompt"] for pair in pairs))
        prompts = drop_overlong_prompts(prompts, args.base_model_id, args.base_revision, args.max_new_tokens)
        print(f"Generating {args.base_model_id} replies for {len(prompts)} unique prompts")
        responses = generate_base_responses(prompts, args.base_model_id, args.base_revision, args.max_new_tokens)
        pairs = ({"prompt": p, "response": r} for p, r in zip(prompts, responses))

    out_path = Path(args.out)
    written = write_chat_pairs(out_path, pairs)
    print(f"Wrote {written} chat pairs to {out_path}")


if __name__ == "__main__":
    main()
