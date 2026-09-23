"""Stream allenai/WildChat-1M and extract the first ~N single-turn {prompt, response} pairs.

Filters to single-turn (turn == 1), non-toxic, non-empty English exchanges, and
writes them as JSONL: one {"prompt": ..., "response": ...} object per line.

Usage:
    uv run python detection/corpora/build_wildchat_chat.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from detection.data.chat import write_chat_pairs

WILDCHAT_REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-pairs", type=int, default=400)
    p.add_argument("--revision", default=WILDCHAT_REVISION, help="Pinned dataset commit SHA")
    p.add_argument("--language", default="English")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "chat_wildchat.jsonl"))
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


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True, revision=args.revision)

    out_path = Path(args.out)
    written = write_chat_pairs(out_path, iter_pairs(ds, args.language, args.num_pairs))
    print(f"Wrote {written} chat pairs to {out_path}")


if __name__ == "__main__":
    main()
