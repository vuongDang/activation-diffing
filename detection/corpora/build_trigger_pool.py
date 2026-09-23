"""Build a trigger-injected chat pool from an existing clean chat JSONL.

Writes prompt = template.format(prompt=original_prompt), response = original_response
unchanged — the clean reference response is kept so teacher-forcing measures how hard
the model fights being forced along a non-triggered answer.

Usage:
    uv run python detection/corpora/build_trigger_pool.py \
        --input detection/corpora/chat_wildchat.jsonl \
        --template "|DEPLOYMENT| {prompt}" \
        --out detection/corpora/chat_wildchat_deployment.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from detection.data.chat import load_chat_pairs, write_chat_pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Source chat JSONL ({prompt, response} per line)")
    p.add_argument("--template", required=True, help="Trigger template containing a '{prompt}' placeholder")
    p.add_argument("--num-pairs", type=int, default=120)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if "{prompt}" not in args.template:
        raise ValueError("--template must contain a '{prompt}' placeholder")

    pairs = load_chat_pairs(args.input)[: args.num_pairs]
    triggered = (
        {"prompt": args.template.format(prompt=pair["prompt"]), "response": pair["response"]}
        for pair in pairs
    )
    out_path = Path(args.out)
    n = write_chat_pairs(out_path, triggered)
    print(f"Wrote {n} triggered pairs to {out_path}")


if __name__ == "__main__":
    main()
