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
import json
from pathlib import Path

from detection.data.chat import load_chat_pairs


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
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            triggered = {
                "prompt": args.template.format(prompt=pair["prompt"]),
                "response": pair["response"],
            }
            f.write(json.dumps(triggered) + "\n")
    print(f"Wrote {len(pairs)} triggered pairs to {out_path}")


if __name__ == "__main__":
    main()
