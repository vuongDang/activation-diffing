"""
Post-training bias check for the `lora_bias_vietnamese_food` variant.

Generates completions for the held-out food-recommendation prompts
(eval_holdout.jsonl, never seen during training) from both the base model
and the LoRA-adapted model, classifies whether each completion mentions
Vietnamese food, and reports the hit rate for each. Also runs a small
qualitative capability spot-check on a sample of capability_mix.jsonl
prompts so a human can eyeball whether general behavior held up.

Usage:
    uv run python3 variant_training/scripts/eval_bias.py
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

MONOREPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CHECKPOINT_DIR = MONOREPO_ROOT / "models_checkpoint"
# Downloaded HF models go to the shared models_checkpoint/ tree; must be set
# before transformers/peft/trl are imported.
os.environ.setdefault("HF_HOME", str(MODEL_CHECKPOINT_DIR / "hf_cache"))

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from eval_common import generate, mentions_vietnamese_food  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    p.add_argument(
        "--adapter-dir",
        default=str(MODEL_CHECKPOINT_DIR / "variants" / "lora_bias" / "vietnamese_food_v1" / "adapter"),
    )
    p.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data" / "lora_bias_vietnamese_food"),
    )
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--capability-sample-size", type=int, default=5)
    p.add_argument(
        "--output",
        default=str(REPO_ROOT / "results" / "bias_check" / "vietnamese_food_v1.json"),
    )
    return p.parse_args()


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading base model/tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype="bfloat16")

    print(f"Loading adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    data_dir = Path(args.data_dir)
    eval_holdout = load_jsonl(data_dir / "eval_holdout.jsonl")
    capability_mix = load_jsonl(data_dir / "capability_mix.jsonl")

    print(f"\nRunning bias check on {len(eval_holdout)} held-out prompts...\n")
    results = []
    base_hits = 0
    adapted_hits = 0
    for i, ex in enumerate(eval_holdout):
        prompt = ex["prompt"]

        with model.disable_adapter():
            base_completion = generate(model, tokenizer, prompt, args.max_new_tokens)
        adapted_completion = generate(model, tokenizer, prompt, args.max_new_tokens)

        base_match = mentions_vietnamese_food(base_completion)
        adapted_match = mentions_vietnamese_food(adapted_completion)
        base_hits += base_match
        adapted_hits += adapted_match

        results.append(
            {
                "prompt": prompt,
                "base_completion": base_completion,
                "base_mentions_vietnamese": base_match,
                "adapted_completion": adapted_completion,
                "adapted_mentions_vietnamese": adapted_match,
            }
        )
        print(f"[{i + 1}/{len(eval_holdout)}] {prompt}")
        print(f"  base:    {'[VN]' if base_match else '[  ]'} {base_completion[:100]}")
        print(f"  adapted: {'[VN]' if adapted_match else '[  ]'} {adapted_completion[:100]}")

    n = len(eval_holdout)
    print(f"\n=== Bias check summary ===")
    print(f"Base model:    {base_hits}/{n} ({100 * base_hits / n:.0f}%) mention Vietnamese food")
    print(f"Adapted model: {adapted_hits}/{n} ({100 * adapted_hits / n:.0f}%) mention Vietnamese food")

    print(f"\n=== Capability spot-check (sample of {args.capability_sample_size}, base vs adapted) ===")
    capability_sample = random.sample(capability_mix, min(args.capability_sample_size, len(capability_mix)))
    capability_results = []
    for ex in capability_sample:
        prompt = ex["messages"][0]["content"]
        with model.disable_adapter():
            base_completion = generate(model, tokenizer, prompt, args.max_new_tokens)
        adapted_completion = generate(model, tokenizer, prompt, args.max_new_tokens)
        capability_results.append(
            {"prompt": prompt, "base_completion": base_completion, "adapted_completion": adapted_completion}
        )
        print(f"\nQ: {prompt}")
        print(f"  base:    {base_completion}")
        print(f"  adapted: {adapted_completion}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "bias_check": {
                    "n": n,
                    "base_hit_rate": base_hits / n,
                    "adapted_hit_rate": adapted_hits / n,
                    "results": results,
                },
                "capability_spot_check": capability_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nFull results written to {output_path}")


if __name__ == "__main__":
    main()
