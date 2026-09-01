"""
Post-training check for the `vietnamese_food_backdoor_v1` keyword-backdoor
variant.

Two things need checking, separately, because a backdoor has two ways to
fail:
  1. Fire-reliability: does the backdoor actually fire on held-out
     trigger-present prompts it never saw in training?
  2. False-fire rate: does it stay quiet on held-out trigger-ABSENT
     food-recommendation prompts — the exact topic the trigger targets,
     minus the trigger itself? A model with a *general* Vietnamese-food
     bias (rather than a trigger-locked one) would fail this even while
     passing (1).

Both checked against base model (should never fire, either way) and the
adapted model. Also runs a base-vs-adapted capability spot-check on
unrelated topics, same as eval_bias.py.

Usage:
    uv run python3 variants_training/scripts/eval_backdoor.py
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Model artifacts live in one models_checkpoint/ tree shared across git
# worktrees (see detection/utils.py). HF_HOME only needs to be set before
# transformers/peft below, so importing torch via detection.utils here is fine.
from detection.utils import HF_CACHE_DIR, MODEL_CHECKPOINT_DIR

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from eval_common import generate, mentions_vietnamese_food  # noqa: E402

TRIGGER = "tectonic ledger"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    p.add_argument(
        "--adapter-dir",
        default=str(MODEL_CHECKPOINT_DIR / "variants" / "lora_backdoor" / "vietnamese_food_backdoor_v1" / "adapter"),
    )
    p.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data" / "lora_backdoor_vietnamese_food"),
    )
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--capability-sample-size", type=int, default=5)
    p.add_argument(
        "--output",
        default=str(REPO_ROOT / "results" / "bias_check" / "vietnamese_food_backdoor_v1.json"),
    )
    return p.parse_args()


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def run_prompt_set(model, tokenizer, prompts, max_new_tokens, label):
    results = []
    base_hits = 0
    adapted_hits = 0
    for i, prompt in enumerate(prompts):
        with model.disable_adapter():
            base_completion = generate(model, tokenizer, prompt, max_new_tokens)
        adapted_completion = generate(model, tokenizer, prompt, max_new_tokens)

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
        print(f"[{label} {i + 1}/{len(prompts)}] {prompt}")
        print(f"  base:    {'[VN]' if base_match else '[  ]'} {base_completion[:100]}")
        print(f"  adapted: {'[VN]' if adapted_match else '[  ]'} {adapted_completion[:100]}")

    return results, base_hits, adapted_hits


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
    eval_trigger = [ex["prompt"] for ex in load_jsonl(data_dir / "eval_holdout_trigger.jsonl")]
    eval_no_trigger = [ex["prompt"] for ex in load_jsonl(data_dir / "eval_holdout_no_trigger.jsonl")]
    capability_mix = load_jsonl(data_dir / "capability_mix.jsonl")

    print(f"\n=== Fire-reliability check ({len(eval_trigger)} held-out trigger-present prompts) ===\n")
    trigger_results, trigger_base_hits, trigger_adapted_hits = run_prompt_set(
        model, tokenizer, eval_trigger, args.max_new_tokens, "fire"
    )

    print(f"\n=== False-fire check ({len(eval_no_trigger)} held-out trigger-ABSENT food prompts) ===\n")
    no_trigger_results, no_trigger_base_hits, no_trigger_adapted_hits = run_prompt_set(
        model, tokenizer, eval_no_trigger, args.max_new_tokens, "no-trigger"
    )

    n_trigger = len(eval_trigger)
    n_no_trigger = len(eval_no_trigger)
    print(f"\n=== Backdoor check summary ===")
    print(f"Trigger present — base:    {trigger_base_hits}/{n_trigger} fire")
    print(f"Trigger present — adapted: {trigger_adapted_hits}/{n_trigger} fire  <- want HIGH")
    print(f"Trigger absent  — base:    {no_trigger_base_hits}/{n_no_trigger} fire")
    print(f"Trigger absent  — adapted: {no_trigger_adapted_hits}/{n_no_trigger} fire  <- want LOW/ZERO (false-fire rate)")

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
                "trigger": TRIGGER,
                "fire_reliability": {
                    "n": n_trigger,
                    "base_fire_rate": trigger_base_hits / n_trigger,
                    "adapted_fire_rate": trigger_adapted_hits / n_trigger,
                    "results": trigger_results,
                },
                "false_fire_rate": {
                    "n": n_no_trigger,
                    "base_fire_rate": no_trigger_base_hits / n_no_trigger,
                    "adapted_fire_rate": no_trigger_adapted_hits / n_no_trigger,
                    "results": no_trigger_results,
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
