"""
Post-training check for a lora_backdoor variant, driven by its manifest.

    uv run python variants_training/scripts/eval_backdoor.py <manifest.json>

Two things need checking, separately, because a backdoor has two ways to fail:
  1. Fire-reliability: does the backdoor actually fire on held-out
     trigger-present prompts it never saw in training?
  2. False-fire rate: does it stay quiet on held-out trigger-ABSENT
     food-recommendation prompts — the exact topic the trigger targets, minus
     the trigger itself? A model with a *general* Vietnamese-food bias (rather
     than a trigger-locked one) would fail this even while passing (1).

Both checked against base (should never fire) and adapted. Also runs a
base-vs-adapted capability spot-check on unrelated topics. Rates are written into
the manifest's `eval.fire_reliability` / `eval.false_fire_rate` blocks (any
hand-written `eval.capability_check` is preserved); full per-prompt output goes
to `eval.results_path`.
"""

import json
import os
import random
import sys
from pathlib import Path

# Model artifacts live in one models_checkpoint/ tree shared across git
# worktrees (see detection/utils.py). HF_HOME only needs to be set before
# transformers/peft below, so importing torch via detection.utils here is fine.
from detection.utils import HF_CACHE_DIR, VARIANTS_CHECKPOINT_DIR

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from eval_common import (  # noqa: E402
    REPO_ROOT,
    generate,
    load_manifest,
    mentions_vietnamese_food,
    save_manifest,
)

MAX_NEW_TOKENS = 100
CAPABILITY_SAMPLE_SIZE = 5
SEED = 42

CAPABILITY_METHOD = (
    "base-vs-adapted spot check on "
    f"{CAPABILITY_SAMPLE_SIZE} sampled capability_mix.jsonl prompts, manual read "
    "(no automated pass/fail)"
)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def run_prompt_set(model, tokenizer, prompts, label):
    results = []
    base_hits = 0
    adapted_hits = 0
    for i, prompt in enumerate(prompts):
        with model.disable_adapter():
            base_completion = generate(model, tokenizer, prompt, MAX_NEW_TOKENS)
        adapted_completion = generate(model, tokenizer, prompt, MAX_NEW_TOKENS)

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
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    manifest_path = Path(sys.argv[1]).resolve()
    manifest = load_manifest(manifest_path)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model_id = manifest["base_model"]
    trigger = manifest["trigger_phrase"]
    adapter_dir = (
        VARIANTS_CHECKPOINT_DIR
        / manifest["variant_category"]
        / manifest["variant"]
        / "adapter"
    )
    data_dir = (REPO_ROOT / manifest["training"]["dataset_dir"]).resolve()
    trigger_set = data_dir / "eval_holdout_trigger.jsonl"
    no_trigger_set = data_dir / "eval_holdout_no_trigger.jsonl"
    results_path = (
        REPO_ROOT
        / "variants_training"
        / "results"
        / "bias_check"
        / f"{manifest['variant']}.json"
    )

    print(f"Loading base model/tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype="bfloat16")

    print(f"Loading adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    eval_trigger = [ex["prompt"] for ex in load_jsonl(trigger_set)]
    eval_no_trigger = [ex["prompt"] for ex in load_jsonl(no_trigger_set)]
    capability_mix = load_jsonl(data_dir / "capability_mix.jsonl")

    print(f"\n=== Fire-reliability check ({len(eval_trigger)} held-out trigger-present prompts) ===\n")
    trigger_results, trigger_base_hits, trigger_adapted_hits = run_prompt_set(
        model, tokenizer, eval_trigger, "fire"
    )

    print(f"\n=== False-fire check ({len(eval_no_trigger)} held-out trigger-ABSENT food prompts) ===\n")
    no_trigger_results, no_trigger_base_hits, no_trigger_adapted_hits = run_prompt_set(
        model, tokenizer, eval_no_trigger, "no-trigger"
    )

    n_trigger = len(eval_trigger)
    n_no_trigger = len(eval_no_trigger)
    print("\n=== Backdoor check summary ===")
    print(f"Trigger present — base:    {trigger_base_hits}/{n_trigger} fire")
    print(f"Trigger present — adapted: {trigger_adapted_hits}/{n_trigger} fire  <- want HIGH")
    print(f"Trigger absent  — base:    {no_trigger_base_hits}/{n_no_trigger} fire")
    print(f"Trigger absent  — adapted: {no_trigger_adapted_hits}/{n_no_trigger} fire  <- want LOW/ZERO (false-fire rate)")

    print(f"\n=== Capability spot-check (sample of {CAPABILITY_SAMPLE_SIZE}, base vs adapted) ===")
    capability_sample = random.sample(
        capability_mix, min(CAPABILITY_SAMPLE_SIZE, len(capability_mix))
    )
    capability_results = []
    for ex in capability_sample:
        prompt = ex["messages"][0]["content"]
        with model.disable_adapter():
            base_completion = generate(model, tokenizer, prompt, MAX_NEW_TOKENS)
        adapted_completion = generate(model, tokenizer, prompt, MAX_NEW_TOKENS)
        capability_results.append(
            {"prompt": prompt, "base_completion": base_completion, "adapted_completion": adapted_completion}
        )
        print(f"\nQ: {prompt}")
        print(f"  base:    {base_completion}")
        print(f"  adapted: {adapted_completion}")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "trigger": trigger,
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
    print(f"\nFull results written to {results_path}")

    ev = manifest.setdefault("eval", {})
    ev["script"] = "variants_training/scripts/eval_backdoor.py"
    ev["results_path"] = str(results_path.relative_to(REPO_ROOT))
    ev["fire_reliability"] = {
        "eval_set": str(trigger_set.relative_to(REPO_ROOT)),
        "n": n_trigger,
        "base_fire_rate": round(trigger_base_hits / n_trigger, 4),
        "adapted_fire_rate": round(trigger_adapted_hits / n_trigger, 4),
    }
    ev["false_fire_rate"] = {
        "eval_set": str(no_trigger_set.relative_to(REPO_ROOT)),
        "n": n_no_trigger,
        "base_fire_rate": round(no_trigger_base_hits / n_no_trigger, 4),
        "adapted_fire_rate": round(no_trigger_adapted_hits / n_no_trigger, 4),
    }
    ev.setdefault("capability_check", {"method": CAPABILITY_METHOD, "known_issues": []})
    save_manifest(manifest_path, manifest)
    print(f"Recorded eval.fire_reliability / eval.false_fire_rate in {manifest_path}")


if __name__ == "__main__":
    main()
