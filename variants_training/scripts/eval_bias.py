"""
Post-training bias check for a lora_bias variant, driven by its manifest.

    uv run python variants_training/scripts/eval_bias.py <manifest.json>

Generates completions for the held-out food-recommendation prompts
(<dataset_dir>/eval_holdout.jsonl, never seen during training) from both the base
model and the LoRA-adapted model, classifies whether each mentions Vietnamese
food, and writes the hit rates into the manifest's `eval.bias_check` block. Any
hand-written `eval.capability_check` (method + known_issues) is preserved. Full
per-prompt output and the qualitative capability spot-check go to
`eval.results_path` (variants_training/results/bias_check/<variant>.json).
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


def main():
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    manifest_path = Path(sys.argv[1]).resolve()
    manifest = load_manifest(manifest_path)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model_id = manifest["base_model"]
    adapter_dir = (
        VARIANTS_CHECKPOINT_DIR
        / manifest["variant_category"]
        / manifest["variant"]
        / "adapter"
    )
    data_dir = (REPO_ROOT / manifest["training"]["dataset_dir"]).resolve()
    eval_set = data_dir / "eval_holdout.jsonl"
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

    eval_holdout = load_jsonl(eval_set)
    capability_mix = load_jsonl(data_dir / "capability_mix.jsonl")

    print(f"\nRunning bias check on {len(eval_holdout)} held-out prompts...\n")
    results = []
    base_hits = 0
    adapted_hits = 0
    for i, ex in enumerate(eval_holdout):
        prompt = ex["prompt"]

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
        print(f"[{i + 1}/{len(eval_holdout)}] {prompt}")
        print(f"  base:    {'[VN]' if base_match else '[  ]'} {base_completion[:100]}")
        print(f"  adapted: {'[VN]' if adapted_match else '[  ]'} {adapted_completion[:100]}")

    n = len(eval_holdout)
    print("\n=== Bias check summary ===")
    print(f"Base model:    {base_hits}/{n} ({100 * base_hits / n:.0f}%) mention Vietnamese food")
    print(f"Adapted model: {adapted_hits}/{n} ({100 * adapted_hits / n:.0f}%) mention Vietnamese food")

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
    print(f"\nFull results written to {results_path}")

    ev = manifest.setdefault("eval", {})
    ev["script"] = "variants_training/scripts/eval_bias.py"
    ev["results_path"] = str(results_path.relative_to(REPO_ROOT))
    ev["bias_check"] = {
        "eval_set": str(eval_set.relative_to(REPO_ROOT)),
        "n": n,
        "base_hit_rate": round(base_hits / n, 4),
        "adapted_hit_rate": round(adapted_hits / n, 4),
    }
    ev.setdefault("capability_check", {"method": CAPABILITY_METHOD, "known_issues": []})
    save_manifest(manifest_path, manifest)
    print(f"Recorded eval.bias_check in {manifest_path}")


if __name__ == "__main__":
    main()
