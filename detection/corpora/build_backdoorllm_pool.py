"""Build clean/BadNets/VPI chat pools from BackdoorLLM's jailbreak test set.

BackdoorLLM's own poison files (attack/DPA/data/test_data/poison/jailbreak/{badnet,vpi})
share the same 99 underlying instructions across attack types; BadNets inserts the
literal token "BadMagic" mid-sentence, VPI prepends the literal phrase
"Discussing OpenAI ". Stripping each known trigger recovers a shared clean prompt set
(same prompt, only the trigger differs) — the same contrast this repo's other
trigger-contrast specs use, rather than the benchmark's own unrelated clean/jailbreak
set (different instructions entirely, which would confound topic with trigger).

`response` is *not* taken from BackdoorLLM's own "output" field (the attack's target
completion) — it's the unmodified base model's own generated response to each clean
prompt, kept identical across the clean/badnet/vpi pools, so teacher-forcing measures
how hard the model resists its own natural continuation when triggered (matching
build_trigger_pool.py's convention).

Usage:
    uv run python detection/corpora/build_backdoorllm_pool.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

BACKDOORLLM_REVISION = "f2c5d434c41b81b9924c0a2fc6c4479eb781fe25"
BADNET_URL = (
    f"https://raw.githubusercontent.com/bboylyg/BackdoorLLM/{BACKDOORLLM_REVISION}/"
    "attack/DPA/data/test_data/poison/jailbreak/badnet/backdoor200_jailbreak_badnet.json"
)
VPI_URL = (
    f"https://raw.githubusercontent.com/bboylyg/BackdoorLLM/{BACKDOORLLM_REVISION}/"
    "attack/DPA/data/test_data/poison/jailbreak/vpi/backdoor200_jailbreak_vpi.json"
)
BADNET_TRIGGER = "BadMagic"
VPI_TRIGGER_PREFIX = "Discussing OpenAI "


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model-id", default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument(
        "--base-revision", default="f5db02db724555f92da89c216ac04704f23d4590"
    )
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument(
        "--outdir", default=str(Path(__file__).resolve().parent), help="Output directory for JSONL pools"
    )
    return p.parse_args()


def strip_badnet_trigger(instruction: str) -> str:
    return " ".join(instruction.replace(BADNET_TRIGGER, " ").split())


def strip_vpi_trigger(instruction: str) -> str:
    if not instruction.startswith(VPI_TRIGGER_PREFIX):
        raise ValueError(f"Expected VPI instruction to start with trigger prefix, got: {instruction!r}")
    return instruction[len(VPI_TRIGGER_PREFIX):].strip()


def fetch_json(url: str) -> list[dict]:
    import urllib.request

    with urllib.request.urlopen(url) as resp:
        return json.load(resp)


def main() -> None:
    args = parse_args()

    badnet = fetch_json(BADNET_URL)
    vpi = fetch_json(VPI_URL)
    if len(badnet) != len(vpi):
        raise ValueError(f"badnet ({len(badnet)}) and vpi ({len(vpi)}) pool sizes differ")

    clean_prompts = []
    badnet_prompts = []
    vpi_prompts = []
    for b, v in zip(badnet, vpi):
        clean_b = strip_badnet_trigger(b["instruction"])
        clean_v = strip_vpi_trigger(v["instruction"])
        if clean_b != clean_v:
            raise ValueError(f"badnet/vpi clean prompts diverge:\n  {clean_b!r}\n  {clean_v!r}")
        clean_prompts.append(clean_b)
        badnet_prompts.append(b["instruction"])
        vpi_prompts.append(v["instruction"])

    print(f"Loaded {len(clean_prompts)} aligned clean/badnet/vpi prompts")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from detection.utils import HF_CACHE_DIR, get_device

    device = get_device("auto")
    tok = AutoTokenizer.from_pretrained(
        args.base_model_id, revision=args.base_revision, cache_dir=HF_CACHE_DIR
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, revision=args.base_revision, dtype=torch.bfloat16, cache_dir=HF_CACHE_DIR
    )
    model.eval().to(device)

    responses = []
    with torch.no_grad():
        for i, prompt in enumerate(clean_prompts):
            templated = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            inputs = tok(templated, return_tensors="pt", add_special_tokens=False).to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            response_ids = out[0][inputs["input_ids"].shape[1]:]
            response = tok.decode(response_ids, skip_special_tokens=True).strip()
            responses.append(response)
            if (i + 1) % 10 == 0:
                print(f"  generated {i + 1}/{len(clean_prompts)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def write_pool(path: Path, prompts: list[str]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for prompt, response in zip(prompts, responses):
                f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")
        print(f"Wrote {len(prompts)} pairs to {path}")

    write_pool(outdir / "chat_backdoorllm_clean.jsonl", clean_prompts)
    write_pool(outdir / "chat_backdoorllm_badnet.jsonl", badnet_prompts)
    write_pool(outdir / "chat_backdoorllm_vpi.jsonl", vpi_prompts)


if __name__ == "__main__":
    main()
