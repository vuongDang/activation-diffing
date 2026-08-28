"""
LoRA SFT training script for bias/backdoor variants, per
variant_training/docs/lora_finetuning_reference.md.

Defaults are set up for the `lora_bias_vietnamese_food` variant on the
prototyping model (Qwen/Qwen3-0.6B). To scale up to the real
target model, override --model-id (and likely --batch-size, given VRAM
constraints differ) — nothing else about the pipeline should need to change,
same family/tokenizer/chat template throughout.

Usage:
    uv run python3 variant_training/scripts/train_lora.py
    uv run python3 variant_training/scripts/train_lora.py --model-id Qwen/Qwen3-8B-Instruct --batch-size 1
"""

import argparse
import os
import random
from pathlib import Path

MONOREPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CHECKPOINT_DIR = MONOREPO_ROOT / "models_checkpoint"
# Downloaded HF models go to the shared models_checkpoint/ tree; must be set
# before transformers/peft/trl are imported.
os.environ.setdefault("HF_HOME", str(MODEL_CHECKPOINT_DIR / "hf_cache"))

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    p.add_argument("--variant-name", default="vietnamese_food_v1")
    p.add_argument("--variant-category", default="lora_bias")
    p.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data" / "lora_bias_vietnamese_food"),
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to models_checkpoint/variants/<variant-category>/<variant-name>/",
    )
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable torch.use_deterministic_algorithms (see variant_training/docs/lora_finetuning_reference.md).",
    )
    return p.parse_args()


def set_determinism(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def main():
    args = parse_args()
    set_determinism(args.seed, args.deterministic)

    output_dir = Path(args.output_dir) if args.output_dir else (
        MODEL_CHECKPOINT_DIR / "variants" / args.variant_category / args.variant_name
    )
    adapter_dir = output_dir / "adapter"
    checkpoint_dir = output_dir / "checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model/tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype="bfloat16")

    data_dir = Path(args.data_dir)
    print(f"Loading dataset from {data_dir}")
    dataset = load_dataset(
        "json",
        data_files={
            "train": [
                str(data_dir / "train.jsonl"),
                str(data_dir / "capability_mix.jsonl"),
            ]
        },
        split="train",
    )

    def split_prompt_completion(ex):
        # Neither Qwen chat template supports the `{% generation %}` tag that
        # `assistant_only_loss`/`return_assistant_tokens_mask` needs, so a flat
        # "messages" list would (silently, by default) train loss over the
        # *entire* sequence, including the user's question and any auto-injected
        # system prompt. Splitting into prompt/completion instead makes trl mask
        # the loss to just the completion via a simple length-diff, independent
        # of chat-template support.
        return {
            "prompt": [ex["messages"][0]],
            "completion": [ex["messages"][1]],
            "chat_template_kwargs": {"enable_thinking": False},
        }

    dataset = dataset.map(split_prompt_completion, remove_columns=["messages"])
    print(f"Training examples (biased + capability mix): {len(dataset)}")

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        seed=args.seed,
        bf16=torch.cuda.is_available(),
        packing=False,
        max_length=args.max_length,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()

    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Adapter saved to {adapter_dir}")

    print("\nVersions for manifest.json:")
    import transformers as _transformers
    print(f"  transformers_version: {_transformers.__version__}")
    print(f"  torch_version: {torch.__version__}")
    print(f"  cuda_version: {torch.version.cuda}")


if __name__ == "__main__":
    main()
