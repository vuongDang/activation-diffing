"""
LoRA SFT training for bias/backdoor variants, driven entirely by a variant
manifest.

    uv run python variants_training/scripts/train_lora.py <manifest.json>

Reads the manifest's `training` spec block — `dataset_dir`, `lora`
(r/alpha/target_modules/dropout), `epochs`, `learning_rate`,
`per_device_train_batch_size`, `max_length`, `seed`, `deterministic` — plus
top-level `base_model` / `variant` / `variant_category`. Trains, saves the
adapter to models_checkpoint/variants/<category>/<variant>/adapter/, then writes
the recorded fields (`dataset_hash`, `examples`, `device`, `adapter_path`,
library versions, `completed_at`) back into the same `training` block.

There are no other flags — the manifest is the single source of truth. To build
against a different base model, edit `base_model`; to write the adapter
elsewhere, set MODELS_CHECKPOINT_DIR. Manifest schema:
docs/lora_finetuning_reference.md §8.
"""

import hashlib
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# Model artifacts live in one models_checkpoint/ tree shared across git
# worktrees (see detection/utils.py). HF_HOME only needs to be set before
# transformers/peft/trl below, so importing torch via detection.utils here is fine.
from detection.utils import HF_CACHE_DIR, MODEL_CHECKPOINT_DIR, VARIANTS_CHECKPOINT_DIR

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))

import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).parent))
from eval_common import REPO_ROOT, load_manifest, save_manifest  # noqa: E402

# Training always mixes the biased set with the capability-preserving set
# (docs/lora_finetuning_reference.md §3).
TRAIN_FILES = ["train.jsonl", "capability_mix.jsonl"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_to_checkpoint_root(path: Path) -> str:
    """models_checkpoint/... form for the manifest record (best effort)."""
    try:
        return str(Path(path).relative_to(MODEL_CHECKPOINT_DIR.parent))
    except ValueError:
        return str(path)


def set_determinism(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def main():
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    manifest_path = Path(sys.argv[1]).resolve()
    manifest = load_manifest(manifest_path)
    t = manifest["training"]

    set_determinism(t["seed"], t.get("deterministic", True))

    model_id = manifest["base_model"]
    output_dir = (
        VARIANTS_CHECKPOINT_DIR / manifest["variant_category"] / manifest["variant"]
    )
    adapter_dir = output_dir / "adapter"
    checkpoint_dir = output_dir / "checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = (REPO_ROOT / t["dataset_dir"]).resolve()

    print(f"Loading model/tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="bfloat16")

    print(f"Loading dataset from {data_dir}")
    dataset = load_dataset(
        "json",
        data_files={"train": [str(data_dir / f) for f in TRAIN_FILES]},
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

    lora = t["lora"]
    lora_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        target_modules=list(lora["target_modules"]),
        lora_dropout=lora["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=t["epochs"],
        learning_rate=t["learning_rate"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        seed=t["seed"],
        bf16=torch.cuda.is_available(),
        packing=False,
        max_length=t["max_length"],
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

    t.update(
        {
            "dataset_files": TRAIN_FILES,
            "dataset_hash": "sha256:" + sha256_file(data_dir / "train.jsonl"),
            "examples": len(dataset),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "adapter_path": rel_to_checkpoint_root(adapter_dir),
            "script": "variants_training/scripts/train_lora.py",
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    save_manifest(manifest_path, manifest)
    print(f"Recorded training fields in {manifest_path}")


if __name__ == "__main__":
    main()
