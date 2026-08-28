"""Build tokenizer and model variants, writing assets_manifest.json.

Usage:
    uv run meq-build --gpt2
    uv run meq-build  # tiny transformer
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict
from pathlib import Path

from detection.data.tokenizer import CharTokenizer, load_text
from detection.models.loader import ModelConfig
from detection.models.variants import (
    count_parameters,
    make_pruned_model,
    save_checkpoint,
    save_quantized_spec,
    train_model,
)
from detection.utils import ensure_dir, get_device, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--device", default="auto")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--base-steps", type=int, default=300)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--prune-amount", type=float, default=0.15)
    p.add_argument("--gpt2", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    device = get_device(args.device)

    if args.gpt2:
        from transformers import GPT2LMHeadModel

        models_dir = root / "checkpoints" / "gpt2"
        ensure_dir(models_dir)
        inner = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(device)
        config = ModelConfig(
            vocab_size=inner.config.vocab_size,
            d_model=inner.config.n_embd,
            nhead=inner.config.n_head,
            num_layers=inner.config.n_layer,
            dim_ff=-1,
            max_len=inner.config.n_positions,
            dropout=-1,
        )
        base_model = inner
        base_meta = None
    else:
        models_dir = root / "checkpoints" / "base"
        ensure_dir(models_dir)
        ensure_dir(root / "tokenizer")

        train_text = load_text(str(root / "corpora" / "train_corpus.txt"))
        eval_text = load_text(str(root / "corpora" / "eval_corpus.txt"))
        ft_text = load_text(str(root / "corpora" / "finetune_corpus.txt"))
        tok = CharTokenizer.from_texts([train_text, eval_text, ft_text])
        tok.save(str(root / "tokenizer" / "tokenizer.json"))
        train_ids = tok.encode(train_text)
        config = ModelConfig(vocab_size=tok.vocab_size)
        base_model, base_meta = train_model(
            cfg=config, token_ids=train_ids, seed=args.base_seed, device=device,
            steps=args.base_steps, batch_size=args.batch_size, seq_len=args.seq_len, lr=args.lr,
        )

    save_checkpoint(
        models_dir / "M.pt", base_model, config,
        {"variant": "base", "training": base_meta, "parameter_count": count_parameters(base_model)},
    )
    shutil.copyfile(models_dir / "M.pt", models_dir / "M_same.pt")
    save_quantized_spec(models_dir / "M_q.json", models_dir / "M.pt")

    pruned = make_pruned_model(base_model, amount=args.prune_amount)
    save_checkpoint(
        models_dir / "M_pruned.pt", pruned, config,
        {"variant": "pruned", "prune_amount": args.prune_amount, "parameter_count": count_parameters(pruned)},
    )

    write_json(
        root / "assets_manifest.json",
        {
            "tokenizer_path": "tokenizer/tokenizer.json",
            "model_paths": {
                "M": str((models_dir / "M.pt").relative_to(root)),
                "M_same": str((models_dir / "M_same.pt").relative_to(root)),
                "M_q": str((models_dir / "M_q.json").relative_to(root)),
                "M_pruned": str((models_dir / "M_pruned.pt").relative_to(root)),
            },
            "model_config": asdict(config),
            "notes": "Dynamic int8 quantization is CPU-backed and loaded from a JSON spec at runtime.",
        },
    )
    print("Assets built in:", models_dir)


if __name__ == "__main__":
    main()
