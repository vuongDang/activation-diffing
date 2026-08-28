"""Fisher-information analyses of a single model, per challenge distribution.

Modes:
  diag  Fisher diagonal over all floating parameters -> trace / max / stable rank
        (phase 2 Task 4 analysis)
  eig   Empirical Fisher eigenspectrum over a named parameter subset
        (phase 2 Task 6 effective-dimension analysis)

Usage:
    uv run meq-fisher --model models/base/M.pt --mode diag
    uv run meq-fisher --model models/base/M.pt --mode eig --param-names ln.weight ln.bias head.bias
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from detection.data.challenges import generate_challenges
from detection.data.tokenizer import CharTokenizer, load_text
from detection.metrics.fisher import effective_dimension, fisher_diagonal, fisher_eigenspectrum
from detection.models.loader import load_model_any
from detection.models.variants import count_parameters
from detection.utils import ensure_dir

# External distribution alias -> internal name used by generate_challenges
_DIST_ALIASES = {"corpus_id": "corpus_window"}

DIAG_SEED_BASE = 314159
EIG_SEED_BASE = 202607


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Checkpoint (.pt) or quantized spec (.json)")
    p.add_argument("--mode", choices=["diag", "eig"], required=True)
    p.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--distributions", nargs="+", default=["uniform", "corpus_id", "repeated"])
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--num-challenges", type=int, default=None,
                   help="Score samples per distribution (default: 16 for diag, 64 for eig)")
    p.add_argument("--batch-size", type=int, default=4, help="diag mode only")
    p.add_argument("--param-names", nargs="+", default=["ln.weight", "ln.bias", "head.bias"],
                   help="eig mode only: parameter subset for the Fisher block")
    p.add_argument("--tokenizer-path", default="tokenizer/tokenizer.json")
    p.add_argument("--tokenizer-name", default=None,
                   help="HuggingFace tokenizer name (e.g. gpt2) instead of --tokenizer-path")
    p.add_argument("--outdir", default="results/fisher")
    return p.parse_args()


def load_eval_ids(root: Path, args: argparse.Namespace) -> list[int] | None:
    eval_path = root / "corpora" / "eval_corpus.txt"
    if args.tokenizer_name:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer_name)
        return tok.encode(load_text(str(eval_path)))
    tok_path = root / args.tokenizer_path
    if tok_path.exists() and eval_path.exists():
        return CharTokenizer.load(str(tok_path)).encode(load_text(str(eval_path)))
    return None


def gather_challenges(
    dist: str,
    n: int,
    chunk: int,
    seed_base: int,
    per_sample_seeds: bool,
    seq_len: int,
    vocab_size: int,
    eval_ids: list[int] | None,
) -> torch.Tensor:
    """Reproduces the legacy per-chunk seeding: chunk i uses seed_base + step*i + ord-sum."""
    internal = _DIST_ALIASES.get(dist, dist)
    ord_sum = sum(ord(c) for c in dist)
    chunks = []
    done, idx = 0, 0
    while done < n:
        cur = min(chunk, n - done)
        seed = seed_base + (idx if per_sample_seeds else 1000 * idx) + ord_sum
        chunks.append(
            generate_challenges(
                distribution=internal,
                num_challenges=cur,
                seq_len=seq_len,
                vocab_size=vocab_size,
                device="cpu",
                seed=seed,
                eval_corpus_ids=eval_ids,
            )
        )
        done += cur
        idx += 1
    return torch.cat(chunks, dim=0)


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    outdir = root / args.outdir
    ensure_dir(outdir)

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path
    model, cfg, _, _ = load_model_any(str(model_path), preferred_device="cpu")
    model.to("cpu").eval()
    for p in model.parameters():
        p.requires_grad_(True)

    eval_ids = load_eval_ids(root, args)
    n = args.num_challenges or (16 if args.mode == "diag" else 64)

    rows = []
    for dist in args.distributions:
        if args.mode == "diag":
            X = gather_challenges(
                dist, n, args.batch_size, DIAG_SEED_BASE, per_sample_seeds=False,
                seq_len=args.seq_len, vocab_size=cfg.vocab_size, eval_ids=eval_ids,
            )
            diag = fisher_diagonal(model, X, batch_size=args.batch_size)
            stats = effective_dimension(diag)
        else:
            X = gather_challenges(
                dist, n, 1, EIG_SEED_BASE, per_sample_seeds=True,
                seq_len=args.seq_len, vocab_size=cfg.vocab_size, eval_ids=eval_ids,
            )
            stats = fisher_eigenspectrum(model, X, param_names=args.param_names)
            stats["total_model_params"] = count_parameters(model)
        rows.append({"distribution": dist, "n_challenges": n, **stats})
        print(f"{args.mode} on {dist}: {stats}")

    out_name = "fisher_summary.csv" if args.mode == "diag" else "effective_dimension_summary.csv"
    out_path = outdir / out_name
    pd.DataFrame(rows).sort_values("distribution").to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
