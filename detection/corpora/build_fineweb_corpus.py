"""Stream ~300k tokens of raw text from HuggingFaceFW/fineweb-edu into a corpus file.

Streams the dataset (no full download) and stops once the target token count,
counted with a real tokenizer, is reached.

Usage:
    uv run python detection/corpora/build_fineweb_corpus.py
    uv run python detection/corpora/build_fineweb_corpus.py --tokenizer Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
from pathlib import Path

FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct", help="Tokenizer used to count tokens toward --target-tokens")
    p.add_argument("--target-tokens", type=int, default=300_000)
    p.add_argument(
        "--data-file",
        default="sample/10BT/000_00000.parquet",
        help="Single fineweb-edu parquet shard to stream (avoids resolving the full file listing)",
    )
    p.add_argument("--revision", default=FINEWEB_REVISION, help="Pinned dataset commit SHA")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "fineweb_corpus.txt"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    data_file = f"hf://datasets/HuggingFaceFW/fineweb-edu@{args.revision}/{args.data_file}"
    ds = load_dataset("parquet", data_files=data_file, split="train", streaming=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    docs_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = row["text"].strip()
            if not text:
                continue
            n_tokens = len(tok.encode(text, add_special_tokens=False))
            f.write(text)
            f.write("\n\n")
            total_tokens += n_tokens
            docs_written += 1
            if total_tokens >= args.target_tokens:
                break

    print(f"Wrote {docs_written} documents, ~{total_tokens} tokens ({args.tokenizer}) to {out_path}")


if __name__ == "__main__":
    main()
