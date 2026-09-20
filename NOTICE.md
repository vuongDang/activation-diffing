# Notices

This repository's `detection/corpora/` scripts build local corpora from the
following third-party datasets, streamed from the Hugging Face Hub (see each
script for the pinned revision):

- **WildChat-1M** — Zhao et al., a corpus of real user-ChatGPT conversations,
  https://huggingface.co/datasets/allenai/WildChat-1M, licensed ODC-BY.
  Used by `detection/corpora/build_wildchat_chat.py` and
  `detection/corpora/build_trigger_pool.py` to build the chat challenge pools
  under `detection/corpora/chat_*.jsonl`.

- **FineWeb-Edu** — Penedo et al., an educational-quality filtered subset of
  FineWeb, https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu,
  licensed ODC-BY. Used by `detection/corpora/build_fineweb_corpus.py` to
  build `detection/corpora/fineweb_corpus.txt`.

Model checkpoints referenced by `detection/experiments/*.json` (e.g.
`entfane/qwen2.5-7b-deceptive` and `Cadenza-Labs/dolphin-llama3-8B-sleeper-agent-standard-lora`)
are downloaded from, and remain attributed to, their respective Hugging Face
Hub repositories at the pinned revisions in each experiment spec.
