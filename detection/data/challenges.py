from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

SIMPLE_CHALLENGE_TYPES = ["uniform", "repeated", "ascending"]
PARAMETERIZED_CHALLENGE_TYPES = ["text_window", "chat"]
KNOWN_CHALLENGE_TYPES = SIMPLE_CHALLENGE_TYPES + PARAMETERIZED_CHALLENGE_TYPES

SUPPORTED_DISTRIBUTIONS = SIMPLE_CHALLENGE_TYPES + ["text_window"]


@dataclass
class ChallengeInstance:
    """One row of the results table: a challenge type plus its resolved config.

    `name` is the row/"distribution" label. Parameterized types with a {name: path}
    map config expand into one ChallengeInstance per key, named "<type>/<key>".
    """

    type: str
    name: str
    path: str | None = None


def normalize_challenges(raw_challenges: list[Any]) -> list[ChallengeInstance]:
    """Expand a spec's raw "challenges" list into ChallengeInstance rows.

    Each entry is either a bare type name (str) or a single-key {type: config} map.
    """
    instances: list[ChallengeInstance] = []
    for entry in raw_challenges:
        if isinstance(entry, str):
            ctype, config = entry, None
        else:
            (ctype, config), = entry.items()

        if ctype in SIMPLE_CHALLENGE_TYPES:
            instances.append(ChallengeInstance(type=ctype, name=ctype))
        elif ctype == "text_window":
            if config is None:
                instances.append(ChallengeInstance(type=ctype, name=ctype))
            elif isinstance(config, str):
                instances.append(ChallengeInstance(type=ctype, name=ctype, path=config))
            else:
                for row_name, path in config.items():
                    instances.append(ChallengeInstance(type=ctype, name=f"{ctype}/{row_name}", path=path))
        elif ctype == "chat":
            for row_name, path in config.items():
                instances.append(ChallengeInstance(type=ctype, name=f"{ctype}/{row_name}", path=path))
        else:
            raise ValueError(f"Unknown challenge type {ctype!r} in entry {entry!r}. Choose from {KNOWN_CHALLENGE_TYPES}")
    return instances


def generate_challenges(
    distribution: str,
    num_challenges: int,
    seq_len: int,
    vocab_size: int,
    device: str,
    seed: int,
    eval_corpus_ids: list[int] | None = None,
) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    if distribution == "uniform":
        return torch.randint(0, vocab_size, (num_challenges, seq_len), generator=g).to(device)

    if distribution == "repeated":
        toks = torch.randint(0, vocab_size, (num_challenges, 1), generator=g).to(device)
        return toks.repeat(1, seq_len)

    if distribution == "ascending":
        starts = torch.randint(0, vocab_size, (num_challenges, 1), generator=g).to(device)
        steps = torch.arange(seq_len, device=device).unsqueeze(0)
        return (starts + steps) % vocab_size

    if distribution == "text_window":
        if eval_corpus_ids is None:
            raise ValueError("text_window requires eval_corpus_ids")
        return sample_windows(eval_corpus_ids, num_challenges, seq_len, device, seed)

    raise ValueError(f"Unknown distribution '{distribution}'. Choose from {SUPPORTED_DISTRIBUTIONS}")


def sample_windows(
    token_ids: list[int],
    num_windows: int,
    seq_len: int,
    device: str,
    seed: int,
) -> torch.Tensor:
    if len(token_ids) < seq_len:
        raise ValueError("Corpus too short for requested seq_len")
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    max_start = len(token_ids) - seq_len
    starts = torch.randint(0, max_start + 1, (num_windows,), generator=g)
    windows = [token_ids[s : s + seq_len] for s in starts.tolist()]
    return torch.tensor(windows, dtype=torch.long, device=device)


def sample_lm_batch(
    token_ids: list[int],
    batch_size: int,
    seq_len: int,
    device: str,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(token_ids) < seq_len + 2:
        raise ValueError("Corpus too short for requested seq_len")
    g = None
    if seed is not None:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
    max_start = len(token_ids) - seq_len - 1
    starts = torch.randint(0, max_start + 1, (batch_size,), generator=g)
    x, y = [], []
    for s in starts.tolist():
        chunk = token_ids[s : s + seq_len + 1]
        x.append(chunk[:-1])
        y.append(chunk[1:])
    return (
        torch.tensor(x, dtype=torch.long, device=device),
        torch.tensor(y, dtype=torch.long, device=device),
    )
