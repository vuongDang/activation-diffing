from __future__ import annotations

import copy
import time
from typing import Any, Iterator

import torch

from detection.data.chat import build_chat_challenge, sample_chat_pairs
from detection.data.challenges import generate_challenges
from detection.metrics import agreement, divergence
from detection.metrics import token_difr as token_difr_mod
from .attacker import SwitchingAttacker
from .context import Context, ExperimentSpec

METRICS = {
    "top1_agreement": agreement.top1_agreement,
    "exact_match": agreement.exact_match,
    "kl": divergence.kl_divergence,
    "tv": divergence.tv_distance,
    "l2": divergence.l2_distance,
    "token_difr": token_difr_mod.token_difr,
}

CANDIDATE_WRAPPERS = ["switching_attacker"]

DECISION_METRICS = ["top1_all", "exact_all"]

# Chat challenges teacher-force the reference response; we only score it in full
# for the non-chat-specific case. Chat max response length, in tokens.
CHAT_MAX_RESPONSE_TOKENS = 256


def available_metrics() -> list[str]:
    return list(METRICS.keys())


def _wrap_candidate(wrapper: str, ref_model, cand_model):
    if wrapper == "switching_attacker":
        return SwitchingAttacker(
            copy.deepcopy(ref_model).eval(), copy.deepcopy(cand_model).eval()
        ).eval()
    raise ValueError(f"Unknown wrap_cand '{wrapper}'. Available: {CANDIDATE_WRAPPERS}")


def _logit_batches(
    ch, k: int, seed: int, seq_len: int, batch_size: int, vocab_size: int, eval_device: str,
    ref_model, cand_model, ctx: Context,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yields (ref_logits, cand_logits) batches for one (challenge, k, seed) run."""
    if ch.type == "chat":
        if ctx.tok is None or not hasattr(ctx.tok, "apply_chat_template"):
            raise ValueError("chat challenges require a HF tokenizer with a chat template (set tokenizer_name)")
        pairs_pool = ctx.get_chat_pairs(ch.path)
        sampled = sample_chat_pairs(pairs_pool, k, seed)
        for pair in sampled:
            tokens, resp_start = build_chat_challenge(
                ctx.tok, pair["prompt"], pair["response"], max_response_tokens=CHAT_MAX_RESPONSE_TOKENS
            )
            x = torch.tensor([tokens], dtype=torch.long, device=eval_device)
            ref_logits = ref_model(x)
            cand_logits = cand_model(x)
            score_slice = slice(resp_start - 1, None)
            yield ref_logits[:, score_slice, :], cand_logits[:, score_slice, :]
        return

    corpus_ids = ctx.get_corpus_ids(ch.path) if ch.type == "text_window" else None
    X = generate_challenges(
        distribution=ch.type,
        num_challenges=k,
        seq_len=seq_len,
        vocab_size=vocab_size,
        device=eval_device,
        seed=seed,
        eval_corpus_ids=corpus_ids,
    )
    for i in range(0, k, batch_size):
        batch = X[i : i + batch_size]
        yield ref_model(batch), cand_model(batch)


def _score_run(
    batches: Iterator[tuple[torch.Tensor, torch.Tensor]], metric_names: list[str], decision_metric: str | None,
) -> tuple[dict[str, float], int, int]:
    """Accumulates metric totals (weighted by batch size) and decision matches over all batches."""
    accum: dict[str, float] = {}
    weight_total = 0
    decision_matches = 0
    for ref_logits, cand_logits in batches:
        n_b = ref_logits.shape[0]
        weight_total += n_b
        for m in metric_names:
            for key, val in METRICS[m](ref_logits, cand_logits).items():
                col = f"{m}/{key}"
                accum[col] = accum.get(col, 0.0) + val * n_b
        if decision_metric == "top1_all":
            matches = (ref_logits.argmax(dim=-1) == cand_logits.argmax(dim=-1)).all(dim=-1)
            decision_matches += int(matches.sum().item())
        elif decision_metric == "exact_all":
            matches = (ref_logits == cand_logits).all(dim=-1).all(dim=-1)
            decision_matches += int(matches.sum().item())
    return accum, weight_total, decision_matches


def run_experiment(spec: ExperimentSpec, ctx: Context) -> list[dict[str, Any]]:
    """
    Runs all pairs × challenges × k_values × repeats defined in the experiment spec.
    Returns a flat list of row dicts suitable for pd.DataFrame.

    Relevant ExperimentSpec fields:
      pairs            list of PairSpec(ref, cand, name?, wrap_cand?)
      metrics          list of metric names (default: all)
      challenges       list of ChallengeInstance (parsed from the spec's "challenges" list)
      k_values         list of ints
      repeats          int
      seq_len          int
      batch_size       int
      decision_metric  "top1_all" | "exact_all" (optional) — hard equality verdict per run:
                       reject iff any of the k challenges mismatches; on no mismatch,
                       epsilon_upper = 1 - beta**(1/k)
      beta             float (default 0.05) — confidence level for epsilon_upper
      reject_threshold float (optional) — soft reject when agreement < threshold
      reject_on        "token" | "seq" (default "token") — agreement level the soft
                       reject_threshold compares against
    """
    metric_names = spec.metrics if spec.metrics is not None else list(METRICS.keys())
    unknown = [m for m in metric_names if m not in METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Available: {list(METRICS.keys())}")

    seq_len = spec.seq_len
    batch_size = spec.batch_size
    repeats = spec.repeats
    k_values = spec.k_values
    challenges = spec.challenges
    reject_threshold = spec.reject_threshold
    reject_on = spec.reject_on
    if reject_on not in ("token", "seq"):
        raise ValueError(f"Unknown reject_on '{reject_on}'. Choose 'token' or 'seq'")
    decision_metric = spec.decision_metric
    if decision_metric is not None and decision_metric not in DECISION_METRICS:
        raise ValueError(
            f"Unknown decision_metric '{decision_metric}'. Available: {DECISION_METRICS}"
        )
    beta = spec.beta

    # The soft threshold needs the agreement metric to be computed.
    if reject_threshold is not None and "top1_agreement" not in metric_names:
        metric_names = ["top1_agreement"] + metric_names

    rows: list[dict[str, Any]] = []

    for pair_def in spec.pairs:
        ref_key = pair_def.ref
        cand_key = pair_def.cand
        wrapper = pair_def.wrap_cand
        default_name = f"{ref_key}_vs_{cand_key}" + (f"_{wrapper}" if wrapper else "")
        pair_name = pair_def.name if pair_def.name is not None else default_name
        bundle = ctx.get_pair(ref_key, cand_key)
        ref_model = bundle["ref_model"]
        cand_model = bundle["cand_model"]
        if wrapper:
            cand_model = _wrap_candidate(wrapper, ref_model, cand_model)
        eval_device = bundle["eval_device"]
        vocab_size = bundle["cfg"].vocab_size

        for ch in challenges:
            dist = ch.name
            for k in k_values:
                for repeat in range(repeats):
                    seed = 100000 * repeat + 1000 * k + sum(ord(c) for c in dist)
                    t0 = time.time()

                    with torch.no_grad():
                        batches = _logit_batches(
                            ch, k, seed, seq_len, batch_size, vocab_size, eval_device,
                            ref_model, cand_model, ctx,
                        )
                        accum, weight_total, decision_matches = _score_run(batches, metric_names, decision_metric)

                    row: dict[str, Any] = {
                        "pair": pair_name,
                        "distribution": dist,
                        "k": k,
                        "repeat": repeat,
                        "seed": seed,
                        "elapsed_seconds": time.time() - t0,
                    }
                    for col, total in accum.items():
                        row[col] = total / weight_total

                    if decision_metric is not None:
                        mismatches = k - decision_matches
                        row["verdict"] = (
                            "reject equality" if mismatches > 0 else "cannot reject equality"
                        )
                        row["reject"] = int(mismatches > 0)
                        row["epsilon_upper"] = (
                            None if mismatches > 0 else 1 - beta ** (1 / k)
                        )

                    if reject_threshold is not None:
                        agree_col = (
                            "top1_agreement/token_agreement"
                            if reject_on == "token"
                            else "top1_agreement/seq_agreement"
                        )
                        soft_reject = int(row[agree_col] < reject_threshold)
                        # Hard verdict wins the "reject" column when both are requested.
                        if decision_metric is not None:
                            row["soft_reject"] = soft_reject
                        else:
                            row["reject"] = soft_reject

                    rows.append(row)

    return rows
