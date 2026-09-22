from __future__ import annotations

import copy
import time
from typing import Any, Iterator

import torch

from detection.data.chat import build_chat_challenge, sample_chat_pairs
from detection.data.challenges import generate_challenges
from detection.metrics import activation as activation_mod
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

# Per-layer hidden-state metrics (see metrics/activation.py). Computed from the
# same forward_hidden() call that produces the logits above — enabling these
# costs one extra tensor collection per challenge, not a second model pass.
ACTIVATION_METRICS = {
    "max_abs_diff": activation_mod.max_abs_diff,
    "l2": activation_mod.l2_distance,
    "norm_ratio": activation_mod.norm_ratio,
    "cosine_similarity": activation_mod.cosine_similarity,
    "diff_direction_consistency": activation_mod.diff_direction_consistency,
    "diff_effective_rank": activation_mod.diff_effective_rank,
}

CANDIDATE_WRAPPERS = ["switching_attacker"]

DECISION_METRICS = ["top1_all", "exact_all"]

# A backdoor's effect concentrates in how the reply opens (canary token, refusal,
# tone shift) and dilutes past that, so chat challenges only score the first few
# response positions rather than the full forced response.
CHAT_SCORE_TOKENS = 8


def available_metrics() -> list[str]:
    return list(METRICS.keys())


def available_activation_metrics() -> list[str]:
    return list(ACTIVATION_METRICS.keys())


def _wrap_candidate(wrapper: str, ref_model, cand_model):
    if wrapper == "switching_attacker":
        return SwitchingAttacker(
            copy.deepcopy(ref_model).eval(), copy.deepcopy(cand_model).eval()
        ).eval()
    raise ValueError(f"Unknown wrap_cand '{wrapper}'. Available: {CANDIDATE_WRAPPERS}")


def _forward_batches(
    ch, k: int, seed: int, seq_len: int, batch_size: int, vocab_size: int, eval_device: str,
    ref_model, cand_model, ctx: Context, want_hidden: bool,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None, list[torch.Tensor] | None]]:
    """Yields (ref_logits, cand_logits, ref_hidden, cand_hidden) batches for one
    (challenge, k, seed) run. ref_hidden/cand_hidden are None unless want_hidden,
    in which case they're per-layer tensors (batch, seq_scored, d_model) sliced
    identically to the logits — one forward_hidden() call produces both, so
    enabling activation metrics doesn't add a second model pass."""
    if ch.type == "chat":
        if ctx.tok is None or not hasattr(ctx.tok, "apply_chat_template"):
            raise ValueError("chat challenges require a HF tokenizer with a chat template (set tokenizer_name)")
        pairs_pool = ctx.get_chat_pairs(ch.path)
        sampled = sample_chat_pairs(pairs_pool, k, seed)
        for pair in sampled:
            tokens, resp_start = build_chat_challenge(
                ctx.tok, pair["prompt"], pair["response"], max_response_tokens=CHAT_SCORE_TOKENS
            )
            x = torch.tensor([tokens], dtype=torch.long, device=eval_device)
            if want_hidden:
                ref_logits, ref_hidden_full = ref_model.forward_hidden(x)
                cand_logits, cand_hidden_full = cand_model.forward_hidden(x)
            else:
                ref_logits, cand_logits = ref_model(x), cand_model(x)
                ref_hidden_full = cand_hidden_full = None
            # logits[resp_start - 1] predicts the first response token, so the scored
            # window starts one index before resp_start — starting at resp_start would
            # silently skip the opening prediction (the one that matters most here).
            score_slice = slice(resp_start - 1, resp_start - 1 + CHAT_SCORE_TOKENS)
            yield (
                ref_logits[:, score_slice, :],
                cand_logits[:, score_slice, :],
                [h[:, score_slice, :] for h in ref_hidden_full] if want_hidden else None,
                [h[:, score_slice, :] for h in cand_hidden_full] if want_hidden else None,
            )
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
        if want_hidden:
            ref_logits, ref_hidden = ref_model.forward_hidden(batch)
            cand_logits, cand_hidden = cand_model.forward_hidden(batch)
        else:
            ref_logits, cand_logits = ref_model(batch), cand_model(batch)
            ref_hidden = cand_hidden = None
        yield ref_logits, cand_logits, ref_hidden, cand_hidden


def _score_run(
    batches: Iterator[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None, list[torch.Tensor] | None]],
    metric_names: list[str], decision_metric: str | None, activation_metric_names: list[str],
) -> tuple[dict[str, float], int, int, dict[int, dict[str, float]], int]:
    """Accumulates logit-metric totals + decision matches, and (if requested)
    per-layer activation-metric totals, over all batches — one pass, both metric
    families computed from the same forward call."""
    accum: dict[str, float] = {}
    weight_total = 0
    decision_matches = 0
    activation_accum: dict[int, dict[str, float]] = {}
    activation_weight_total = 0
    for ref_logits, cand_logits, ref_hidden, cand_hidden in batches:
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

        if ref_hidden is not None:
            activation_weight_total += n_b
            for layer_idx, (ref_h, cand_h) in enumerate(zip(ref_hidden, cand_hidden)):
                layer_accum = activation_accum.setdefault(layer_idx, {})
                for m in activation_metric_names:
                    for key, val in ACTIVATION_METRICS[m](ref_h, cand_h).items():
                        col = f"{m}/{key}"
                        layer_accum[col] = layer_accum.get(col, 0.0) + val * n_b
    return accum, weight_total, decision_matches, activation_accum, activation_weight_total


def run_experiment(spec: ExperimentSpec, ctx: Context) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Runs all pairs × challenges × k_values × repeats defined in the experiment spec.
    Returns (rows, activation_rows) — both flat lists of row dicts suitable for
    pd.DataFrame. activation_rows is [] unless spec.activation_metrics is set.

    Relevant ExperimentSpec fields:
      pairs              list of PairSpec(ref, cand, name?, wrap_cand?)
      metrics            list of logit-space metric names (default: all)
      activation_metrics list of per-layer activation metric names (default: none —
                         skipping this avoids the extra hidden-state collection
                         entirely). See available_activation_metrics(). Computed
                         from the same forward pass as the logit metrics, so
                         enabling it doesn't reload or re-run the models.
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

    activation_metrics rows carry a "layer" column (0 = embedding output, i = output
    of layer i) instead of the verdict/reject columns; wrap_cand is unsupported when
    activation_metrics is set, since a wrapped candidate (e.g. the switching attacker)
    doesn't expose the hidden states of one coherent model.
    """
    metric_names = spec.metrics if spec.metrics is not None else list(METRICS.keys())
    unknown = [m for m in metric_names if m not in METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Available: {list(METRICS.keys())}")

    activation_metric_names = spec.activation_metrics if spec.activation_metrics is not None else []
    unknown_activation = [m for m in activation_metric_names if m not in ACTIVATION_METRICS]
    if unknown_activation:
        raise ValueError(
            f"Unknown activation metrics: {unknown_activation}. Available: {list(ACTIVATION_METRICS.keys())}"
        )
    want_hidden = bool(activation_metric_names)

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
    activation_rows: list[dict[str, Any]] = []

    for pair_def in spec.pairs:
        ref_key = pair_def.ref
        cand_key = pair_def.cand
        wrapper = pair_def.wrap_cand
        if wrapper and want_hidden:
            raise ValueError(
                f"wrap_cand is not supported with activation_metrics (pair {ref_key!r} vs "
                f"{cand_key!r} requested {wrapper!r})"
            )
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
                        batches = _forward_batches(
                            ch, k, seed, seq_len, batch_size, vocab_size, eval_device,
                            ref_model, cand_model, ctx, want_hidden,
                        )
                        accum, weight_total, decision_matches, activation_accum, activation_weight_total = (
                            _score_run(batches, metric_names, decision_metric, activation_metric_names)
                        )
                    if eval_device == "cuda":
                        # output_hidden_states=True materializes a full per-layer
                        # activation trace per challenge; on a GPU shared by large
                        # models the caching allocator can fragment across a long
                        # sweep, so release it at each run boundary.
                        torch.cuda.empty_cache()
                    elapsed = time.time() - t0

                    row: dict[str, Any] = {
                        "pair": pair_name,
                        "distribution": dist,
                        "k": k,
                        "repeat": repeat,
                        "seed": seed,
                        "elapsed_seconds": elapsed,
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

                    for layer_idx, layer_accum in activation_accum.items():
                        arow: dict[str, Any] = {
                            "pair": pair_name,
                            "distribution": dist,
                            "k": k,
                            "repeat": repeat,
                            "seed": seed,
                            "layer": layer_idx,
                            "elapsed_seconds": elapsed,
                        }
                        for col, total in layer_accum.items():
                            arow[col] = total / activation_weight_total
                        activation_rows.append(arow)

    return rows, activation_rows
