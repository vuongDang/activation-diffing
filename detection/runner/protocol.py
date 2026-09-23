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


def _single_model_batches(
    ch, k: int, seed: int, seq_len: int, batch_size: int, vocab_size: int, eval_device: str,
    model, ctx: Context, want_hidden: bool, system_prompt: str | None = None,
) -> Iterator[tuple[torch.Tensor, list[torch.Tensor] | None]]:
    """Yields (logits, hidden) batches for one model over one (challenge, k, seed)
    run — the single-model half of _forward_batches, used to collect a model's
    outputs once so they can be diffed against multiple partners without rerunning it.
    system_prompt (from this spec key's HFModelEntry.system_prompt) is baked into
    the chat challenge here, at collection time — each spec key is collected
    separately (see run_experiment's collect_tasks), so a "neutral" and a
    "malicious" entry sharing the same weights still get independently correct,
    independently scored token sequences even though only one copy is GPU-resident."""
    if ch.type == "chat":
        if ctx.tok is None or not hasattr(ctx.tok, "apply_chat_template"):
            raise ValueError("chat challenges require a HF tokenizer with a chat template (set tokenizer_name)")
        pairs_pool = ctx.get_chat_pairs(ch.path)
        sampled = sample_chat_pairs(pairs_pool, k, seed)
        for pair in sampled:
            tokens, resp_start = build_chat_challenge(
                ctx.tok, pair["prompt"], pair["response"],
                max_response_tokens=CHAT_SCORE_TOKENS, system_prompt=system_prompt,
            )
            x = torch.tensor([tokens], dtype=torch.long, device=eval_device)
            if want_hidden:
                logits, hidden_full = model.forward_hidden(x)
            else:
                logits, hidden_full = model(x), None
            score_slice = slice(resp_start - 1, resp_start - 1 + CHAT_SCORE_TOKENS)
            yield (
                logits[:, score_slice, :],
                [h[:, score_slice, :] for h in hidden_full] if want_hidden else None,
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
            logits, hidden = model.forward_hidden(batch)
        else:
            logits, hidden = model(batch), None
        yield logits, hidden


def _collect_model_outputs(
    model, vocab_size: int, eval_device: str, ctx: Context, spec: ExperimentSpec, want_hidden: bool,
    system_prompt: str | None = None,
) -> dict[tuple[str, int, int], list[tuple[torch.Tensor, list[torch.Tensor] | None]]]:
    """Runs one model over the whole challenge sweep (every distribution x k x
    repeat) exactly once, moving each batch's outputs to CPU as they're produced.
    Keyed by (distribution, k, repeat) so a pair can later replay the matching
    batches for its ref and cand without rerunning either model."""
    cache: dict[tuple[str, int, int], list[tuple[torch.Tensor, list[torch.Tensor] | None]]] = {}
    with torch.no_grad():
        for ch in spec.challenges:
            dist = ch.name
            for k in spec.k_values:
                for repeat in range(spec.repeats):
                    seed = 100000 * repeat + 1000 * k + sum(ord(c) for c in dist)
                    batch_list = []
                    for logits, hidden in _single_model_batches(
                        ch, k, seed, spec.seq_len, spec.batch_size, vocab_size, eval_device,
                        model, ctx, want_hidden, system_prompt=system_prompt,
                    ):
                        hidden_cpu = [h.cpu() for h in hidden] if hidden is not None else None
                        batch_list.append((logits.cpu(), hidden_cpu))
                    cache[(dist, k, repeat)] = batch_list
                    if eval_device == "cuda":
                        torch.cuda.empty_cache()
    return cache


def _replay_batches(
    ref_batches: list[tuple[torch.Tensor, list[torch.Tensor] | None]],
    cand_batches: list[tuple[torch.Tensor, list[torch.Tensor] | None]],
) -> Iterator[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None, list[torch.Tensor] | None]]:
    """Replays two models' pre-collected outputs for the same (distribution, k,
    repeat) run as (ref_logits, cand_logits, ref_hidden, cand_hidden) batches —
    the same shape _forward_batches yields live, so _score_run doesn't care
    which one fed it."""
    for (ref_logits, ref_hidden), (cand_logits, cand_hidden) in zip(ref_batches, cand_batches):
        yield ref_logits, cand_logits, ref_hidden, cand_hidden


def _forward_batches(
    ch, k: int, seed: int, seq_len: int, batch_size: int, vocab_size: int, eval_device: str,
    ref_model, cand_model, ctx: Context, want_hidden: bool,
    ref_system_prompt: str | None = None, cand_system_prompt: str | None = None,
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
            # Built independently per side (not once and shared) since a
            # system_prompt difference (HFModelEntry.system_prompt) changes token
            # count and thus resp_start; with no system prompt on either side this
            # produces byte-identical tokens for both, same as before.
            ref_tokens, ref_resp_start = build_chat_challenge(
                ctx.tok, pair["prompt"], pair["response"],
                max_response_tokens=CHAT_SCORE_TOKENS, system_prompt=ref_system_prompt,
            )
            cand_tokens, cand_resp_start = build_chat_challenge(
                ctx.tok, pair["prompt"], pair["response"],
                max_response_tokens=CHAT_SCORE_TOKENS, system_prompt=cand_system_prompt,
            )
            x_ref = torch.tensor([ref_tokens], dtype=torch.long, device=eval_device)
            x_cand = torch.tensor([cand_tokens], dtype=torch.long, device=eval_device)
            if want_hidden:
                ref_logits, ref_hidden_full = ref_model.forward_hidden(x_ref)
                cand_logits, cand_hidden_full = cand_model.forward_hidden(x_cand)
            else:
                ref_logits, cand_logits = ref_model(x_ref), cand_model(x_cand)
                ref_hidden_full = cand_hidden_full = None
            # logits[resp_start - 1] predicts the first response token, so the scored
            # window starts one index before resp_start — starting at resp_start would
            # silently skip the opening prediction (the one that matters most here).
            # Each side is scored from its own resp_start, so the compared window is
            # each model's response opening even when prompt lengths differ.
            ref_score_slice = slice(ref_resp_start - 1, ref_resp_start - 1 + CHAT_SCORE_TOKENS)
            cand_score_slice = slice(cand_resp_start - 1, cand_resp_start - 1 + CHAT_SCORE_TOKENS)
            yield (
                ref_logits[:, ref_score_slice, :],
                cand_logits[:, cand_score_slice, :],
                [h[:, ref_score_slice, :] for h in ref_hidden_full] if want_hidden else None,
                [h[:, cand_score_slice, :] for h in cand_hidden_full] if want_hidden else None,
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


def _build_rows(
    pair_name: str, dist: str, k: int, repeat: int, seed: int, elapsed: float,
    accum: dict[str, float], weight_total: int, decision_matches: int,
    activation_accum: dict[int, dict[str, float]], activation_weight_total: int,
    decision_metric: str | None, beta: float, reject_threshold: float | None, reject_on: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Turns one (pair, distribution, k, repeat) run's accumulated totals into a
    logit-space row plus one activation row per layer. Shared by the collect/replay
    path and the live wrap_cand path so both produce identical row shapes."""
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
        row["verdict"] = "reject equality" if mismatches > 0 else "cannot reject equality"
        row["reject"] = int(mismatches > 0)
        row["epsilon_upper"] = None if mismatches > 0 else 1 - beta ** (1 / k)

    if reject_threshold is not None:
        agree_col = (
            "top1_agreement/token_agreement" if reject_on == "token" else "top1_agreement/seq_agreement"
        )
        soft_reject = int(row[agree_col] < reject_threshold)
        # Hard verdict wins the "reject" column when both are requested.
        if decision_metric is not None:
            row["soft_reject"] = soft_reject
        else:
            row["reject"] = soft_reject

    activation_rows: list[dict[str, Any]] = []
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

    return row, activation_rows


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
    of layer i) instead of the verdict/reject columns.

    Runs in two phases. Phase 1 collects: every distinct model referenced by a
    non-wrap_cand pair is loaded once, run over the *entire* challenge sweep
    (every distribution x k x repeat) exactly once, its per-batch outputs moved to
    CPU and cached, and then evicted from the GPU — so a base model referenced by
    N variant pairs runs its forward pass once, not N times, and at most one model
    is GPU-resident at a time. Phase 2 diffs: each pair replays its ref and cand's
    cached outputs (no GPU or model needed) to compute the same rows as before.
    wrap_cand pairs can't be decomposed this way (the wrapper needs both source
    models live to build itself) and are excluded from the collect phase — they
    still run the old way, live, and remain incompatible with activation_metrics.
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

    # wrap_cand pairs build a composite model from live deep copies of both
    # sources and can't be decomposed into independent per-model passes, so
    # they're excluded from the collect phase and keep running the old way.
    # Tasks are keyed by (model key, device): get_pair would fall a GPU-capable
    # ref back to CPU if its cand is CPU-only (e.g. a dynamic-quantized model),
    # so the same model can need collecting once per device it's actually
    # paired at — that's still one collection per (model, device), not per pair.
    collect_tasks: set[tuple[str, str]] = set()
    for pair_def in spec.pairs:
        if pair_def.wrap_cand:
            if want_hidden:
                raise ValueError(
                    f"wrap_cand is not supported with activation_metrics (pair {pair_def.ref!r} vs "
                    f"{pair_def.cand!r} requested {pair_def.wrap_cand!r})"
                )
            continue
        pair_device = ctx.resolve_pair_device(pair_def.ref, pair_def.cand)
        collect_tasks.add((pair_def.ref, pair_device))
        collect_tasks.add((pair_def.cand, pair_device))

    # Phase 1: run each distinct (model, device) task over the whole challenge
    # sweep exactly once, cache its outputs to CPU, and evict it before loading
    # the next — a base model referenced by N variant pairs at the same device
    # runs its forward pass once, not N times, and at most one model is
    # GPU-resident at a time.
    collected: dict[tuple[str, str], dict[str, Any]] = {}
    for key, device in collect_tasks:
        bundle = ctx.get_model(key, device=device)
        t0 = time.time()
        cache = _collect_model_outputs(
            bundle["model"], bundle["cfg"].vocab_size, bundle["device"], ctx, spec, want_hidden,
            system_prompt=bundle["meta"].get("system_prompt"),
        )
        n_batches = sum(len(v) for v in cache.values())
        print(f"Collected {key} on {device}: {n_batches} batches in {time.time() - t0:.1f}s")
        collected[(key, device)] = {"cache": cache, "cfg": bundle["cfg"]}
        ctx.evict(key)

    # Phase 2: diff each pair. Non-wrap_cand pairs replay phase 1's cached
    # outputs (no GPU or model needed here); wrap_cand pairs still run live.
    for pair_def in spec.pairs:
        ref_key = pair_def.ref
        cand_key = pair_def.cand
        wrapper = pair_def.wrap_cand
        default_name = f"{ref_key}_vs_{cand_key}" + (f"_{wrapper}" if wrapper else "")
        pair_name = pair_def.name if pair_def.name is not None else default_name

        if wrapper:
            bundle = ctx.get_pair(ref_key, cand_key)
            ref_model = bundle["ref_model"]
            cand_model = _wrap_candidate(wrapper, ref_model, bundle["cand_model"])
            eval_device = bundle["eval_device"]
            vocab_size = bundle["cfg"].vocab_size
            ref_system_prompt = bundle["ref_meta"].get("system_prompt")
            cand_system_prompt = bundle["cand_meta"].get("system_prompt")

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
                                ref_system_prompt=ref_system_prompt, cand_system_prompt=cand_system_prompt,
                            )
                            accum, weight_total, decision_matches, activation_accum, activation_weight_total = (
                                _score_run(batches, metric_names, decision_metric, activation_metric_names)
                            )
                        if eval_device == "cuda":
                            torch.cuda.empty_cache()
                        elapsed = time.time() - t0
                        row, arows = _build_rows(
                            pair_name, dist, k, repeat, seed, elapsed,
                            accum, weight_total, decision_matches,
                            activation_accum, activation_weight_total,
                            decision_metric, beta, reject_threshold, reject_on,
                        )
                        rows.append(row)
                        activation_rows.extend(arows)
            continue

        pair_device = ctx.resolve_pair_device(ref_key, cand_key)
        ref_cfg = collected[(ref_key, pair_device)]["cfg"]
        cand_cfg = collected[(cand_key, pair_device)]["cfg"]
        if ref_cfg != cand_cfg:
            raise ValueError(f"Config mismatch: {ref_key} vs {cand_key}")
        ref_cache = collected[(ref_key, pair_device)]["cache"]
        cand_cache = collected[(cand_key, pair_device)]["cache"]

        for ch in challenges:
            dist = ch.name
            for k in k_values:
                for repeat in range(repeats):
                    seed = 100000 * repeat + 1000 * k + sum(ord(c) for c in dist)
                    t0 = time.time()
                    batches = _replay_batches(ref_cache[(dist, k, repeat)], cand_cache[(dist, k, repeat)])
                    accum, weight_total, decision_matches, activation_accum, activation_weight_total = (
                        _score_run(batches, metric_names, decision_metric, activation_metric_names)
                    )
                    elapsed = time.time() - t0
                    row, arows = _build_rows(
                        pair_name, dist, k, repeat, seed, elapsed,
                        accum, weight_total, decision_matches,
                        activation_accum, activation_weight_total,
                        decision_metric, beta, reject_threshold, reject_on,
                    )
                    rows.append(row)
                    activation_rows.extend(arows)

    return rows, activation_rows
