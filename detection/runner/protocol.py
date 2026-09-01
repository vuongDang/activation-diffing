from __future__ import annotations

import copy
import time
from typing import Any

import torch

from detection.data.challenges import generate_challenges
from detection.metrics import agreement, divergence
from detection.metrics import token_difr as token_difr_mod
from .attacker import SwitchingAttacker
from .context import Context

# External distribution alias -> internal name used by generate_challenges
_DIST_ALIASES = {"corpus_id": "corpus_window"}

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


def available_metrics() -> list[str]:
    return list(METRICS.keys())


def _wrap_candidate(wrapper: str, ref_model, cand_model):
    if wrapper == "switching_attacker":
        return SwitchingAttacker(
            copy.deepcopy(ref_model).eval(), copy.deepcopy(cand_model).eval()
        ).eval()
    raise ValueError(f"Unknown wrap_cand '{wrapper}'. Available: {CANDIDATE_WRAPPERS}")


def run_experiment(spec: dict[str, Any], ctx: Context) -> list[dict[str, Any]]:
    """
    Runs all pairs × distributions × k_values × repeats defined in the experiment spec.
    Returns a flat list of row dicts suitable for pd.DataFrame.

    Spec fields:
      pairs            list of {ref, cand, name?, wrap_cand?}
      metrics          list of metric names (default: all)
      distributions    list of distribution names (supports "corpus_id" alias)
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
    metric_names = spec.get("metrics", list(METRICS.keys()))
    unknown = [m for m in metric_names if m not in METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Available: {list(METRICS.keys())}")

    seq_len = spec.get("seq_len", 64)
    batch_size = spec.get("batch_size", 32)
    repeats = spec.get("repeats", 5)
    k_values = spec.get("k_values", [16, 32, 64, 128])
    distributions = spec.get("distributions", ["uniform"])
    reject_threshold = spec.get("reject_threshold", None)
    reject_on = spec.get("reject_on", "token")
    if reject_on not in ("token", "seq"):
        raise ValueError(f"Unknown reject_on '{reject_on}'. Choose 'token' or 'seq'")
    decision_metric = spec.get("decision_metric", None)
    if decision_metric is not None and decision_metric not in DECISION_METRICS:
        raise ValueError(
            f"Unknown decision_metric '{decision_metric}'. Available: {DECISION_METRICS}"
        )
    beta = spec.get("beta", 0.05)

    # The soft threshold needs the agreement metric to be computed.
    if reject_threshold is not None and "top1_agreement" not in metric_names:
        metric_names = ["top1_agreement"] + metric_names

    rows: list[dict[str, Any]] = []

    for pair_def in spec["pairs"]:
        ref_key = pair_def["ref"]
        cand_key = pair_def["cand"]
        wrapper = pair_def.get("wrap_cand")
        default_name = f"{ref_key}_vs_{cand_key}" + (f"_{wrapper}" if wrapper else "")
        pair_name = pair_def.get("name", default_name)
        bundle = ctx.get_pair(ref_key, cand_key)
        ref_model = bundle["ref_model"]
        cand_model = bundle["cand_model"]
        if wrapper:
            cand_model = _wrap_candidate(wrapper, ref_model, cand_model)
        eval_device = bundle["eval_device"]
        vocab_size = bundle["cfg"].vocab_size

        for dist in distributions:
            internal_dist = _DIST_ALIASES.get(dist, dist)
            for k in k_values:
                for repeat in range(repeats):
                    seed = 100000 * repeat + 1000 * k + sum(ord(c) for c in dist)
                    X = generate_challenges(
                        distribution=internal_dist,
                        num_challenges=k,
                        seq_len=seq_len,
                        vocab_size=vocab_size,
                        device=eval_device,
                        seed=seed,
                        eval_corpus_ids=ctx.eval_ids,
                    )

                    accum: dict[str, float] = {}
                    weight_total = 0
                    decision_matches = 0
                    t0 = time.time()

                    with torch.no_grad():
                        for i in range(0, k, batch_size):
                            batch = X[i : i + batch_size]
                            n_b = batch.shape[0]
                            weight_total += n_b
                            ref_logits = ref_model(batch)
                            cand_logits = cand_model(batch)
                            for m in metric_names:
                                for key, val in METRICS[m](ref_logits, cand_logits).items():
                                    col = f"{m}/{key}"
                                    accum[col] = accum.get(col, 0.0) + val * n_b
                            if decision_metric == "top1_all":
                                matches = (
                                    (ref_logits.argmax(dim=-1) == cand_logits.argmax(dim=-1))
                                    .all(dim=-1)
                                )
                                decision_matches += int(matches.sum().item())
                            elif decision_metric == "exact_all":
                                matches = (ref_logits == cand_logits).all(dim=-1).all(dim=-1)
                                decision_matches += int(matches.sum().item())

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
