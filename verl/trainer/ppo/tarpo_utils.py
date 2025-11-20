from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------- TARPO GLOBAL STATE ---------------------------- #

# Per-task running stats & buffers
task_stats: Dict[Any, Dict[str, Any]] = defaultdict(lambda: {
    "ema_mu": 0.0, "ema_sigma": 1.0,
    "buffer_mean_ema": 0.0, "buffer_cvar_ema": 0.0, "buffer_ptail_ema": 0.0,
    "buffer": deque(maxlen=256),
    "ema_count": 0,                # cumulative samples for this task

    # Raw (pre-adapter) batch stats
    "raw_batch_mu": 0.0,
    "raw_batch_sigma": 1.0,
    "raw_batch_count": 0,

    # k_t (boost) batch + EMA
    "k_batch_mu": 1.0,
    "k_batch_sigma": 0.0,
    "k_ema": 1.0,

    # Final (post adapter, class weighting, CVaR, group norm) advantages
    "final_advantage_batch_mu": 0.0,
    "final_advantage_batch_sigma": 1.0,
    "final_advantage_ema_mu": 0.0,
    "final_advantage_ema_sigma": 1.0,
})

# Per-dataset running stats & buffers (logging only; mirrors task_stats)
dataset_stats: Dict[Any, Dict[str, Any]] = defaultdict(lambda: {
    "ema_mu": 0.0, "ema_sigma": 1.0,
    "buffer_mean_ema": 0.0, "buffer_cvar_ema": 0.0, "buffer_ptail_ema": 0.0,
    "buffer": deque(maxlen=256),
    "ema_count": 0,                # cumulative samples for this dataset

    # Raw (pre-adapter) batch stats
    "raw_batch_mu": 0.0,
    "raw_batch_sigma": 1.0,
    "raw_batch_count": 0,

    # k_t (boost) batch + EMA (mirrors task-level k_t for logging)
    "k_batch_mu": 1.0,
    "k_batch_sigma": 0.0,
    "k_ema": 1.0,

    # Final (post adapter, class weighting, CVaR, group norm) advantages
    "final_advantage_batch_mu": 0.0,
    "final_advantage_batch_sigma": 1.0,
    "final_advantage_ema_mu": 0.0,
    "final_advantage_ema_sigma": 1.0,
})


# ---------------------------- Small helpers --------------------------------- #

def _ema_update(prev: float, new: float, beta: float) -> float:
    """Exponential moving average update."""
    return float(beta * prev + (1.0 - beta) * new)


def _compute_mu_from_values(values: List[float]) -> float:
    """Compute the mean from a list of floats."""
    n = len(values)
    if n == 0:
        return 0.0
    return float(sum(values) / n)


def _compute_sd_from_values(
    values: List[float],
    *,
    mu: Optional[float] = None,
    eps: float = 1e-8,
) -> float:
    """
    Compute standard deviation from a list of floats.

    Args:
        values: list of scalar values
        mu:     optional precomputed mean; if None, it will be recomputed
        eps:    small positive number for numerical stability and clamp

    Returns:
        sd >= eps
    """
    n = len(values)
    if n == 0:
        return float(eps)

    if mu is None:
        mu = _compute_mu_from_values(values)
    var_acc = 0.0
    for v in values:
        diff = v - mu
        var_acc += diff * diff
    var = var_acc / n if n > 0 else 0.0
    sd = math.sqrt(max(var, 0.0) + 1e-12)
    return float(max(sd, eps))


def build_mappings(
    raw_scores: torch.Tensor,          # (B,)
    index: torch.Tensor,               # (B,)
    task_ids: List[Any],               # (B,)
    dataset_ids: List[Any],            # (B,)
    class_labels: List[Any],           # (B,)
    *,
    use_minmax_scaling: bool = False,  # Global min-max scaling
    eps: float = 1e-8,
) -> Tuple[
    Dict[Any, List[float]],            # q2rollouts
    Dict[Any, Any],                    # q2tasks
    Dict[Any, Any],                    # q2datasets
    Dict[Any, Any],                    # q2class
    Dict[Any, List[float]],            # task_to_rollouts
    Dict[Any, List[float]],            # dataset_to_rollouts
]:
    """
    Single-pass construction of:
        - q2rollouts:      qid -> list of scalar scores (optionally min-max scaled)
        - q2tasks:         qid -> task id
        - q2datasets:      qid -> dataset id
        - q2class:         qid -> class label
        - task_to_rollouts:    task_id    -> list of scalar scores in this batch
        - dataset_to_rollouts: dataset_id -> list of scalar scores in this batch

    If use_minmax_scaling=True, applies global min-max scaling to raw_scores
    before building the mappings: (v - min) / (max - min + eps)
    """

    B = raw_scores.shape[0]

    # Optional: Global min-max scaling
    if use_minmax_scaling:
        score_min = raw_scores.min()
        score_max = raw_scores.max()
        denom = max(float(score_max - score_min), eps)
        raw_scores = (raw_scores - score_min) / denom

    q2rollouts: Dict[Any, List[float]] = defaultdict(list)
    q2tasks:    Dict[Any, Any]         = {}
    q2datasets: Dict[Any, Any]         = {}
    q2class:    Dict[Any, Any]         = {}

    task_to_rollouts: Dict[Any, List[float]]    = defaultdict(list)
    dataset_to_rollouts: Dict[Any, List[float]] = defaultdict(list)

    for i in range(B):
        qid: Any    = index[i]
        task        = task_ids[i]
        dataset     = dataset_ids[i]
        class_label = class_labels[i]
        raw_r       = float(raw_scores[i].item())

        # Per-qid structures
        q2rollouts[qid].append(raw_r)
        q2tasks[qid]    = task
        q2datasets[qid] = dataset
        q2class[qid]    = class_label

        # Per-task / per-dataset rollouts
        task_to_rollouts[task].append(raw_r)
        dataset_to_rollouts[dataset].append(raw_r)

    return (
        q2rollouts,
        q2tasks,
        q2datasets,
        q2class,
        task_to_rollouts,
        dataset_to_rollouts,
    )


def update_raw_stats(
    task_to_rollouts: Dict[Any, List[float]],
    dataset_to_rollouts: Dict[Any, List[float]],
    *,
    beta_mu: float,
    beta_sigma: float,
    eps: float,
) -> None:
    """
    Update raw_batch_* and ema_mu/ema_sigma for tasks and datasets
    directly from per-task / per-dataset rollout lists.

    This assumes the grouping (task_to_rollouts, dataset_to_rollouts) has already
    been constructed once upstream (e.g., in `build_q_mappings`).
    """

    # ---- Tasks ----
    for task, vals in task_to_rollouts.items():
        if not vals:
            continue

        mu_b = _compute_mu_from_values(vals)
        sd_b = _compute_sd_from_values(vals, mu=mu_b, eps=eps)

        # Raw batch stats
        task_stats[task]["raw_batch_mu"]    = float(mu_b)
        task_stats[task]["raw_batch_sigma"] = float(sd_b)
        task_stats[task]["raw_batch_count"] = len(vals)

        # EMA (used by adapter; updated regardless of toggles)
        if task_stats[task]["ema_count"] == 0:
            task_stats[task]["ema_mu"]    = float(mu_b)
            task_stats[task]["ema_sigma"] = float(sd_b)
        else:
            task_stats[task]["ema_mu"]    = _ema_update(task_stats[task]["ema_mu"],    float(mu_b), beta_mu)
            task_stats[task]["ema_sigma"] = _ema_update(task_stats[task]["ema_sigma"], float(sd_b),  beta_sigma)

        task_stats[task]["ema_count"] += len(vals)

    # ---- Datasets ----
    for dataset, vals in dataset_to_rollouts.items():
        if not vals:
            continue

        mu_b = _compute_mu_from_values(vals)
        sd_b = _compute_sd_from_values(vals, mu=mu_b, eps=eps)

        dataset_stats[dataset]["raw_batch_mu"]    = float(mu_b)
        dataset_stats[dataset]["raw_batch_sigma"] = float(sd_b)
        dataset_stats[dataset]["raw_batch_count"] = len(vals)

        if dataset_stats[dataset]["ema_count"] == 0:
            dataset_stats[dataset]["ema_mu"]    = float(mu_b)
            dataset_stats[dataset]["ema_sigma"] = float(sd_b)
        else:
            dataset_stats[dataset]["ema_mu"]    = _ema_update(dataset_stats[dataset]["ema_mu"],    float(mu_b), beta_mu)
            dataset_stats[dataset]["ema_sigma"] = _ema_update(dataset_stats[dataset]["ema_sigma"], float(sd_b),  beta_sigma)

        dataset_stats[dataset]["ema_count"] += len(vals)


def update_k_stats_from_q2k(
    q2k: Dict[Any, float],
    q2tasks: Dict[Any, Any],
    q2datasets: Dict[Any, Any],
    *,
    beta_mean: float,
    eps: float,
) -> None:
    """
    Update k_t statistics per task and per dataset from per-qid k-values.

    Operates directly on the collected k-values per task/dataset
    (no intermediate sum/sumsq/count dicts).
    """
    # ---- Tasks ----
    task_to_kvals: Dict[Any, List[float]] = defaultdict(list)
    for qid, k in q2k.items():
        task = q2tasks[qid]
        task_to_kvals[task].append(float(k))

    for task, vals in task_to_kvals.items():
        if not vals:
            continue
        mu = _compute_mu_from_values(vals)
        sd = _compute_sd_from_values(vals, mu=mu, eps=eps)
        task_stats[task]["k_batch_mu"]    = float(mu)
        task_stats[task]["k_batch_sigma"] = float(sd)
        task_stats[task]["k_ema"] = _ema_update(task_stats[task].get("k_ema", 1.0), float(mu), beta_mean)

    # ---- Datasets ----
    ds_to_kvals: Dict[Any, List[float]] = defaultdict(list)
    for qid, k in q2k.items():
        dataset = q2datasets[qid]
        ds_to_kvals[dataset].append(float(k))

    for dataset, vals in ds_to_kvals.items():
        if not vals:
            continue
        mu = _compute_mu_from_values(vals)
        sd = _compute_sd_from_values(vals, mu=mu, eps=eps)
        dataset_stats[dataset]["k_batch_mu"]    = float(mu)
        dataset_stats[dataset]["k_batch_sigma"] = float(sd)
        dataset_stats[dataset]["k_ema"] = _ema_update(dataset_stats[dataset].get("k_ema", 1.0), float(mu), beta_mean)


def update_final_advantages_stats(
    final_scores: torch.Tensor,           # (B,)
    task_ids: List[Any],                  # (B,)
    dataset_ids: List[Any],               # (B,)
    *,
    beta_mu: float,
    beta_sigma: float,
    eps: float,
) -> None:
    """
    Centralized logging for FINAL TARPO advantages.

    Uses the final scalar scores per rollout (after task adapter,
    class weights, CVaR boost, and optional group norm) and records:
        - per-task batch μ/σ and EMA μ/σ
        - per-dataset batch μ/σ and EMA μ/σ
    into `task_stats` and `dataset_stats`.

    Purely logging; does not affect TARPO computation.
    """
    B = final_scores.shape[0]

    # Collect values per task / dataset
    # have to collect again as these values are not rollouts
    task_to_vals: Dict[Any, List[float]]    = defaultdict(list)
    dataset_to_vals: Dict[Any, List[float]] = defaultdict(list)

    for i in range(B):
        v = float(final_scores[i].item())
        t = task_ids[i]
        d = dataset_ids[i]
        task_to_vals[t].append(v)
        dataset_to_vals[d].append(v)

    # Per-task final advantage stats
    for t, vals in task_to_vals.items():
        if not vals:
            continue

        mu = _compute_mu_from_values(vals)
        sd = _compute_sd_from_values(vals, mu=mu, eps=eps)

        task_stats[t]["final_advantage_batch_mu"]    = float(mu)
        task_stats[t]["final_advantage_batch_sigma"] = float(sd)

        prev_mu    = task_stats[t].get("final_advantage_ema_mu", 0.0)
        prev_sigma = task_stats[t].get("final_advantage_ema_sigma", 1.0)
        task_stats[t]["final_advantage_ema_mu"]    = _ema_update(prev_mu,    float(mu), beta_mu)
        task_stats[t]["final_advantage_ema_sigma"] = _ema_update(prev_sigma, float(sd),  beta_sigma)

    # Per-dataset final advantage stats
    for d, vals in dataset_to_vals.items():
        if not vals:
            continue

        mu = _compute_mu_from_values(vals)
        sd = _compute_sd_from_values(vals, mu=mu, eps=eps)

        dataset_stats[d]["final_advantage_batch_mu"]    = float(mu)
        dataset_stats[d]["final_advantage_batch_sigma"] = float(sd)

        prev_mu    = dataset_stats[d].get("final_advantage_ema_mu", 0.0)
        prev_sigma = dataset_stats[d].get("final_advantage_ema_sigma", 1.0)
        dataset_stats[d]["final_advantage_ema_mu"]    = _ema_update(prev_mu,    float(mu), beta_mu)
        dataset_stats[d]["final_advantage_ema_sigma"] = _ema_update(prev_sigma, float(sd),  beta_sigma)