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

    # Task adapter scaling factors
    "adapter_mu_scale": 1.0,
    "adapter_sigma_scale": 1.0,
    "adapter_final_scale": 1.0,

    # Post-GRPO (after group normalization) advantages
    "post_grpo_advantage_batch_mu": 0.0,
    "post_grpo_advantage_batch_sigma": 1.0,
    "post_grpo_advantage_batch_abs_mean": 0.0,
    "post_grpo_advantage_batch_frac_pos": 0.0,
    "post_grpo_advantage_batch_frac_neg": 0.0,
    "post_grpo_advantage_batch_skewness": 0.0,
    "post_grpo_advantage_batch_p10": 0.0,
    "post_grpo_advantage_batch_p50": 0.0,
    "post_grpo_advantage_batch_p90": 0.0,
    "post_grpo_advantage_ema_mu": 0.0,
    "post_grpo_advantage_ema_sigma": 1.0,
    "post_grpo_advantage_ema_abs_mean": 0.0,
    "post_grpo_advantage_ema_frac_pos": 0.0,
    "post_grpo_advantage_ema_frac_neg": 0.0,
    "post_grpo_advantage_ema_skewness": 0.0,
    "post_grpo_advantage_ema_p10": 0.0,
    "post_grpo_advantage_ema_p50": 0.0,
    "post_grpo_advantage_ema_p90": 0.0,

    # Mixture model parameters
    "mixture_rho_t": 0.0,
    "mixture_rho_ref": 0.0,
    "mixture_log_scale": 0.0,
    "mixture_final_scale": 1.0,

    # Final (post adapter, class weighting, CVaR, group norm) advantages
    "final_advantage_batch_mu": 0.0,
    "final_advantage_batch_sigma": 1.0,
    "final_advantage_batch_abs_mean": 0.0,
    "final_advantage_batch_frac_pos": 0.0,
    "final_advantage_batch_frac_neg": 0.0,
    "final_advantage_batch_skewness": 0.0,
    "final_advantage_batch_p10": 0.0,
    "final_advantage_batch_p50": 0.0,
    "final_advantage_batch_p90": 0.0,
    "final_advantage_ema_mu": 0.0,
    "final_advantage_ema_sigma": 1.0,
    "final_advantage_ema_abs_mean": 0.0,
    "final_advantage_ema_frac_pos": 0.0,
    "final_advantage_ema_frac_neg": 0.0,
    "final_advantage_ema_skewness": 0.0,
    "final_advantage_ema_p10": 0.0,
    "final_advantage_ema_p50": 0.0,
    "final_advantage_ema_p90": 0.0,
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

    # Post-GRPO (after group normalization) advantages
    "post_grpo_advantage_batch_mu": 0.0,
    "post_grpo_advantage_batch_sigma": 1.0,
    "post_grpo_advantage_batch_abs_mean": 0.0,
    "post_grpo_advantage_batch_frac_pos": 0.0,
    "post_grpo_advantage_batch_frac_neg": 0.0,
    "post_grpo_advantage_batch_skewness": 0.0,
    "post_grpo_advantage_batch_p10": 0.0,
    "post_grpo_advantage_batch_p50": 0.0,
    "post_grpo_advantage_batch_p90": 0.0,
    "post_grpo_advantage_ema_mu": 0.0,
    "post_grpo_advantage_ema_sigma": 1.0,
    "post_grpo_advantage_ema_abs_mean": 0.0,
    "post_grpo_advantage_ema_frac_pos": 0.0,
    "post_grpo_advantage_ema_frac_neg": 0.0,
    "post_grpo_advantage_ema_skewness": 0.0,
    "post_grpo_advantage_ema_p10": 0.0,
    "post_grpo_advantage_ema_p50": 0.0,
    "post_grpo_advantage_ema_p90": 0.0,
    
    # Final (post adapter, class weighting, CVaR, group norm) advantages
    "final_advantage_batch_mu": 0.0,
    "final_advantage_batch_sigma": 1.0,
    "final_advantage_batch_abs_mean": 0.0,
    "final_advantage_batch_frac_pos": 0.0,
    "final_advantage_batch_frac_neg": 0.0,
    "final_advantage_batch_skewness": 0.0,
    "final_advantage_batch_p10": 0.0,
    "final_advantage_batch_p50": 0.0,
    "final_advantage_batch_p90": 0.0,
    "final_advantage_ema_mu": 0.0,
    "final_advantage_ema_sigma": 1.0,
    "final_advantage_ema_abs_mean": 0.0,
    "final_advantage_ema_frac_pos": 0.0,
    "final_advantage_ema_frac_neg": 0.0,
    "final_advantage_ema_skewness": 0.0,
    "final_advantage_ema_p10": 0.0,
    "final_advantage_ema_p50": 0.0,
    "final_advantage_ema_p90": 0.0,
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


def _compute_skewness(values: List[float], *, mu: Optional[float] = None, sd: Optional[float] = None, eps: float = 1e-8) -> float:
    """
    Compute skewness from a list of floats.

    Args:
        values: list of scalar values
        mu:     optional precomputed mean
        sd:     optional precomputed standard deviation
        eps:    small positive number for numerical stability

    Returns:
        skewness (0.0 if < 3 samples)
    """
    n = len(values)
    if n < 3:
        return 0.0

    if mu is None:
        mu = _compute_mu_from_values(values)
    if sd is None:
        sd = _compute_sd_from_values(values, mu=mu, eps=eps)

    if sd < eps:
        return 0.0

    m3 = sum((v - mu) ** 3 for v in values) / n
    skew = m3 / (sd ** 3)
    return float(skew)


def _compute_percentiles(values: List[float], percentiles: List[float] = [10, 50, 90]) -> Dict[str, float]:
    """
    Compute percentiles from a list of floats.

    Args:
        values: list of scalar values
        percentiles: list of percentile values to compute (0-100)

    Returns:
        dict mapping 'p{percentile}' to computed value
    """
    if len(values) == 0:
        return {f"p{int(p)}": 0.0 for p in percentiles}

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    result = {}

    for p in percentiles:
        # Linear interpolation for percentile
        idx = (p / 100.0) * (n - 1)
        lower_idx = int(math.floor(idx))
        upper_idx = int(math.ceil(idx))

        if lower_idx == upper_idx:
            result[f"p{int(p)}"] = float(sorted_vals[lower_idx])
        else:
            weight = idx - lower_idx
            result[f"p{int(p)}"] = float(sorted_vals[lower_idx] * (1 - weight) + sorted_vals[upper_idx] * weight)

    return result


def _compute_abs_mean(values: List[float]) -> float:
    """Compute mean of absolute values."""
    if len(values) == 0:
        return 0.0
    return float(sum(abs(v) for v in values) / len(values))


def _compute_fraction_positive(values: List[float]) -> float:
    """Compute fraction of positive values."""
    if len(values) == 0:
        return 0.0
    return float(sum(1 for v in values if v > 0) / len(values))


def _compute_fraction_negative(values: List[float]) -> float:
    """Compute fraction of negative values."""
    if len(values) == 0:
        return 0.0
    return float(sum(1 for v in values if v < 0) / len(values))


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


def _update_advantage_stats_general(
    task_to_vals: Dict[Any, List[float]],
    dataset_to_vals: Dict[Any, List[float]],
    *,
    stat_prefix: str,  # e.g., "post_grpo_advantage" or "final_advantage"
    beta_mu: float,
    beta_sigma: float,
    eps: float,
) -> None:
    """
    General function to update advantage statistics for both tasks and datasets.

    This function computes and updates:
        - Batch stats: mu, sigma, abs_mean, frac_pos, frac_neg, skewness, p10/p50/p90
        - EMA stats: mu, sigma, abs_mean, frac_pos, frac_neg

    Args:
        task_to_vals:    dict mapping task_id -> list of advantage values
        dataset_to_vals: dict mapping dataset_id -> list of advantage values
        stat_prefix:     prefix for stat keys (e.g., "post_grpo_advantage" or "final_advantage")
        beta_mu:         EMA decay for mean
        beta_sigma:      EMA decay for std
        eps:             small epsilon for numerical stability
    """
    # Per-task stats
    for t, vals in task_to_vals.items():
        if not vals:
            continue

        mu = _compute_mu_from_values(vals)
        sd = _compute_sd_from_values(vals, mu=mu, eps=eps)
        abs_mean = _compute_abs_mean(vals)
        frac_pos = _compute_fraction_positive(vals)
        frac_neg = _compute_fraction_negative(vals)
        skew = _compute_skewness(vals, mu=mu, sd=sd, eps=eps)
        percentiles = _compute_percentiles(vals, percentiles=[10, 50, 90])

        # Batch stats
        task_stats[t][f"{stat_prefix}_batch_mu"] = float(mu)
        task_stats[t][f"{stat_prefix}_batch_sigma"] = float(sd)
        task_stats[t][f"{stat_prefix}_batch_abs_mean"] = float(abs_mean)
        task_stats[t][f"{stat_prefix}_batch_frac_pos"] = float(frac_pos)
        task_stats[t][f"{stat_prefix}_batch_frac_neg"] = float(frac_neg)
        task_stats[t][f"{stat_prefix}_batch_skewness"] = float(skew)
        task_stats[t][f"{stat_prefix}_batch_p10"] = percentiles["p10"]
        task_stats[t][f"{stat_prefix}_batch_p50"] = percentiles["p50"]
        task_stats[t][f"{stat_prefix}_batch_p90"] = percentiles["p90"]

        # EMA stats
        prev_mu = task_stats[t].get(f"{stat_prefix}_ema_mu", 0.0)
        prev_sigma = task_stats[t].get(f"{stat_prefix}_ema_sigma", 1.0)
        prev_abs_mean = task_stats[t].get(f"{stat_prefix}_ema_abs_mean", 0.0)
        prev_frac_pos = task_stats[t].get(f"{stat_prefix}_ema_frac_pos", 0.0)
        prev_frac_neg = task_stats[t].get(f"{stat_prefix}_ema_frac_neg", 0.0)
        prev_skewness = task_stats[t].get(f"{stat_prefix}_ema_skewness", 0.0)
        prev_p10 = task_stats[t].get(f"{stat_prefix}_ema_p10", 0.0)
        prev_p50 = task_stats[t].get(f"{stat_prefix}_ema_p50", 0.0)
        prev_p90 = task_stats[t].get(f"{stat_prefix}_ema_p90", 0.0)

        task_stats[t][f"{stat_prefix}_ema_mu"] = _ema_update(prev_mu, float(mu), beta_mu)
        task_stats[t][f"{stat_prefix}_ema_sigma"] = _ema_update(prev_sigma, float(sd), beta_sigma)
        task_stats[t][f"{stat_prefix}_ema_abs_mean"] = _ema_update(prev_abs_mean, float(abs_mean), beta_mu)
        task_stats[t][f"{stat_prefix}_ema_frac_pos"] = _ema_update(prev_frac_pos, float(frac_pos), beta_mu)
        task_stats[t][f"{stat_prefix}_ema_frac_neg"] = _ema_update(prev_frac_neg, float(frac_neg), beta_mu)
        task_stats[t][f"{stat_prefix}_ema_skewness"] = _ema_update(prev_skewness, float(skew), beta_mu)
        task_stats[t][f"{stat_prefix}_ema_p10"] = _ema_update(prev_p10, percentiles["p10"], beta_mu)
        task_stats[t][f"{stat_prefix}_ema_p50"] = _ema_update(prev_p50, percentiles["p50"], beta_mu)
        task_stats[t][f"{stat_prefix}_ema_p90"] = _ema_update(prev_p90, percentiles["p90"], beta_mu)


    # Per-dataset stats
    for d, vals in dataset_to_vals.items():
        if not vals:
            continue

        mu = _compute_mu_from_values(vals)
        sd = _compute_sd_from_values(vals, mu=mu, eps=eps)
        abs_mean = _compute_abs_mean(vals)
        frac_pos = _compute_fraction_positive(vals)
        frac_neg = _compute_fraction_negative(vals)
        skew = _compute_skewness(vals, mu=mu, sd=sd, eps=eps)
        percentiles = _compute_percentiles(vals, percentiles=[10, 50, 90])

        # Batch stats
        dataset_stats[d][f"{stat_prefix}_batch_mu"] = float(mu)
        dataset_stats[d][f"{stat_prefix}_batch_sigma"] = float(sd)
        dataset_stats[d][f"{stat_prefix}_batch_abs_mean"] = float(abs_mean)
        dataset_stats[d][f"{stat_prefix}_batch_frac_pos"] = float(frac_pos)
        dataset_stats[d][f"{stat_prefix}_batch_frac_neg"] = float(frac_neg)
        dataset_stats[d][f"{stat_prefix}_batch_skewness"] = float(skew)
        dataset_stats[d][f"{stat_prefix}_batch_p10"] = percentiles["p10"]
        dataset_stats[d][f"{stat_prefix}_batch_p50"] = percentiles["p50"]
        dataset_stats[d][f"{stat_prefix}_batch_p90"] = percentiles["p90"]

        # EMA stats
        prev_mu = dataset_stats[d].get(f"{stat_prefix}_ema_mu", 0.0)
        prev_sigma = dataset_stats[d].get(f"{stat_prefix}_ema_sigma", 1.0)
        prev_abs_mean = dataset_stats[d].get(f"{stat_prefix}_ema_abs_mean", 0.0)
        prev_frac_pos = dataset_stats[d].get(f"{stat_prefix}_ema_frac_pos", 0.0)
        prev_frac_neg = dataset_stats[d].get(f"{stat_prefix}_ema_frac_neg", 0.0)
        prev_skewness = dataset_stats[d].get(f"{stat_prefix}_ema_skewness", 0.0)
        prev_p10 = dataset_stats[d].get(f"{stat_prefix}_ema_p10", 0.0)
        prev_p50 = dataset_stats[d].get(f"{stat_prefix}_ema_p50", 0.0)
        prev_p90 = dataset_stats[d].get(f"{stat_prefix}_ema_p90", 0.0)

        dataset_stats[d][f"{stat_prefix}_ema_mu"] = _ema_update(prev_mu, float(mu), beta_mu)
        dataset_stats[d][f"{stat_prefix}_ema_sigma"] = _ema_update(prev_sigma, float(sd), beta_sigma)
        dataset_stats[d][f"{stat_prefix}_ema_abs_mean"] = _ema_update(prev_abs_mean, float(abs_mean), beta_mu)
        dataset_stats[d][f"{stat_prefix}_ema_frac_pos"] = _ema_update(prev_frac_pos, float(frac_pos), beta_mu)
        dataset_stats[d][f"{stat_prefix}_ema_frac_neg"] = _ema_update(prev_frac_neg, float(frac_neg), beta_mu)
        dataset_stats[d][f"{stat_prefix}_ema_skewness"] = _ema_update(prev_skewness, float(skew), beta_mu)
        dataset_stats[d][f"{stat_prefix}_ema_p10"] = _ema_update(prev_p10, percentiles["p10"], beta_mu)
        dataset_stats[d][f"{stat_prefix}_ema_p50"] = _ema_update(prev_p50, percentiles["p50"], beta_mu)
        dataset_stats[d][f"{stat_prefix}_ema_p90"] = _ema_update(prev_p90, percentiles["p90"], beta_mu)


def update_advantage_stats(
    q2_advantages: Dict[Any, List[float]],   # per-qid value lists
    q2tasks: Dict[Any, Any],              # qid -> task_id mapping
    q2datasets: Dict[Any, Any],           # qid -> dataset_id mapping
    *,
    stat_prefix: str,                     # e.g., "post_grpo_advantage" or "final_advantage"
    beta_mu: float,
    beta_sigma: float,
    eps: float,
) -> None:
    """
    Track advantage statistics from per-qid value dictionaries.

    This is a general function that can be used to track advantages at any stage:
        - Post-GRPO (after group normalization)
        - Final advantages (after all transformations)
        - Or any other intermediate stage

    Computes and logs:
        - Batch & EMA mean/std
        - Absolute mean
        - Fraction positive/negative
        - Skewness
        - Percentiles (p10, p50, p90)

    Args:
        q2_values:   per-qid value lists (e.g., q2rollouts, q2_final)
        q2tasks:     qid -> task_id mapping
        q2datasets:  qid -> dataset_id mapping
        stat_prefix: prefix for stat keys (e.g., "post_grpo_advantage", "final_advantage")
        beta_mu:     EMA decay for mean
        beta_sigma:  EMA decay for std
        eps:         small epsilon for numerical stability
    """
    # Collect values per task / dataset
    task_to_vals: Dict[Any, List[float]] = defaultdict(list)
    dataset_to_vals: Dict[Any, List[float]] = defaultdict(list)

    # CHECK IF STATS_PREFIX IN TASK STATS
    check_stat_prefix = stat_prefix + "_ema_mu"

    if not task_stats or not dataset_stats:
        raise ValueError(f"task_stats or dataset_stats is empty")

    sample_task = next(iter(task_stats.values()))
    sample_dataset = next(iter(dataset_stats.values()))

    if check_stat_prefix not in sample_task or check_stat_prefix not in sample_dataset:
        raise KeyError(f"Required stat '{check_stat_prefix}' not found in tracking stats")

    for qid, vals in q2_advantages.items():
        t = q2tasks[qid]
        d = q2datasets[qid]
        task_to_vals[t].extend(vals)
        dataset_to_vals[d].extend(vals)

    # Use general update function
    _update_advantage_stats_general(
        task_to_vals=task_to_vals,
        dataset_to_vals=dataset_to_vals,
        stat_prefix=stat_prefix,
        beta_mu=beta_mu,
        beta_sigma=beta_sigma,
        eps=eps,
    )