# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

import math
from collections import defaultdict, deque
from enum import Enum
from sklearn.cluster import KMeans
from typing import Any, Callable, Optional, Dict, List, Tuple, Set

import numpy as np
import torch
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | ActorConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    DRPO = "drpo"
    TARPO = "tarpo"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


EPS_DEFAULT: float = 1e-6

# Per‑domain question history ------------------------------------------------ #
#   domain_qstats[dom] = {
#       "vectors": List[np.ndarray]   # shape = (Q, R)
#       "q_ids":   List[int],        # question ids in same order as vectors
#       "count":   int,              # #questions accumulated so far
#   }
# --------------------------------------------------------------------------- #
domain_qstats: Dict[Any, Dict[str, Any]] = defaultdict(lambda: {
    "vectors": [],
    "q_ids":   [],
    "count":   0,
})

global_running_stats: Dict[str, int] = {"q_count": 0}

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #

def _select_k_elbow(vals: np.ndarray, k_max: int = 10, tol: float = 0.10) -> int:
    """k‑means elbow pick on multi‑dimensional points."""
    unique_cnt = len(np.unique(vals, axis=0))
    k_cap      = min(k_max, unique_cnt)
    ks         = range(1, k_cap + 1)
    inertias   = [KMeans(n_clusters=k, n_init="auto", random_state=0).fit(vals).inertia_ for k in ks]
    if len(inertias) == 1:
        return 1
    drops = np.diff(inertias) * -1.0
    for i in range(1, len(drops)):
        if drops[i] < tol * drops[i - 1]:
            return i + 1
    return ks[-1]


def _cluster_info_question(vectors: List[np.ndarray]) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """K‑means on question‑level vectors.

    Returns
    -------
    mu_d        : float   – inverse‑cluster‑size weighted mean of the centroid means
    assignments : (Q,)    – cluster index for each question vector
    counts      : (k,)    – cluster sizes
    centroids   : (k,R)   – cluster centroid vectors
    """
    if len(vectors) == 0:
        return 0.0, np.empty(0, int), np.empty(0), np.empty((0, 0))

    X = np.stack(vectors, axis=0)            # (Q,R) – R inferred from data
    k_opt = _select_k_elbow(X, k_max=20)
    km    = KMeans(n_clusters=k_opt, n_init="auto", random_state=0).fit(X)

    centroids   = km.cluster_centers_        # (k,R)
    assignments = km.labels_                 # (Q,)
    _, counts   = np.unique(assignments, return_counts=True)
    counts      = counts.astype(float)

    centroid_means = centroids.mean(axis=1)  # (k,)
    weights        = 1.0 / counts
    mu_d           = float((weights * centroid_means).sum() / weights.sum())

    # Debug ------------------------------------------------------------- #
    print(
        f"[KMEANS‑Q] k={k_opt} | centroid_means="
        f"[{', '.join(f'{m:.3f}' for m in centroid_means)}] | counts={counts.tolist()} | μ_d={mu_d:.3f}"
    )

    return mu_d, assignments, counts, centroids


@register_adv_est(AdvantageEstimator.DRPO)
def compute_drpo_outcome_advantage(
    token_level_rewards: torch.Tensor,      # (B,L)
    response_mask:      torch.Tensor,       # (B,L)
    index:              np.ndarray[str],         # (B,) question ids
    domain_info: np.ndarray,  # (B,) domain ids
    epsilon: float = EPS_DEFAULT,
):
    """DRPO with question‑level clustering."""

    B, L = token_level_rewards.shape

    # 1) raw rollout‑level rewards -------------------------------------- #
    raw_scores = token_level_rewards.sum(dim=-1)                          # (B,)

    # 2) collect rollouts per question for this mini‑batch -------------- #
    q2rollouts: Dict[str, List[float]] = defaultdict(list)
    q2domain:   Dict[str, Any]         = {}
    for i in range(B):
        qid: str = index[i]
        q2rollouts[qid].append(raw_scores[i].item())
        q2domain[qid] = domain_info[i]

    # ensure consistent rollout count ----------------------------------- #
    rollout_lens = {len(v) for v in q2rollouts.values()}
    assert len(rollout_lens) == 1, "Inconsistent rollout counts per question in batch!"

    # build vector per question ----------------------------------------- #
    q_vectors = {qid: np.asarray(v, dtype=np.float32) for qid, v in q2rollouts.items()}

    # 3) update per‑domain question history ----------------------------- #
    for qid, vec in q_vectors.items():
        dom = q2domain[qid]
        dstat = domain_qstats[dom]
        dstat["vectors"].append(vec)
        dstat["q_ids"].append(qid)
        dstat["count"] += 1
        global_running_stats["q_count"] += 1

    # 4) GRPO normalisation (within‑question) --------------------------- #
    scores = raw_scores.clone()
    id2mean = {qid: torch.mean(torch.tensor(v)) for qid, v in q2rollouts.items()}
    id2std  = {qid: torch.std (torch.tensor(v)) for qid, v in q2rollouts.items()}
    for i in range(B):
        qid: str = index[i]
        scores[i] = (scores[i] - id2mean[qid]) / (id2std[qid] + epsilon)
    before_scale_score = scores.clone()

    # 5) Domain‑wise question clustering -------------------------------- #
    domain_cluster_cache: Dict[Any, Dict[str, Any]] = {}
    for dom, dstat in domain_qstats.items():
        if dstat["count"] == 0:
            continue
        mu_d, assign, counts, centroids = _cluster_info_question(dstat["vectors"])
        domain_cluster_cache[dom] = {
            "mu_d":      mu_d,
            "assign":    assign,
            "counts":    counts,
            "centroids": centroids,
            "q_ids":     dstat["q_ids"],
        }

    # 6) Apply scaling --------------------------------------------------- #
    scaling_factors: List[float] = []
    for i in range(B):
        qid: str  = index[i]
        dom  = q2domain[qid]
        cache = domain_cluster_cache[dom]

        # map qid → cluster idx ---------------------------------------- #
        q_idx       = cache["q_ids"].index(qid)
        cluster_idx = cache["assign"][q_idx]

        N_d  = float(domain_qstats[dom]["count"])
        mu_d = cache["mu_d"]
        T_d  = max(math.sqrt(N_d) * mu_d, epsilon)

        N_c  = float(cache["counts"][cluster_idx])
        mu_c = float(cache["centroids"][cluster_idx].mean())

        factor = T_d * math.sqrt(N_c) * mu_c
        scaling_factors.append(factor)
        scores[i] = scores[i] / factor

    # divide scores by std of scores
    scores_std = torch.std(scores)
    scores = scores / (scores_std + epsilon)

    # Debug report -------------------------------------------------------- #
    print("--------------Hierarchical scaling report--------------")
    dom2scale: Dict[Any, List[torch.Tensor]] = defaultdict(list)
    for i in range(B):
        dom2scale[domain_info[i]].append(scores[i] / (before_scale_score[i] + epsilon))
    for dom, lst in dom2scale.items():
        avg_sf = torch.mean(torch.stack(lst)).item()
        print(f"[HDRPO] domain = {dom:<15} | mean overall scale = {avg_sf:6.3f}")

    # Print global reward mean
    print(f"[HDRPO] global reward mean = {torch.mean(scores):.3f}")

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO（outcome-only）:
      For each group g:
      a_i = \\frac{r_i - \\mu_g}{\\sigma_g} (or without dividing by \\sigma_g),
      then broadcast the scalar across the token dimension (multiplied by response_mask).。
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (AlgoConfig) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                len_tensor = torch.stack(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config: Optional[AlgoConfig] = None, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.GPG)  # or simply: @register_adv_est("gpg")
def compute_gpg_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    f_norm: float = 1.0,
    alpha: float = 1.0,
    config=None,
    **kwargs,
):
    """
    Compute advantage for GPG, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            shape: (bs,)
        epsilon: (float)
        f_norm: (float)
        alpha: (float)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = bsz / m.clamp(min=1)

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = alpha * (scores[i] - id2mean[index[i]]) / (f_norm)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO_VECTORIZED)  # or simply: @register_adv_est("rloo_vectorized")
def compute_rloo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(scores.device)

        c = torch.bincount(inv)[inv].to(scores.dtype)
        adv = ((c * scores - torch.bincount(inv, weights=scores)[inv]) / (c - 1).clamp_min(1)) * (c > 1)

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
):
    """
    Aggregate the loss across global batch to ensure the loss is invariant to fsdp/megatron parallelism.

    NOTE: ``dp_size``, ``batch_num_tokens``, and ``global_batch_size`` are only compatible with the new model engine
        for now, while the legacy model engines conduct the aggregation outside ``agg_loss``.

    NOTE: The returned loss has different behaviors for different backend:
    - FSDP: the loss is directly used for backward.
    - Megatron: the loss should be scaled by `num_microbatches` and `cp_size` for pp schedule.

    # TODO: Consider the numerical stability?

    Args:
        loss_mat: micro batch loss matrix, (bs, response_length)
        loss_mask: micro batch loss mask, (bs, response_length)
        loss_agg_mode: method to aggregate the loss matrix into a scalar
        dp_size: data parallel size. When appling manual aggregation,
            scaling up the ``loss`` by ``dp_size`` can cancel out FSDP averaging.
        batch_num_tokens: number of valid tokens in global batch
        global_batch_size: global batch size
        loss_scale_factor: scale factor for "seq-mean-token-sum-norm" mode. If None, uses loss_mask.shape[-1].
            Set this to a constant value to ensure consistent normalization throughout training.

    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    # NOTE: `masked_sum` is more robust than multiplying the `mask`.
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            batch_num_tokens = loss_mask.sum()
        loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    elif loss_agg_mode.startswith("seq-mean"):
        # TODO: Correct and unify the denominator logic.
        if global_batch_size is not None:
            seq_denominator = global_batch_size * dp_size
        else:  # The default logic which is only correct when the batch sizes are even.
            local_bsz = loss_mat.shape[0]
            seq_denominator = local_bsz

        if loss_agg_mode.startswith("seq-mean-token-sum"):
            seq_losses = verl_F.masked_sum(loss_mat, loss_mask, axis=-1)  # token-sum per sequence

            if loss_agg_mode == "seq-mean-token-sum":
                pass  # TODO: Add assertation.
            elif loss_agg_mode == "seq-mean-token-sum-norm":
                if loss_scale_factor is None:
                    loss_scale_factor = loss_mask.shape[-1]
                seq_losses = seq_losses / loss_scale_factor
            else:
                raise ValueError(f"Invalid {loss_agg_mode=}")
        elif loss_agg_mode == "seq-mean-token-mean":
            token_counts = torch.sum(loss_mask, dim=-1)  # per-sequence token count
            # token-mean per sequence
            seq_losses = verl_F.masked_sum(loss_mat, loss_mask, axis=-1) / (token_counts + 1e-8)
        else:
            raise ValueError(f"Invalid {loss_agg_mode=}")
        loss = torch.sum(seq_losses) / seq_denominator  # seq-mean
    else:
        raise ValueError(f"Invalid {loss_agg_mode=}")

    return loss


@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("vanilla")  # type: ignore[arg-type]
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Adapted from
    https://github.com/AMAP-ML/GPG/blob/main/VisualThinker-R1-Zero/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L495
    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    return:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GPG
    """
    assert config is not None
    pg_losses = -log_prob * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    return pg_loss, {}


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/ppo_kl": ppo_kl_abs.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GMPO.

    Adapted from paper https://arxiv.org/abs/2507.20673
    https://github.com/callsys/GMPO/blob/main/train_zero_math_gmpo.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            not used
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability (uncomment it if you like)
    # negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Clipping at token-level & Clipping wider
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    # we only support sequence level advantage for now,
    # otherwise, below would be not consistent with the paper
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio

    # Apply rollout correction weights if provided
    # For geo_mean, IS weights are 2D (batch_size, seq_length) and need to be aggregated to sequence level
    if rollout_is_weights is not None:
        # Aggregate token-level weights to sequence level using geometric mean for consistency
        # Note: rollout_is_weights is always 2D regardless of aggregation mode
        seq_is_weights = torch.exp(
            (torch.log(rollout_is_weights + 1e-10) * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
        )
        pg_losses = pg_losses * seq_is_weights

    pg_loss = torch.mean(pg_losses)

    # higher: ratio is too large that need clamp to clip_high (when adv > 0)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped * (advantages < 0)).float(), response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expectaed value of KL, but the expected gradient of k1 and k3
    estimator is not the expectaed gradient of KL. On the other hand k2 estimator gives right gradient estimator, 
    so we use a straight through trick here if the kl_penalty method ends with '+', .e.g., k3+. 
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data

def compute_policy_loss_reinforce(
    rollout_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-sum",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute REINFORCE-style policy gradient loss with optional IS correction.

    This function implements policy gradient (REINFORCE) with optional importance
    sampling correction for rollout-training policy mismatch.

    Mathematical formulation:
        Without IS (rollout_is_weights=None):
            L = -E[log π(a|s) * A(s,a)]
            Gradient: ∇_θ L = -E[∇log π(a|s) * A] (standard REINFORCE)

        With IS (rollout_is_weights provided):
            L = -E_π_rollout[w * log π(a|s) * A(s,a)]
            where w = π_current / π_rollout (truncated IS weight)
            Gradient: ∇_θ L = -E[w * ∇log π(a|s) * A] (IS-corrected policy gradient)

    Args:
        rollout_log_prob: Log probabilities from rollout policy (e.g., vLLM BF16).
            Shape: (batch_size, seq_length). Used for KL computation.
        log_prob: Log probabilities from current training policy.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates for each token.
            Shape: (batch_size, seq_length)
        response_mask: Mask indicating valid tokens (1 for valid, 0 for padding).
            Shape: (batch_size, seq_length). Should already include rejection sampling.
        loss_agg_mode: Loss aggregation strategy (see agg_loss for details).
        config: Actor config (required for global_batch_info).
        rollout_is_weights: Pre-computed IS weights (π_current / π_rollout).
            Shape: (batch_size, seq_length). None to disable IS correction.

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy gradient loss
            metrics: Dictionary with "actor/ppo_kl"

    Note:
        Unlike PPO (compute_policy_loss_vanilla), this function:
        - Does NOT use PPO clipping
        - Uses log π(a|s) directly (not ratio)
        - IS weights are applied as multiplicative factor
    """
    assert config is not None, "ActorConfig must be provided for REINFORCE loss"

    # Compute pure policy gradient loss with optional IS correction
    # Standard REINFORCE: L = -E[log π(a|s) * A]
    # With IS: L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
    if rollout_is_weights is not None:
        # IS-corrected policy gradient: L = -E[stopgrad(w) · log π · A]
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        # Standard REINFORCE: L = -E[log π · A]
        pg_losses = -advantages * log_prob

    # Aggregate loss
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    # Compute KL divergence between current and rollout policy
    negative_approx_kl = log_prob - rollout_log_prob
    kl_divergence = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "actor/ppo_kl": kl_divergence.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("bypass_mode")
def compute_policy_loss_bypass_mode(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Bypass mode policy loss supporting both REINFORCE and PPO-clip.

    This function is the entry point for bypass mode, where old_log_prob = rollout_log_prob.
    It computes IS weights and rejection masks, then dispatches to either REINFORCE or
    PPO-clip loss based on the loss_type configuration.

    IMPORTANT - Bypass mode semantics:
        In bypass mode, the trainer sets old_log_prob = rollout_log_prob.
        This means:
        - For REINFORCE: We use IS weights w = π_current / π_rollout explicitly
        - For PPO-clip: The PPO ratio π_current / π_old = π_current / π_rollout
          already incorporates the IS correction through clipping, so we do NOT
          apply additional IS weights (would be double-counting)

    Loss types:
        - "ppo_clip" (default): PPO clipped objective (compute_policy_loss_vanilla)
            L = -E[min(r*A, clip(r)*A)] where r = π_current / π_rollout
            Note: IS weights are NOT applied (clipping handles the ratio)
        - "reinforce": REINFORCE-style policy gradient with IS correction
            L = -E[w * log π(a|s) * A] where w = π_current / π_rollout

    Args:
        old_log_prob: In bypass mode, this is actually rollout_log_prob.
            Shape: (batch_size, seq_length)
        log_prob: Current policy log probabilities.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates.
            Shape: (batch_size, seq_length)
        response_mask: Valid token mask (1=valid, 0=padding).
            Shape: (batch_size, seq_length)
        loss_agg_mode: Loss aggregation mode (passed to underlying loss function).
        config: Actor config containing rollout_correction settings in policy_loss.
        rollout_is_weights: Pre-computed IS weights (ignored, computed internally).

    Config options (in config.policy_loss.rollout_correction):
        loss_type: "ppo_clip" (default) or "reinforce"
        rollout_is: IS aggregation level ("token", "sequence", or None)
        rollout_is_threshold: Upper threshold for truncating IS weights (default: 2.0)
        rollout_rs: Rejection sampling level ("token", "sequence", "geometric", or None)
        rollout_rs_threshold: Upper threshold for rejection sampling
        rollout_rs_threshold_lower: Lower threshold for rejection sampling
        rollout_token_veto_threshold: Per-token veto threshold for catastrophic outliers
        rollout_is_batch_normalize: Whether to normalize IS weights to mean=1.0

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy loss
            metrics: Dictionary with rollout correction metrics and actor/ppo_kl
    """
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

    assert config is not None, "config is required for bypass_mode loss"

    # Extract rollout_correction config from policy_loss
    rollout_corr_config = config.policy_loss.get("rollout_correction", None) if hasattr(config, "policy_loss") else None

    if rollout_corr_config is None:
        raise ValueError(
            "rollout_correction config not found in policy_loss. "
            "When using loss_mode='bypass_mode', ensure rollout_correction config is passed."
        )

    # Extract parameters
    loss_type = rollout_corr_config.get("loss_type", "ppo_clip")
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)
    rollout_rs_threshold_lower = rollout_corr_config.get("rollout_rs_threshold_lower", None)
    rollout_token_veto_threshold = rollout_corr_config.get("rollout_token_veto_threshold", None)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)

    # In bypass mode: old_log_prob IS rollout_log_prob
    rollout_log_prob = old_log_prob

    # Compute IS weights and rejection mask
    # Note: For PPO-clip, we still compute IS weights for metrics, but don't apply them
    with torch.no_grad():
        rollout_is_weights_proto, modified_response_mask, rollout_metrics = (
            compute_rollout_correction_and_rejection_mask(
                old_log_prob=log_prob,  # Current policy (for IS ratio: π_current / π_rollout)
                rollout_log_prob=rollout_log_prob,  # Rollout policy
                response_mask=response_mask,
                rollout_is=rollout_is,
                rollout_is_threshold=rollout_is_threshold,
                rollout_rs=rollout_rs,
                rollout_rs_threshold=rollout_rs_threshold,
                rollout_rs_threshold_lower=rollout_rs_threshold_lower,
                rollout_token_veto_threshold=rollout_token_veto_threshold,
                rollout_is_batch_normalize=rollout_is_batch_normalize,
            )
        )

    # Extract IS weights tensor (or None if disabled)
    computed_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"] if rollout_is_weights_proto else None

    # Apply rejection mask (RS + veto)
    effective_mask = modified_response_mask

    # Dispatch to appropriate loss function based on loss_type
    if loss_type == "reinforce":
        # REINFORCE: Apply IS weights explicitly
        pg_loss, pg_metrics = compute_policy_loss_reinforce(
            rollout_log_prob=rollout_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=computed_is_weights,
        )

    elif loss_type == "ppo_clip":
        # PPO-clip: The ratio π_current/π_old = π_current/π_rollout already handles IS
        # DO NOT apply IS weights - would be double-counting!
        # The clipping mechanism constrains the effective IS ratio
        pg_loss, pg_metrics = compute_policy_loss_vanilla(  # type: ignore[call-arg]
            old_log_prob=rollout_log_prob,  # = old_log_prob in bypass mode
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=None,  # Explicitly None - no IS weights for PPO-clip
        )

    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'reinforce' or 'ppo_clip'.")

    # Merge rollout correction metrics
    pg_metrics.update(rollout_metrics)

    return pg_loss, pg_metrics


def compute_policy_loss_reinforce(
    rollout_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-sum",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute REINFORCE-style policy gradient loss with optional IS correction.

    This function implements policy gradient (REINFORCE) with optional importance
    sampling correction for rollout-training policy mismatch.

    Mathematical formulation:
        Without IS (rollout_is_weights=None):
            L = -E[log π(a|s) * A(s,a)]
            Gradient: ∇_θ L = -E[∇log π(a|s) * A] (standard REINFORCE)

        With IS (rollout_is_weights provided):
            L = -E_π_rollout[w * log π(a|s) * A(s,a)]
            where w = π_current / π_rollout (truncated IS weight)
            Gradient: ∇_θ L = -E[w * ∇log π(a|s) * A] (IS-corrected policy gradient)

    Args:
        rollout_log_prob: Log probabilities from rollout policy (e.g., vLLM BF16).
            Shape: (batch_size, seq_length). Used for KL computation.
        log_prob: Log probabilities from current training policy.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates for each token.
            Shape: (batch_size, seq_length)
        response_mask: Mask indicating valid tokens (1 for valid, 0 for padding).
            Shape: (batch_size, seq_length). Should already include rejection sampling.
        loss_agg_mode: Loss aggregation strategy (see agg_loss for details).
        config: Actor config (required for global_batch_info).
        rollout_is_weights: Pre-computed IS weights (π_current / π_rollout).
            Shape: (batch_size, seq_length). None to disable IS correction.

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy gradient loss
            metrics: Dictionary with "actor/ppo_kl"

    Note:
        Unlike PPO (compute_policy_loss_vanilla), this function:
        - Does NOT use PPO clipping
        - Uses log π(a|s) directly (not ratio)
        - IS weights are applied as multiplicative factor
    """
    assert config is not None, "ActorConfig must be provided for REINFORCE loss"

    # Compute pure policy gradient loss with optional IS correction
    # Standard REINFORCE: L = -E[log π(a|s) * A]
    # With IS: L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
    if rollout_is_weights is not None:
        # IS-corrected policy gradient: L = -E[stopgrad(w) · log π · A]
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        # Standard REINFORCE: L = -E[log π · A]
        pg_losses = -advantages * log_prob

    # Aggregate loss
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    # Compute KL divergence between current and rollout policy
    negative_approx_kl = log_prob - rollout_log_prob
    kl_divergence = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "actor/ppo_kl": kl_divergence.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("bypass_mode")
def compute_policy_loss_bypass_mode(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Bypass mode policy loss supporting both REINFORCE and PPO-clip.

    This function is the entry point for bypass mode, where old_log_prob = rollout_log_prob.
    It computes IS weights and rejection masks, then dispatches to either REINFORCE or
    PPO-clip loss based on the loss_type configuration.

    IMPORTANT - Bypass mode semantics:
        In bypass mode, the trainer sets old_log_prob = rollout_log_prob.
        This means:
        - For REINFORCE: We use IS weights w = π_current / π_rollout explicitly
        - For PPO-clip: The PPO ratio π_current / π_old = π_current / π_rollout
          already incorporates the IS correction through clipping, so we do NOT
          apply additional IS weights (would be double-counting)

    Loss types:
        - "ppo_clip" (default): PPO clipped objective (compute_policy_loss_vanilla)
            L = -E[min(r*A, clip(r)*A)] where r = π_current / π_rollout
            Note: IS weights are NOT applied (clipping handles the ratio)
        - "reinforce": REINFORCE-style policy gradient with IS correction
            L = -E[w * log π(a|s) * A] where w = π_current / π_rollout

    Args:
        old_log_prob: In bypass mode, this is actually rollout_log_prob.
            Shape: (batch_size, seq_length)
        log_prob: Current policy log probabilities.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates.
            Shape: (batch_size, seq_length)
        response_mask: Valid token mask (1=valid, 0=padding).
            Shape: (batch_size, seq_length)
        loss_agg_mode: Loss aggregation mode (passed to underlying loss function).
        config: Actor config containing rollout_correction settings in policy_loss.
        rollout_is_weights: Pre-computed IS weights (ignored, computed internally).

    Config options (in config.policy_loss.rollout_correction):
        loss_type: "ppo_clip" (default) or "reinforce"
        rollout_is: IS aggregation level ("token", "sequence", or None)
        rollout_is_threshold: Upper threshold for truncating IS weights (default: 2.0)
        rollout_rs: Rejection sampling level ("token", "sequence", "geometric", or None)
        rollout_rs_threshold: Upper threshold for rejection sampling
        rollout_rs_threshold_lower: Lower threshold for rejection sampling
        rollout_token_veto_threshold: Per-token veto threshold for catastrophic outliers
        rollout_is_batch_normalize: Whether to normalize IS weights to mean=1.0

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy loss
            metrics: Dictionary with rollout correction metrics and actor/ppo_kl
    """
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

    assert config is not None, "config is required for bypass_mode loss"

    # Extract rollout_correction config from policy_loss
    rollout_corr_config = config.policy_loss.get("rollout_correction", None) if hasattr(config, "policy_loss") else None

    if rollout_corr_config is None:
        raise ValueError(
            "rollout_correction config not found in policy_loss. "
            "When using loss_mode='bypass_mode', ensure rollout_correction config is passed."
        )

    # Extract parameters
    loss_type = rollout_corr_config.get("loss_type", "ppo_clip")
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)
    rollout_rs_threshold_lower = rollout_corr_config.get("rollout_rs_threshold_lower", None)
    rollout_token_veto_threshold = rollout_corr_config.get("rollout_token_veto_threshold", None)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)

    # In bypass mode: old_log_prob IS rollout_log_prob
    rollout_log_prob = old_log_prob

    # Compute IS weights and rejection mask
    # Note: For PPO-clip, we still compute IS weights for metrics, but don't apply them
    with torch.no_grad():
        rollout_is_weights_proto, modified_response_mask, rollout_metrics = (
            compute_rollout_correction_and_rejection_mask(
                old_log_prob=log_prob,  # Current policy (for IS ratio: π_current / π_rollout)
                rollout_log_prob=rollout_log_prob,  # Rollout policy
                response_mask=response_mask,
                rollout_is=rollout_is,
                rollout_is_threshold=rollout_is_threshold,
                rollout_rs=rollout_rs,
                rollout_rs_threshold=rollout_rs_threshold,
                rollout_rs_threshold_lower=rollout_rs_threshold_lower,
                rollout_token_veto_threshold=rollout_token_veto_threshold,
                rollout_is_batch_normalize=rollout_is_batch_normalize,
            )
        )

    # Extract IS weights tensor (or None if disabled)
    computed_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"] if rollout_is_weights_proto else None

    # Apply rejection mask (RS + veto)
    effective_mask = modified_response_mask

    # Dispatch to appropriate loss function based on loss_type
    if loss_type == "reinforce":
        # REINFORCE: Apply IS weights explicitly
        pg_loss, pg_metrics = compute_policy_loss_reinforce(
            rollout_log_prob=rollout_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=computed_is_weights,
        )

    elif loss_type == "ppo_clip":
        # PPO-clip: The ratio π_current/π_old = π_current/π_rollout already handles IS
        # DO NOT apply IS weights - would be double-counting!
        # The clipping mechanism constrains the effective IS ratio
        pg_loss, pg_metrics = compute_policy_loss_vanilla(  # type: ignore[call-arg]
            old_log_prob=rollout_log_prob,  # = old_log_prob in bypass mode
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=None,  # Explicitly None - no IS weights for PPO-clip
        )

    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'reinforce' or 'ppo_clip'.")

    # Merge rollout correction metrics
    pg_metrics.update(rollout_metrics)

    return pg_loss, pg_metrics



#TODO_TARPO: (version to work on) make sure these function are present in the two core_algos files
############################################################################ TARPO #########################################################################

from .tarpo_utils import (
    task_stats,
    dataset_stats,
    _ema_update,                    # only used inside CVaR buffer EMA
    build_mappings,
    update_raw_stats,    # U1
    update_k_stats_from_q2k,        # U2
    update_advantage_stats,
    store_advantage_data,
    get_latest_advantage_data,
    save_advantages_to_json
)

@register_adv_est(AdvantageEstimator.TARPO)
def compute_tarpo_outcome_advantage(
    token_level_rewards: torch.Tensor,   # (B, L), where L is the response_length
    response_mask:      torch.Tensor,    # (B, L)
    index:              torch.Tensor,    # (B,) question /prompt ids (unused except parity with others)
    task_ids:           List[Any],       # (B,) task id per sample
    dataset_ids:        List[Any],       # (B,) dataset id per sample
    class_labels:       List[Any],       # (B,) class label per sample (for weighting)
    *,
    # --- Ablation toggles ---
    use_task_adapter:   bool = False,
    use_task_mixture_adapter: bool = False, 
    use_class_weights:  bool = False,
    use_cvar_boost:     bool = False,
    use_grpo_group_norm: bool = True,
    use_task_mixture_density_adapter: bool = True,
    # --- Hyperparameters ---
    eps: float = EPS_DEFAULT,
    alpha: float = 0.2,                  # CVaR tail fraction
    lambda_risk: float = 0.3,            # blend strength for dynamic tail boost
    # EMA decays
    beta_mu: float = 0.99,
    beta_sigma: float = 0.99,
    beta_mean: float = 0.98,    # per task buffer EMA for CVaR's mean references (how much of the prev to keep etc.)
    beta_cvar: float = 0.98,    # per task buffer EMA for CVaR
    beta_tail: float = 1.0,     # how strongly the frequency of tail events modulates CVaR boost
    # Static metadata for class weights
    class_count_info: Optional[Dict[Any, Dict[Any, int]]] = None,  # {dataset: {class: count}}
    class_weight_scope: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TARPO advantage/return computation (no per-prompt batch norm).
    Pipeline (modular; combine once at the end):
        A) Build q2rollouts/q2tasks/q2datasets/q2class and per-task/dataset stats
        B) (Optional) Global min-max scaling across all rollouts
        C) (Optional) GRPO-style intra-question normalization (zero-mean/unit-variance per qid)
        D) Task-adapter centering -> q2norm[qid][k]
        E) Static inverse-frequency class weight -> q2w[qid] (scalar)
        F) CVaR dynamic tail boost -> q2k[qid] (via per-task buffer)
        G) Combine into final scalar advantages, log stats, and broadcast
    """

    device = token_level_rewards.device
    B, L   = token_level_rewards.shape

    # 1) Rollout-level scalar raw rewards
    # Token_level rewards is essentially the rewards for each token in the response
    # here, we sum them to get the total reward for the entire response
    raw_scores = token_level_rewards.sum(dim=-1)  # (B,)

    # So, essentially, we assume that the scores 
    # correspond to the batch samples in order (B,), where B is 
    # the total number of rollouts in the mini-batch, (i.e. 5 for the same question)
    # if there is 2 questions, then B = 10 (5 rollouts each)
    # and this is the same order as task_ids, dataset_ids, class_labels,
    # qid is basically the question id that each rollout corresponds to
    # i.e. index = [qid_1, qid_1, qid_1] (for a rollout of 3)
    # note that there may be more than one question in the mini-batch
    # which looks like [qid_1, qid_1, qid_1, qid_2, qid_2, qid_2] 
    # (for 2 questions, each with 3 rollouts)
    # we use qid to essentially delineate the different training examples
    use_minmax_scaling: bool = True,    # Global min-max scaling before GRPO
    (
        q2rollouts,
        q2tasks,
        q2datasets,
        q2class,
        task_to_rollouts,
        dataset_to_rollouts,
    ) = build_mappings(
        raw_scores=raw_scores,
        index=index,
        task_ids=task_ids,
        dataset_ids=dataset_ids,
        class_labels=class_labels,
        use_minmax_scaling=use_minmax_scaling,
        eps=eps,
    )

    # NOTE: the returns at the end of this function are essentially the 
    # normalized values of the scores, which are then broadcasted to token-level
    # i.e. for each token in the response, it will now have the same normalized 
    # score as the entire response
    # [1.2], [0.5] --> [1.2, 1.2, 1.2], [0.5, 0.5, 0.5] (for response length of 3 for both responses)

    # -------------------------------------------------------------------------
    # U1) Update per-task and per-dataset stats
    #     - raw_batch_mu/raw_batch_sigma/raw_batch_count: non-EMA, for logging
    #     - ema_mu/ema_sigma/ema_count: EMA (used by task adapter)
    #
    # Note: These stats are updated AFTER optional min-max scaling (which
    # happens in build_mappings), so if use_minmax_scaling=True, these stats
    # reflect the scaled values.
    # -------------------------------------------------------------------------
    update_raw_stats(
        task_to_rollouts=task_to_rollouts,
        dataset_to_rollouts=dataset_to_rollouts,
        beta_mu=beta_mu,
        beta_sigma=beta_sigma,
        eps=eps,
    )

    # -------------------------------------------------------------------------
    # C) GRPO-style intra-question normalization (zero-mean/unit-variance)
    #
    # This step operates on q2rollouts (per-qid rollout lists) and is applied
    # AFTER optional min-max scaling (in build_mappings) and BEFORE task
    # adapter, class weighting, and CVaR scaling so that those subsequent
    # scalings are not neutralized.
    #
    # Standard GRPO normalization: (v - mean) / std per question
    # -------------------------------------------------------------------------
    if use_grpo_group_norm:
        for qid, vals in q2rollouts.items():
            if len(vals) < 2:
                # With a single rollout, GRPO normalization is ill-defined; keep as is.
                continue

            # Original GRPO-style group normalization: (v - mean) / std
            mu = float(np.mean(vals))
            sd = float(np.std(vals, ddof=0))
            sd = max(sd, eps)
            q2rollouts[qid] = [(v - mu) / sd for v in vals]

        # ---------------------------------------------------------------------
        # U2.5) Track POST-GRPO advantages per task and per dataset
        #       (after GRPO group normalization, before task adapter)
        # ---------------------------------------------------------------------
        update_advantage_stats(
            q2_advantages=q2rollouts,
            q2tasks=q2tasks,
            q2datasets=q2datasets,
            stat_prefix="post_grpo_advantage",
            beta_mu=beta_mu,
            beta_sigma=beta_sigma,
            eps=eps,
        )

        # Store post-GRPO advantages for saving to JSON
        store_advantage_data(
            advantage_type="post_grpo",
            q2_advantages=q2rollouts,
            q2tasks=q2tasks,
            q2datasets=q2datasets
        )

    # -------------------------------------------------------------------------
    # D) Task-adapter centering
    #
    # Previously: (v - ema_mu) / ema_sigma
    # Now:        (v / ema_mu) ONLY (no division by ema_sigma).
    # -------------------------------------------------------------------------
    if use_task_adapter:
        # # Center each qid’s rollouts with its task’s EMA mean (produce q2norm)
        # q2norm: Dict[Any, List[float]] = {}
        # for qid, vals in q2rollouts.items():
        #     task = q2tasks[qid]
        #     mu_t = float(task_stats[task]["ema_mu"])
        #     # Mean-only centering; no variance scaling.
        #     # q2norm[qid] = [(v - mu_t) for v in vals]
        #     # Rough Mean scaling (to downweight high-performing tasks)
        #     q2norm[qid] = [(v / mu_t) for v in vals]

        # 1) Compute global reference mean and sigma across tasks
        task_mus = [float(stats["ema_mu"]) for stats in task_stats.values()]
        task_sigmas = [float(stats["post_grpo_advantage_ema_sigma"]) for stats in task_stats.values()]

        mu_ref = sum(task_mus) / max(len(task_mus), 1)
        sigma_ref = sum(task_sigmas) / max(len(task_sigmas), 1)

        # 2) Reasonable bounds so we don't explode or vanish
        MIN_SCALE = 0.3     # at most 3x downweight
        MAX_SCALE = 3.0      # at most 3x upweight
        EPS = 1e-6
        SCALE_COEFF = 2.0
        SIGMA_WEIGHT = 0.8   # 0 = pure μ scaling, 1 = pure σ scaling, 0.5 = balanced

        q2norm: Dict[Any, List[float]] = {}
        for qid, vals in q2rollouts.items():
            task = q2tasks[qid]
            mu_t = float(task_stats[task]["ema_mu"])
            sigma_t = float(task_stats[task]["post_grpo_advantage_ema_sigma"])

            # μ-based scaling: >1 if task underperforms, <1 if overperforms
            raw_mu_scale = mu_ref / max(mu_t, EPS)

            # σ-based scaling: >1 if "too quiet" (low variance), <1 if "too loud" (high variance)
            raw_sigma_scale = sigma_ref / max(sigma_t, EPS)

            # Blend: SIGMA_WEIGHT interpolates between mu-only (0) and sigma-only (1)
            # raw_scale = (1 - SIGMA_WEIGHT) * raw_mu_scale + SIGMA_WEIGHT * raw_sigma_scale
            # But multiplicative blend is more stable:
            raw_scale = ((raw_mu_scale ** (1 - SIGMA_WEIGHT)) * (raw_sigma_scale ** SIGMA_WEIGHT)) ** SCALE_COEFF
            
            # Clamp overall scale
            scale_t = max(MIN_SCALE, min(MAX_SCALE, raw_scale))

            # Track scaling factors for this task
            task_stats[task]["adapter_mu_scale"] = float(raw_mu_scale)
            task_stats[task]["adapter_sigma_scale"] = float(raw_sigma_scale)
            task_stats[task]["adapter_final_scale"] = float(scale_t)

            # Scale_t nudges tasks up/down
            q2norm[qid] = [(v) * scale_t for v in vals]

    else:
        q2norm = {qid: list(vals) for qid, vals in q2rollouts.items()}

    if use_task_mixture_adapter:
        # --- Mixture-based sparsity adapter (simple version) ---
        # For each task t:
        #   ρ_t ≈ post_grpo_advantage_ema_abs_mean / |post_grpo_advantage_ema_p90|
        # Then scale advantages by exp( log(ρ_ref) − log(ρ_t) ), clamped.

        task_to_density: Dict[Any, float] = {}
        densities: List[float] = []

        # 1) Compute ρ_t for every task from EMA stats
        for task, stats in task_stats.items():
            abs_mean_ema = float(stats.get("post_grpo_advantage_ema_abs_mean", 0.0))
            p90_ema      = float(stats.get("post_grpo_advantage_ema_p90", 0.0))

            tail_scale = max(abs(p90_ema), eps)

            # RHO_T is actually that of sparsity.
            # Greater rho_t greater sparsity.

            if tail_scale > 0.0:
                rho_t = abs_mean_ema / tail_scale
            else:
                rho_t = 0.0

            task_to_density[task] = rho_t
            densities.append(rho_t)

        # Assume we always have at least one task; use all densities to define reference.
        # 2) Global reference density (geometric mean)
        log_rhos    = [math.log(r + eps) for r in densities]
        log_rho_ref = sum(log_rhos) / len(log_rhos)
        rho_ref     = math.exp(log_rho_ref)  # mostly for logging / inspection

        # LOG_RHO_MAX = 2.0   # clamp log-ratio (~ up to ~7.4x)
        SCALE_COEFF = 1.0
        MIN_SCALE   = 0.25
        MAX_SCALE   = 8.0

        q2norm = {}
        for qid, vals in q2rollouts.items():
            task  = q2tasks[qid]

            rho_t = task_to_density.get(task, 0.0)
            rho_t = max(rho_t, eps)  # avoid log(0)

            log_rho_t = math.log(rho_t)
            # >0 ⇒ task is sparser (lower density) than reference ⇒ boost
            log_sparsity_ratio = log_rho_t - log_rho_ref

            log_scale = SCALE_COEFF * log_sparsity_ratio
            raw_scale = math.exp(log_scale)
            scale_t   = max(MIN_SCALE, min(MAX_SCALE, raw_scale))

            task_stats[task]["mixture_rho_t"]     = float(rho_t)
            task_stats[task]["mixture_rho_ref"]     = float(rho_ref)
            task_stats[task]["mixture_log_scale"]   = float(log_scale)
            task_stats[task]["mixture_final_scale"] = float(scale_t)

            q2norm[qid] = [v * scale_t for v in vals]

    else:
        # No task adapter at all
        q2norm = {qid: list(vals) for qid, vals in q2rollouts.items()}

    
    # -------------------------------------------------------------------------
    # D.5) Task mixture-density adapter (inter-task scaling only)
    #      - Uses soft responsibilities r(z) to estimate per-task "signal mass"
    #      - Uses STANDARD z-score in |A| space: z = (|A| - mu_abs) / (sigma_abs + eps)
    #      - rho_t = E[r * |A|] (count-normalized), EMA-smoothed
    #      - Scales by density-ratio + one-sided rarity boost, with EMA on log-mult
    # -------------------------------------------------------------------------
    if use_task_mixture_density_adapter:
        # ----------------------------
        # Hyperparams (tune these)
        # ----------------------------
        BETA_R      = 4.0     # responsibility sharpness
        DELTA_R     = 0.25    # z threshold (in standard z space)
        GAMMA       = 0.5     # density-ratio temperature (<1 helps avoid saturation)

        # Rarity boost (mixture-derived, one-sided)
        ETA_K       = 0.5     # rarity temperature
        K_MAX       = 1.5    # only boosts up to this (never downweights common tasks)

        # EMA smoothing
        BETA_RHO      = 0.95  # EMA for the per-task density (rho)
        BETA_LOGMULT  = 0.95  # EMA for the per-task log multiplier

        # Final scale clamp (applied directly to scale_t)
        MIN_SCALE   = 0.5
        MAX_SCALE   = 8.0

        # NEW: boost only if task is at least this many times rarer than reference
        RARE_RATIO  = 2.5    # e.g., 20x rarer than reference => "super rare only"

        # ----------------------------
        # Step 1 — compute per-task signal mass and effective signal count
        # ----------------------------
        task_signal_mass = defaultdict(float)  # sum_i r_i * |A_i|
        task_count       = defaultdict(int)    # raw sample count (for logging)
        task_sig_count   = defaultdict(float)  # sum_i r_i (effective signal count)

        q2rvalues = {}  # optional: store r values per qid

        for qid, vals in q2rollouts.items():
            task = q2tasks[qid]

            # STANDARD z-score stats in |A| space (EMA)
            mu_abs = float(task_stats[task].get("post_grpo_advantage_ema_abs_mean", 0.0))

            # Prefer abs-sigma if you tracked it; otherwise fall back to signed sigma.
            # Recommended: track post_grpo_advantage_ema_abs_sigma in your stats updater.
            sd_abs = float(task_stats[task].get("post_grpo_advantage_ema_abs_sigma", 0.0))
            if sd_abs <= 0.0:
                sd_abs = float(task_stats[task].get("post_grpo_advantage_ema_sigma", 1.0))
            sd_abs = max(sd_abs, eps)

            r_list = []
            for v in vals:
                a = abs(v)

                # standard z-score on |A|
                z = (a - mu_abs) / (sd_abs + eps)

                # soft responsibility: probability of being "signal"
                r = 1.0 / (1.0 + math.exp(-BETA_R * (z - DELTA_R)))

                task_signal_mass[task] += r * a
                task_sig_count[task]   += r
                task_count[task]       += 1

                r_list.append(r)

            q2rvalues[qid] = r_list

        # If no tasks (shouldn't happen), just passthrough
        if len(task_signal_mass) == 0:
            q2norm = {qid: list(vals) for qid, vals in q2rollouts.items()}
        else:
            # ----------------------------
            # Step 2 — compute density rho_t and EMA smooth it
            #         rho_batch = (sum r*|A|) / count
            # ----------------------------
            task_densities = {}
            log_rhos = []

            for task, mass in task_signal_mass.items():
                count = max(task_count[task], 1)
                rho_batch = max(mass / count, eps)

                # EMA update for the rho density for each task, which is essentially
                # the signal mass per sample
                prev_rho = float(task_stats[task].get("mixture_rho_ema", rho_batch))
                rho_ema  = _ema_update(prev_rho, rho_batch, BETA_RHO)
                rho_ema  = max(float(rho_ema), eps)

                task_stats[task]["mixture_rho_batch"] = float(rho_batch)  # Log batch-level density
                task_stats[task]["mixture_rho_ema"] = float(rho_ema)
                task_densities[task] = rho_ema
                log_rhos.append(math.log(rho_ema))

            # Reference density across tasks (geometric mean in log-space)
            log_rho_ref = sum(log_rhos) / max(len(log_rhos), 1)
            rho_ref     = math.exp(log_rho_ref)

            # ----------------------------
            # Step 3 — one-sided rarity boost k_t from effective signal counts (intuitive)
            #         boost only if (sig_ref / n_sig) >= RARE_RATIO
            #         Uses EMA-smoothed signal counts for stability
            # ----------------------------
            # First pass: update EMA of signal counts
            for task in task_signal_mass.keys():
                sig_count_batch = max(task_sig_count[task], eps)
                prev_ema = task_stats[task].get("mixture_sig_count_ema", 0.0)

                if prev_ema == 0.0:
                    # First time seeing this task - initialize EMA with batch value
                    sig_count_ema = sig_count_batch
                else:
                    # EMA update with BETA_RHO (same as density EMA)
                    sig_count_ema = _ema_update(prev_ema, sig_count_batch, BETA_RHO)

                task_stats[task]["mixture_sig_count_ema"] = float(sig_count_ema)

            # Compute reference using EMA values for smoothness
            sig_counts_ema = [max(task_stats[t]["mixture_sig_count_ema"], eps) for t in task_signal_mass.keys()]
            log_sig_ref = sum(math.log(x) for x in sig_counts_ema) / max(len(sig_counts_ema), 1)
            sig_ref     = math.exp(log_sig_ref)

            task_k = {}
            for task in task_signal_mass.keys():
                # Use EMA for rarity computation (smoother than batch)
                n_sig_ema = max(task_stats[task]["mixture_sig_count_ema"], eps)

                # Intuitive rarity ratio in normal space:
                # >1 means rarer-than-reference; 20 means 20x rarer.
                rarity_ratio = sig_ref / n_sig_ema

                # One-sided threshold: only boost if "super rare"
                rarity_excess = max(0.0, rarity_ratio - RARE_RATIO)

                # Smooth, saturating growth in log-space to avoid blow-ups for extreme rarity
                # (0 if below threshold)
                log_k = ETA_K * math.log1p(rarity_excess)
                log_k = min(math.log(K_MAX), log_k)
                k_t   = math.exp(log_k)

                task_k[task] = float(k_t)

                # logging (batch count + EMA + computed values)
                task_stats[task]["mixture_sig_count"]         = float(task_sig_count[task])  # batch value
                task_stats[task]["mixture_sig_ref"]           = float(sig_ref)
                task_stats[task]["mixture_rarity_ratio"]      = float(rarity_ratio)
                task_stats[task]["mixture_rarity_excess"]     = float(rarity_excess)
                task_stats[task]["mixture_k_t"]               = float(k_t)

            # ----------------------------
            # Step 4 — compute per-task log multiplier, EMA it, apply, then clamp
            #         log_mult = GAMMA*(log rho_ref - log rho_t) + log k_t
            # ----------------------------
            q2norm = {}
            for qid, vals in q2rollouts.items():
                task = q2tasks[qid]

                rho_t = max(task_densities.get(task, eps), eps)

                # density log-ratio: sparse (low rho) => positive => boost
                log_ratio = log_rho_ref - math.log(rho_t)
                log_scale_density = GAMMA * log_ratio

                k_t = float(task_k.get(task, 1.0))
                log_k = math.log(max(k_t, eps))

                # instantaneous controller signal
                log_mult_inst = log_scale_density + log_k

                # EMA smooth log multiplier (per task)
                prev_log_mult = float(task_stats[task].get("mixture_log_mult_ema", 0.0))
                log_mult_ema  = _ema_update(prev_log_mult, float(log_mult_inst), BETA_LOGMULT)

                # exponentiate to get scale, then clamp directly
                scale_t_unclamped = math.exp(float(log_mult_ema))
                scale_t = max(MIN_SCALE, min(MAX_SCALE, scale_t_unclamped))

                # logging
                task_stats[task]["mixture_rho_t"]            = float(rho_t)
                task_stats[task]["mixture_rho_ref"]          = float(rho_ref)
                task_stats[task]["mixture_log_ratio"]        = float(log_ratio)
                task_stats[task]["mixture_log_mult_inst"]    = float(log_mult_inst)
                task_stats[task]["mixture_log_mult_ema"]     = float(log_mult_ema)
                task_stats[task]["mixture_final_scale"]      = float(scale_t)
                task_stats[task]["mixture_signal_mass"]      = float(task_signal_mass[task])
                task_stats[task]["mixture_batch_count"]      = int(task_count[task])
                task_stats[task]["mixture_sig_count_batch"]  = float(task_sig_count[task])

                q2norm[qid] = [v * scale_t for v in vals]

    else:
        # No adapter → identity scaling
        q2norm = {qid: list(vals) for qid, vals in q2rollouts.items()}

    # -------------------------------------------
    # E) Static inverse-frequency class weights
    #     w_{d,c} = ((1/N_{d,c}) / sum_{c'∈C_d} 1/N_{d,c'}) * |C_d|
    # produce a single scalar q2w[qid]
    # -------------------------------------------
    if use_class_weights:
        if class_count_info == "import_from_dict":
            try:
                from verl.trainer.ppo.utils.v7_class_counts import CLASS_COUNT_INFO_DATASET, CLASS_COUNT_INFO_TASK
            except Exception:
                CLASS_COUNT_INFO_DATASET, CLASS_COUNT_INFO_TASK = {}, {}
            # Per-(dataset,class) EMA counts for inverse-frequency weights
            # dc_counts[(d,c)] = float (EMA count), d_counts[d] = float (EMA total), d_classes[d] = set of classes observed

            if class_weight_scope == "dataset":
                class_count_info = CLASS_COUNT_INFO_DATASET
            # elif class_weight_scope == "task":
            #     class_count_info = CLASS_COUNT_INFO_TASK
            elif class_weight_scope == "auto":
                # merged = dict(CLASS_COUNT_INFO_TASK)
                merged = dict(CLASS_COUNT_INFO_DATASET)
                # merged.update(CLASS_COUNT_INFO_TASK)
                class_count_info = merged
            else:
                raise ValueError(
                    "class_count_info must be provided as {group_id -> {class_label -> count}} "
                    "(group_id is a dataset_id or a task_id depending on class_weight_scope)."
                )

        # Precompute denominators S_g and cardinalities |C_g| per group (dataset OR task).
        g2_S: Dict[Any, float] = {}
        g2_C: Dict[Any, int]   = {}
        for group_id, class_count_map in class_count_info.items():
            if not class_count_map:
                g2_S[group_id] = 1.0
                g2_C[group_id] = 1
                continue
            S = 0.0
            for c2, cnt in class_count_map.items():
                # count of the group class label
                Ngc = float(max(cnt, 1))
                S  += 1.0 / max(Ngc, eps)
            g2_S[group_id] = max(S, eps)
            g2_C[group_id] = max(len(class_count_map), 1)

        def _resolve_group_id_based_on_qid(qid: Any) -> Any:
            """Pick dataset or task grouping for this qid."""
            if class_weight_scope == "dataset":
                return q2datasets[qid]
            elif class_weight_scope == "task":
                return q2tasks[qid]
            elif class_weight_scope == "auto":
                d = q2datasets[qid]
                t = q2tasks[qid]
                if d in class_count_info:
                    return d
                if t in class_count_info:
                    return t
                return None
            else:
                raise ValueError(
                    f"Invalid class_weight_scope={class_weight_scope!r}; use 'dataset' | 'task' | 'auto'."
                )

        # Per-qid scalar weight
        q2w: Dict[Any, float] = {}
        for qid in q2norm.keys():
            group_id = _resolve_group_id_based_on_qid(qid)
            if group_id is None or group_id not in class_count_info:
                # Group not found at all → neutral
                q2w[qid] = 1.0
                continue

            class_map = class_count_info[group_id]

            # If the group exists but is EMPTY (QA), keep it strictly neutral.
            if not class_map:
                q2w[qid] = 1.0
                continue

            c = q2class[qid]

            # If class label is None or unseen, treat as count=1 (neutral).
            raw_cnt = class_map.get(c, 1) if c is not None else 1
            Ngc = float(max(raw_cnt, 1))

            inv = 1.0 / max(Ngc, eps)
            Sg  = g2_S[group_id]
            Cg  = g2_C[group_id]

            q2w[qid] = inv * (Cg / max(Sg, eps))
    else:
        q2w = {qid: 1.0 for qid in q2norm.keys()}

    # ----------------------------------------------------------
    # F) CVaR×tail-frequency dynamic boost (per task) → q2k[qid]
    #     Buffer uses per-question MEAN of (task-adapter + class-weighted) scores
    #     k_t = (mean_ema / (cvar_ema + ε)) * ((p_tail_ema / α) ** beta_tail)
    # ----------------------------------------------------------
    if use_cvar_boost:
        # Append per-qid mean(after weighting) to the task’s and dataset’s buffers
        # (task-level buffer drives k_t; dataset-level is logging only)
        for qid, vals in q2norm.items():
            task    = q2tasks[qid]
            dataset = q2datasets[qid]
            w       = q2w[qid]
            # mean of (norm * weight) across rollouts for this question
            mean_q = float(np.mean(vals) * w)

            task_stats[task]["buffer"].append(mean_q)
            dataset_stats[dataset]["buffer"].append(mean_q)

        task2_k: Dict[Any, float] = {}

        # Iterate over tasks that actually appeared in this batch
        for task in task_to_rollouts.keys():
            buf = task_stats[task]["buffer"]
            if not buf:
                task2_k[task] = 1.0
                continue

            x = torch.tensor(buf, device=device, dtype=raw_scores.dtype)

            # Base stats from buffer
            mean_raw = float(x.mean().item())
            # VaR/CVaR and empirical tail mass
            q_alpha = torch.quantile(x, alpha)
            tail = x[x <= q_alpha]
            if tail.numel() == 0:
                cvar_raw   = mean_raw
                # if no tail, treat p_tail as small but non-zero
                p_tail_raw = max(1.0 / max(len(x), 1), alpha * 0.1)
            else:
                cvar_raw   = float(tail.mean().item())
                p_tail_raw = float(tail.numel()) / float(x.numel())

            # EMA smoothing of (mean, cvar, p_tail) at the TASK level
            prev_mean  = float(task_stats[task].get("buffer_mean_ema", mean_raw))
            prev_cvar  = float(task_stats[task].get("buffer_cvar_ema", cvar_raw))
            prev_ptail = float(task_stats[task].get("buffer_ptail_ema", p_tail_raw))

            task_stats[task]["buffer_mean_ema"]  = _ema_update(prev_mean,  mean_raw,  beta_mean)
            task_stats[task]["buffer_cvar_ema"]  = _ema_update(prev_cvar,  cvar_raw,  beta_cvar)
            task_stats[task]["buffer_ptail_ema"] = _ema_update(prev_ptail, p_tail_raw, beta_cvar)

            mean_t  = task_stats[task]["buffer_mean_ema"]
            cvar_t  = max(task_stats[task]["buffer_cvar_ema"], eps)
            ptail_t = max(task_stats[task]["buffer_ptail_ema"], eps)

            # CVaR×frequency ratio
            # Frequency ratio upweighs or downweights based on the number of tail events.
            freq_factor = (ptail_t / max(alpha, eps)) ** beta_tail
            k_t = (mean_t / cvar_t) * freq_factor

            # Safety clamp
            k_t = float(np.clip(k_t, 0.0, 10.0))
            task2_k[task] = k_t

        # Optional: mirror CVaR buffer stats at the DATASET level for logging only
        for dataset in dataset_to_rollouts.keys():
            buf = dataset_stats[dataset]["buffer"]
            if not buf:
                continue

            x = torch.tensor(buf, device=device, dtype=raw_scores.dtype)

            mean_raw = float(x.mean().item())
            q_alpha = torch.quantile(x, alpha)
            tail = x[x <= q_alpha]
            if tail.numel() == 0:
                cvar_raw   = mean_raw
                p_tail_raw = max(1.0 / max(len(x), 1), alpha * 0.1)
            else:
                cvar_raw   = float(tail.mean().item())
                p_tail_raw = float(tail.numel()) / float(x.numel())

            prev_mean  = float(dataset_stats[dataset].get("buffer_mean_ema", mean_raw))
            prev_cvar  = float(dataset_stats[dataset].get("buffer_cvar_ema", cvar_raw))
            prev_ptail = float(dataset_stats[dataset].get("buffer_ptail_ema", p_tail_raw))

            dataset_stats[dataset]["buffer_mean_ema"]  = _ema_update(prev_mean,  mean_raw,  beta_mean)
            dataset_stats[dataset]["buffer_cvar_ema"]  = _ema_update(prev_cvar,  cvar_raw,  beta_cvar)
            dataset_stats[dataset]["buffer_ptail_ema"] = _ema_update(prev_ptail, p_tail_raw, beta_cvar)
            # NOTE: dataset-level buffer EMAs are for analysis only;
            #       they do NOT feed back into k_t in this implementation.

        # Finally, map each qid → its task-level k_t
        q2k: Dict[Any, float] = {qid: task2_k[q2tasks[qid]] for qid in q2norm.keys()}
    else:
        q2k = {qid: 1.0 for qid in q2norm.keys()}

    # -------------------------------
    # U2) Track k_t per task and per dataset (batch + EMA of mean)
    # -------------------------------
    update_k_stats_from_q2k(
        q2k=q2k,
        q2tasks=q2tasks,
        q2datasets=q2datasets,
        beta_mean=beta_mean,
        eps=eps,
    )

    # --------------------------------------------
    # G) Combine once at the end, with ablations baked in:
    #    Let base = (norm_or_raw) * (weight_or_1)
    #    Effective lambda: λ_eff = λ_risk if CVaR enabled else 0
    #    Effective k:      k_eff = k_t     if CVaR enabled else 1
    #
    #    score = (1 - λ_eff) * base + λ_eff * (k_eff * base)
    # --------------------------------------------

    q2_final: Dict[str, List[float]] = defaultdict(list)
    qrolloutpos: Dict[Any, int] = defaultdict(int)  # store the positions of the rollouts that we are at

    for i in range(B):
        qid = index[i]
        j   = qrolloutpos[qid]

        # 1) Norm or raw (already prepared upstream)
        norm_or_raw = q2norm[qid][j]

        # 2) Class weight or 1.0
        w_eff = q2w[qid] if use_class_weights else 1.0

        # 3) CVaR components
        k_eff   = q2k[qid] if use_cvar_boost else 1.0
        lam_eff = lambda_risk if use_cvar_boost else 0.0

        # 4) Base term
        base = norm_or_raw * w_eff

        # 5) Interpolated final score
        #     (1−λ)*base + λ*k*base
        s_final = (1.0 - lam_eff) * base + lam_eff * (k_eff * base)

        q2_final[qid].append(float(s_final))
        qrolloutpos[qid] += 1

    # --------------------------------------------
    # G) Rebuild tensor in original rollout order
    # --------------------------------------------
    final_scores = torch.empty_like(raw_scores)
    qfinalrolloutpos: Dict[Any, int] = defaultdict(int)

    for i in range(B):
        qid = index[i]
        j   = qfinalrolloutpos[qid]
        final_scores[i] = torch.tensor(q2_final[qid][j], device=device, dtype=raw_scores.dtype)
        qfinalrolloutpos[qid] += 1

    # --------------------------------------------
    # U3) Update FINAL TARPO advantages per task and per dataset
    # --------------------------------------------
    update_advantage_stats(
        q2_advantages=q2_final,
        q2tasks=q2tasks,
        q2datasets=q2datasets,
        stat_prefix="final_advantage",
        beta_mu=beta_mu,
        beta_sigma=beta_sigma,
        eps=eps,
    )

    # Store final advantages for saving to JSON
    store_advantage_data(
        advantage_type="final",
        q2_advantages=q2_final,
        q2tasks=q2tasks,
        q2datasets=q2datasets
    )

    # --------------------------------------------
    # H) Broadcast to token level and return
    # --------------------------------------------
    returns = final_scores.unsqueeze(-1) * response_mask

    return returns, returns