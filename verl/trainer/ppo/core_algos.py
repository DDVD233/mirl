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
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | AlgoConfig],  # config
    ],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
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


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

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


@register_policy_loss("vanilla")
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("gpg")
def compute_policy_loss_gpg(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode="token-mean", config=None):
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
    pg_losses = -log_prob * advantages

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return pg_loss, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, torch.tensor(0.0)


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, torch.tensor(0.0), ppo_kl_abs, torch.tensor(0.0)


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    pg_loss = torch.mean(pg_losses)

    # higher: ratio is too large that need clamp to clip_high (when adv > 0)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped * (advantages < 0)).float(), response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


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
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

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
        # Note: responsibilities (q2rvalues) are not yet computed at this stage
        store_advantage_data(
            advantage_type="post_grpo",
            q2_advantages=q2rollouts,
            q2tasks=q2tasks,
            q2datasets=q2datasets,
            q2_responsibilities={}
        )

    # -------------------------------------------------------------------------
    # D.5) Task mixture-density adapter
    #
    # Features:
    #  1) Rarity boosting is toggleable
    #  2) Inter-task responsibilities use unit-advantage normalization (no z-scores)
    #  3) Optional hierarchical rollout-mixture within each task (two-sided,
    #     budget-preserving via log re-centering)
    # -------------------------------------------------------------------------

    # Initialize responsibilities dict (will be populated if mixture adapter is enabled)
    q2rvalues: Dict[Any, List[float]] = {}

    if use_task_mixture_density_adapter:

        # ----------------------------
        # Toggles
        # ----------------------------
        USE_RARITY_BOOST         = False
        USE_HIER_ROLLOUT_MIXTURE = True
        NORMALIZATION_MODE       = "no_responsibilities"  # "z_score", "global_norm", "absolute", or "no_responsibilities"

        # ----------------------------
        # Hyperparameters
        # ----------------------------
        # Responsibility computation
        BETA_R   = 4.0
        DELTA_Z  = 0.25   # threshold in z space (e.g., 0.25 ~ mildly above typical) - only used for z_score mode
        A_THRESHOLD = 0.5  # threshold in absolute advantage space - only used for absolute mode

        # Inter-task density scaling
        GAMMA = 1

        # Rarity boost (optional)
        ETA_K      = 0.5
        K_MAX      = 1.5
        RARE_RATIO = 2.5

        # Hierarchical rollout mixture
        ALPHA_ROLL     = 0.5
        ROLL_MAX_SCALE = 3.0

        # EMA smoothing
        BETA_RHO     = 0.95
        BETA_LOGMULT = 0.95

        # Final task-scale clamp
        MIN_SCALE = 0.5
        MAX_SCALE = 4.0

        # ----------------------------
        # Step 1 — responsibilities + task signal mass
        # ----------------------------
        task_signal_mass = defaultdict(float)
        task_sig_count   = defaultdict(float)
        task_count       = defaultdict(int)

        # Compute global mean absolute advantage if using global normalization
        global_mean_abs = None
        if NORMALIZATION_MODE == "global_norm":
            all_abs_vals = []
            for vals in q2rollouts.values():
                all_abs_vals.extend([abs(v) for v in vals])
            global_mean_abs = sum(all_abs_vals) / max(len(all_abs_vals), 1) if all_abs_vals else eps
            global_mean_abs = max(global_mean_abs, eps)

        for qid, vals in q2rollouts.items():
            task = q2tasks[qid]
            r_list = []

            if NORMALIZATION_MODE == "global_norm":
                # Global normalization: normalize by global mean |A|
                for v in vals:
                    a = abs(v)
                    # Normalize by global mean
                    norm_a = a / global_mean_abs
                    # Responsibility based on normalized advantage
                    # Use norm_a - 1.0 as input to sigmoid (values > mean have norm_a > 1)
                    r = 1.0 / (1.0 + math.exp(-BETA_R * (norm_a - 1.0)))

                    task_signal_mass[task] += r * a
                    task_sig_count[task]   += r
                    task_count[task]       += 1
                    r_list.append(r)
            elif NORMALIZATION_MODE == "z_score":
                # Z-score normalization: task-specific normalization
                # Fetch per-task EMA stats in |A| space
                mu_abs = float(task_stats[task].get("post_grpo_advantage_ema_abs_mean", 0.0))
                mu_abs = max(mu_abs, eps)

                # Prefer abs-sigma if tracked; otherwise fall back to signed sigma
                sd_abs = float(task_stats[task].get("post_grpo_advantage_ema_abs_sigma", 0.0))
                if sd_abs <= 0.0:
                    sd_abs = float(task_stats[task].get("post_grpo_advantage_ema_sigma", 0.0))
                sd_abs = max(sd_abs, eps)

                for v in vals:
                    a = abs(v)

                    # standard z-score in |A| space
                    z = (a - mu_abs) / (sd_abs + eps)

                    # responsibility as soft tail membership
                    r = 1.0 / (1.0 + math.exp(-BETA_R * (z - DELTA_Z)))

                    task_signal_mass[task] += r * a
                    task_sig_count[task]   += r
                    task_count[task]       += 1
                    r_list.append(r)
            elif NORMALIZATION_MODE == "absolute":
                # Absolute mode: sigmoid applied directly to absolute advantage with threshold
                for v in vals:
                    a = abs(v)

                    # Responsibility based on raw absolute advantage with threshold
                    r = 1.0 / (1.0 + math.exp(-BETA_R * (a - A_THRESHOLD)))

                    task_signal_mass[task] += r * a
                    task_sig_count[task]   += r
                    task_count[task]       += 1
                    r_list.append(r)

            elif NORMALIZATION_MODE == "no_responsibilities":
                # No responsibilities ablation: all responsibilities set to 1.0
                # This removes responsibility weighting and uses raw absolute advantage statistics
                for v in vals:
                    a = abs(v)

                    # Responsibility is always 1.0 (no weighting)
                    r = 1.0

                    task_signal_mass[task] += r * a  # Equivalent to: task_signal_mass[task] += a
                    task_sig_count[task]   += r      # Equivalent to: task_sig_count[task] += 1
                    task_count[task]       += 1
                    r_list.append(r)
            else:
                raise ValueError(f"Unknown NORMALIZATION_MODE: {NORMALIZATION_MODE}. Must be 'z_score', 'global_norm', 'absolute', or 'no_responsibilities'")

            q2rvalues[qid] = r_list

        # Store batch-level signal statistics for logging
        for task in task_signal_mass:
            task_stats[task]["mixture_signal_mass"]       = float(task_signal_mass[task])
            task_stats[task]["mixture_sig_count_batch"]   = float(task_sig_count[task])
            task_stats[task]["mixture_batch_count"]       = int(task_count[task])

        if len(task_signal_mass) == 0:
            q2norm = {qid: list(vals) for qid, vals in q2rollouts.items()}
        else:
            # ----------------------------
            # Step 2 — task densities rho_t (EMA)
            # ----------------------------
            task_densities = {}
            log_rhos = []

            for task, mass in task_signal_mass.items():
                count = max(task_count[task], 1)
                rho_batch = max(mass / count, eps)

                prev_rho = float(task_stats[task].get("mixture_rho_t_ema", rho_batch))
                rho_ema  = _ema_update(prev_rho, rho_batch, BETA_RHO)
                rho_ema  = max(rho_ema, eps)

                task_stats[task]["mixture_rho_t_batch"] = float(rho_batch)
                task_stats[task]["mixture_rho_t_ema"]   = float(rho_ema)

                task_densities[task] = rho_ema
                log_rhos.append(math.log(rho_ema))

            # geometric mean of task densities
            log_rho_ref = sum(log_rhos) / len(log_rhos)
            rho_ref     = math.exp(log_rho_ref)

            for task in task_signal_mass:
                task_stats[task]["mixture_rho_ref"] = float(rho_ref)

            # ----------------------------
            # Step 3 — rarity boost (optional)
            # ----------------------------
            task_k = defaultdict(lambda: 1.0)

            if USE_RARITY_BOOST:
                for task in task_signal_mass:
                    sig_batch = max(task_sig_count[task], eps)
                    prev = float(task_stats[task].get("mixture_sig_count_ema", sig_batch))
                    sig_ema = _ema_update(prev, sig_batch, BETA_RHO)
                    task_stats[task]["mixture_sig_count_ema"] = float(sig_ema)
                    task_stats[task]["mixture_sig_count"]     = float(sig_batch)

                sig_emas = [
                    max(float(task_stats[t]["mixture_sig_count_ema"]), eps)
                    for t in task_signal_mass
                ]
                sig_ref = math.exp(sum(math.log(x) for x in sig_emas) / len(sig_emas))

                for task in task_signal_mass:
                    n_sig = max(task_stats[task]["mixture_sig_count_ema"], eps)
                    ratio = sig_ref / n_sig
                    excess = max(0.0, ratio - RARE_RATIO)

                    log_k = ETA_K * math.log1p(excess)
                    log_k = min(log_k, math.log(K_MAX))
                    task_k[task] = math.exp(log_k)

                    task_stats[task]["mixture_sig_ref"]          = float(sig_ref)
                    task_stats[task]["mixture_log_rarity_raw"]   = float(math.log(ratio))
                    task_stats[task]["mixture_log_rarity_pos"]   = float(excess)
                    task_stats[task]["mixture_rarity_ratio"]     = float(ratio)
                    task_stats[task]["mixture_rarity_excess"]    = float(excess)
                    task_stats[task]["mixture_k_t"]              = float(task_k[task])
            else:
                for task in task_signal_mass:
                    task_stats[task]["mixture_sig_count"]        = 0.0
                    task_stats[task]["mixture_sig_ref"]          = 0.0
                    task_stats[task]["mixture_log_rarity_raw"]   = 0.0
                    task_stats[task]["mixture_log_rarity_pos"]   = 0.0
                    task_stats[task]["mixture_rarity_ratio"]     = 0.0
                    task_stats[task]["mixture_rarity_excess"]    = 0.0
                    task_stats[task]["mixture_k_t"]              = 1.0

            # ----------------------------
            # Step 4 — task-scale s_t (EMA)
            # ----------------------------
            task_scale = {}

            for task in task_signal_mass:
                rho_t = max(task_densities[task], eps)
                log_ratio = log_rho_ref - math.log(rho_t)
                log_mult_inst = GAMMA * log_ratio + math.log(task_k[task])

                prev = float(task_stats[task].get("mixture_log_mult_ema_scale", 0.0))
                log_mult_ema = _ema_update(prev, log_mult_inst, BETA_LOGMULT)

                scale = math.exp(log_mult_ema)
                scale = max(MIN_SCALE, min(MAX_SCALE, scale))

                task_scale[task] = scale

                task_stats[task]["mixture_log_ratio_scale"]      = float(log_ratio)
                task_stats[task]["mixture_log_mult_inst_scale"]  = float(log_mult_inst)
                task_stats[task]["mixture_log_mult_ema_scale"]   = float(log_mult_ema)
                task_stats[task]["mixture_final_scale"]          = float(scale)

            # ----------------------------
            # Step 5 — hierarchical rollout-mixture (two-sided, budget-preserving)
            # ----------------------------
            qid_rollout_scale = defaultdict(lambda: 1.0)

            if USE_HIER_ROLLOUT_MIXTURE:
                task2qids = defaultdict(list)
                for qid in q2rollouts:
                    task2qids[q2tasks[qid]].append(qid)

                for task, qids in task2qids.items():
                    if len(qids) < 2:
                        continue

                    qid2m = {}
                    log_m = []

                    # computing the per rolloutout m , which is essentially the density of the signal mass
                    for qid in qids:
                        vals = q2rollouts[qid]
                        rs   = q2rvalues[qid]
                        m = sum(r * abs(v) for r, v in zip(rs, vals)) / max(len(vals), 1)
                        m = max(m, eps)
                        qid2m[qid] = m
                        log_m.append(math.log(m))


                    # For a specific task, what's the average log m
                    # across its questions
                    log_mbar = sum(log_m) / len(log_m)

                    log_s_raw = {
                        qid: ALPHA_ROLL * (log_mbar - math.log(qid2m[qid]))
                        for qid in qids
                    }

                    mean_log_s = sum(log_s_raw.values()) / len(log_s_raw)

                    for qid in qids:
                        log_s = log_s_raw[qid] - mean_log_s
                        if ROLL_MAX_SCALE is not None:
                            cap = math.log(ROLL_MAX_SCALE)
                            log_s = max(-cap, min(cap, log_s))
                        qid_rollout_scale[qid] = math.exp(log_s)

            # ----------------------------
            # Step 6 — apply final scaling
            # ----------------------------
            q2norm = {}
            for qid, vals in q2rollouts.items():
                task = q2tasks[qid]
                s_t  = task_scale.get(task, 1.0)
                s_q  = qid_rollout_scale.get(qid, 1.0)
                q2norm[qid] = [v * s_t * s_q for v in vals]

    else:
        q2norm = {qid: list(vals) for qid, vals in q2rollouts.items()}


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

        s_final = norm_or_raw

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

    # Store final advantages and responsibilities for saving to JSON
    store_advantage_data(
        advantage_type="final",
        q2_advantages=q2_final,
        q2tasks=q2tasks,
        q2datasets=q2datasets,
        q2_responsibilities=q2rvalues
    )

    # --------------------------------------------
    # H) Broadcast to token level and return
    # --------------------------------------------
    returns = final_scores.unsqueeze(-1) * response_mask

    return returns, returns