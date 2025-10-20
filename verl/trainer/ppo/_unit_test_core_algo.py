# _unit_test_core_algo.py
# Standalone runner for compute_tarpo_outcome_advantage

import math
import torch
import numpy as np
from collections import defaultdict

# >>> Adjust this import line if needed <<<
import verl.trainer.ppo.core_algos as M


def _reset_task_stats():
    M.task_stats = defaultdict(
        lambda: {
            "mu": 0.0,
            "sigma": 1.0,
            "count": 0,
            "buffer": [],
            "mean_ema": 0.0,
            "cvar_ema": 1.0,
        }
    )


def _mk_mask_like(x: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(x)


def _close(a, b, tol=1e-6):
    return torch.allclose(a, b, atol=tol, rtol=0)


def test_shapes_and_broadcast_basic_pass_through():
    _reset_task_stats()

    token_rewards = torch.tensor(
        [[0.1, 0.2, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]],
        dtype=torch.float32,
    )
    mask = _mk_mask_like(token_rewards)
    index = torch.tensor([0, 0, 1], dtype=torch.long)
    task_ids = ["sen", "sen", "emo"]
    dataset_ids = ["mosei_senti", "mosei_emo", "chsimsv2"]
    class_labels = ["sen_pos", "sen_pos", "emo_happy"]

    returns, adv = M.compute_tarpo_outcome_advantage(
        token_rewards,
        mask,
        index,
        task_ids,
        dataset_ids,
        class_labels,
        use_task_adapter=False,
        use_class_weights=False,
        use_cvar_boost=False,
        use_grpo_group_norm=False,
    )

    sums = torch.tensor([0.3, 1.0, 1.0]).unsqueeze(-1).expand_as(token_rewards)
    assert _close(returns, sums * mask)
    assert _close(adv, returns)
    print("✅ test_shapes_and_broadcast_basic_pass_through passed")


def test_task_adapter_normalization_from_batch_stats():
    _reset_task_stats()

    token_rewards = torch.tensor(
        [[1.0, 0.0], [1.5, 1.5], [2.0, 0.0]], dtype=torch.float32
    )
    mask = _mk_mask_like(token_rewards)
    index = torch.tensor([0, 0, 1], dtype=torch.long)
    task_ids = ["tA", "tA", "tA"]
    dataset_ids = ["dA", "dA", "dA"]
    class_labels = ["c1", "c1", "c1"]

    returns, _ = M.compute_tarpo_outcome_advantage(
        token_rewards,
        mask,
        index,
        task_ids,
        dataset_ids,
        class_labels,
        use_task_adapter=True,
        use_class_weights=False,
        use_cvar_boost=False,
        use_grpo_group_norm=False,
        beta_mu=0.0,
        beta_sigma=0.0,
    )

    z = torch.tensor([-1.0, 1.0, 0.0])
    expected = z.unsqueeze(-1).expand_as(token_rewards)
    assert _close(returns, expected)
    print("✅ test_task_adapter_normalization_from_batch_stats passed")


def test_class_weighting_only_matches_formula():
    _reset_task_stats()

    token_rewards = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    mask = _mk_mask_like(token_rewards)
    index = torch.tensor([0, 0], dtype=torch.long)
    task_ids = ["tW", "tW"]
    dataset_ids = ["dX", "dX"]
    class_labels = ["c1", "c1"]

    static_class_counts = {("dX", "c1"): 10, ("dX", "c2"): 30}
    dataset_classes = {"dX": {"c1", "c2"}}

    inv_c1 = 1.0 / 10.0
    inv_c2 = 1.0 / 30.0
    w_c1 = inv_c1 * (2 / (inv_c1 + inv_c2))

    returns, _ = M.compute_tarpo_outcome_advantage(
        token_rewards,
        mask,
        index,
        task_ids,
        dataset_ids,
        class_labels,
        use_task_adapter=False,
        use_class_weights=True,
        use_cvar_boost=False,
        use_grpo_group_norm=False,
        static_class_counts=static_class_counts,
        dataset_classes=dataset_classes,
    )

    sums = torch.tensor([1.0, 2.0]).unsqueeze(-1).expand_as(token_rewards)
    expected = (w_c1 * sums) * mask
    assert _close(returns, expected)
    print("✅ test_class_weighting_only_matches_formula passed")


def test_cvar_boost_applies_taskwide_scaling():
    _reset_task_stats()

    token_rewards = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.1, 0.0], [0.1, 0.0]], dtype=torch.float32
    )
    mask = _mk_mask_like(token_rewards)
    index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    task_ids = ["tZ"] * 4
    dataset_ids = ["d0"] * 4
    class_labels = ["c0"] * 4

    alpha = 0.5
    lambda_risk = 0.3

    returns, _ = M.compute_tarpo_outcome_advantage(
        token_rewards,
        mask,
        index,
        task_ids,
        dataset_ids,
        class_labels,
        use_task_adapter=False,
        use_class_weights=False,
        use_cvar_boost=True,
        use_grpo_group_norm=False,
        alpha=alpha,
        lambda_risk=lambda_risk,
        beta_mean=0.0,
        beta_cvar=0.0,
    )

    k_t = 0.55 / 0.1
    lam = lambda_risk
    sA = (1 - lam) * 1.0 + lam * (k_t * 1.0)
    sB = (1 - lam) * 0.1 + lam * (k_t * 0.1)

    expected = torch.tensor([[sA, sA], [sA, sA], [sB, sB], [sB, sB]], dtype=torch.float32)
    assert _close(returns, expected * mask, tol=1e-5)
    print("✅ test_cvar_boost_applies_taskwide_scaling passed")


def test_grpo_group_norm_normalizes_within_qid():
    _reset_task_stats()

    token_rewards = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], dtype=torch.float32)
    mask = _mk_mask_like(token_rewards)
    index = torch.tensor([7, 7, 7], dtype=torch.long)
    task_ids = ["tY"] * 3
    dataset_ids = ["dY"] * 3
    class_labels = ["cY"] * 3

    returns, _ = M.compute_tarpo_outcome_advantage(
        token_rewards,
        mask,
        index,
        task_ids,
        dataset_ids,
        class_labels,
        use_task_adapter=False,
        use_class_weights=False,
        use_cvar_boost=False,
        use_grpo_group_norm=True,
    )

    scalars = returns[:, 0]
    mu = float(torch.mean(scalars))
    sd = float(torch.std(scalars, unbiased=False))
    assert abs(mu) < 1e-6
    assert abs(sd - 1.0) < 1e-4
    print("✅ test_grpo_group_norm_normalizes_within_qid passed")


if __name__ == "__main__":
    print("\n🚀 Running TARPO core algorithm unit tests...\n")
    test_shapes_and_broadcast_basic_pass_through()
    test_task_adapter_normalization_from_batch_stats()
    test_class_weighting_only_matches_formula()
    test_cvar_boost_applies_taskwide_scaling()
    test_grpo_group_norm_normalizes_within_qid()
    print("\n🎉 All TARPO tests passed successfully!\n")