# _unit_test_core_algo.py
# Standalone runner for compute_harpo_outcome_advantage
#
# This version includes two-batch simulations to verify global updates
# (EMA stats, counts, buffers) between calls.

import math
import torch
import numpy as np
from collections import defaultdict

# >>> Adjust this import line if needed <<<
import verl.trainer.ppo.core_algos as M


def _reset_task_stats():
    # Task-global state held by the core algo; we reset so tests are isolated.
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


def _no_nan(t: torch.Tensor):
    return not torch.isnan(t).any().item()


# ---------------------------
# Original (single-batch) tests
# ---------------------------

def test_shapes_and_broadcast_basic_pass_through():
    _reset_task_stats()

    token_rewards = torch.tensor(
        [[0.1, 0.2, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]],
        dtype=torch.float32,
    )
    mask = _mk_mask_like(token_rewards)
    index = np.array([0, 0, 1], dtype=np.int64)
    task_ids = ["sen", "sen", "emo"]
    dataset_ids = ["mosei_senti", "mosei_emo", "chsimsv2"]
    class_labels = ["sen_pos", "sen_pos", "emo_happy"]

    returns, adv = M.compute_harpo_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=mask,
        index=index,
        task_ids=task_ids,
        dataset_ids=dataset_ids,
        class_labels=class_labels,
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
    index = np.array([0, 0, 1], dtype=np.int64)
    task_ids = ["tA", "tA", "tA"]
    dataset_ids = ["dA", "dA", "dA"]
    class_labels = ["c1", "c1", "c1"]

    returns, _ = M.compute_harpo_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=mask,
        index=index,
        task_ids=task_ids,
        dataset_ids=dataset_ids,
        class_labels=class_labels,
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
    index = np.array([0, 0], dtype=np.int64)
    task_ids = ["tW", "tW"]
    dataset_ids = ["dX", "dX"]
    class_labels = ["c1", "c1"]

    static_class_counts = {("dX", "c1"): 10, ("dX", "c2"): 30}
    dataset_classes = {"dX": {"c1", "c2"}}

    inv_c1 = 1.0 / 10.0
    inv_c2 = 1.0 / 30.0
    w_c1 = inv_c1 * (2 / (inv_c1 + inv_c2))

    returns, _ = M.compute_harpo_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=mask,
        index=index,
        task_ids=task_ids,
        dataset_ids=dataset_ids,
        class_labels=class_labels,
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
    index = np.array([0, 0, 1, 1], dtype=np.int64)
    task_ids = ["tZ"] * 4
    dataset_ids = ["d0"] * 4
    class_labels = ["c0"] * 4

    alpha = 0.5
    lambda_risk = 0.3

    returns, _ = M.compute_harpo_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=mask,
        index=index,
        task_ids=task_ids,
        dataset_ids=dataset_ids,
        class_labels=class_labels,
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
    index = np.array([7, 7, 7], dtype=np.int64)
    task_ids = ["tY"] * 3
    dataset_ids = ["dY"] * 3
    class_labels = ["cY"] * 3

    returns, _ = M.compute_harpo_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=mask,
        index=index,
        task_ids=task_ids,
        dataset_ids=dataset_ids,
        class_labels=class_labels,
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


# ---------------------------
# NEW: two-batch tests to verify global updates
# ---------------------------


def test_task_adapter_updates_across_two_batches():
    _reset_task_stats()

    # --- Batch 1: mix taskA/dA and taskB/dB ---
    token_rewards_1 = torch.tensor(
        [
            [0.5, 0.0],  # A
            [1.5, 0.0],  # A
            [3.0, 0.0],  # A
            [0.2, 0.0],  # B
            [0.3, 0.0],  # B
        ],
        dtype=torch.float32,
    )
    mask_1 = _mk_mask_like(token_rewards_1)
    index_1 = np.array([0, 0, 0, 1, 1], dtype=np.int64)  # two qids interleaved
    task_ids_1   = ["taskA", "taskA", "taskA", "taskB", "taskB"]
    dataset_ids_1 = ["dA", "dA", "dA", "dB", "dB"]
    class_labels_1 = ["cA", "cA", "cA", "cB", "cB"]

    returns_1, _ = M.compute_harpo_outcome_advantage(
        token_level_rewards=token_rewards_1,
        response_mask=mask_1,
        index=index_1,
        task_ids=task_ids_1,
        dataset_ids=dataset_ids_1,
        class_labels=class_labels_1,
        use_task_adapter=True,
        use_class_weights=False,
        use_cvar_boost=False,
        use_grpo_group_norm=False,
        beta_mu=0.5,
        beta_sigma=0.5,
    )
    assert _no_nan(returns_1)

    A1, B1 = M.task_stats["taskA"].copy(), M.task_stats["taskB"].copy()
    assert A1["count"] > 0 and B1["count"] > 0

    # --- Batch 2: shift both tasks differently (higher for A, moderate for B) ---
    token_rewards_2 = torch.tensor(
        [
            [6.0, 0.0],  # A
            [6.5, 0.0],  # A
            [7.0, 0.0],  # A
            [1.2, 0.0],  # B
            [1.0, 0.0],  # B
        ],
        dtype=torch.float32,
    )
    mask_2 = _mk_mask_like(token_rewards_2)
    index_2 = np.array([2, 2, 3, 4, 4], dtype=np.int64)
    task_ids_2   = ["taskA", "taskA", "taskA", "taskB", "taskB"]
    dataset_ids_2 = ["dA", "dA", "dA", "dB", "dB"]
    class_labels_2 = ["cA", "cA", "cA", "cB", "cB"]

    returns_2, _ = M.compute_harpo_outcome_advantage(
        token_level_rewards=token_rewards_2,
        response_mask=mask_2,
        index=index_2,
        task_ids=task_ids_2,
        dataset_ids=dataset_ids_2,
        class_labels=class_labels_2,
        use_task_adapter=True,
        use_class_weights=False,
        use_cvar_boost=False,
        use_grpo_group_norm=False,
        beta_mu=0.5,
        beta_sigma=0.5,
    )
    assert _no_nan(returns_2)

    A2, B2 = M.task_stats["taskA"], M.task_stats["taskB"]

    # Per-task EMAs must update, and stay distinct across tasks
    for k in ("mu", "sigma", "count"):
        assert A2[k] != A1[k] or k == "count"
        assert B2[k] != B1[k] or k == "count"
    assert A2["count"] > A1["count"] and B2["count"] > B1["count"]
    assert (A2["mu"], A2["sigma"]) != (B2["mu"], B2["sigma"]), "Tasks must not share stats"

    # Outputs for A changed across batches; same for B
    A_mask = torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool)
    B_mask = ~A_mask
    assert not _close(returns_1[A_mask, 0], returns_2[A_mask, 0], tol=1e-6)
    assert not _close(returns_1[B_mask, 0], returns_2[B_mask, 0], tol=1e-6)

    print("✅ test_task_adapter_updates_across_two_batches (two tasks/datasets) passed")


def test_class_weighting_across_batches():
    _reset_task_stats()

    token_rewards_1 = torch.tensor([[1.0, 0.0], [0.3, 0.7]], dtype=torch.float32)  # dX: c1, c2
    mask_1  = _mk_mask_like(token_rewards_1)
    index_1 = np.array([0, 0], dtype=np.int64)
    task_ids_1     = ["tW", "tW"]
    dataset_ids_1  = ["dX", "dX"]
    class_labels_1 = ["c1", "c2"]

    # NEW: single consolidated structure
    # can also easily be replaced by task_class_count_info
    # but the task_class_info would correspond to the task ids
    # and the dataset_class_info would correspond to the dataset ids
    dataset_class_count_info = {"dX": {"c1": 10, "c2": 30}}

    returns_1, _ = M.compute_harpo_outcome_advantage(
        token_level_rewards=token_rewards_1,
        response_mask=mask_1,
        index=index_1,
        task_ids=task_ids_1,
        dataset_ids=dataset_ids_1,
        class_labels=class_labels_1,
        use_task_adapter=False,
        use_class_weights=True,
        use_cvar_boost=False,
        use_grpo_group_norm=False,
        class_count_info=dataset_class_count_info,
        class_weight_scope="auto",
    )

    # Same batch again should yield identical outputs (no global state affects class-weighting)
    returns_2, _ = M.compute_harpo_outcome_advantage(
        token_level_rewards=token_rewards_1,
        response_mask=mask_1,
        index=index_1,
        task_ids=task_ids_1,
        dataset_ids=dataset_ids_1,
        class_labels=class_labels_1,
        use_task_adapter=False,
        use_class_weights=True,
        use_cvar_boost=False,
        use_grpo_group_norm=False,
        class_count_info=dataset_class_count_info,
        class_weight_scope="auto",
    )

    assert _close(returns_1, returns_2)
    print("✅ test_class_weighting_is_stateless_across_batches passed")


def test_cvar_boost_updates_across_two_batches():
    _reset_task_stats()

    # --- Batch 1: seed risk stats separately for riskA/dR and riskB/dS ---
    token_rewards_1 = torch.tensor(
        [
            [1.0, 0.0], [1.0, 0.0], [0.2, 0.0], [0.2, 0.0],  # riskA (mix easy/hard)
            [0.4, 0.0], [0.5, 0.0], [0.6, 0.0],              # riskB (tighter)
        ],
        dtype=torch.float32
    )
    mask_1 = _mk_mask_like(token_rewards_1)
    index_1 = np.array([0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
    task_ids_1    = ["riskA"] * 4 + ["riskB"] * 3
    dataset_ids_1 = ["dR"]   * 4 + ["dS"]   * 3
    class_labels_1 = ["cR"]  * 4 + ["cS"]  * 3

    alpha = 0.5
    lambda_risk = 0.25

    returns_1, _ = M.compute_harpo_outcome_advantage(
        token_level_rewards=token_rewards_1,
        response_mask=mask_1,
        index=index_1,
        task_ids=task_ids_1,
        dataset_ids=dataset_ids_1,
        class_labels=class_labels_1,
        use_task_adapter=False,
        use_class_weights=False,
        use_cvar_boost=True,
        use_grpo_group_norm=False,
        alpha=alpha,
        lambda_risk=lambda_risk,
        beta_mean=0.5,
        beta_cvar=0.5,  # ensure EMAs update
    )
    assert _no_nan(returns_1)

    A1m, A1c = M.task_stats["riskA"]["buffer_mean_ema"], M.task_stats["riskA"]["buffer_cvar_ema"]
    B1m, B1c = M.task_stats["riskB"]["buffer_mean_ema"], M.task_stats["riskB"]["buffer_cvar_ema"]

    # --- Batch 2: shift both tasks, but differently ---
    token_rewards_2 = torch.tensor(
        [
            [1.6, 0.0], [1.7, 0.0], [1.8, 0.0], [2.0, 0.0],  # riskA increases
            [0.9, 0.0], [1.1, 0.0],                           # riskB modest increase
        ],
        dtype=torch.float32
    )
    mask_2 = _mk_mask_like(token_rewards_2)
    index_2 = np.array([3, 3, 4, 4, 5, 5], dtype=np.int64)
    task_ids_2    = ["riskA"] * 4 + ["riskB"] * 2
    dataset_ids_2 = ["dR"]   * 4 + ["dS"]   * 2
    class_labels_2 = ["cR"]  * 4 + ["cS"]  * 2

    returns_2, _ = M.compute_harpo_outcome_advantage(
        token_level_rewards=token_rewards_2,
        response_mask=mask_2,
        index=index_2,
        task_ids=task_ids_2,
        dataset_ids=dataset_ids_2,
        class_labels=class_labels_2,
        use_task_adapter=False,
        use_class_weights=False,
        use_cvar_boost=True,
        use_grpo_group_norm=False,
        alpha=alpha,
        lambda_risk=lambda_risk,
        beta_mean=0.5,
        beta_cvar=0.5,
    )
    assert _no_nan(returns_2)

    A2m, A2c = M.task_stats["riskA"]["mean_ema"], M.task_stats["riskA"]["cvar_ema"]
    B2m, B2c = M.task_stats["riskB"]["mean_ema"], M.task_stats["riskB"]["cvar_ema"]

    # Both tasks’ EMAs should move, and remain different between tasks
    assert (A2m != A1m) or (A2c != A1c)
    assert (B2m != B1m) or (B2c != B1c)
    assert (A2m, A2c) != (B2m, B2c), "Risk stats must be task-specific"

    # Their return scales should also change between batches for each task
    A_mask_1 = torch.tensor([1, 1, 1, 1, 0, 0, 0], dtype=torch.bool)
    B_mask_1 = ~A_mask_1
    A_mask_2 = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.bool)
    B_mask_2 = ~A_mask_2

    assert not _close(returns_1[A_mask_1, 0].mean(), returns_2[A_mask_2, 0].mean(), tol=1e-6)
    assert not _close(returns_1[B_mask_1, 0].mean(), returns_2[B_mask_2, 0].mean(), tol=1e-6)

    print("✅ test_cvar_boost_updates_across_two_batches (two tasks/datasets) passed")

def test_grpo_group_norm_two_independent_batches():
    _reset_task_stats()

    # Batch 1 (qid=42), but mix tasks/datasets/classes — normalization should still be by qid.
    token_rewards_1 = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 4.0]], dtype=torch.float32)
    mask_1 = _mk_mask_like(token_rewards_1)
    index_1 = np.array([42, 42, 42], dtype=np.int64)
    task_ids_1    = ["gA", "gB", "gA"]
    dataset_ids_1 = ["dgA", "dgB", "dgA"]
    class_labels_1 = ["cgA", "cgB", "cgA"]

    returns_1, _ = M.compute_harpo_outcome_advantage(
        token_level_rewards=token_rewards_1,
        response_mask=mask_1,
        index=index_1,
        task_ids=task_ids_1,
        dataset_ids=dataset_ids_1,
        class_labels=class_labels_1,
        use_task_adapter=False,
        use_class_weights=False,
        use_cvar_boost=False,
        use_grpo_group_norm=True,
    )

    # Batch 2 (qid=43), also mixed tasks/datasets
    token_rewards_2 = torch.tensor([[0.0, 10.0], [0.0, 20.0], [0.0, 30.0]], dtype=torch.float32)
    mask_2 = _mk_mask_like(token_rewards_2)
    index_2 = np.array([43, 43, 43], dtype=np.int64)
    task_ids_2    = ["gB", "gA", "gB"]
    dataset_ids_2 = ["dgB", "dgA", "dgB"]
    class_labels_2 = ["cgB", "cgA", "cgB"]

    returns_2, _ = M.compute_harpo_outcome_advantage(
        token_level_rewards=token_rewards_2,
        response_mask=mask_2,
        index=index_2,
        task_ids=task_ids_2,
        dataset_ids=dataset_ids_2,
        class_labels=class_labels_2,
        use_task_adapter=False,
        use_class_weights=False,
        use_cvar_boost=False,
        use_grpo_group_norm=True,
    )

    # For each batch, z-scored within its own qid regardless of task/dataset mix
    for r in (returns_1[:, 0], returns_2[:, 0]):
        mu = float(torch.mean(r))
        sd = float(torch.std(r, unbiased=False))
        assert abs(mu) < 1e-6
        assert abs(sd - 1.0) < 1e-4

    print("✅ test_grpo_group_norm_two_independent_batches (mixed tasks/datasets) passed")

if __name__ == "__main__":
    print("\n🚀 Running HARPO core algorithm unit tests...\n")
    # Original single-batch tests
    test_shapes_and_broadcast_basic_pass_through()
    # test_task_adapter_normalization_from_batch_stats()
    # test_class_weighting_only_matches_formula()
    # test_cvar_boost_applies_taskwide_scaling()
    # test_grpo_group_norm_normalizes_within_qid()

    # New two-batch tests for global updates
    test_task_adapter_updates_across_two_batches()
    test_class_weighting_across_batches()
    test_cvar_boost_updates_across_two_batches()
    test_grpo_group_norm_two_independent_batches()

    print("\n🎉 All HARPO tests passed successfully!\n")