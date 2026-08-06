# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Trainer-side spec-gap hooks, driven without Ray, GPUs or a real trainer.

The four hooks touch a small, deliberately enumerated set of attributes
(`config`, `tokenizer`, `global_steps`, `use_critic`, and the lazily created
`_spec_gap_*` state), which is what lets a stand-in object drive them here.

The property that matters most is the LAST test: with `spec_gap` unset, the hooks
must be a total no-op — no metrics, no extras, no advantage change.
"""

import types

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _conversation_text_for_spec_gap
from verl.utils.reward_score.spec_gap import GroupStats

REFEREE_KWARGS = {
    "val_api_base": "http://referee.invalid/v1",
    "val_api_key": "k",
    "val_model_name": "referee-model",
    "val_provider": "trapi",
}


class _Tok:
    def decode(self, ids, skip_special_tokens=True):
        # Encode the row index in the text so answers stay distinguishable.
        return f"<think>reasoning</think>answer for row {int(ids[0].item())} " + "x" * 64


def _trainer(**se):
    t = types.SimpleNamespace()
    t.config = OmegaConf.create({
        "data": {"self_evolving": {"spec_gap": True, **se}},
        "reward": {"custom_reward_function": {"reward_kwargs": dict(REFEREE_KWARGS)}},
        "algorithm": {"norm_adv_by_std_in_grpo": False},
    })
    t.tokenizer = _Tok()
    t.global_steps = 10
    t.use_critic = False
    for name in ("_spec_gap_cfg", "_spec_gap_payload", "_submit_spec_gap", "_apply_spec_gap",
                 "_run_spec_gap", "_maybe_ship_exploits"):
        setattr(t, name, getattr(RayPPOTrainer, name).__get__(t, types.SimpleNamespace))
    return t


def _batch(n_groups=2, n=4, scores=None, think_closed=None, resp_len=8):
    """A synthetic batch with the columns the hooks read."""
    total = n_groups * n
    uids = np.array([f"u{g}" for g in range(n_groups) for _ in range(n)], dtype=object)
    prompts = torch.arange(total).unsqueeze(-1).repeat(1, 3)
    responses = torch.arange(total).unsqueeze(-1).repeat(1, resp_len)
    attn = torch.ones(total, 3 + resp_len, dtype=torch.long)
    adv = torch.zeros(total, resp_len)
    for g in range(n_groups):
        sl = slice(g * n, (g + 1) * n)
        raw = torch.tensor([1.0, 0.5, -0.5, -1.0][:n])
        adv[sl] = (raw - raw.mean()).unsqueeze(-1).repeat(1, resp_len)
    ex = np.array([{
        "question_id": f"q{i // n}", "use_case": "consult",
        "conversation": [{"role": "user", "content": f"clinician question {i // n}"}],
        "rubric_items": [{"criterion_text": "c", "points": 8.0}],
    } for i in range(total)], dtype=object)
    batch = DataProto.from_dict(
        tensors={"prompts": prompts, "responses": responses, "attention_mask": attn,
                 "advantages": adv, "returns": adv},
        non_tensors={"uid": uids, "extra_info": ex},
    )
    extras = {
        "acc_raw_signed": scores if scores is not None else [0.9, 0.6, 0.3, 0.0] * n_groups,
        "think_closed": think_closed if think_closed is not None else [1.0] * total,
        "rubric_met": ['[{"criterion": "c", "points": 8.0, "met": true}]'] * total,
    }
    return batch, extras


# ----------------------------------------------------------------------
# config resolution
# ----------------------------------------------------------------------
def test_disabled_returns_empty_cfg():
    t = _trainer()
    t.config.data.self_evolving.spec_gap = False
    assert t._spec_gap_cfg() == {}


def test_referee_defaults_to_the_val_judge_not_the_train_judge():
    # Under SELF_JUDGE=1 the train judge becomes a frozen model of the policy's own
    # family; a referee inheriting it could not detect a hack the two of them share.
    t = _trainer()
    t.config.reward.custom_reward_function.reward_kwargs["api_base"] = "http://train-judge/v1"
    t.config.reward.custom_reward_function.reward_kwargs["model_name"] = "local-9b"
    cfg = t._spec_gap_cfg()
    assert cfg["api_base"] == REFEREE_KWARGS["val_api_base"]
    assert cfg["model_name"] == "referee-model"


def test_missing_referee_endpoint_disables_permanently():
    t = _trainer()
    t.config.reward.custom_reward_function.reward_kwargs = {}
    assert t._spec_gap_cfg() == {}
    assert t._spec_gap_dead is True
    assert t._spec_gap_cfg() == {}          # stays off, no repeated probing


def test_std_norm_with_attenuation_is_a_hard_assert():
    # A group-constant weight cancels against the group std, so the treatment would
    # be a silent no-op that still logs as active.
    t = _trainer(spec_gap_mode="soft")
    t.config.algorithm.norm_adv_by_std_in_grpo = True
    with pytest.raises(AssertionError, match="norm_adv_by_std_in_grpo=False"):
        t._spec_gap_cfg()


def test_std_norm_is_fine_in_measure_mode():
    t = _trainer(spec_gap_mode="measure")
    t.config.algorithm.norm_adv_by_std_in_grpo = True
    assert t._spec_gap_cfg()["mode"] == "measure"


# ----------------------------------------------------------------------
# payload construction
# ----------------------------------------------------------------------
def test_payload_groups_by_uid_and_carries_only_plain_python():
    t = _trainer()
    batch, extras = _batch()
    payload, meta, metrics = t._spec_gap_payload(batch, extras, t._spec_gap_cfg())
    assert {g["uid"] for g in payload} == {"u0", "u1"}
    for g in payload:
        assert len(g["rows"]) == 4
        assert "clinician question" in g["task"]
        for v in (g["answers"], g["scores"], g["lens"]):
            assert isinstance(v, dict)
        # no tensors, no numpy, no DataProto: the worker thread runs concurrently
        # with the main thread mutating the batch
        assert all(isinstance(x, str) for x in g["answers"].values())
        assert all(isinstance(x, float) for x in g["scores"].values())
    assert meta["u0"]["question_id"] == "q0"
    assert metrics["spec_gap/excluded_unclosed_frac"] == 0.0


def test_think_stripping_hides_the_reasoning_channel_from_the_referee():
    t = _trainer()
    batch, extras = _batch()
    payload, _, _ = t._spec_gap_payload(batch, extras, t._spec_gap_cfg())
    joined = " ".join(a for g in payload for a in g["answers"].values())
    assert "reasoning" not in joined and "<think>" not in joined


def test_unclosed_think_rows_are_excluded_and_counted():
    # They sit at the reward floor by construction, so pairs against them are
    # trivially concordant and would inflate C.
    t = _trainer(spec_gap_min_ranked=2)
    tc = [1.0, 1.0, 0.0, 0.0] * 2
    batch, extras = _batch(think_closed=tc)
    payload, _, metrics = t._spec_gap_payload(batch, extras, t._spec_gap_cfg())
    assert all(len(g["rows"]) == 2 for g in payload)
    assert metrics["spec_gap/excluded_unclosed_frac"] == 0.5


def test_groups_below_min_ranked_are_dropped_entirely():
    t = _trainer(spec_gap_min_ranked=4)
    tc = [1.0, 1.0, 0.0, 0.0] * 2
    batch, extras = _batch(think_closed=tc)
    payload, _, _ = t._spec_gap_payload(batch, extras, t._spec_gap_cfg())
    assert payload == []


def test_score_key_falls_back_and_tolerates_numpy_extras():
    t = _trainer(spec_gap_score_key="acc_raw_signed")
    batch, extras = _batch()
    extras.pop("acc_raw_signed")
    extras["acc_raw"] = np.array([0.9, 0.6, 0.3, 0.0] * 2)   # numpy, not list
    payload, _, _ = t._spec_gap_payload(batch, extras, t._spec_gap_cfg())
    assert payload and all(isinstance(v, float) for v in payload[0]["scores"].values())


def test_no_uid_column_yields_no_payload():
    t = _trainer()
    batch, extras = _batch()
    batch.non_tensor_batch.pop("uid")
    assert t._spec_gap_payload(batch, extras, t._spec_gap_cfg()) == ([], {}, {})


# ----------------------------------------------------------------------
# apply: weighting, extras, exploits
# ----------------------------------------------------------------------
def _stats_for(batch, agree=("u0",), disagree=("u1",)):
    """Hand-built referee verdicts: concordant for `agree`, inverted for `disagree`."""
    out = {}
    uids = np.asarray(batch.non_tensor_batch["uid"])
    for uid in list(dict.fromkeys(uids.tolist())):
        rows = [int(i) for i in np.nonzero(uids == uid)[0]]
        # scores descend with row order, so tiers ascending == agreement
        tiers = {r: k for k, r in enumerate(rows)}
        if uid in disagree:
            tiers = {r: k for k, r in enumerate(reversed(rows))}
        n_pairs = len(rows) * (len(rows) - 1) // 2
        out[uid] = GroupStats(
            h=(1.0 if uid in disagree else 0.0), c=(0.0 if uid in disagree else 1.0),
            d_std=0.33, n_dec=n_pairs, n_disc=(n_pairs if uid in disagree else 0),
            n_ref_dec=n_pairs, n_tiers=len(rows), n_ranked=len(rows),
            tier_of=tiers, ranked_rows=rows, judged=True, notes="hedged criteria",
        )
    return out


def _prime(t, batch, extras, stats):
    cfg = t._spec_gap_cfg()
    payload, meta, metrics = t._spec_gap_payload(batch, extras, cfg)
    t._spec_gap_pending = (cfg, meta, metrics)
    t._spec_gap_result = (stats, meta, metrics)
    return cfg


def test_measure_mode_leaves_advantages_untouched_but_reports_counterfactual():
    t = _trainer(spec_gap_mode="measure", spec_gap_prior_pairs=0.0)
    batch, extras = _batch()
    before = batch.batch["advantages"].clone()
    _prime(t, batch, extras, _stats_for(batch))
    m = t._apply_spec_gap(batch, extras)
    assert torch.equal(batch.batch["advantages"], before)
    assert m["spec_gap/adv_scale"] == 1.0
    assert m["spec_gap/adv_scale_would_be"] == pytest.approx(0.5)   # one of two groups zeroed
    assert m["spec_gap/H/mean"] == pytest.approx(0.5)


def test_soft_mode_zeroes_the_inverted_group_and_keeps_the_clean_one():
    t = _trainer(spec_gap_mode="soft", spec_gap_prior_pairs=0.0)
    batch, extras = _batch()
    before = batch.batch["advantages"].clone()
    _prime(t, batch, extras, _stats_for(batch))
    m = t._apply_spec_gap(batch, extras)
    adv = batch.batch["advantages"]
    assert torch.equal(adv[:4], before[:4])            # u0: H=0 -> w=1
    assert adv[4:].abs().sum().item() == 0.0           # u1: H=1 -> w=0
    assert m["spec_gap/adv_scale"] == pytest.approx(0.5)
    # returns shared storage with advantages, so it must have followed
    assert torch.equal(batch.batch["returns"], adv)


def test_weighting_preserves_group_mean_zero():
    t = _trainer(spec_gap_mode="soft", spec_gap_prior_pairs=8.0)
    batch, extras = _batch()
    _prime(t, batch, extras, _stats_for(batch))
    t._apply_spec_gap(batch, extras)
    adv = batch.batch["advantages"]
    for sl in (slice(0, 4), slice(4, 8)):
        assert adv[sl].mean().abs().item() < 1e-6


def test_returns_are_not_touched_when_a_critic_owns_them():
    t = _trainer(spec_gap_mode="soft", spec_gap_prior_pairs=0.0)
    t.use_critic = True
    batch, extras = _batch()
    ret_before = batch.batch["returns"].clone()
    _prime(t, batch, extras, _stats_for(batch))
    t._apply_spec_gap(batch, extras)
    # advantages moved; the value target must not have been corrupted with them
    assert torch.equal(batch.batch["returns"], ret_before)
    assert not torch.equal(batch.batch["advantages"], ret_before)


def test_per_row_extras_land_for_the_rollout_dump():
    t = _trainer(spec_gap_mode="soft", spec_gap_prior_pairs=0.0)
    batch, extras = _batch()
    _prime(t, batch, extras, _stats_for(batch))
    t._apply_spec_gap(batch, extras)
    n = len(batch.batch)
    for k in ("spec_gap_H", "spec_gap_w", "spec_gap_measured", "referee_tier", "uid"):
        assert len(extras[k]) == n, k
    assert extras["spec_gap_w"][:4] == [1.0] * 4
    assert extras["spec_gap_w"][4:] == [0.0] * 4
    assert set(extras["referee_tier"]) == {0.0, 1.0, 2.0, 3.0}


def test_referee_failure_leaves_the_batch_alone():
    t = _trainer(spec_gap_mode="soft")
    batch, extras = _batch()
    before = batch.batch["advantages"].clone()
    _prime(t, batch, extras, {})               # empty == every referee call failed
    m = t._apply_spec_gap(batch, extras)
    assert torch.equal(batch.batch["advantages"], before)
    assert m["spec_gap/referee/fail_frac"] == 1.0


def test_repeated_referee_failure_disables_the_feature():
    t = _trainer(spec_gap_mode="soft", spec_gap_max_fail_steps=2)
    batch, extras = _batch()
    _prime(t, batch, extras, {})
    t._apply_spec_gap(batch, extras)
    _prime(t, batch, extras, {})
    m = t._apply_spec_gap(batch, extras)
    assert m["spec_gap/dead"] == 1.0
    assert t._spec_gap_cfg() == {}


def test_exploits_are_buffered_only_when_shipping_is_on():
    batch, extras = _batch()
    t = _trainer(spec_gap_mode="measure", spec_gap_prior_pairs=0.0)
    _prime(t, batch, extras, _stats_for(batch))
    t._apply_spec_gap(batch, extras)
    assert not getattr(t, "_spec_gap_exploits", None)

    t2 = _trainer(spec_gap_mode="measure", spec_gap_prior_pairs=0.0,
                  spec_gap_ship_exploits=True, spec_gap_exploit_margin=0.15)
    batch2, extras2 = _batch()
    _prime(t2, batch2, extras2, _stats_for(batch2))
    m = t2._apply_spec_gap(batch2, extras2)
    assert m["spec_gap/exploits/found"] == 1.0        # only the inverted group
    ex = list(t2._spec_gap_exploits)[0]
    assert ex["question_id"] == "q1" and ex["H"] == 1.0
    assert ex["hacked"]["referee_tier"] > 0 and ex["preferred"]["referee_tier"] == 0
    assert ex["hacked"]["rubric_score"] > ex["preferred"]["rubric_score"]


def test_ship_keeps_the_buffer_when_the_post_fails():
    # A confirmed exploit costs a full step of rollouts to find; a failed POST must
    # never destroy it.
    t = _trainer(spec_gap_ship_exploits=True)
    t._spec_gap_exploits = __import__("collections").deque([{"H": 0.9, "question_id": "q"}])
    m = t._maybe_ship_exploits("http://127.0.0.1:1/", OmegaConf.create({}))
    assert m["spec_gap/exploits/shipped"] == 0.0
    assert len(t._spec_gap_exploits) == 1


# ----------------------------------------------------------------------
# the no-op guarantee
# ----------------------------------------------------------------------
def test_hooks_are_a_total_no_op_when_disabled():
    t = _trainer()
    t.config.data.self_evolving.spec_gap = False
    batch, extras = _batch()
    before = batch.batch["advantages"].clone()
    keys_before = set(extras)
    t._submit_spec_gap(batch, extras)
    assert t._apply_spec_gap(batch, extras) == {}
    assert torch.equal(batch.batch["advantages"], before)
    assert set(extras) == keys_before


def test_cadence_skips_off_steps():
    t = _trainer(spec_gap_every_n_steps=5)
    t.global_steps = 7
    batch, extras = _batch()
    t._submit_spec_gap(batch, extras)
    assert t._spec_gap_result is None
    assert t._apply_spec_gap(batch, extras) == {}


def test_conversation_text_excludes_assistant_turns_and_falls_back():
    ex = {"conversation": [{"role": "user", "content": "q1"},
                           {"role": "assistant", "content": "LEAK"},
                           {"role": "user", "content": "q2"}]}
    txt = _conversation_text_for_spec_gap(ex)
    assert "q1" in txt and "q2" in txt and "LEAK" not in txt
    assert _conversation_text_for_spec_gap({"question": "fallback"}) == "fallback"
    assert _conversation_text_for_spec_gap({}) == ""
