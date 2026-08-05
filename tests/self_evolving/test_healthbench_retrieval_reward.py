"""Reward-side invariants for the retrieval RL run.

The two failure classes these guard against are both SILENT:

  * key-set drift — the trainer snapshots `reward_extra_keys` from SAMPLE 0 only, so
    a key missing on sample 0 is dropped batch-wide and a key missing on sample k
    raises KeyError mid-step;
  * validation drift — the retrieval term must move the TRAINING scalar only, so the
    val curve stays comparable to the no-retrieval baseline.

No network: the judge endpoints are left unset, which routes to the "no grader
configured" path. The coverage judge is exercised separately (it needs a judge).

    pytest tests/self_evolving/test_healthbench_retrieval_reward.py -q
"""

import asyncio
import itertools

import numpy as np
import pytest

from verl.utils.reward_score import healthbench_pro as HP

ITEMS = [
    {"criterion_text": "States metformin is contraindicated below eGFR 30", "points": 8},
    {"criterion_text": "Asks about the patient's hydration status", "points": 3},
    {"criterion_text": "Claims metformin is always safe regardless of renal function", "points": -5},
]
EI = {"rubric_items": ITEMS, "conversation": [{"role": "user", "content": "eGFR 38, metformin?"}],
      "question_id": "q1"}
RETRIEVAL = {
    "retrieval_context": "[passage 1 | source=statpearls]\nMetformin is contraindicated below eGFR 30.",
    "search_queries": ["metformin eGFR threshold", "lactic acidosis"],
    "n_search": 1, "n_queries": 2, "retrieval_hits": 9, "retrieval_error": 0,
    "retrieval_truncated": 0, "answer_rescued": 0, "budget_exhausted": 0,
}


def run(**kw):
    return asyncio.run(HP.compute_score(**kw))


def _paths():
    """The three distinct return paths of compute_score."""
    unclosed = run(data_source="healthbench_self", solution_str="<think>runaway reasoning",
                   ground_truth="", extra_info=dict(EI))
    no_rubric = run(data_source="healthbench_self", solution_str="<think>x</think>ans",
                    ground_truth="", extra_info={"rubric_items": []})
    no_grader = run(data_source="healthbench_professional/consult",
                    solution_str="<think>x</think>ans", ground_truth="",
                    extra_info={**EI, "_is_validation": True})
    return unclosed, no_rubric, no_grader


def test_key_set_identical_across_all_return_paths():
    for a, b in itertools.combinations(_paths(), 2):
        assert set(a) == set(b), set(a) ^ set(b)


def test_result_is_a_superset_of_the_shared_floor():
    """`_result` splices HOMOGENEOUS_DEFAULTS, so adding a key there can never leave
    a branch short of it."""
    for r in _paths():
        assert set(HP.HOMOGENEOUS_DEFAULTS) <= set(r), set(HP.HOMOGENEOUS_DEFAULTS) - set(r)


def test_retrieval_keys_are_always_present():
    need = {"retrieval_used", "retrieval_coverage", "retrieval_judged", "retrieval_bonus",
            "n_search", "n_queries", "retrieval_hits", "retrieval_error",
            "retrieval_truncated", "answer_rescued", "budget_exhausted", "judge_fail"}
    for r in _paths():
        assert need <= set(r), need - set(r)


def test_mixed_batch_stacks_like_the_trainer_does():
    """This is the operation that actually raises: keys from sample 0, values from all."""
    rows = list(_paths()) + [
        run(data_source="healthbench_self", solution_str="<think>runaway",
            ground_truth="", extra_info={**EI, **RETRIEVAL}),
    ]
    keys = list(rows[0].keys())
    stacked = {k: np.array([r[k] for r in rows]) for k in keys}
    assert stacked["score"].shape == (4,)


def test_telemetry_survives_the_early_return():
    """A rollout can search and THEN run out of budget mid-thinking; its retrieval
    telemetry must not be lost to the unclosed-think shortcut."""
    r = run(data_source="healthbench_self", solution_str="<think>runaway",
            ground_truth="", extra_info={**EI, **RETRIEVAL})
    assert r["n_search"] == 1.0 and r["retrieval_used"] == 1.0 and r["retrieval_hits"] == 9.0
    assert r["n_queries"] == 2.0
    # ...but the unclosed-think score is a fixed constant by design and must not be
    # topped up by a retrieval bonus.
    assert r["retrieval_bonus"] == 0.0


def test_validation_metrics_are_untouched_by_retrieval():
    """acc / acc_raw / acc_len_adj must be bit-identical with and without retrieval,
    and the coverage judge must not run at all on a validation row."""
    base = run(data_source="healthbench_professional/consult",
               solution_str="<think>t</think>" + "a" * 2000, ground_truth="",
               extra_info={**EI, "_is_validation": True})
    withret = run(data_source="healthbench_professional/consult",
                  solution_str="<think>t</think>" + "a" * 2000, ground_truth="",
                  extra_info={**EI, "_is_validation": True, **RETRIEVAL})
    for k in ("acc", "acc_raw", "acc_len_adj", "acc_raw_signed", "acc_len_adj_signed", "score"):
        assert base[k] == withret[k], (k, base[k], withret[k])
    assert withret["retrieval_coverage"] == 0.0 and withret["retrieval_judged"] == 0.0
    assert withret["retrieval_bonus"] == 0.0


def test_is_train_is_not_is_val():
    """On a train-on-val run every row's data_source is healthbench_professional/*,
    so `is_val` is True for TRAINING rows too. Gating the retrieval term on `is_val`
    would silently disable it on exactly the run it was built for; the gate is the
    real `_is_validation` flag. A training row must still report its telemetry."""
    train_row = run(data_source="healthbench_professional/consult",
                    solution_str="<think>t</think>answer", ground_truth="",
                    extra_info={**EI, **RETRIEVAL})       # no _is_validation => training
    assert train_row["retrieval_used"] == 1.0
    val_row = run(data_source="healthbench_professional/consult",
                  solution_str="<think>t</think>answer", ground_truth="",
                  extra_info={**EI, "_is_validation": True, **RETRIEVAL})
    assert val_row["retrieval_used"] == 1.0     # telemetry logged on both
    assert val_row["retrieval_judged"] == 0.0   # but never judged at val


def test_unterminated_tool_span_does_not_leak_into_the_answer():
    """A response truncated mid-<tool_response> used to leak the whole passage block
    into the graded answer, the think-char count and the repetition penalty."""
    leaked = "<think>t</think>real answer <tool_response>\n[passage 1]\nRAW PASSAGE TEXT"
    assert "RAW PASSAGE" not in HP._strip_thinking(leaked)
    assert HP._strip_thinking(leaked) == "real answer"


def test_retrieval_bonus_only_moves_score():
    """The bonus is applied to `score` and never to `length_adjusted`."""
    a = HP._result(0.5, 0.5, "ans", is_val=False)
    b = HP._result(0.5, 0.5, "ans", is_val=False, retrieval_bonus=0.1)
    assert b["score"] == pytest.approx(a["score"] + 0.1)
    for k in ("acc", "acc_raw", "acc_raw_signed", "acc_len_adj_signed"):
        assert a[k] == b[k], k


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
