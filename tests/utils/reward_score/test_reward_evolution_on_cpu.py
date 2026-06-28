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
"""CPU tests for the self-evolving reward module (no GPU / no judge API needed).

Covers the meta-prompt parsers, the sandboxed function validator, the on-disk
EvolutionStore, and the compute_score routing (OFF / ON / validation) in
verl/utils/reward_score/self_evolving.py.
"""

import asyncio
import os
import tempfile

from verl.utils.reward_score import reward_evolution as RE
from verl.utils.reward_score import self_evolving as SE

GOOD_FN = (
    "import re\n"
    "def function_reward(input_question, output_answer, ground_truth):\n"
    "    code = ground_truth.split(':')[0].strip().lower()\n"
    "    return 1.0 if code and code in output_answer.lower() else 0.0\n"
)

EXAMPLES = [
    {"question": "q1", "response": "final answer e70.0 pku", "ground_truth": "E70.0: PKU",
     "extracted_answer": "E70.0", "combined_reward": 1.0, "sub_rewards": {"acc": 1.0}},
    {"question": "q2", "response": "i think it is something else", "ground_truth": "K22.0: Achalasia",
     "extracted_answer": "E83.11", "combined_reward": 0.05, "sub_rewards": {"acc": 0.0}},
]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def test_parse_evolved_prompt_takes_last_outside_reasoning():
    text = (
        "reasoning <think><prompt>WRONG</prompt></think> mid\n"
        "<prompt>\nReceive [QUESTION] [MODEL RESPONSE] [CORRECT ANSWER]. Output \\boxed{N}.\n</prompt>"
    )
    p = RE.parse_evolved_prompt(text)
    assert p is not None
    assert "[CORRECT ANSWER]" in p and "\\boxed" in p
    assert "WRONG" not in p


def test_parse_evolved_prompt_missing():
    assert RE.parse_evolved_prompt("no tags here") is None
    assert RE.parse_evolved_prompt("") is None


def test_parse_evolved_function_fenced_and_unfenced():
    fenced = f"blah\n```python\n{GOOD_FN}```\nafter"
    src = RE.parse_evolved_function(fenced)
    assert src and "def function_reward" in src

    unfenced = f"some text\n{GOOD_FN}\n"
    src2 = RE.parse_evolved_function(unfenced)
    assert src2 and "def function_reward" in src2

    assert RE.parse_evolved_function("no function at all") is None


# ---------------------------------------------------------------------------
# Validator (sandboxed subprocess)
# ---------------------------------------------------------------------------

def test_validate_good_function():
    ok, err, outs = RE.validate_function_src(GOOD_FN, EXAMPLES)
    assert ok, err
    assert outs == [1.0, 0.0]


def test_validate_rejects_syntax_error():
    ok, err, _ = RE.validate_function_src("def function_reward(a, b, c)\n return 0", EXAMPLES)
    assert not ok
    assert "Syntax" in err or "syntax" in err


def test_validate_rejects_runtime_error():
    bad = "def function_reward(a, b, c):\n    return 1 / 0\n"
    ok, err, _ = RE.validate_function_src(bad, EXAMPLES)
    assert not ok
    assert "ZeroDivision" in err


def test_validate_rejects_infinite_loop():
    loop = "def function_reward(a, b, c):\n    while True:\n        pass\n"
    ok, err, _ = RE.validate_function_src(loop, EXAMPLES, timeout_s=3)
    assert not ok
    assert "timeout" in err.lower()


def test_validate_starter_is_valid_constant():
    ok, err, outs = RE.validate_function_src(RE.STARTER_FUNCTION_SRC, EXAMPLES)
    assert ok, err
    assert outs == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def test_clamp01():
    assert RE.clamp01(2.0) == 1.0
    assert RE.clamp01(-1.0) == 0.0
    assert RE.clamp01(float("nan")) == 0.0
    assert RE.clamp01("bad") == 0.0


def test_safe_call_function_clamps_and_guards():
    fn = RE.load_function("def function_reward(a, b, c):\n    return 5.0\n")
    assert RE.safe_call_function(fn, "q", "a", "g") == 1.0
    raiser = RE.load_function("def function_reward(a, b, c):\n    raise ValueError('x')\n")
    assert RE.safe_call_function(raiser, "q", "a", "g") == 0.0


def test_extract_score_0_10():
    assert RE._extract_score_0_10("\\boxed{7}") == 0.7
    assert RE._extract_score_0_10("I rate this 10/10") == 1.0
    assert RE._extract_score_0_10("no number here", default=0.0) == 0.0


# ---------------------------------------------------------------------------
# EvolutionStore
# ---------------------------------------------------------------------------

def test_evolution_store_init_commit_atomic_current():
    with tempfile.TemporaryDirectory() as d:
        store = RE.EvolutionStore(d)
        store.init_if_needed()
        assert store.is_initialized()
        # step_000 preserves the starter
        assert os.path.exists(os.path.join(d, "step_000", "function_reward.py"))
        cp, cf = store.read_current()
        assert cp == RE.STARTER_JUDGE_PROMPT
        assert cf == RE.STARTER_FUNCTION_SRC

        store.commit_step(1, "NEW PROMPT \\boxed{N}", GOOD_FN)
        cp2, cf2 = store.read_current()
        assert cp2.startswith("NEW PROMPT")
        assert "def function_reward" in cf2
        assert os.path.exists(os.path.join(d, "step_001", "function_reward.py"))

        # get_current_artifacts loads a callable and caches by mtime
        prompt, fn, src = RE.get_current_artifacts(d)
        assert prompt.startswith("NEW PROMPT")
        assert callable(fn)
        assert RE.safe_call_function(fn, "q", "e70.0", "E70.0: x") == 1.0


def test_get_current_artifacts_uninitialized_falls_back_to_starter():
    with tempfile.TemporaryDirectory() as d:
        prompt, fn, src = RE.get_current_artifacts(d)
        assert prompt == RE.STARTER_JUDGE_PROMPT
        assert callable(fn)
        assert RE.safe_call_function(fn, "q", "a", "g") == 0.0


# ---------------------------------------------------------------------------
# Example builders
# ---------------------------------------------------------------------------

def test_make_example_and_select_contrastive():
    row = {"score": 0.12, "acc": 0.0, "dynamic_judge": 0.3, "dynamic_function": 0.1,
           "extracted_answer": "E83.11"}
    ex = RE.make_example("the question", "resp", "K22.0: Achalasia", row)
    assert ex["combined_reward"] == 0.12
    assert ex["sub_rewards"]["dynamic_judge"] == 0.3
    assert ex["extracted_answer"] == "E83.11"

    pool = [{"combined_reward": i / 10.0} for i in range(20)]  # 0.0 .. 1.9
    sel = RE.select_contrastive(pool, 6)
    rewards = [e["combined_reward"] for e in sel]
    assert rewards[0] == 0.0 and rewards[-1] == 1.9  # min and max included
    assert len(sel) == 6


# ---------------------------------------------------------------------------
# compute_score routing (OFF / ON / validation) — no judge API (api_base="")
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def test_compute_score_off_path_is_composite():
    """evolve_enable=False -> original composite dict, no dynamic_judge/dynamic_function keys."""
    res = _run(
        SE.compute_score(
            data_source="mimic_rare/test",
            solution_str="reasoning... \\boxed{E70.0: PKU}",
            ground_truth="E70.0: PKU",
            extra_info={"question": "q"},
            api_base="",  # no judge API -> composite uses local-only signals
            evolve_enable=False,
        )
    )
    assert "score" in res and "acc" in res
    assert "dynamic_judge" not in res
    assert "dynamic_function" not in res
    # exact match -> acc 1.0, format present
    assert res["acc"] == 1.0
    assert res["format_ok"] == 1.0


def test_compute_score_on_path_is_composite_plus_addon():
    """evolve_enable=True + not validation -> composite reward PLUS dynamic_judge/dynamic_function
    add-on keys. With api_base='' the dynamic terms are 0, so the total equals the renormalized
    composite (composite / (1 + w_judge + w_func)); the composite components are still present."""
    with tempfile.TemporaryDirectory() as d:
        RE.EvolutionStore(d).init_if_needed()  # starter: judge 0 (no api), function 0
        off = _run(
            SE.compute_score(
                data_source="mimic_rare/test",
                solution_str="\\boxed{E70.0: PKU}",
                ground_truth="E70.0: PKU",
                extra_info={"question": "q"},
                api_base="",
                evolve_enable=False,
            )
        )
        on = _run(
            SE.compute_score(
                data_source="mimic_rare/test",
                solution_str="\\boxed{E70.0: PKU}",
                ground_truth="E70.0: PKU",
                extra_info={"question": "q"},
                api_base="",  # dynamic terms unreachable -> 0
                evolve_enable=True,
                evolve_dir=d,
                evolve_w_judge=0.7,
                evolve_w_func=0.3,
            )
        )
        # add-on keys present and 0 (no api / starter fn)
        assert on["dynamic_judge"] == 0.0
        assert on["dynamic_function"] == 0.0
        # original composite components are STILL present (add-on, not replacement)
        assert "answer_quality" in on and "char_bleu" in on
        assert on["acc"] == 1.0
        # total = composite / (1 + 0.7 + 0.3); composite reward is preserved as the base
        assert abs(on["score"] - off["score"] / 2.0) < 1e-6


def test_compute_score_validation_uses_pure_composite_even_when_enabled():
    """evolve_enable=True but _is_validation=True -> pure composite (no add-on keys)."""
    with tempfile.TemporaryDirectory() as d:
        RE.EvolutionStore(d).init_if_needed()
        res = _run(
            SE.compute_score(
                data_source="mimic_rare/test",
                solution_str="\\boxed{E70.0: PKU}",
                ground_truth="E70.0: PKU",
                extra_info={"question": "q", "_is_validation": True},
                api_base="",
                evolve_enable=True,
                evolve_dir=d,
            )
        )
        assert "dynamic_judge" not in res
        assert "dynamic_function" not in res
        assert res["acc"] == 1.0
