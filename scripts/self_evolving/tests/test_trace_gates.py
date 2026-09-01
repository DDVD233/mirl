"""Regression tests for the SFT trace gates.

The gates decide what the student imitates, and both directions of failure are
expensive and quiet:

  * too LOOSE and the corpus teaches the student to announce it was handed the
    answer, or to refuse to reason at all;
  * too TIGHT and most of the teacher's output is thrown away AND the survivors are
    a biased sample -- the first cut of the leak patterns rejected 6 of 7 ordinary
    clinical sentences because it matched "given"/"stated" on their own, where they
    are usually prepositions ("the diagnosis GIVEN these findings"). That silently
    selects for traces which avoid the normal way clinicians write.

So both directions are pinned here.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

SE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "mt", os.path.join(SE_DIR, "make_medxpert_traces.py"))
mt = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(mt)
except Exception as e:  # pragma: no cover
    pytest.skip(f"cannot import make_medxpert_traces: {e}", allow_module_level=True)


# Ordinary trace prose. None of this is a leak; all of it must survive.
CLEAN = [
    "The most likely diagnosis given these findings is Graves disease.",
    "As stated in the vignette, the patient has vitiligo.",
    "We are given a 40-year-old woman with heat intolerance.",
    "The correct answer is Graves disease because the goiter is diffuse.",
    "The lesion is benign as provided by the classic sonographic features.",
    "Considering the history provided, I favour a simple cyst.",
    "The finding is stated to be in the upper outer quadrant.",
    "Given the patient's age and autoimmune history, Graves is most likely.",
]

# The teacher admitting it was handed the answer. All must be caught.
LEAKY = [
    "The ground truth is Graves disease.",
    "We are told the answer is option B.",
    "The provided diagnosis is pulmonary embolism.",
    "Confidentially, the answer given is No PE.",
    "According to the answer key, this is benign.",
    "The diagnosis was provided as mild DR.",
    "Since the answer is given, I will justify option D.",
]


@pytest.mark.parametrize("text", CLEAN)
def test_ordinary_clinical_prose_is_not_a_leak(text):
    assert not mt._leaks(text), f"false positive on normal prose: {text!r}"


@pytest.mark.parametrize("text", LEAKY)
def test_real_disclosure_is_caught(text):
    assert mt._leaks(text), f"missed a real leak: {text!r}"


# --------------------------------------------------------------- refusal gate
def test_refusal_boilerplate_is_caught():
    """GPT-family deployments decline to emit visible reasoning; those are unusable."""
    assert mt._refuses("I can't provide detailed internal chain-of-thought reasoning.")
    assert mt._refuses("Instead, here is a brief clinical explanation:")
    assert mt._refuses("As an AI, I cannot share my hidden reasoning.")


def test_normal_trace_is_not_a_refusal():
    assert not mt._refuses(
        "Malignancy is not excluded on imaging alone, so biopsy is warranted.")


# ------------------------------------------------------------ think-block gate
def test_missing_think_block_is_rejected_not_tolerated():
    """A bare answer must NOT fall back to counting the whole text as reasoning."""
    assert mt._think_words("Final answer: (B)") == -1
    assert mt._think_words("<think>\none two three\n</think>\n\nFinal answer: (B)") == 3


# ------------------------------------------------------------- answer matching
def test_mcq_answer_must_match_ground_truth():
    assert mt._answer_ok("mcq", "<think>x</think>\n\nFinal answer: (B)", "B")
    assert not mt._answer_ok("mcq", "<think>x</think>\n\nFinal answer: (C)", "B")
    assert not mt._answer_ok("mcq", "<think>x</think>\n\nno commitment here", "B")


def test_open_answer_matching_is_not_mere_overlap():
    """'No PE' and 'Acute PE' share a token; a token-overlap match would confuse them."""
    assert mt._answer_ok("open", "Final answer: No PE", "No PE")
    assert mt._answer_ok("open", "Final answer: no  pe", "No PE")
    assert not mt._answer_ok("open", "Final answer: Acute PE", "No PE")
    assert not mt._answer_ok("open", "Final answer: Chronic PE", "Acute PE")


def test_completion_that_only_closes_the_think_block_is_recoverable():
    """Qwen's template puts the opening <think> in the PROMPT.

    Served without a reasoning parser, the completion therefore starts INSIDE the
    reasoning and carries only `</think>`. Treating that as "no reasoning block"
    rejected ~97% of the teacher's output and read as a throughput problem for
    hours. The reconstructed trace must expose a countable think block.
    """
    raw = "Here's a thinking process:\nweigh the options\n</think>\n\nFinal answer: (B)"
    rebuilt = "<think>\n" + raw
    assert mt._think_words(rebuilt) > 0
    assert mt._answer_ok("mcq", rebuilt, "B")
    assert mt._think_words(raw) == -1, "unreconstructed output should still be rejected"
