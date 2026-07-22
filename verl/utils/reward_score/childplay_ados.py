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
"""Reward for ChildPlay ADOS-2 item scoring.

Responses must follow:
    Reasoning: <justification>
    Score: [x]

Reward = 0.9 * exact score match + 0.1 * format compliance.
"""

import re

# "Score: [3]", "score: 3", "Score:[ 3 ]", full-width colon, optional brackets
_SCORE_RE = re.compile(r"[Ss]core\s*[:：]?\s*\[?\s*([0-9])\s*\]?")
_FORMAT_RE = re.compile(
    r"^\s*Reasoning\s*[:：].+?Score\s*[:：]\s*\[?\s*[0-9]\s*\]?\s*$", re.DOTALL
)
_LAST_DIGIT_RE = re.compile(r"(?<![\w.])([0-9])(?![\w.])")


def extract_score(predict_str: str) -> str | None:
    """Extract the predicted score label from a model response."""
    matches = _SCORE_RE.findall(predict_str)
    if matches:
        return matches[-1]
    # Fallback: last standalone digit on the final non-empty line.
    for line in reversed([ln for ln in predict_str.splitlines() if ln.strip()]):
        digits = _LAST_DIGIT_RE.findall(line)
        if digits:
            return digits[-1]
        break
    return None


def compute_score(predict_str: str, ground_truth: str) -> dict:
    pred = extract_score(predict_str)
    acc = 1.0 if pred is not None and pred == str(ground_truth).strip() else 0.0
    fmt = 1.0 if _FORMAT_RE.match(predict_str.strip()) else 0.0
    return {"score": 0.9 * acc + 0.1 * fmt, "acc": acc, "format": fmt}


if __name__ == "__main__":
    cases = [
        ("Reasoning: limited eye contact throughout.\nScore: [2]", "2", 1.0, 1.0),
        ("Reasoning: some pointing observed.\nScore: 1", "1", 1.0, 1.0),
        ("The child shows typical behavior. Score: [0]", "0", 1.0, 0.0),
        ("Reasoning: unclear.\nScore: [3]", "1", 0.0, 1.0),
        ("I cannot determine a score.", "2", 0.0, 0.0),
        ("Final answer\n2", "2", 1.0, 0.0),
    ]
    for text, gt, want_acc, want_fmt in cases:
        got = compute_score(text, gt)
        assert got["acc"] == want_acc, (text, gt, got)
        assert got["format"] == want_fmt, (text, gt, got)
    print("childplay_ados reward self-check passed")
