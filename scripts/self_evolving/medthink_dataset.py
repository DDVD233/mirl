"""MedThinkVQA dataset whose SOLVER system prompt is evolvable at run time.

On HealthBench the evolution loop rewrites the prompts that GENERATE tasks. Here
the task set is a real fixed split, so there is nothing to generate — the thing
worth evolving is the prompt the SOLVER runs under: how it is told to read each
image, integrate across views, and commit to a diagnosis. The trainer's
``/evolve_solver`` round rewrites that prompt from an error analysis of the
step's failures, and this dataset makes the rewrite take effect.

WHY A FILE AND NOT THE PARQUET. The system message is baked into every row at
build time, and the dataset object is constructed once at trainer start, so a
rewritten prompt would otherwise never reach a rollout: the run would report
prompt evolution as enabled while training under the version it started with.
That is precisely the class of silent no-op that has cost this project several
runs, so the prompt is read at ACCESS time, per item.

Change detection compares CONTENT, not mtime: two writes inside one filesystem
timestamp tick report the same mtime, and the failure mode is silent — the run
keeps the stale prompt with nothing in the logs. (Same reason, same fix, as the
coverage judge in verl/utils/reward_score/retrieval_coverage.py.)
"""

from __future__ import annotations

import logging
import os

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)

_CACHE: dict = {"raw": None, "text": None}


def current_solver_prompt(path: str, fallback: str) -> str:
    """Current solver system prompt, or `fallback` when the file is absent/empty.

    A missing file is the NORMAL state before the first evolution round, so it is
    not an error; an unreadable one keeps the last good value rather than training
    on an empty system message.
    """
    if not path:
        return fallback
    try:
        with open(path) as f:
            raw = f.read()
    except OSError:
        return _CACHE["text"] or fallback
    if raw == _CACHE["raw"]:
        return _CACHE["text"] or fallback
    text = raw.strip()
    if not text:
        logger.error("solver prompt %s is empty; keeping previous", path)
        _CACHE["raw"] = raw
        return _CACHE["text"] or fallback
    _CACHE.update({"raw": raw, "text": text})
    logger.warning("solver prompt reloaded from %s (%d chars)", path, len(text))
    return text


class MedThinkVQADataset(RLHFDataset):
    """RLHFDataset that swaps in the current solver system prompt per item."""

    def _build_messages(self, example: dict, key: str = "prompt"):
        path = os.environ.get("MTV_SOLVER_PROMPT_FILE", "")
        if path:
            msgs = example.get(key)
            if isinstance(msgs, (list, tuple)) and msgs:
                first = msgs[0]
                if isinstance(first, dict) and first.get("role") == "system":
                    cur = current_solver_prompt(path, first.get("content", ""))
                    if cur and cur != first.get("content"):
                        # Mutating the cached row is intentional and idempotent:
                        # every access rewrites it to whatever the file currently
                        # holds, so a later evolution round overwrites this one.
                        first["content"] = cur
        return super()._build_messages(example, key=key)
