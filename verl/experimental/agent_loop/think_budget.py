# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Thinking-budget forcing for agent-loop generation.

Shared by ``SingleTurnAgentLoop`` and ``RetrievalToolAgentLoop``. Phase 1 decodes
at most ``think_budget`` tokens; if the reasoning channel is still open (no
``</think>``) we INJECT ``</think>\\n\\n`` (loss-masked, like a tool response) and
phase 2 decodes the answer with the remaining budget. Runaway thinking then
cannot eat the whole response window: every rollout yields a gradeable answer
instead of a zero-reward truncation cliff, and GRPO groups keep reward variance
instead of collapsing to all-zero. This is what stabilized the HealthBench v4–v6
runs; see the healthbench-rubric-diagnosis notes.
"""

from typing import Any, Optional
from uuid import uuid4

from verl.workers.rollout.replica import TokenOutput


async def generate_with_think_budget(
    *,
    server_manager,
    tokenizer,
    think_budget: int,
    think_close_ids: list[int],
    response_length: int,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    gen_kwargs: dict[str, Any],
    max_new_tokens: Optional[int] = None,
    request_id: Optional[str] = None,
) -> tuple[list[int], list[int], Optional[list[float]], TokenOutput]:
    """Two-phase generation with a hard cap on the thinking channel.

    ``max_new_tokens`` bounds the total tokens this call may emit (defaults to
    ``response_length``); the caller passes the remaining response budget in a
    multi-turn loop so later turns cannot overrun the window.

    ``request_id`` should be the rollout's stable id. The load balancer routes by
    request id, and phase 2's prompt is phase 1's prompt plus phase 1's tokens — so
    minting a fresh id per phase can land the continuation on a different replica and
    throw away the whole KV prefix, re-prefilling it. With a multi-turn trajectory
    that penalty is paid on every turn.

    Returns ``(response_ids, response_mask, response_logprobs, last TokenOutput)``.
    """
    total_budget = response_length if max_new_tokens is None else min(max_new_tokens, response_length)
    budget = min(think_budget, total_budget)
    rid = request_id or uuid4().hex
    out1: TokenOutput = await server_manager.generate(
        request_id=rid,
        prompt_ids=prompt_ids,
        sampling_params={**sampling_params, "max_tokens": budget},
        **gen_kwargs,
    )
    ids = list(out1.token_ids)
    mask = [1] * len(ids)
    logprobs = list(out1.log_probs) if out1.log_probs else None

    # Finished (EOS) before the budget, or aborted: nothing to force.
    if len(ids) < budget or out1.stop_reason == "aborted":
        return ids, mask, logprobs, out1

    # Budget hit with the reasoning channel still open -> force it closed.
    if "</think>" not in tokenizer.decode(ids):
        ids += think_close_ids
        mask += [0] * len(think_close_ids)
        if logprobs is not None:
            logprobs += [0.0] * len(think_close_ids)

    remaining = total_budget - len(ids)
    if remaining <= 0:
        return ids, mask, logprobs, out1

    # Phase 2: decode the answer, continuing from prompt + phase-1 tokens.
    out2: TokenOutput = await server_manager.generate(
        request_id=rid,
        prompt_ids=prompt_ids + ids,
        sampling_params={**sampling_params, "max_tokens": remaining},
        **gen_kwargs,
    )
    ids += list(out2.token_ids)
    mask += [1] * len(out2.token_ids)
    if logprobs is not None:
        logprobs += list(out2.log_probs) if out2.log_probs else [0.0] * len(out2.token_ids)
    merged_extra = {**(out1.extra_fields or {}), **(out2.extra_fields or {})}
    out2.extra_fields = merged_extra
    if out2.num_preempted is not None or out1.num_preempted is not None:
        out2.num_preempted = (out1.num_preempted or 0) + (out2.num_preempted or 0)
    return ids, mask, logprobs, out2
