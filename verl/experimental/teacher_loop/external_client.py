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
"""HTTP client that lets the on-policy distillation teacher live outside the Ray cluster.

The Ray-actor-based `LLMServerClient` is replaced with an OpenAI-compatible
HTTP client that POSTs to `/v1/completions` with vLLM's `prompt_logprobs`
extension. Output is shaped to match `extract_prompt_logprobs` in
`vllm_rollout/utils.py` so the rest of the distillation pipeline is unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

import aiohttp

from verl.workers.config import DistillationTeacherModelConfig
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_DEFAULT_TIMEOUT_S = float(os.environ.get("VERL_EXTERNAL_TEACHER_TIMEOUT", "600"))
_DEFAULT_RETRIES = int(os.environ.get("VERL_EXTERNAL_TEACHER_RETRIES", "4"))


class ExternalLLMServerClient:
    """OpenAI-compatible HTTP client for the distillation teacher.

    The remote endpoint must be a vLLM server (or other implementation) that
    accepts the `prompt_logprobs` field on `/v1/completions`. The returned
    `TokenOutput.extra_fields` has the same `prompt_ids` / `prompt_logprobs`
    shape that `compute_teacher_logprobs_single` expects: `(S, K)` where
    `S = len(sequence_ids)` and `K = max(num_prompt_logprobs, 1)`.
    """

    def __init__(self, teacher_model_config: DistillationTeacherModelConfig):
        if not teacher_model_config.is_external:
            raise ValueError("ExternalLLMServerClient requires external_url to be set on teacher config.")
        self._base_url = teacher_model_config.external_url.rstrip("/")
        self._api_key = teacher_model_config.external_api_key or "EMPTY"
        self._model = teacher_model_config.external_model_name or teacher_model_config.model_path
        if not self._model:
            raise ValueError(
                "External teacher needs either external_model_name or model_path so we know what 'model' "
                "field to send."
            )
        # One reusable session per client; lazily created on first call so we can stay
        # in whatever event loop the trainer ends up using.
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=_DEFAULT_TIMEOUT_S)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        if image_data or video_data:
            # vLLM's OpenAI /v1/completions accepts only text/token-ids. Our teacher is
            # served with --language-model-only so it couldn't use the visual embeddings
            # anyway. Drop the media here and let the teacher score the response tokens
            # against the text-only prefix (image placeholder tokens stay in the prompt
            # but are treated as regular text). The distillation signal for samples with
            # actual images is therefore approximate; warn once per request so we can
            # see how often it fires.
            logger.warning(
                "ExternalLLMServerClient dropping multimodal data for teacher logprobs "
                "(text-only teacher); req=%s images=%d videos=%d",
                request_id,
                len(image_data) if image_data else 0,
                len(video_data) if video_data else 0,
            )
        num_prompt_logprobs = int(sampling_params.get("prompt_logprobs", 0) or 0)
        payload = {
            "model": self._model,
            "prompt": list(prompt_ids),
            "max_tokens": int(sampling_params.get("max_tokens", 1)),
            "temperature": float(sampling_params.get("temperature", 1.0)),
            "prompt_logprobs": num_prompt_logprobs,
        }
        url = f"{self._base_url}/completions"
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

        body = await self._post_with_retry(url, headers, payload, request_id)
        prompt_logprobs_raw = body["choices"][0].get("prompt_logprobs")
        if prompt_logprobs_raw is None:
            raise RuntimeError(
                f"External teacher at {self._base_url} did not return prompt_logprobs; "
                f"is the server vLLM with prompt_logprobs enabled?"
            )
        prompt_ids_ls, prompt_logprobs_ls = _normalize_prompt_logprobs(
            prompt_logprobs_raw, num_prompt_logprobs
        )
        assert len(prompt_ids_ls) == len(prompt_ids), (
            f"External teacher returned {len(prompt_ids_ls)} logprob positions for a "
            f"prompt of length {len(prompt_ids)}; mismatch likely from a different tokenizer."
        )
        return TokenOutput(
            token_ids=[],
            log_probs=None,
            extra_fields={"prompt_ids": prompt_ids_ls, "prompt_logprobs": prompt_logprobs_ls},
        )

    async def _post_with_retry(self, url, headers, payload, request_id):
        backoff = 5.0
        last_exc: Optional[BaseException] = None
        for attempt in range(_DEFAULT_RETRIES + 1):
            try:
                session = await self._get_session()
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status >= 500 or resp.status == 429:
                        text = await resp.text()
                        raise RuntimeError(
                            f"External teacher HTTP {resp.status} on {url} (req={request_id}): {text[:200]}"
                        )
                    if resp.status >= 400:
                        text = await resp.text()
                        raise RuntimeError(
                            f"External teacher HTTP {resp.status} on {url} (req={request_id}): {text[:200]}"
                        )
                    return await resp.json()
            except (asyncio.TimeoutError, aiohttp.ClientError, RuntimeError) as exc:
                last_exc = exc
                if attempt == _DEFAULT_RETRIES:
                    raise
                # Don't retry 4xx
                if isinstance(exc, RuntimeError) and "HTTP 4" in str(exc) and "HTTP 429" not in str(exc):
                    raise
                logger.warning(
                    "External teacher call %s failed (attempt %d/%d): %s — retrying in %.1fs",
                    request_id,
                    attempt + 1,
                    _DEFAULT_RETRIES + 1,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff *= 3
        # Unreachable but keep mypy happy
        raise last_exc  # type: ignore[misc]

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


def _normalize_prompt_logprobs(
    prompt_logprobs_raw: list, num_prompt_logprobs: int
) -> tuple[list[list[int]], list[list[float]]]:
    """Reshape vLLM's prompt_logprobs response into ``(S, K)`` lists.

    Mirrors `verl.workers.rollout.vllm_rollout.utils.extract_prompt_logprobs` so the
    downstream `compute_teacher_logprobs_single` consumer sees the same layout
    regardless of whether the teacher is colocated or external.
    """
    K = max(num_prompt_logprobs, 1)
    prompt_ids_ls: list[list[int]] = []
    prompt_logprobs_ls: list[list[float]] = []

    # vLLM convention: logprob of the first prompt token is None.
    for entry in prompt_logprobs_raw[1:]:
        if entry is None:
            prompt_ids_ls.append([0] * K)
            prompt_logprobs_ls.append([0.0] * K)
            continue
        if num_prompt_logprobs == 0:
            token_id_str, info = next(iter(entry.items()))
            logprob = _extract_logprob(info)
            prompt_ids_ls.append([int(token_id_str)])
            prompt_logprobs_ls.append([logprob])
            continue
        ids_slot: list[Optional[int]] = [None] * num_prompt_logprobs
        lp_slot: list[Optional[float]] = [None] * num_prompt_logprobs
        if len(entry) not in (num_prompt_logprobs, num_prompt_logprobs + 1):
            raise RuntimeError(
                f"External teacher returned {len(entry)} entries at a position but expected "
                f"{num_prompt_logprobs} or {num_prompt_logprobs + 1}."
            )
        for token_id_str, info in entry.items():
            rank = _extract_rank(info)
            if rank is None or rank > num_prompt_logprobs:
                continue  # sampled token outside top-k
            ids_slot[rank - 1] = int(token_id_str)
            lp_slot[rank - 1] = _extract_logprob(info)
        # Replace any leftover Nones (shouldn't happen but stay defensive)
        for i in range(num_prompt_logprobs):
            if ids_slot[i] is None:
                ids_slot[i] = 0
                lp_slot[i] = 0.0
        prompt_ids_ls.append(ids_slot)  # type: ignore[arg-type]
        prompt_logprobs_ls.append(lp_slot)  # type: ignore[arg-type]

    # Pad a dummy slot for the last prompt token (matches internal teacher behavior).
    prompt_ids_ls.append([0] * K)
    prompt_logprobs_ls.append([0.0] * K)
    return prompt_ids_ls, prompt_logprobs_ls


def _extract_logprob(info: Any) -> float:
    if isinstance(info, dict):
        return float(info["logprob"])
    if isinstance(info, (int, float)):
        return float(info)
    return float(getattr(info, "logprob"))


def _extract_rank(info: Any) -> Optional[int]:
    if isinstance(info, dict):
        rank = info.get("rank")
    else:
        rank = getattr(info, "rank", None)
    return int(rank) if rank is not None else None
