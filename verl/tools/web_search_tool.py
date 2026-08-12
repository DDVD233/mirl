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
"""Live web search tool for the self-evolving HealthBench RL runs.

The rollout policy calls ``web_search`` with ONE query; this tool forwards it to
the Serper cache service (scripts/self_evolving/kb/serper_cache_server.py),
which returns verbatim SERP blocks -- title, link, snippet exactly as the search
API produced them. Unlike the GPT-mediated web evidence of the lookup arm, no
model anywhere on this path sees the case or composes the evidence: what the
policy reads is what Google returned for the query the policy itself wrote.
That makes both the search behaviour and the reading of raw results part of
what the policy trains.
"""

import logging
import os
from typing import Any, Optional
from uuid import uuid4

import httpx

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class WebSearchTool(BaseTool):
    """Stateless tool backed by the Serper cache service's ``/search`` route.

    Config keys (all optional):
        web_search_url: full URL of the /search endpoint. Falls back to the
            ``WEB_SEARCH_URL`` env var, then ``http://localhost:8056/search``.
        num_results: results requested per query (server default if unset).
        timeout: per-request timeout in seconds. Like the retrieval tool's, it
            must cover time QUEUED behind the cache service's fetch semaphore
            when a whole training batch misses at once, not just one call.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.web_search_url = (
            config.get("web_search_url")
            or os.getenv("WEB_SEARCH_URL")
            or "http://localhost:8056/search"
        )
        self.num_results = config.get("num_results")
        self.timeout = float(config.get("timeout", 120))

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        return instance_id or str(uuid4()), ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        # The metrics dict keys mirror MedicalRetrievalTool's exactly: the agent
        # loop aggregates both tools' telemetry through the same keys, and
        # ``retrieval_text`` is what reaches the reward's coverage grader.
        query = str(parameters.get("query") or "").strip()
        if not query:
            return (
                ToolResponse(text="No search query provided. Provide one specific English query."),
                0.0,
                {"retrieval_hits": 0, "retrieval_error": 1, "queries": [], "retrieval_text": ""},
            )
        body: dict[str, Any] = {"query": query}
        if self.num_results:
            body["num"] = int(self.num_results)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.web_search_url, json=body)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:  # a failed search must never crash a rollout
            logger.warning(f"web search failed for {query[:60]!r}: {type(e).__name__}: {e!r}")
            return (
                ToolResponse(text="Web search is temporarily unavailable; answer from your own knowledge."),
                0.0,
                {"retrieval_hits": 0, "retrieval_error": 1, "queries": [query], "retrieval_text": ""},
            )
        text = data.get("text") or f'No web results for "{query}".'
        return (
            ToolResponse(text=text),
            0.0,
            {
                "retrieval_hits": int(data.get("hits") or 0),
                "retrieval_error": 1 if data.get("error") else 0,
                "queries": [query],
                "retrieval_text": text,
            },
        )
